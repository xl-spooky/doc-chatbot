"""Main application window for the Doc Chatbot UI.

This module defines the `App` class which provides the Tkinter GUI for
managing documents, building/updating the retrieval index, and chatting with
an LLM over your local document collection.
"""

from __future__ import annotations

import contextlib
import os
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import TYPE_CHECKING

import numpy as np

from app.config import load_config
from app.watcher import DocsWatcher
from rag import (
    build_or_update_index,
    chat_answer,
    ensure_dirs,
    list_docs,
    load_index_cached,
    paths,
)

from .dialogs import SettingsDialog
from .docs_manager import DocsManager

if TYPE_CHECKING:
    from rag import IndexItem


class App(tk.Tk):
    """Main application window for interacting with the Doc Chatbot."""

    def __init__(self) -> None:
        """Initialize the UI, client, cached index, and file watcher.

        Builds widgets, reads the API key, initializes the OpenAI client
        when available, attempts to load a cached index, and starts a
        background watcher to reindex when documents change.
        """
        super().__init__()
        self.title("Doc Chatbot")
        self.geometry("820x600")
        self.minsize(720, 500)

        ensure_dirs()

        # Local mode only; no remote client needed
        self.items: list[IndexItem] = []
        self.mat: np.ndarray | None = None
        self.config_data = load_config()
        self._indexing = False
        self._pending_reindex = False
        self._reindex_timer: threading.Timer | None = None
        self._watcher = DocsWatcher(self._schedule_reindex_debounced)
        self._force_full_next = False

        self._build_ui()

        # No API key or remote client required in local mode

        # Try to load existing index, then auto-index if docs exist
        self._load_index()
        try:
            has_docs = bool(list_docs())
            if has_docs:
                self._reindex_background(tag="Indexing documents...")
        except Exception:
            pass

        # Start watcher for live updates
        if self._watcher.start():
            self._set_status("Ready (watching docs for changes)")
        else:
            self._set_status("Ready (watching disabled: install watchdog)")

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        """Create and lay out widgets for the main window."""
        top = tk.Frame(self)
        top.pack(side=tk.TOP, fill=tk.X, padx=10, pady=8)

        self.btn_reindex = tk.Button(top, text="Rebuild Index", command=self.on_reindex)
        self.btn_reindex.pack(side=tk.LEFT)

        self.btn_open_docs = tk.Button(top, text="Open Docs Folder", command=self.on_open_docs)
        self.btn_open_docs.pack(side=tk.LEFT, padx=(8, 0))

        self.btn_manage_docs = tk.Button(top, text="Manage Docs", command=self.on_manage_docs)
        self.btn_manage_docs.pack(side=tk.LEFT, padx=(8, 0))

        # No API key needed in local mode

        self.btn_clear = tk.Button(top, text="Clear Chat", command=self.on_clear)
        self.btn_clear.pack(side=tk.LEFT, padx=(8, 0))

        self.btn_settings = tk.Button(top, text="Settings", command=self.on_settings)
        self.btn_settings.pack(side=tk.RIGHT)

        body = tk.Frame(self)
        body.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=4)
        self.txt_chat = tk.Text(body, wrap=tk.WORD, state=tk.DISABLED)
        self.txt_chat.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        bottom = tk.Frame(self)
        bottom.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=8)
        self.entry = tk.Entry(bottom)
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.entry.bind('<Return>', lambda e: self.on_send())
        self.btn_send = tk.Button(bottom, text="Send", command=self.on_send)
        self.btn_send.pack(side=tk.LEFT, padx=(8, 0))

        self.status = tk.StringVar(value="Ready")
        status_frame = tk.Frame(self)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=(0, 6))
        self.progress = ttk.Progressbar(status_frame, orient=tk.HORIZONTAL, mode="determinate")
        self.progress.pack(side=tk.RIGHT, fill=tk.X, expand=False, padx=(8, 0))
        bar = tk.Label(status_frame, textvariable=self.status, anchor="w")
        bar.pack(side=tk.LEFT, fill=tk.X, expand=True)

    def _set_status(self, msg: str) -> None:
        """Update the status bar text.

        Args:
            msg: Message to display in the footer.
        """
        self.status.set(msg)
        self.update_idletasks()

    def _set_progress(self, cur: int, total: int) -> None:
        """Update the progress bar given the current and total values.

        Args:
            cur: Current progress value.
            total: Total value used to compute percentage; non-positive clears.
        """
        try:
            if total <= 0:
                self.progress['value'] = 0
                self.progress.update_idletasks()
                return
            pct = max(0, min(100, int(cur * 100 / total)))
            self.progress['value'] = pct
            self.progress.update_idletasks()
        except Exception:
            pass

    def on_manage_docs(self) -> None:
        """Open the documents manager and reindex when changes occur."""
        try:
            DocsManager(
                self, on_change=lambda: self._reindex_background(tag="Docs changed: reindexing...")
            )
        except Exception as e:
            messagebox.showerror("Docs", f"Failed to open manager: {e}")

    def on_settings(self) -> None:
        """Open the settings dialog populated from the current config."""
        try:
            SettingsDialog(self, self.config_data, self._on_settings_saved)
        except Exception as e:
            messagebox.showerror("Settings", f"Failed to open settings: {e}")

    def _on_settings_saved(self, new_cfg: dict) -> None:
        """Persist updated settings and rebuild index if needed.

        Triggers a full rebuild when embedding model, chunk size, or overlap
        change; otherwise just updates the status.

        Args:
            new_cfg: The updated configuration mapping.
        """
        old = dict(self.config_data)
        self.config_data.update(new_cfg)
        if (
            old.get("embed_model") != self.config_data.get("embed_model")
            or int(old.get("chunk_chars", 1200)) != int(self.config_data.get("chunk_chars", 1200))
            or int(old.get("overlap", 200)) != int(self.config_data.get("overlap", 200))
        ):
            self._force_full_next = True
            self._reindex_background(tag="Settings changed: rebuilding index...")
        else:
            self._set_status("Settings saved.")

    def _append_chat(self, role: str, text: str) -> None:
        """Append a message to the transcript.

        Args:
            role: Label for the message author (e.g., "You", "Assistant").
            text: Message body to display.
        """
        self.txt_chat.configure(state=tk.NORMAL)
        self.txt_chat.insert(tk.END, f"{role}:\n{text}\n\n")
        self.txt_chat.see(tk.END)
        self.txt_chat.configure(state=tk.DISABLED)

    # No remote client initialization required

    def _load_index(self) -> None:
        """Load the cached index into memory and update status."""
        try:
            self.items, self.mat = load_index_cached()
            if self.items:
                self._set_status(f"Index loaded ({len(self.items)} chunks)")
            else:
                self._set_status("No index yet. Put PDFs in docs/ and click Rebuild Index.")
        except Exception as e:
            messagebox.showerror("Index", f"Failed to load index: {e}")

    def _schedule_reindex_debounced(self, delay: float = 1.0) -> None:
        """Debounce scheduling of a background reindex operation.

        Args:
            delay: Timer delay in seconds before triggering a rebuild.
        """

        def fire() -> None:
            """Timer callback that kicks off background reindexing."""
            self._reindex_timer = None
            self._reindex_background(tag="Detected changes: reindexing...")

        if self._reindex_timer is not None:
            with contextlib.suppress(Exception):
                self._reindex_timer.cancel()
        self._reindex_timer = threading.Timer(delay, lambda: self.after(0, fire))
        self._reindex_timer.daemon = True
        self._reindex_timer.start()

    def on_open_docs(self) -> None:
        """Open the documents directory in the system file explorer."""
        d = paths()["docs"]
        os.makedirs(d, exist_ok=True)
        try:
            os.startfile(d)
        except Exception:
            messagebox.showinfo("Docs", f"Folder: {d}")

    def on_reload_key(self) -> None:
        """Reload the API key from disk and reinitialize the client."""
        self.api_key = read_api_key()
        if not self.api_key:
            messagebox.showwarning("API Key", "api_key.txt missing or empty.")
            return
        self._init_client()
        self._set_status("API key loaded.")

    def on_clear(self) -> None:
        """Clear all text from the chat transcript widget."""
        self.txt_chat.configure(state=tk.NORMAL)
        self.txt_chat.delete("1.0", tk.END)
        self.txt_chat.configure(state=tk.DISABLED)

    def on_reindex(self) -> None:
        """Kick off a background index rebuild if a client is ready."""
        provider = str(self.config_data.get("provider", "local"))
        if provider == "openai" and not self.client:
            messagebox.showwarning("OpenAI", "Load API key first.")
            return
        self._reindex_background()

    def _reindex_background(self, tag: str | None = None) -> None:
        """Rebuild or update the index in a background thread.

        Args:
            tag: Optional status message to display while indexing.
        """
        if self._indexing:
            self._pending_reindex = True
            return
        self._indexing = True
        if tag:
            self._set_status(tag)

        def worker() -> None:
            """Background worker that builds/updates the vector index."""
            try:

                def cb(msg: str) -> None:
                    """Status callback used by the indexing routine."""
                    self.after(0, lambda: self._set_status(msg))

                def pcb(stage: str, cur: int, total: int) -> None:
                    """Progress callback used by the indexing routine."""
                    self.after(0, lambda: self._set_progress(cur, total))

                cfg = self.config_data
                items, mat = build_or_update_index(
                    status_cb=cb,
                    progress_cb=pcb,
                    embed_model_local=str(cfg.get("embed_model_local", "BAAI/bge-small-en-v1.5")),
                    chunk_chars=int(cfg.get("chunk_chars", 1200)),
                    overlap=int(cfg.get("overlap", 200)),
                    force_full=self._force_full_next,
                )
                self.items, self.mat = items, mat
            except Exception as e:
                # Don't close over the exception variable; it is cleared after the except block.
                msg = str(e)
                self.after(0, lambda: messagebox.showerror("Index", msg))
            finally:

                def done() -> None:
                    """Finalize UI state after indexing completes."""
                    self._indexing = False
                    self._force_full_next = False
                    self._set_progress(0, 0)
                    if self._pending_reindex:
                        self._pending_reindex = False
                        self._reindex_background(tag="Detected changes: reindexing...")
                    else:
                        self._set_status(f"Index ready ({len(self.items)} chunks)")

                self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def on_send(self) -> None:
        """Send the user's query, run RAG, and display the answer."""
        q = self.entry.get().strip()
        if not q:
            return
        self.entry.delete(0, tk.END)
        self._append_chat("You", q)

        # Local mode only; no remote client required
        if not self.items:
            if messagebox.askyesno("No Index", "No index found. Rebuild now?"):
                self.on_reindex()
            return

        self._set_status("Thinking...")

        def worker() -> None:
            """Background worker that generates an answer for the query."""
            try:
                cfg = self.config_data
                mat = self.mat
                if mat is None:
                    self.after(0, lambda: messagebox.showwarning("Chat", "Index not ready."))
                    self.after(0, lambda: self._set_status("Ready"))
                    return
                answer, sources = chat_answer(
                    q,
                    self.items,
                    mat,
                    embed_model_local=str(cfg.get("embed_model_local", "BAAI/bge-small-en-v1.5")),
                    top_k=int(cfg.get("top_k", 4)),
                    mmr=bool(cfg.get("mmr", True)),
                    mmr_lambda=float(cfg.get("mmr_lambda", 0.5)),
                    hybrid=bool(cfg.get("hybrid", False)),
                    hybrid_weight=float(cfg.get("hybrid_weight", 0.5)),
                )
                txt = answer
                if sources:
                    txt += "\n\nSources:\n" + "\n".join(sources)
                self.after(0, lambda: self._append_chat("Assistant", txt))
                self.after(0, lambda: self._set_status("Ready"))
            except Exception as e:
                msg = str(e)
                self.after(0, lambda: messagebox.showerror("Chat", msg))
                self.after(0, lambda: self._set_status("Ready"))

        threading.Thread(target=worker, daemon=True).start()

    def _on_close(self) -> None:
        """Stop timers and watchers, then close the window."""
        try:
            if self._reindex_timer is not None:
                self._reindex_timer.cancel()
        except Exception:
            pass
        with contextlib.suppress(Exception):
            self._watcher.stop()
        self.destroy()

    def _ensure_sample_pdf_if_missing(self) -> bool:
        """Ensure the docs folder exists; no sample is created.

        Returns:
            bool: Always ``False`` to indicate no sample was created.
        """
        # No-op: samples are not auto-created; users add their own docs.
        os.makedirs(paths()["docs"], exist_ok=True)
        return False

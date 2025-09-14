"""Documents manager window for the UI.

Provides a simple list view to refresh, delete, and open documents.
"""

from __future__ import annotations

import contextlib
import os
import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox

from rag import list_docs, paths


class DocsManager(tk.Toplevel):
    """Window for listing, deleting, and opening documents."""

    def __init__(self, master: tk.Tk, on_change: Callable[[], None] | None) -> None:
        """Initialize the manager and populate the document list.

        Args:
            master: Parent window.
            on_change: Callback invoked after changes that affect the index.
        """
        super().__init__(master)
        self.title("Manage Documents")
        self.resizable(True, True)
        self.on_change = on_change
        self.docs_dir = paths()["docs"]

        frm = tk.Frame(self)
        frm.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        self.listbox = tk.Listbox(frm, selectmode=tk.EXTENDED)
        self.listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = tk.Scrollbar(frm, orient=tk.VERTICAL, command=self.listbox.yview)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.listbox.configure(yscrollcommand=scroll.set)

        btns = tk.Frame(self)
        btns.pack(fill=tk.X, padx=12, pady=(0, 12))
        tk.Button(btns, text="Refresh", command=self._refresh).pack(side=tk.LEFT)
        tk.Button(btns, text="Delete Selected", command=self._delete_selected).pack(
            side=tk.LEFT, padx=(8, 0)
        )
        tk.Button(btns, text="Open Folder", command=self._open_folder).pack(side=tk.RIGHT)

        self._refresh()
        self.grab_set()
        self.transient(master)

    def _refresh(self) -> None:
        """Refresh the document list from the docs directory."""
        self.listbox.delete(0, tk.END)
        docs = list_docs()
        base = self.docs_dir
        for p in docs:
            try:
                rel = os.path.relpath(p, base)
            except Exception:
                rel = p
            self.listbox.insert(tk.END, rel)

    def _delete_selected(self) -> None:
        """Delete selected files and notify ``on_change`` if provided."""
        sel = list(self.listbox.curselection())
        if not sel:
            return
        items = [self.listbox.get(i) for i in sel]
        if not messagebox.askyesno("Delete", f"Delete {len(items)} file(s)?"):
            return
        errors = 0
        for rel in items:
            p = os.path.join(self.docs_dir, rel)
            try:
                os.remove(p)
            except Exception:
                errors += 1
        self._refresh()
        if errors:
            messagebox.showwarning("Delete", f"Some files could not be deleted ({errors}).")
        if callable(self.on_change):
            with contextlib.suppress(Exception):
                self.on_change()

    def _open_folder(self) -> None:
        """Open the docs directory in the system file explorer."""
        try:
            os.startfile(self.docs_dir)
        except Exception:
            messagebox.showinfo("Docs", f"Folder: {self.docs_dir}")

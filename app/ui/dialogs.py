"""Dialogs for editing application settings."""

from __future__ import annotations

import tkinter as tk
from collections.abc import Callable
from tkinter import messagebox
from typing import Any

from app.config import save_config


class SettingsDialog(tk.Toplevel):
    """Modal dialog for editing model and retrieval settings."""

    def __init__(
        self, master: tk.Tk, cfg: dict[str, Any], on_save: Callable[[dict[str, Any]], None]
    ) -> None:
        """Create the settings dialog and populate fields from ``cfg``.

        Args:
            master: Parent window.
            cfg: Current configuration mapping.
            on_save: Callback invoked with the updated configuration.
        """
        super().__init__(master)
        self.title("Settings")
        self.resizable(False, False)
        self.on_save = on_save

        frm = tk.Frame(self)
        frm.pack(padx=12, pady=12)

        def add_row(row: int, label: str, widget: tk.Widget) -> None:
            """Add a labeled widget row to the settings grid."""
            tk.Label(frm, text=label, anchor="w").grid(row=row, column=0, sticky="w", pady=4)
            widget.grid(row=row, column=1, sticky="ew", pady=4)

        self.var_chat = tk.StringVar(value=str(cfg.get("chat_model", "gpt-4o-mini")))
        self.var_embed = tk.StringVar(value=str(cfg.get("embed_model", "text-embedding-3-small")))
        self.var_topk = tk.StringVar(value=str(int(cfg.get("top_k", 4))))
        self.var_chunk = tk.StringVar(value=str(int(cfg.get("chunk_chars", 1200))))
        self.var_overlap = tk.StringVar(value=str(int(cfg.get("overlap", 200))))
        self.var_mmr = tk.BooleanVar(value=bool(cfg.get("mmr", True)))
        self.var_lambda = tk.StringVar(value=str(float(cfg.get("mmr_lambda", 0.5))))

        e_chat = tk.Entry(frm, textvariable=self.var_chat, width=30)
        e_embed = tk.Entry(frm, textvariable=self.var_embed, width=30)
        e_topk = tk.Entry(frm, textvariable=self.var_topk, width=10)
        e_chunk = tk.Entry(frm, textvariable=self.var_chunk, width=10)
        e_overlap = tk.Entry(frm, textvariable=self.var_overlap, width=10)
        e_lambda = tk.Entry(frm, textvariable=self.var_lambda, width=10)
        c_mmr = tk.Checkbutton(frm, text="Use MMR (diversified retrieval)", variable=self.var_mmr)

        add_row(0, "Chat model", e_chat)
        add_row(1, "Embedding model", e_embed)
        add_row(2, "Top K", e_topk)
        add_row(3, "Chunk size", e_chunk)
        add_row(4, "Overlap", e_overlap)
        add_row(5, "MMR lambda (0..1)", e_lambda)
        c_mmr.grid(row=6, column=0, columnspan=2, sticky="w", pady=(4, 8))

        btns = tk.Frame(self)
        btns.pack(padx=12, pady=(0, 12), fill=tk.X)
        tk.Button(btns, text="Cancel", command=self.destroy).pack(side=tk.RIGHT)
        tk.Button(btns, text="Save", command=self._save).pack(side=tk.RIGHT, padx=(0, 8))

        self.grab_set()
        self.transient(master)

    def _save(self) -> None:
        """Validate inputs, persist config, and notify the caller."""
        try:
            new_cfg = {
                "chat_model": self.var_chat.get().strip(),
                "embed_model": self.var_embed.get().strip(),
                "top_k": int(self.var_topk.get().strip()),
                "chunk_chars": int(self.var_chunk.get().strip()),
                "overlap": int(self.var_overlap.get().strip()),
                "mmr": bool(self.var_mmr.get()),
                "mmr_lambda": float(self.var_lambda.get().strip()),
            }
            if new_cfg["top_k"] <= 0:
                new_cfg["top_k"] = 1
            if new_cfg["chunk_chars"] < 200:
                new_cfg["chunk_chars"] = 200
            if new_cfg["overlap"] < 0:
                new_cfg["overlap"] = 0
            if new_cfg["mmr_lambda"] < 0.0:
                new_cfg["mmr_lambda"] = 0.0
            if new_cfg["mmr_lambda"] > 1.0:
                new_cfg["mmr_lambda"] = 1.0
        except Exception as e:
            messagebox.showerror("Settings", f"Invalid value: {e}")
            return
        save_config(new_cfg)
        try:
            self.on_save(new_cfg)
        finally:
            self.destroy()

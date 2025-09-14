"""Filesystem watcher for the docs directory.

This module optionally uses ``watchdog`` to observe changes in the documents folder and
invokes a provided callback when files with supported extensions are modified,
created, or deleted. The watcher is used to trigger background reindexing.
"""

import os
from collections.abc import Callable
from typing import Any

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from rag import SUPPORTED_EXTS, paths


class DocsWatcher:
    """Watches the docs directory and triggers a callback on changes.

    The watcher monitors only files that end with one of ``SUPPORTED_EXTS``.
    """

    def __init__(self, on_change: Callable[[], None]):
        """Initialize the watcher.

        Args:
            on_change: Callback to invoke when a relevant change is observed.
        """
        # Keep observer as Any to satisfy static analysis without importing
        # watchdog at module import time. watchdog is imported lazily in start().
        self._observer: Any | None = None
        self._on_change = on_change

    def start(self) -> bool:
        """Start observing the docs directory.

        Ensures the directory exists, installs a local event handler, and
        starts a background ``Observer``.

        Returns:
            bool: ``True`` if the observer started successfully.
        """
        # Import watchdog lazily so the app can run without it installed.
        docs_dir = paths()["docs"]
        os.makedirs(docs_dir, exist_ok=True)

        class Handler(FileSystemEventHandler):
            """Internal watchdog handler that filters and forwards events."""

            def __init__(self, cb: Callable[[], None]) -> None:
                """Store callback for later invocation.

                Args:
                    cb: Callback to call when a matching file event occurs.
                """
                super().__init__()
                self._cb: Callable[[], None] = cb

            def on_any_event(self, event: FileSystemEvent) -> None:
                """Forward file events for supported extensions to the callback."""
                if event.is_directory:
                    return
                p = (getattr(event, 'src_path', '') or getattr(event, 'dest_path', '')).lower()
                for ext in SUPPORTED_EXTS:
                    if p.endswith(ext):
                        self._cb()
                        break

        handler = Handler(self._on_change)
        self._observer = Observer()
        self._observer.schedule(handler, docs_dir, recursive=True)
        self._observer.start()
        return True

    def stop(self):
        """Stop the observer and release resources."""
        try:
            if self._observer is not None:
                self._observer.stop()
                self._observer.join(timeout=1.5)
        finally:
            self._observer = None

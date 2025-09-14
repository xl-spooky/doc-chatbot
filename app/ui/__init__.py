"""UI package for the Doc Chatbot application.

Exposes the main application window and related dialogs.
"""

from .dialogs import SettingsDialog
from .docs_manager import DocsManager
from .main_window import App

__all__ = [
    "App",
    "SettingsDialog",
    "DocsManager",
]

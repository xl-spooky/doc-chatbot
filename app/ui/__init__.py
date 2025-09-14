"""UI package for the Doc Chatbot application.

Exposes the main application window and related dialogs.
"""

from .main_window import App
from .dialogs import SettingsDialog
from .docs_manager import DocsManager

__all__ = [
    "App",
    "SettingsDialog",
    "DocsManager",
]


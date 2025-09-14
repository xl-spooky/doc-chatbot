"""Build a Windows release zip with exe and user-facing files.

This script assumes PyInstaller has already produced `dist/DocChatbot.exe`.
It assembles:

- DocChatbot.exe
- docs/ (empty placeholder folder)
- api_key.txt (empty placeholder file)
- README.txt (usage instructions)

and zips them into `DocChatbot-win64.zip` inside the project root.
"""

from __future__ import annotations

import os
import shutil
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def main() -> None:
    dist_exe = os.path.join(ROOT, "dist", "DocChatbot.exe")
    if not os.path.exists(dist_exe):
        raise SystemExit(
            "Missing dist/DocChatbot.exe. Build it first with PyInstaller "
            "(e.g., `pyinstaller --noconfirm --onefile --name DocChatbot main.py`)."
        )

    stage = os.path.join(ROOT, "release")
    if os.path.isdir(stage):
        shutil.rmtree(stage)
    ensure_dir(stage)

    # Copy executable
    shutil.copy2(dist_exe, os.path.join(stage, "DocChatbot.exe"))

    # Create docs folder
    ensure_dir(os.path.join(stage, "docs"))

    # Create api_key.txt placeholder
    api_path = os.path.join(stage, "api_key.txt")
    if not os.path.exists(api_path):
        with open(api_path, "w", encoding="utf-8") as f:
            f.write("")

    # Create README.txt with instructions
    readme_path = os.path.join(stage, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(
            "DocChatbot – Quick Start\n"
            "\n"
            "1) Put your OpenAI API key in api_key.txt (first line only).\n"
            "2) Place your PDFs (and supported docs) into the docs/ folder.\n"
            "3) Double‑click DocChatbot.exe to run.\n"
            "\n"
            "Supported file types: PDF, DOCX, TXT/MD, CSV, HTML/HTM, PPTX, XLSX, RTF.\n"
            "You can add/remove files in docs/ at any time; the app can reindex.\n"
            "\n"
            "Note: No data leaves your machine except calls to OpenAI APIs for\n"
            "embeddings and chat. Ensure your API key has the proper access.\n"
        )

    # Create zip
    zip_path = os.path.join(ROOT, "DocChatbot-win64.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(stage):
            for fn in files:
                abspath = os.path.join(root, fn)
                rel = os.path.relpath(abspath, stage)
                zf.write(abspath, rel)

    print(f"Created {zip_path}")


if __name__ == "__main__":
    main()


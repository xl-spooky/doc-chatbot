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
    dist_dir = os.path.join(ROOT, "dist", "DocChatbot")
    if not (os.path.exists(dist_exe) or os.path.isdir(dist_dir)):
        raise SystemExit(
            "Missing PyInstaller output. Build it first:"
            " onefile -> dist/DocChatbot.exe or onedir -> dist/DocChatbot/"
        )

    stage = os.path.join(ROOT, "release")
    if os.path.isdir(stage):
        shutil.rmtree(stage)
    ensure_dir(stage)

    # Copy app output (prefer onedir build)
    if os.path.isdir(dist_dir):
        target_app = os.path.join(stage, "DocChatbot")
        shutil.copytree(dist_dir, target_app)
    else:
        target_app = stage
        shutil.copy2(dist_exe, os.path.join(stage, "DocChatbot.exe"))

    # Create docs folder with a placeholder so it appears in the ZIP
    docs_dir = os.path.join(target_app, "docs")
    ensure_dir(docs_dir)
    placeholder = os.path.join(docs_dir, "PUT_YOUR_DOCUMENTS_HERE.txt")
    if not os.path.exists(placeholder):
        with open(placeholder, "w", encoding="utf-8") as f:
            f.write(
                "Place your PDFs and supported documents in this folder.\n"
                "Supported: PDF, DOCX, TXT/MD, CSV, HTML/HTM, PPTX, XLSX, RTF.\n"
            )

    # Create README.txt with instructions
    readme_path = os.path.join(target_app, "README.txt")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(
            "DocChatbot – Quick Start (Local Mode)\n"
            "\n"
            "1) Place your PDFs (and supported docs) into the docs/ folder.\n"
            "2) Double‑click DocChatbot.exe to run.\n"
            "\n"
            "The app indexes your documents locally and answers by selecting the\n"
            "closest relevant chunk. No API keys or network services are used.\n"
            "\n"
            "Supported file types: PDF, DOCX, TXT/MD, CSV, HTML/HTM, PPTX, XLSX, RTF.\n"
            "You can add/remove files in docs/ at any time; the app can reindex.\n"
        )

    # Create zip
    zip_path = os.path.join(ROOT, "DocChatbot-win64.zip")
    if os.path.exists(zip_path):
        os.remove(zip_path)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        base = target_app
        for root, _, files in os.walk(base):
            for fn in files:
                abspath = os.path.join(root, fn)
                rel = os.path.relpath(abspath, base)
                # Place contents under DocChatbot/ inside the ZIP
                rel = os.path.join("DocChatbot", rel)
                zf.write(abspath, rel)

    print(f"Created {zip_path}")


if __name__ == "__main__":
    main()

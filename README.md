Doc Chatbot (Windows .exe)

Runs in a free, offline local mode. Drop your documents in `docs/`, run the app, and it finds the closest matching chunk as the answer.

Overview
- Drop your documents into the `docs` folder.
- Optional: put your OpenAI API key in `api_key.txt` (first line only).
- Run the app (Python or packaged .exe). It indexes your docs and answers using them with citations.
- Supports `.pdf`, `.docx`, `.txt`, `.md`, `.csv`, `.html/.htm`, `.pptx`, `.xlsx`, `.rtf`.

Folder Layout
- `main.py` - minimal entry point
- `app/ui.py` - Tkinter UI (App, Settings dialog, Docs manager)
- `app/config.py` - load/save user settings (models, retrieval, chunking)
- `app/watcher.py` - live folder watcher (auto reindex)
- `rag.py` - indexing, embeddings, retrieval, prompt construction
- `pyproject.toml` - dependencies, Ruff config
- `DocChatbot.spec` - PyInstaller spec used by CI
- `.github/workflows/windows-build.yml` - GitHub Actions workflow building the .exe
- `docs/` - place your docs here (auto-created)
- `storage/` - local index and embeddings (auto-created)
- `api_key.txt` - your OpenAI API key (no quotes/spaces)
- `storage/config.json` - saved settings

For End Users (no Python required)
- Download `DocChatbot-win64.zip` from GitHub Releases (or the latest Actions run artifact) and unzip it.
- Inside you’ll find:
  - `DocChatbot.exe`
  - `docs/` (empty) — put your files here (subfolders are fine)
  - `README.txt` — quick instructions
- Double-click `DocChatbot.exe`. It auto-indexes on startup and watches the folder in real time.
- Ask questions; answers are grounded in your documents with sources.

Build the .exe (single method via GitHub Actions)
- The repo includes `.github/workflows/windows-build.yml` which builds a self‑contained EXE and uploads it.
- Trigger a build:
  - Manually: GitHub → Actions → build-windows-exe → Run workflow
  - Or tag: `git tag v0.1.0 && git push origin v0.1.0` (creates a Release with assets)
- Download:
  - Artifacts: GitHub → Actions → open the latest run → Artifacts → DocChatbot-win
  - Releases (tagged): GitHub → Releases → pick the tag → download `DocChatbot.exe`

Optional local run (developer machine)
- Install Python 3.10–3.12 and Poetry
- `poetry install --no-root`
- Run without building: `poetry run python main.py`

Local mode
- Provider defaults to `local` (free/offline).
- Embeddings use `sentence-transformers` (default: `BAAI/bge-small-en-v1.5`).
- Retrieval returns the closest chunk(s) as the answer, with sources.
- Optional hybrid retrieval (BM25 + embeddings) can be enabled in `storage/config.json`.

OpenAI mode
- Removed — the app runs fully locally without API keys.

Config keys (storage/config.json)
```
{
  "provider": "local",
  "embed_model_local": "BAAI/bge-small-en-v1.5",
  // No cloud models are used in local mode
  "top_k": 4,
  "chunk_chars": 1200,
  "overlap": 200,
  "mmr": true,
  "mmr_lambda": 0.5,
  "hybrid": false,
  "hybrid_weight": 0.5
}
```

Packaging notes
- The EXE bundles Python and all dependencies using PyInstaller and `DocChatbot.spec`.
- Hidden imports include watchdog, pdfminer, python-docx, python-pptx (lxml, Pillow), openpyxl, striprtf.
- Actions artifacts include a placeholder `api_key.txt`; end users should replace it with their key.

Usage Tips
- If you change or add files, click "Rebuild Index" to update (or rely on auto-reindex).
- If `api_key.txt` changes, click "Reload API Key".
- Click "Open Docs Folder" to quickly drop files in place.

Live Updates (Auto-Reindex)
- The app watches the `docs/` folder in real time. When you add, modify, rename, or delete files while the app is running, it automatically reindexes after a short delay.
- Status bar shows "Detected changes: reindexing..." during updates.
- If multiple changes happen quickly, the app batches them (debounced) to avoid repeated rebuilding.
- A progress bar indicates embedding progress during indexing.

Troubleshooting
- Index shows 0 chunks: Ensure your PDFs have selectable text. Image-only/scanned PDFs need OCR (not included). Try DOCX/TXT/MD, or export PDFs with text. The app falls back to a second parser (pdfminer) if basic extraction fails.
 - Build artifacts: Actions artifacts expire over time. Prefer tagged Releases for durable downloads.

Default Models
- Chat: `gpt-4o-mini`
- Embeddings: `text-embedding-3-small`

Settings
- Change chat and embedding models.
- Adjust retrieval Top-K and toggle MMR diversification with a lambda value.
- Control chunk size and overlap. Changing embedding model or chunking forces a full rebuild.

Document Manager
- Click "Manage Docs" to see a list of files in `docs/`.
- Delete selected files directly; the app reindexes automatically.

FAQ
- Where does my data go? Locally under `storage/` (JSON index + embeddings). Queries are sent to OpenAI to generate answers.
- How big can docs be? Start with a handful of files. Extremely large files can be slow to index.
- What file types are supported? PDF, DOCX, TXT, MD, CSV, HTML/HTM, PPTX, XLSX, RTF.
- Auto-reindex isn't working? Ensure `watchdog` is installed (included via Poetry). If building your own PyInstaller spec, make sure watchdog is bundled.

Codebase notes
- HTML parsing uses Python's stdlib `html.parser` (no BeautifulSoup).
- Type hints use modern builtins (e.g., `dict[str, ...]`, `X | Y`).
- Ruff enforces formatting and import order; VS Code is configured to format on save.

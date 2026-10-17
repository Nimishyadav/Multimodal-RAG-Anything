import argparse
import os
import pickle
import subprocess
import sys
from pathlib import Path


PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = PROJECT_FOLDER / "data"
STORAGE_FOLDER = PROJECT_FOLDER / "storage"

SUPPORTED_EXTENSIONS = (
    ".pdf",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif",
    ".xlsx", ".xls",
    ".docx",
    ".pptx",
)


def _data_files():
    DATA_FOLDER.mkdir(exist_ok=True)
    return [
        path.name
        for path in sorted(DATA_FOLDER.iterdir(), key=lambda item: item.name.casefold())
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not path.name.startswith("~$")
    ]


def _load_pickle(filename, fallback):
    try:
        with (STORAGE_FOLDER / filename).open("rb") as file:
            return pickle.load(file)
    except (FileNotFoundError, OSError, pickle.PickleError, EOFError):
        return fallback


def _processed_sources():
    source_text = _load_pickle("source_text.pkl", {})
    chunks = _load_pickle("chunks.pkl", [])
    processed = set(source_text)
    processed.update(chunk.get("source") for chunk in chunks if chunk.get("source"))
    return processed


def _has_required_storage():
    required = [
        STORAGE_FOLDER / "chunks.pkl",
        STORAGE_FOLDER / "source_text.pkl",
        STORAGE_FOLDER / "graph.pkl",
        STORAGE_FOLDER / "index.faiss",
    ]
    return all(path.exists() for path in required)


def _run_visualizer(no_browser=False):
    command = [sys.executable, "visualize_graph_fast.py"]
    if no_browser:
        command.append("--no-browser")
    subprocess.run(command, cwd=PROJECT_FOLDER, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Fast ingestion wrapper that reuses already processed storage."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run the full ingestion pipeline for all supported files.",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Do not regenerate graph HTML after the fast check/full ingestion.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open graph HTML after visualization.",
    )
    args = parser.parse_args()

    STORAGE_FOLDER.mkdir(exist_ok=True)
    files = _data_files()

    if not files:
        print("No supported files found in data/.")
        return

    processed = _processed_sources()
    missing = [filename for filename in files if filename not in processed]

    if args.force or not _has_required_storage() or missing:
        if args.force:
            reason = "--force was requested"
            files_to_process = files
            force_reprocess = True
        elif not _has_required_storage():
            reason = "required storage files are missing"
            files_to_process = files
            force_reprocess = True
        else:
            reason = "new/unprocessed files were found"
            files_to_process = missing
            force_reprocess = False

        print(f"Running full ingestion because {reason}.")
        print("Files:")
        for filename in files_to_process:
            print(f"- {filename}")

        from src.ingest_pipeline import run_ingestion

        result = run_ingestion(
            file_names=files_to_process,
            force_reprocess=force_reprocess,
            progress_callback=lambda message: print(message, flush=True),
        )
        print(result)
    else:
        from src.ingest_pipeline import run_ingestion
        # Shared pipeline also checks file hashes and configuration changes.
        print(run_ingestion())

    if not args.no_visualize:
        _run_visualizer(no_browser=args.no_browser)


if __name__ == "__main__":
    main()

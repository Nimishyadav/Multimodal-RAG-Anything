import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

# MinerU is the parser for every project-supported input.
SUPPORTED_EXTENSIONS = (
    ".pdf",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif",
    ".xlsx", ".xls", ".docx", ".pptx",
)
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif")

_IMAGE_PATTERN = re.compile(r"!\[[^]]*\]\([^)]*\)")
_PAGE_PATTERN = re.compile(r"<!--\s*page(?:\s+|:)[^>]*-->", re.IGNORECASE)
_PAGE_NUMBER = re.compile(r"<!--\s*page(?:\s+|:)\s*(\d+)", re.IGNORECASE)


def _mineru_command():
    executable = shutil.which("mineru")
    if executable:
        return [executable]
    return [sys.executable, "-m", "mineru.cli.main"]


def _segments_from_markdown(markdown):
    segments = []
    text_lines = []
    table_lines = []
    page = None
    in_html_table = False

    def append_segment(text, modality):
        segment = {"text": text, "type": modality}
        if page is not None:
            segment["page"] = page
        segments.append(segment)

    def flush_text():
        text = "\n".join(text_lines).strip()
        if text:
            append_segment(text, "TEXT")
        text_lines.clear()

    def flush_table():
        table = "\n".join(table_lines).strip()
        if table:
            append_segment(table, "TABLE")
        table_lines.clear()

    for line in markdown.splitlines():
        if _PAGE_PATTERN.search(line):
            flush_text()
            flush_table()
            number = _PAGE_NUMBER.search(line)
            page = int(number.group(1)) if number else page
            line = _PAGE_PATTERN.sub("", line)
            if not line.strip():
                continue
        if in_html_table or "<table" in line.lower():
            flush_text()
            table_lines.append(line)
            in_html_table = "</table>" not in line.lower()
            if not in_html_table:
                flush_table()
        elif _IMAGE_PATTERN.search(line):
            flush_text()
            flush_table()
            for piece in re.split(r"(!\[[^]]*\]\([^)]*\))", line):
                if piece.strip():
                    append_segment(piece.strip(), "IMAGE" if _IMAGE_PATTERN.fullmatch(piece) else "TEXT")
        elif line.strip().startswith("|") and line.count("|") >= 2:
            flush_text()
            table_lines.append(line)
        else:
            flush_table()
            text_lines.append(line)

    flush_text()
    flush_table()
    return segments


def _page_count(markdown, payload):
    numbers = [int(number) for number in _PAGE_NUMBER.findall(markdown)]
    if numbers:
        return max(numbers)
    count = len(_PAGE_PATTERN.findall(markdown))
    if count:
        return count

    parse_result = payload.get("parse")
    if isinstance(parse_result, dict):
        for key in ("page_count", "pages"):
            value = parse_result.get(key)
            if isinstance(value, int) and value >= 0:
                return value
    return 0


def read_mineru_document(file_path, progress_callback=None):
    """Parse a supported document using the local MinerU server."""
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Document file not found: {path}")

    def report_progress(message):
        print(message, flush=True)
        if progress_callback:
            progress_callback(message)

    tier = os.environ.get("MINERU_TIER", "flash").strip() or "flash"
    try:
        wait_seconds = int(os.environ.get("MINERU_WAIT_SECONDS", "300"))
    except ValueError as exc:
        raise ValueError("MINERU_WAIT_SECONDS must be a whole number.") from exc
    if wait_seconds <= 0:
        raise ValueError("MINERU_WAIT_SECONDS must be greater than zero.")

    command = _mineru_command() + [
        "parse",
        os.fspath(path),
        "--tier", tier,
        "--json",
        "--wait", str(wait_seconds),
    ]
    if path.suffix.lower() == ".pdf":
        command.extend(["--pages", "all"])

    report_progress(
        f"MinerU is parsing {path.name} locally (tier: {tier}; "
        f"wait limit: {wait_seconds}s)..."
    )
    started_at = time.monotonic()
    timeout_seconds = wait_seconds + 30
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        while True:
            elapsed = time.monotonic() - started_at
            remaining = timeout_seconds - elapsed
            if remaining <= 0:
                process.kill()
                process.communicate()
                raise TimeoutError(
                    f"MinerU timed out while parsing {path.name} after "
                    f"{int(elapsed)} seconds. Increase MINERU_WAIT_SECONDS "
                    "or check the local MinerU server."
                )

            try:
                output, details = process.communicate(
                    timeout=min(10, remaining)
                )
                break
            except subprocess.TimeoutExpired:
                elapsed = int(time.monotonic() - started_at)
                report_progress(
                    f"MinerU is still parsing {path.name} "
                    f"({elapsed}s elapsed)..."
                )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "MinerU is not installed in this Python environment. "
            "Install the project requirements and run this project with that interpreter."
        ) from exc

    output = (output or "").strip()
    details = (details or output).strip()
    if process.returncode != 0:
        if "server_not_running" in details.lower():
            raise RuntimeError(
                "The local MinerU server is not running. Start it with "
                "'mineru server start', then retry ingestion."
            )
        if "no module named mineru" in details.lower():
            raise RuntimeError(
                "MinerU is not installed in the Python environment running this project. "
                "Install it with 'python -m pip install -r requirements.txt'."
            )
        raise RuntimeError(f"MinerU failed to parse {path.name}: {details}")

    try:
        payload = json.loads(output)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"MinerU returned invalid JSON while parsing {path.name}: {details}"
        ) from exc

    if not isinstance(payload, dict):
        raise RuntimeError(f"MinerU returned an unexpected response for {path.name}.")
    if payload.get("error"):
        raise RuntimeError(f"MinerU failed to parse {path.name}: {payload['error']}")

    content = payload.get("content")
    markdown = content.get("content") if isinstance(content, dict) else content
    if not isinstance(markdown, str):
        raise RuntimeError(
            f"MinerU returned no Markdown content for {path.name}. "
            "Check that the local MinerU server completed parsing successfully."
        )

    segments = _segments_from_markdown(markdown)
    if not segments:
        raise RuntimeError(f"MinerU extracted no readable content from {path.name}.")

    stats = {
        "pages": _page_count(markdown, payload),
        "tables": sum(segment["type"] == "TABLE" for segment in segments),
        "images": sum(segment["type"] == "IMAGE" for segment in segments),
        "parser": "MinerU",
        "tier": tier,
    }
    report_progress(f"MinerU finished parsing {path.name}.")
    return segments, stats

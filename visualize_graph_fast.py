import argparse
import html
import os
import re
import webbrowser
import hashlib
from pathlib import Path

from src.knowledge_graph import load_graph
from src.pyvis_graph import render_graph


HIDDEN_RELATIONS = {
    "__node__",
    "contains_document",
    "has_graph_component",
    "belongs_to",
    "contains_source",
    "source_contains_entity",
    "contains_entity",
    "mentions",
}


PROJECT_FOLDER = Path(__file__).resolve().parent
DATA_FOLDER = PROJECT_FOLDER / "data"
STORAGE_FOLDER = PROJECT_FOLDER / "storage"
GRAPH_FOLDER = STORAGE_FOLDER / "graphs"

SUPPORTED_EXTENSIONS = (
    ".pdf",
    ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif",
    ".xlsx", ".xls",
    ".docx",
    ".pptx",
)


def _supported_sources():
    DATA_FOLDER.mkdir(exist_ok=True)
    return [
        path.name
        for path in sorted(DATA_FOLDER.iterdir(), key=lambda item: item.name.casefold())
        if path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not path.name.startswith("~$")
    ]


def _safe_filename(source):
    stem = os.path.splitext(os.path.basename(source))[0]
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._")
    suffix = hashlib.sha256(source.encode("utf-8")).hexdigest()[:10]
    return f"{stem or 'source'}_{suffix}"


def _source_triples(triples, source):
    return [
        triple
        for triple in triples
        if triple.get("source") == source
    ]


def _has_visible_triples(triples):
    return any(
        str(triple.get("relation", "")).strip() not in HIDDEN_RELATIONS
        for triple in triples
    )


def _write_selector(source_graphs, output_path):
    options = ['<option value="graph_all.html">All Sources</option>']

    for source, filename in source_graphs:
        options.append(
            f'<option value="graphs/{html.escape(filename)}">'
            f'{html.escape(source)}'
            "</option>"
        )

    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Fast Knowledge Graph View</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body {{
    margin: 0;
    font-family: Arial, sans-serif;
    background: #f7f7f7;
    color: #111827;
}}
header {{
    padding: 14px 20px;
    background: white;
    border-bottom: 1px solid #e5e7eb;
}}
select {{
    min-width: 420px;
    max-width: 90vw;
    padding: 9px 12px;
    font-size: 15px;
}}
iframe {{
    width: 100%;
    height: calc(100vh - 75px);
    border: 0;
    background: white;
}}
</style>
</head>
<body>
<header>
    <label for="graphSelect"><b>Knowledge Graph:</b></label>
    <select id="graphSelect" onchange="changeGraph(this.value)">
        {"".join(options)}
    </select>
</header>
<iframe id="graphFrame" src="graph_all.html"></iframe>
<script>
function changeGraph(path) {{
    document.getElementById("graphFrame").src = path;
}}
</script>
</body>
</html>
"""

    with open(output_path, "w", encoding="utf-8", newline="\n") as file:
        file.write(document)


def main():
    parser = argparse.ArgumentParser(
        description="Fast graph visualizer that only uses saved storage/graph.pkl."
    )
    parser.add_argument(
        "--source",
        help="Generate/open only one source graph.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the selector page automatically.",
    )
    args = parser.parse_args()

    STORAGE_FOLDER.mkdir(exist_ok=True)
    GRAPH_FOLDER.mkdir(exist_ok=True)

    graph_path = STORAGE_FOLDER / "graph.pkl"
    if not graph_path.exists():
        raise SystemExit("storage/graph.pkl not found. Run python ingest.py first.")

    triples = load_graph(os.fspath(graph_path))
    if not triples:
        raise SystemExit("storage/graph.pkl has no relationships to visualize.")

    sources = _supported_sources()

    if args.source:
        if args.source not in sources:
            raise SystemExit(
                "Unknown source.\nAvailable sources:\n- "
                + "\n- ".join(sources)
            )

        filtered = _source_triples(triples, args.source)
        if not filtered or not _has_visible_triples(filtered):
            raise SystemExit(f"No relationships were found for: {args.source}")

        output_path = GRAPH_FOLDER / (_safe_filename(args.source) + ".html")
        render_graph(
            filtered,
            os.fspath(output_path),
            merge_shared_entities=False,
        )
        print(f"Graph saved: {output_path}")

        if not args.no_browser:
            webbrowser.open("file://" + os.path.abspath(output_path))
        return

    all_graph_path = STORAGE_FOLDER / "graph_all.html"
    render_graph(
        triples,
        os.fspath(all_graph_path),
        merge_shared_entities=True,
    )
    print(f"All-source graph saved: {all_graph_path}")

    source_graphs = []
    for source in sources:
        filtered = _source_triples(triples, source)
        if not filtered or not _has_visible_triples(filtered):
            continue

        output_name = _safe_filename(source) + ".html"
        output_path = GRAPH_FOLDER / output_name
        render_graph(
            filtered,
            os.fspath(output_path),
            merge_shared_entities=False,
        )

        source_graphs.append((source, output_name))
        print(f"Individual graph saved: {output_path}")

    selector_path = STORAGE_FOLDER / "graph_selector_fast.html"
    _write_selector(source_graphs, selector_path)

    print()
    print("Open this file to choose All Sources / individual source:")
    print(f"  {selector_path}")

    if not args.no_browser:
        webbrowser.open("file://" + os.path.abspath(selector_path))


if __name__ == "__main__":
    main()

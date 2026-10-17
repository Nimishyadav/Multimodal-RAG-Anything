import os
import pickle
import hashlib
import tempfile
from pathlib import Path

from src.chunker import chunk_segment
from src.document_reader import (
    DOCUMENT_EXTENSIONS,
    graph_triples_from_segments,
    read_document,
)
from src.embedder import create_embeddings
from src.knowledge_graph import build_graph, load_graph, save_graph
from src.vector_store import VectorStore
from src.config import env_flag, TEXT_MODEL_NAME, VISION_MODEL_NAME


PROJECT_FOLDER = Path(__file__).resolve().parent.parent
DATA_FOLDER = PROJECT_FOLDER / "data"
STORAGE_FOLDER = PROJECT_FOLDER / "storage"


def _report(progress_callback, message):
    print(message, flush=True)
    if progress_callback:
        progress_callback(message)


def _pickle_path(filename):
    return STORAGE_FOLDER / filename


def _load_pickle(filename, fallback):
    try:
        with _pickle_path(filename).open("rb") as file:
            return pickle.load(file)
    except (FileNotFoundError, OSError, pickle.PickleError, EOFError):
        return fallback


def _save_pickle(filename, value, folder=None):
    with ((folder or STORAGE_FOLDER) / filename).open("wb") as file:
        pickle.dump(value, file)


def _supported_files():
    DATA_FOLDER.mkdir(exist_ok=True)
    return [
        path.name
        for path in sorted(DATA_FOLDER.iterdir(), key=lambda item: item.name.casefold())
        if path.is_file()
        and path.name.lower().endswith(DOCUMENT_EXTENSIONS)
        and not path.name.startswith("~$")
    ]


def _select_input_files(file_names=None, force_reprocess=False):
    available = _supported_files()
    available_set = set(available)

    if file_names is None:
        selected = available
    else:
        selected = list(dict.fromkeys(os.path.basename(os.fspath(name)) for name in file_names))
        unknown = set(selected) - available_set
        if unknown:
            raise ValueError(f"Unsupported or missing data files: {', '.join(sorted(unknown))}")

    if force_reprocess:
        return selected

    existing_sources_text = _load_pickle("source_text.pkl", {})
    existing_chunks = _load_pickle("chunks.pkl", [])
    existing_stats = _load_pickle("stats.pkl", {})
    processed = set(existing_sources_text)
    processed.update(
        chunk.get("source")
        for chunk in existing_chunks
        if chunk.get("source")
    )

    return [
        filename
        for filename in selected
        if filename not in processed
        or existing_stats.get(filename, {}).get("_ingestion") != _file_signature(DATA_FOLDER / filename)
    ]


def _file_signature(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return {
        "version": 2,
        "sha256": digest.hexdigest(),
        "visual_enrichment": env_flag("RAG_VISUAL_ENRICHMENT", True),
        "text_model": TEXT_MODEL_NAME,
        "vision_model": VISION_MODEL_NAME,
        "vision_max_side": os.environ.get("RAG_VISION_MAX_SIDE", "1024"),
        "mineru_tier": os.environ.get("MINERU_TIER", "flash"),
    }


def _remove_sources_from_storage(sources):
    sources = set(sources)
    chunks = [
        chunk
        for chunk in _load_pickle("chunks.pkl", [])
        if chunk.get("source") not in sources
    ]
    stats = _load_pickle("stats.pkl", {})
    source_text = _load_pickle("source_text.pkl", {})
    triples = [
        triple
        for triple in load_graph(os.fspath(STORAGE_FOLDER / "graph.pkl"))
        if triple.get("source") not in sources
    ]

    for source in sources:
        stats.pop(source, None)
        source_text.pop(source, None)

    return chunks, stats, source_text, triples


def _build_vector_index(chunks, output_path=None):
    texts = [
        chunk["text"]
        for chunk in chunks
        if str(chunk.get("text", "")).strip()
    ]

    if not texts:
        raise RuntimeError("No readable content was found in data/.")

    embeddings = create_embeddings(texts)
    store = VectorStore(len(embeddings[0]))
    store.add(embeddings)
    store.save(os.fspath(output_path or STORAGE_FOLDER / "index.faiss"))


def run_ingestion(file_names=None, progress_callback=None, force_reprocess=False):
    """
    Ingest supported files from data/ and build vector + knowledge-graph storage.

    By default new or changed files are processed. Set force_reprocess=True to replace
    saved chunks, text, stats, and graph relationships for the selected files.
    """
    DATA_FOLDER.mkdir(exist_ok=True)
    STORAGE_FOLDER.mkdir(exist_ok=True)

    input_files = _select_input_files(
        file_names=file_names,
        force_reprocess=force_reprocess,
    )
    if not (STORAGE_FOLDER / "index.faiss").is_file() or not (STORAGE_FOLDER / "graph.pkl").is_file():
        input_files = _select_input_files(file_names=file_names, force_reprocess=True)

    if not input_files:
        return "These files are already processed; no files were read again."

    # Selected files include changed inputs as well as new ones. Replace their
    # old content in memory and publish only after all model work succeeds.
    all_chunks, stats_by_source, sources_text, existing_triples = (
        _remove_sources_from_storage(input_files)
    )

    new_structured_triples = []
    structured_sources = set()

    for file_index, filename in enumerate(input_files, start=1):
        path = DATA_FOLDER / filename
        extension = path.suffix.lower().lstrip(".").upper() or "FILE"

        _report(
            progress_callback,
            f"[{file_index}/{len(input_files)}] Reading {extension}: {filename}",
        )

        signature = _file_signature(path)
        segments, stats = read_document(path, progress_callback=progress_callback)
        stats["_ingestion"] = signature
        stats_by_source[filename] = stats

        source_triples = graph_triples_from_segments(segments, filename)
        if source_triples:
            structured_sources.add(filename)
            new_structured_triples.extend(source_triples)

        sources_text[filename] = "\n".join(
            str(segment.get("text", ""))
            for segment in segments
            if str(segment.get("text", "")).strip()
        )

        for segment_index, segment in enumerate(segments):
            segment.setdefault("id", f"{filename}:segment:{segment_index}")
            segment.setdefault("modality", segment.get("type", "TEXT"))
            all_chunks.extend(
                chunk_segment(segment, filename, segment_index)
            )

    _report(progress_callback, f"Total chunks in index: {len(all_chunks)}")
    # The FAISS position must correspond to the same position in chunks.pkl.
    all_chunks = [chunk for chunk in all_chunks if str(chunk.get("text", "")).strip()]

    _report(progress_callback, "Building knowledge graph...")
    text_triples = build_graph(
        {source: sources_text[source] for source in input_files},
        all_chunks,
        structured_sources=structured_sources,
    )

    triples = existing_triples + new_structured_triples + text_triples
    with tempfile.TemporaryDirectory(prefix="ingestion-", dir=STORAGE_FOLDER) as staging:
        staging = Path(staging)
        _report(progress_callback, "Building vector index...")
        _build_vector_index(all_chunks, staging / "index.faiss")
        _save_pickle("chunks.pkl", all_chunks, staging)
        _save_pickle("stats.pkl", stats_by_source, staging)
        _save_pickle("source_text.pkl", sources_text, staging)
        save_graph(
            triples, path=os.fspath(staging / "graph.pkl"),
            source_files=sorted(sources_text), chunks=all_chunks,
            stats_by_source=stats_by_source, source_text_by_source=sources_text,
        )
        for artifact in staging.iterdir():
            os.replace(artifact, STORAGE_FOLDER / artifact.name)

    _report(
        progress_callback,
        "Ingestion and knowledge graph generation complete.",
    )

    return (
        f"Done! Processed {len(input_files)} file(s). "
        f"Structured graph relationships: {len(new_structured_triples)}. "
        f"Text relationships: {len(text_triples)}."
    )


def rebuild_knowledge_graph(progress_callback=None):
    """
    Rebuild graph relationships from saved chunks/text without re-reading files.
    """
    STORAGE_FOLDER.mkdir(exist_ok=True)

    chunks = _load_pickle("chunks.pkl", [])
    stats_by_source = _load_pickle("stats.pkl", {})
    sources_text = _load_pickle("source_text.pkl", {})

    if not sources_text:
        return "No saved ingested content found. Run ingestion first."

    structured_triples = []
    structured_sources = set()
    for chunk in chunks:
        graph_triples = chunk.get("graph_triples", [])
        if graph_triples and chunk.get("source"):
            structured_sources.add(chunk["source"])
            for relationship in graph_triples:
                triple = dict(relationship)
                triple["source"] = chunk["source"]
                triple.setdefault("modality", chunk.get("modality", "IMAGE"))
                structured_triples.append(triple)

    _report(progress_callback, "Rebuilding knowledge graph from saved content...")
    text_triples = build_graph(
        sources_text,
        chunks,
        structured_sources=structured_sources,
    )

    triples = structured_triples + text_triples
    with tempfile.TemporaryDirectory(prefix="graph-", dir=STORAGE_FOLDER) as staging:
        staging = Path(staging)
        save_graph(
            triples, path=os.fspath(staging / "graph.pkl"),
            source_files=sorted(sources_text), chunks=chunks,
            stats_by_source=stats_by_source, source_text_by_source=sources_text,
        )
        for artifact in staging.iterdir():
            os.replace(artifact, STORAGE_FOLDER / artifact.name)

    return (
        f"Knowledge graph rebuilt. "
        f"Structured relationships: {len(structured_triples)}. "
        f"Text relationships: {len(text_triples)}."
    )

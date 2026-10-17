"""Unified document ingestion entry point.

MinerU is the primary parser for every supported project document format:
PDF, DOCX, PPTX, XLS/XLSX, and standalone images.

Ollama is NOT used to parse DOCX, PPTX, XLS/XLSX, or PDF files.  Ollama is
only invoked for the separate visual-understanding enrichment step on
standalone image inputs and for the existing LLM/graph-generation stages.
"""

import os

from src.config import env_flag
from src.image_reader import read_image_segments
from src.mineru_reader import (
    IMAGE_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    read_mineru_document,
)

EXCEL_EXTENSIONS = (".xlsx", ".xls")
PDF_EXTENSIONS = (".pdf",)
WORD_EXTENSIONS = (".docx",)
PRESENTATION_EXTENSIONS = (".pptx",)

# The parser contract is intentionally derived from MinerU's supported
# formats used by this project. This prevents native readers/Ollama from
# becoming an accidental parser for one of these document types.
DOCUMENT_EXTENSIONS = tuple(SUPPORTED_EXTENSIONS)


def _merge_visual_graph_data(segments, visual_segments):
    """Attach Ollama visual graph enrichment to a MinerU image segment."""
    visual_triples = []
    visual_nodes = []
    visual_edges = []
    visual_text = []

    for segment in visual_segments:
        if segment.get("vision_error"):
            raise RuntimeError(
                f"Image graph enrichment failed: {segment['vision_error']}. "
                "Retry image reprocessing after checking Ollama."
            )
        if segment.get("text", "").strip():
            visual_text.append(segment["text"].strip())
        visual_triples.extend(segment.get("graph_triples", []))
        visual_nodes.extend(segment.get("graph_nodes", []))
        visual_edges.extend(segment.get("graph_edges", []))

    if not visual_triples and not visual_nodes and not visual_edges and not visual_text:
        return segments

    target = next(
        (segment for segment in segments if segment.get("type") == "IMAGE"),
        None,
    )
    if target is None:
        target = {"text": "", "type": "IMAGE", "modality": "IMAGE"}
        segments.append(target)

    if visual_triples:
        target["graph_triples"] = visual_triples
    if visual_nodes:
        target["graph_nodes"] = visual_nodes
    if visual_edges:
        target["graph_edges"] = visual_edges

    target["visual_parser"] = "Ollama vision enrichment"
    target["text"] = "\n".join(
        [target.get("text", "").strip(), *visual_text]
    ).strip()
    return segments


def read_document(path, progress_callback=None, visual_enrichment=None):
    """Read a supported project file with MinerU as the parser.

    DOCX/PPTX/XLS/XLSX/PDF are passed directly to MinerU and do not call
    Ollama for parsing. Standalone images are also parsed by MinerU first;
    only after that, Ollama vision may enrich visible image relationships.
    """
    extension = os.path.splitext(os.fspath(path))[1].lower()

    if extension not in DOCUMENT_EXTENSIONS:
        raise ValueError(f"Unsupported document format: {extension or path}")

    # One common parsing path for all supported project formats.
    segments, stats = read_mineru_document(path, progress_callback=progress_callback)

    # Ollama is intentionally restricted to standalone-image visual enrichment.
    # It is never the parser for DOCX/PPTX/XLS/XLSX/PDF.
    if visual_enrichment is None:
        visual_enrichment = env_flag("RAG_VISUAL_ENRICHMENT", default=True)
    if extension in IMAGE_EXTENSIONS and visual_enrichment:
        if progress_callback:
            progress_callback(f"Extracting visible image relationships: {os.path.basename(path)}")
        visual_segments = read_image_segments(path)
        segments = _merge_visual_graph_data(segments, visual_segments)
        stats["visual_enrichment"] = "Ollama vision"
    elif extension in IMAGE_EXTENSIONS:
        stats["visual_enrichment"] = "disabled"

    return segments, stats


def graph_triples_from_segments(segments, source):
    """Collect structured graph relationships produced during ingestion."""
    triples = []

    for segment in segments:
        for relationship in segment.get("graph_triples", []):
            triple = dict(relationship)
            triple["source"] = source
            triple.setdefault("modality", segment.get("modality", "IMAGE"))
            triples.append(triple)

    return triples

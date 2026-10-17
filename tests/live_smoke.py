"""Opt-in local MinerU/Ollama smoke test; writes only temporary fixture storage."""
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def write_pdf(path):
    content = b"BT /F1 16 Tf 50 750 Td (Alice manages the Atlas project.) Tj 0 -30 Td (The Atlas project uses the Orion database.) Tj ET"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(content)).encode() + b" >>\nstream\n" + content + b"\nendstream",
    ]
    data = bytearray(b"%PDF-1.4\n")
    offsets = []
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(data))
        data.extend(f"{index} 0 obj\n".encode() + obj + b"\nendobj\n")
    start = len(data)
    data.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode())
    for offset in offsets:
        data.extend(f"{offset:010d} 00000 n \n".encode())
    data.extend(f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{start}\n%%EOF\n".encode())
    path.write_bytes(data)


def main():
    # Verification uses an already cached embedding model; never downloads one.
    os.environ["HF_HUB_OFFLINE"] = "1"
    from src import ingest_pipeline as pipeline
    from src.retriever import search, build_grounded_context
    from src.knowledge_graph import load_graph
    from src.pyvis_graph import render_graph
    from src.llm import generate_answer

    with tempfile.TemporaryDirectory(prefix="live-smoke-", dir=ROOT) as directory:
        base = Path(directory)
        data = base / "data"
        data.mkdir()
        write_pdf(data / "smoke.pdf")
        pipeline.DATA_FOLDER = data
        pipeline.STORAGE_FOLDER = base / "storage"
        previous_cwd = Path.cwd()
        os.chdir(base)
        try:
            print(pipeline.run_ingestion(), flush=True)
            triples = load_graph()
            assert triples, "No text relationships extracted from explicit fixture facts."
            results = search("Who manages Atlas?", use_reranker=False)
            context = build_grounded_context("Atlas", results)
            assert "Alice" in context and "KNOWLEDGE GRAPH FACTS" in context
            answer = generate_answer(context, "Who manages Atlas?")
            assert "Alice" in answer, f"Unexpected grounded answer: {answer}"
            render_graph(triples, "storage/smoke.html")
            assert (base / "storage/smoke.html").is_file()
            print("LIVE SMOKE PASS: MinerU -> chunks -> embeddings/FAISS -> text graph -> combined retrieval -> local answer -> PyVis", flush=True)
        finally:
            os.chdir(previous_cwd)


if __name__ == "__main__":
    main()

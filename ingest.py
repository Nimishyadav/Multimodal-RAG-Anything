"""MinerU-first indexing entry point shared with the Streamlit dashboard."""
import argparse
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the MinerU multimodal FAISS index and knowledge graph.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--force", action="store_true", help="Reparse all selected files, replacing saved content.")
    mode.add_argument("--reprocess-images", action="store_true", help="Reparse standalone images and rebuild their visual relationships.")
    mode.add_argument("--rebuild-graph", action="store_true", help="Rebuild relationships from saved evidence without parsing or embedding again.")
    parser.add_argument("--source", action="append", help="Process a data/ filename; repeat to select multiple files.")
    parser.add_argument("--no-vision", action="store_true", help="Use MinerU parsing without standalone-image visual enrichment.")
    args = parser.parse_args(argv)
    if args.rebuild_graph and (args.source or args.no_vision):
        parser.error("--rebuild-graph uses all saved evidence; --source and --no-vision apply to ingestion.")

    os.chdir(Path(__file__).resolve().parent)
    if args.no_vision:
        os.environ["RAG_VISUAL_ENRICHMENT"] = "false"

    from src.ingest_pipeline import run_ingestion, rebuild_knowledge_graph, _supported_files
    from src.mineru_reader import IMAGE_EXTENSIONS

    if args.rebuild_graph:
        print(rebuild_knowledge_graph())
        return
    sources = args.source
    if args.reprocess_images:
        sources = [name for name in (sources if sources is not None else _supported_files())
                   if name.lower().endswith(IMAGE_EXTENSIONS)]
        if not sources:
            print("No standalone images selected in data/.")
            return
    print(run_ingestion(file_names=sources, force_reprocess=args.force or args.reprocess_images))
    print("Next: python visualize_graph.py")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, TimeoutError) as exc:
        raise SystemExit(f"Ingestion failed: {exc}\nExisting saved graph was not replaced. Fix the error and retry ingestion.") from None

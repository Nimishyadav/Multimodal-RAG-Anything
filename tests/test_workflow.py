import importlib.util
import json
import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.modules.setdefault("dotenv", MagicMock())
sys.modules.setdefault("ollama", MagicMock())
from src.document_reader import read_document
from src.mineru_reader import _segments_from_markdown
from src.knowledge_graph import build_graph, load_graph

ROOT = Path(__file__).resolve().parents[1]


def load_isolated(name, file, dependencies):
    spec = importlib.util.spec_from_file_location(name, ROOT / file)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, dependencies):
        spec.loader.exec_module(module)
    return module


class RoutingTests(unittest.TestCase):
    def test_every_document_format_uses_mineru_without_vision(self):
        for suffix in (".pdf", ".docx", ".pptx", ".xls", ".xlsx"):
            with self.subTest(suffix=suffix), patch("src.document_reader.read_mineru_document", return_value=([{"text": "parsed", "type": "TEXT"}], {})) as mineru, patch("src.document_reader.read_image_segments") as vision:
                segments, _ = read_document("sample" + suffix)
                self.assertEqual(segments[0]["text"], "parsed")
                mineru.assert_called_once()
                vision.assert_not_called()

    def test_image_enrichment_can_be_disabled(self):
        with patch("src.document_reader.read_mineru_document", return_value=([{"text": "OCR text", "type": "TEXT"}], {})), patch("src.document_reader.read_image_segments") as vision:
            segments, stats = read_document("sample.png", visual_enrichment=False)
            vision.assert_not_called()
            self.assertEqual(stats["visual_enrichment"], "disabled")
            self.assertEqual(segments[0]["text"], "OCR text")

    def test_html_table_and_inline_image_keep_provenance(self):
        segments = _segments_from_markdown("<!-- page 2 of 3 -->\n<table>\n<tr><td>A</td></tr>\n</table>\nBefore ![chart](doc:123) After")
        self.assertEqual([s["type"] for s in segments], ["TABLE", "TEXT", "IMAGE", "TEXT"])
        self.assertTrue(all(s["page"] == 2 for s in segments))
        self.assertEqual(segments[-1]["text"], "After")

    def test_long_saved_text_is_not_truncated(self):
        text = "A" * 9500
        with patch("src.knowledge_graph.extract_triples", return_value=[]) as extract:
            build_graph({"sample.pdf": text})
        batches = [call.args[0] for call in extract.call_args_list]
        self.assertEqual("".join(batches), text)
        self.assertTrue(all(len(batch) <= 4000 for batch in batches))


class FakeVectorStore:
    def __init__(self, dimension):
        self.rows = []

    def add(self, embeddings):
        self.rows.extend(embeddings)

    def save(self, path):
        Path(path).write_text(json.dumps(self.rows), encoding="utf-8")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        base = Path(self.temp.name)
        self.data = base / "data"
        self.storage = base / "storage"
        self.data.mkdir()
        self.storage.mkdir()
        self.pipeline = load_isolated("pipeline_test", "src/ingest_pipeline.py", {
            "src.embedder": MagicMock(create_embeddings=lambda texts: [[len(text), 1.0] for text in texts]),
            "src.vector_store": MagicMock(VectorStore=FakeVectorStore),
        })
        self.pipeline.DATA_FOLDER = self.data
        self.pipeline.STORAGE_FOLDER = self.storage

    def ingest(self, text="Alice works on Project", triples=None):
        (self.data / "report.pdf").write_text(text, encoding="utf-8")
        triple = {"subject": "Alice", "relation": "WORKS_ON", "object": "Project", "source": "report.pdf"}
        with patch.object(self.pipeline, "read_document", return_value=([{"text": text, "type": "TEXT"}], {"pages": 1, "tables": 0, "images": 0})), patch.object(self.pipeline, "build_graph", return_value=triples if triples is not None else [triple]):
            return self.pipeline.run_ingestion()

    def test_publishes_complete_storage_and_reuses_unchanged_input(self):
        self.ingest()
        names = {p.name for p in self.storage.iterdir()}
        self.assertTrue({"index.faiss", "chunks.pkl", "source_text.pkl", "stats.pkl", "graph.pkl", "graph.json", "graph_readable.txt"} <= names)
        with patch.object(self.pipeline, "read_document") as reader:
            self.assertIn("already processed", self.pipeline.run_ingestion())
            reader.assert_not_called()
        with (self.storage / "chunks.pkl").open("rb") as file:
            chunks = pickle.load(file)
        self.assertEqual(len(chunks), len(json.loads((self.storage / "index.faiss").read_text())))

    def test_changed_file_replaces_old_chunks_and_graph(self):
        self.ingest()
        self.ingest("New report", triples=[])
        with (self.storage / "chunks.pkl").open("rb") as file:
            chunks = pickle.load(file)
        self.assertEqual([c["text"] for c in chunks], ["New report"])
        self.assertEqual(load_graph(self.storage / "graph.pkl"), [])

    def test_failed_model_leaves_saved_artifacts_unchanged(self):
        self.ingest()
        before = {p.name: p.read_bytes() for p in self.storage.iterdir()}
        (self.data / "report.pdf").write_text("Updated", encoding="utf-8")
        with patch.object(self.pipeline, "read_document", return_value=([{"text": "Updated", "type": "TEXT"}], {})), patch.object(self.pipeline, "build_graph", side_effect=RuntimeError("model failed")):
            with self.assertRaisesRegex(RuntimeError, "model failed"):
                self.pipeline.run_ingestion()
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.storage.iterdir()})

    def test_missing_index_is_rebuilt(self):
        self.ingest()
        (self.storage / "index.faiss").unlink()
        with patch.object(self.pipeline, "read_document", return_value=([{"text": "Restored", "type": "TEXT"}], {})) as reader, patch.object(self.pipeline, "build_graph", return_value=[]):
            self.pipeline.run_ingestion()
            reader.assert_called_once()
        self.assertTrue((self.storage / "index.faiss").is_file())

    def test_failed_embeddings_leave_saved_artifacts_unchanged(self):
        self.ingest()
        before = {p.name: p.read_bytes() for p in self.storage.iterdir()}
        (self.data / "report.pdf").write_text("Updated", encoding="utf-8")
        with patch.object(self.pipeline, "read_document", return_value=([{"text": "Updated", "type": "TEXT"}], {})), patch.object(self.pipeline, "build_graph", return_value=[]), patch.object(self.pipeline, "create_embeddings", side_effect=RuntimeError("embedding failed")):
            with self.assertRaisesRegex(RuntimeError, "embedding failed"):
                self.pipeline.run_ingestion()
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.storage.iterdir()})

    def test_empty_explicit_selection_does_not_process_all_files(self):
        (self.data / "report.pdf").write_text("Report")
        with patch.object(self.pipeline, "read_document") as reader:
            self.pipeline.run_ingestion(file_names=[])
            reader.assert_not_called()

    def test_rebuild_retains_visual_triples_without_reading_images(self):
        (self.data / "chart.png").write_bytes(b"fixture")
        triple = {"subject": "Start", "relation": "leads_to", "object": "End"}
        with patch.object(self.pipeline, "read_document", return_value=([{"text": "Start -> End", "type": "IMAGE", "graph_triples": [triple]}], {})), patch.object(self.pipeline, "build_graph", return_value=[]):
            self.pipeline.run_ingestion()
            with patch.object(self.pipeline, "read_document") as reader:
                self.pipeline.rebuild_knowledge_graph()
                reader.assert_not_called()
        self.assertEqual(load_graph(self.storage / "graph.pkl")[0]["subject"], "Start")


class GroundedContextTests(unittest.TestCase):
    def test_combines_vector_and_graph_evidence_with_branch_and_source_filter(self):
        retriever = load_isolated("retriever_test", "src/retriever.py", {
            "numpy": MagicMock(), "src.embedder": MagicMock(),
            "src.vector_store": MagicMock(), "src.reranker": MagicMock(),
        })
        facts = [{"subject": "Check", "relation": "leads_to", "object": "Approved", "source": "chart.png", "details": "YES"},
                 {"subject": "Check", "relation": "leads_to", "object": "Other", "source": "other.png"}]
        chunks = [{"source": "chart.png", "type": "IMAGE", "text": "Read the visible decision"},
                  {"source": "other.png", "text": "Unrelated evidence"}]
        with patch.object(retriever, "load_graph", return_value=facts):
            context = retriever.build_grounded_context("Check", chunks, source_filter="chart.png")
        self.assertIn("Read the visible decision", context)
        self.assertIn("Approved", context)
        self.assertIn("YES", context)
        self.assertNotIn("other.png", context)


class SharedAnswerTests(unittest.TestCase):
    def setUp(self):
        self.retriever = MagicMock()
        for name in ("is_stats_query", "is_graph_query", "is_image_query", "is_excel_query", "is_summary_query"):
            getattr(self.retriever, name).return_value = False
        self.qa = load_isolated("qa_test", "src/qa.py", {
            "src.retriever": self.retriever, "src.llm": MagicMock(),
        })

    def test_image_count_uses_stats_instead_of_vision_summary(self):
        self.retriever.is_stats_query.return_value = True
        self.retriever.is_image_query.return_value = True
        self.retriever.get_stats_answer.return_value = "2 images"
        with patch.object(self.qa, "save_answer_record"):
            answer, _, _ = self.qa.answer_question("How many images are there?")
        self.assertEqual(answer, "2 images")
        self.qa.generate_summary.assert_not_called()

    def test_graph_overview_respects_requested_source(self):
        self.retriever.is_graph_query.return_value = True
        self.retriever.get_known_sources.return_value = ["a.pdf", "b.pdf"]
        self.retriever.find_mentioned_source.return_value = "a.pdf"
        facts = [{"subject": "Alice", "relation": "USES", "object": "Atlas", "source": "a.pdf"},
                 {"subject": "Bob", "relation": "USES", "object": "Other", "source": "b.pdf"}]
        with patch.object(self.qa, "load_graph", return_value=facts), patch.object(self.qa, "save_answer_record"):
            answer, sources, debug = self.qa.answer_question("Show knowledge graph for a.pdf")
        self.assertIn("Alice", answer)
        self.assertNotIn("Bob", answer)
        self.assertEqual(sources, ["a.pdf"])
        self.assertEqual(debug["graph_triples"][0]["source"], "a.pdf")


if __name__ == "__main__":
    unittest.main()

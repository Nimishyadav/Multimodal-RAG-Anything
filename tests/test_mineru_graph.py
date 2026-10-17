import sys
import json
import unittest
from unittest.mock import MagicMock, patch

# Parser/graph contract tests do not need running local models.
sys.modules.setdefault("dotenv", MagicMock())
sys.modules.setdefault("ollama", MagicMock())

from src.chunker import chunk_segment
from src.document_reader import _merge_visual_graph_data, graph_triples_from_segments
from src.mineru_reader import _segments_from_markdown
from src.knowledge_graph import build_graph, extract_triples
from src.image_reader import read_image_segments


class MinerUGraphTests(unittest.TestCase):
    def test_large_image_is_resized_only_for_inference(self):
        from io import BytesIO
        from PIL import Image
        from src.image_reader import _vision_image_input
        original = BytesIO()
        Image.new("RGB", (2000, 1000), "white").save(original, format="PNG")
        source = original.getvalue()
        with patch.dict("os.environ", {"RAG_VISION_MAX_SIDE": "1024"}):
            prepared = _vision_image_input(image_bytes=source)
        with Image.open(BytesIO(prepared)) as image:
            self.assertEqual(image.size, (1024, 512))
        with Image.open(BytesIO(source)) as image:
            self.assertEqual(image.size, (2000, 1000))

    def test_vision_uses_unformatted_primary_and_disables_thinking(self):
        payload = {"nodes": [{"id": "n1", "label": "Start", "type": "start"},
                             {"id": "n2", "label": "End", "type": "end"}],
                   "edges": [{"from": "n1", "to": "n2", "relation": "leads_to"}]}
        client = MagicMock()
        client.chat.side_effect = [
            iter([{"message": {"content": ""}, "done_reason": "length", "eval_count": 3200}]),
            iter([{"message": {"content": json.dumps(payload)}}]),
        ]
        with patch("src.image_reader._pick_vision_model", return_value="qwen3-vl:4b"), patch("src.image_reader.ollama.Client", return_value=client), patch("src.image_reader._vision_image_input", return_value=b"fixture"):
            segments = read_image_segments(image_bytes=b"fixture", image_name="chart.png")
        requests = [call.kwargs for call in client.chat.call_args_list]
        self.assertNotIn("format", requests[0])
        self.assertEqual(requests[1]["format"], "json")
        self.assertTrue(all(r["think"] is False and r["options"]["num_ctx"] == 6144 for r in requests))
        self.assertTrue(segments[0]["graph_triples"])

    def test_whitespace_only_stream_is_closed_early(self):
        from src.image_reader import _chat_vision
        stream = MagicMock()
        stream.__iter__.return_value = iter([{"message": {"content": " " * 513}}])
        client = MagicMock()
        client.chat.return_value = stream
        with self.assertRaisesRegex(RuntimeError, "whitespace"):
            _chat_vision(client, model="qwen3-vl:4b")
        stream.close.assert_called_once()

    def test_visual_content_and_triples_survive_chunking(self):
        triple = {"subject": "Start", "relation": "leads_to", "object": "End"}
        segments = _merge_visual_graph_data(
            [{"text": "![chart](doc:123)", "type": "IMAGE"}],
            [{"text": "Start -> End", "graph_triples": [triple]}],
        )
        chunks = chunk_segment(segments[0], "chart.png", 0)
        self.assertIn("Start -> End", chunks[0]["text"])
        self.assertEqual(chunks[0]["graph_triples"], [triple])
        self.assertEqual(graph_triples_from_segments(segments, "chart.png")[0]["source"], "chart.png")

    def test_failed_vision_is_not_silently_saved(self):
        with self.assertRaisesRegex(RuntimeError, "model failed"):
            _merge_visual_graph_data([], [{"vision_error": "model failed"}])

    def test_graph_only_segment_survives_chunking(self):
        triple = {"subject": "A", "relation": "leads_to", "object": "B"}
        chunks = chunk_segment({"text": "", "type": "IMAGE", "graph_triples": [triple]}, "chart.png", 0)
        self.assertEqual(chunks[0]["graph_triples"], [triple])

    def test_page_markers_are_not_content(self):
        self.assertEqual(_segments_from_markdown("<!-- page 1 of 1 -->\n![chart](doc:123)"),
                         [{"text": "![chart](doc:123)", "type": "IMAGE", "page": 1}])

    def test_image_links_are_not_sent_to_text_model(self):
        with patch("src.knowledge_graph.extract_triples") as extract:
            self.assertEqual(build_graph({"chart.png": "![chart](doc:123)"},
                                         [{"source": "chart.png", "modality": "IMAGE", "text": "![chart](doc:123)"}]), [])
            extract.assert_not_called()

    def test_invalid_model_responses_fail_explicitly(self):
        for raw in ("", "not JSON", "[]", '{"relationships": {}}'):
            with self.subTest(raw=raw), patch("src.knowledge_graph._ollama_chat", return_value=raw):
                with self.assertRaises(RuntimeError):
                    extract_triples("A leads to B", "doc.pdf")
        with patch("src.knowledge_graph._ollama_chat", return_value='{"relationships": []}'):
            self.assertEqual(extract_triples("No relationships", "doc.pdf"), [])


if __name__ == "__main__":
    unittest.main()

import json
import os
import re
from io import BytesIO
from typing import Any

import ollama
from src.config import VISION_MODEL_NAME


MAX_NODES = 80
MAX_EDGES = 120

_TEXT_MODEL = os.environ.get("RAG_TEXT_MODEL", "llama3")
_VISION_MODEL_ENV = os.environ.get("RAG_VISION_MODEL", "").strip()

_VISION_CANDIDATES = [
    _VISION_MODEL_ENV or VISION_MODEL_NAME,
    "qwen3-vl:4b-instruct",
    "qwen3-vl:2b-instruct",
    "qwen2.5vl:3b",
    "llama3.2-vision:11b",
    "llama3.2-vision",
]


def _model_name(model_obj: Any) -> str:
    if isinstance(model_obj, dict):
        return str(model_obj.get("name") or model_obj.get("model") or "").strip()
    return str(
        getattr(model_obj, "name", None)
        or getattr(model_obj, "model", None)
        or ""
    ).strip()


def _installed_models() -> set[str]:
    try:
        response = ollama.list()
    except Exception as exc:
        raise RuntimeError(
            "Ollama is not reachable. Start Ollama and try again."
        ) from exc

    models = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", [])
    installed = set()
    for model in models or []:
        name = _model_name(model)
        if name:
            installed.add(name)
    return installed


def _pick_vision_model() -> str:
    installed = _installed_models()
    if _VISION_MODEL_ENV and _VISION_MODEL_ENV not in installed:
        raise RuntimeError(f"Configured vision model '{_VISION_MODEL_ENV}' is not installed. Run ollama pull {_VISION_MODEL_ENV}.")

    for candidate in _VISION_CANDIDATES:
        if candidate and candidate in installed:
            return candidate

    available = ", ".join(sorted(installed)) or "none"
    raise RuntimeError(
        "No Ollama vision model is installed.\n"
        "Install one of these and rerun:\n"
        "  ollama pull qwen2.5vl:3b\n"
        "or\n"
        "  ollama pull qwen3-vl:4b-instruct\n\n"
        f"Installed Ollama models: {available}"
    )


def _parse_json(content: str) -> dict:
    """Parse JSON robustly from a vision model response.

    Models sometimes wrap JSON in markdown fences or add a short explanation.
    This function accepts those harmless wrappers but never invents missing data.
    """
    content = str(content or "").strip()
    if not content:
        return {}

    # 1) Exact JSON response.
    try:
        payload = json.loads(content)
        return payload if isinstance(payload, dict) else {}
    except (json.JSONDecodeError, TypeError):
        pass

    # 2) Remove common markdown code fences.
    unfenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
    if unfenced != content:
        try:
            payload = json.loads(unfenced)
            return payload if isinstance(payload, dict) else {}
        except (json.JSONDecodeError, TypeError):
            pass

    # 3) Find the outermost object and progressively test balanced braces.
    start = content.find("{")
    if start < 0:
        start = unfenced.find("{")
        source = unfenced
    else:
        source = content

    if start < 0:
        return {}

    depth = 0
    in_string = False
    escaped = False
    candidates = []

    for index in range(start, len(source)):
        char = source[index]

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidates.append(source[start:index + 1])
                break

    for candidate in candidates:
        try:
            payload = json.loads(candidate)
            return payload if isinstance(payload, dict) else {}
        except (json.JSONDecodeError, TypeError):
            continue

    # 4) Last-resort extraction from the first '{' to the last '}'.
    end = source.rfind("}")
    if end > start:
        try:
            payload = json.loads(source[start:end + 1])
            return payload if isinstance(payload, dict) else {}
        except (json.JSONDecodeError, TypeError):
            pass

    return {}


def _clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _normalize_type(value: Any) -> str:
    value = _clean_text(value).lower()

    mapping = {
        "start": "start",
        "begin": "start",
        "beginning": "start",
        "terminal": "end",
        "finish": "end",
        "stop": "end",
        "end": "end",
        "decision": "decision",
        "condition": "decision",
        "diamond": "decision",
        "process": "process",
        "action": "process",
        "step": "process",
        "input": "input",
        "output": "output",
        "database": "database",
        "document": "document",
        "data": "data",
    }
    return mapping.get(value, "entity")


def _safe_sequence(value: Any, fallback: int) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return fallback


def _image_prompt() -> str:
    return r"""
You are an exhaustive multimodal document/flowchart parser.

Analyze ONLY what is visibly present in the supplied image.

Your job is to convert the visible image into a precise graph structure.

Rules:
1. Read every visible box, circle, ellipse, diamond, database shape, document shape,
   text block, title, label, side note, and visible annotation that is meaningful.
2. For a flowchart, create one node for EVERY distinct visible shape/text step.
3. Preserve the visible wording as closely as possible. Do not paraphrase node labels.
4. Preserve the actual order/sequence shown in the image.
5. Capture EVERY visible arrow/connector between nodes.
6. Preserve arrow direction exactly.
7. When a connector has a visible label such as Yes, No, True, False, Accept,
   Reject, etc., put that exact label in the edge "label" field.
8. Do not invent missing arrows, steps, entities, or domain knowledge.
9. A branch is a real graph edge. Do not collapse branches into a single sequence.
10. If a flow returns to an earlier node, preserve that loop edge.
11. If there are parallel branches, keep every branch.
12. For non-flowchart images, turn meaningful visible entities into nodes and
    meaningful visible relationships into edges, but do not invent relationships.
13. If the image is just a photo/illustration with no meaningful graph structure,
    still extract its visible text into "visible_text" and use nodes=[] and edges=[].
14. Use "type" only for visible shape semantics. Allowed values:
    start, process, decision, end, input, output, database, document, data, entity.
15. "sequence" is the left-to-right/top-to-bottom order in which the node appears
    in the actual flow. Start with 0 for the first/start node.
16. Coordinates are optional. When confident, return center_x and center_y as values
    from 0 to 1 based on the image dimensions. These are only hints for layout.
17. Every edge must reference node ids from the nodes array.
18. Return ONLY valid JSON. Do not add markdown fences.

JSON shape:
{
  "title": "visible title or empty string",
  "description": "complete concise description of visible content",
  "visible_text": ["every important visible text item"],
  "nodes": [
    {
      "id": "n1",
      "label": "exact visible node text",
      "type": "start|process|decision|end|input|output|database|document|data|entity",
      "sequence": 0,
      "center_x": 0.5,
      "center_y": 0.1
    }
  ],
  "edges": [
    {
      "from": "n1",
      "to": "n2",
      "relation": "leads_to",
      "label": "",
      "sequence": 1
    }
  ]
}
"""


def _dedupe_nodes(raw_nodes: Any) -> list[dict]:
    if not isinstance(raw_nodes, list):
        return []

    nodes: list[dict] = []
    seen_ids: set[str] = set()

    for index, raw in enumerate(raw_nodes[:MAX_NODES], start=1):
        if not isinstance(raw, dict):
            continue

        node_id = _clean_text(raw.get("id")) or f"n{index}"
        label = _clean_text(raw.get("label"))

        if not label:
            continue

        while node_id in seen_ids:
            node_id = f"{node_id}_{index}"

        seen_ids.add(node_id)

        node = {
            "id": node_id,
            "label": label,
            "type": _normalize_type(raw.get("type")),
            "sequence": _safe_sequence(raw.get("sequence"), index - 1),
        }

        for coordinate in ("center_x", "center_y"):
            value = raw.get(coordinate)
            try:
                value = float(value)
            except (TypeError, ValueError):
                value = None
            if value is not None and 0 <= value <= 1:
                node[coordinate] = value

        nodes.append(node)

    # Keep the original sequence, but make ties deterministic.
    nodes.sort(key=lambda item: (item["sequence"], item["id"]))
    for fallback_index, node in enumerate(nodes):
        if "sequence" not in node or node["sequence"] < 0:
            node["sequence"] = fallback_index

    return nodes


def _clean_edges(raw_edges: Any, nodes: list[dict]) -> list[dict]:
    if not isinstance(raw_edges, list):
        return []

    node_ids = {node["id"] for node in nodes}
    cleaned = []
    seen = set()

    for raw in raw_edges[:MAX_EDGES]:
        if not isinstance(raw, dict):
            continue

        source_id = _clean_text(raw.get("from"))
        target_id = _clean_text(raw.get("to"))

        if source_id not in node_ids or target_id not in node_ids:
            continue

        if source_id == target_id:
            # Keep a self-loop only when it is explicitly meaningful.
            relation = _clean_text(raw.get("relation")) or "loops_to"
            if relation not in {"loops_to", "returns_to"}:
                continue

        relation = _clean_text(raw.get("relation")) or "leads_to"
        label = _clean_text(raw.get("label"))
        sequence = _safe_sequence(raw.get("sequence"), 0)

        key = (source_id, target_id, relation.casefold(), label.casefold())
        if key in seen:
            continue
        seen.add(key)

        cleaned.append(
            {
                "from": source_id,
                "to": target_id,
                "relation": relation,
                "label": label,
                "sequence": sequence,
            }
        )

    return cleaned


def _build_graph_triples(
    title: str,
    nodes: list[dict],
    edges: list[dict],
    image_name: str,
) -> list[dict]:
    label_by_id = {node["id"]: node["label"] for node in nodes}
    type_by_id = {node["id"]: node["type"] for node in nodes}
    sequence_by_id = {node["id"]: node["sequence"] for node in nodes}

    triples: list[dict] = []

    # A node-only triple makes single-node images possible while remaining invisible
    # as a relationship in the visualizer.
    for node in nodes:
        triples.append(
            {
                "subject": node["label"],
                "relation": "__node__",
                "object": node["label"],
                "source": image_name,
                "modality": "IMAGE",
                "node_type": node["type"],
                "sequence": node["sequence"],
                "node_id": node["id"],
                "center_x": node.get("center_x"),
                "center_y": node.get("center_y"),
                "title": title,
            }
        )

    for edge in edges:
        source_id = edge["from"]
        target_id = edge["to"]

        triples.append(
            {
                "subject": label_by_id[source_id],
                "relation": edge["relation"],
                "object": label_by_id[target_id],
                "details": edge["label"],
                "source": image_name,
                "modality": "IMAGE",
                "subject_id": source_id,
                "object_id": target_id,
                "subject_type": type_by_id[source_id],
                "object_type": type_by_id[target_id],
                "subject_sequence": sequence_by_id[source_id],
                "object_sequence": sequence_by_id[target_id],
                "sequence": edge["sequence"],
                "title": title,
            }
        )

    return triples


def _segment_from_payload(payload: dict, image_name: str) -> dict:
    title = _clean_text(payload.get("title"))
    description = _clean_text(payload.get("description"))

    visible_text = payload.get("visible_text")
    if not isinstance(visible_text, list):
        visible_text = []
    visible_text = [_clean_text(value) for value in visible_text if _clean_text(value)]

    nodes = _dedupe_nodes(payload.get("nodes"))
    edges = _clean_edges(payload.get("edges"), nodes)

    text_lines = []
    if title:
        text_lines.append(f"Title: {title}")
    if description:
        text_lines.append(description)

    if visible_text:
        text_lines.append("Visible text:")
        text_lines.extend(f"- {item}" for item in visible_text)

    if nodes:
        text_lines.append("Graph nodes:")
        for node in nodes:
            text_lines.append(
                f"- {node['label']} [{node['type']}] sequence={node['sequence']}"
            )

    if edges:
        text_lines.append("Graph edges:")
        for edge in edges:
            branch = f" [{edge['label']}]" if edge["label"] else ""
            text_lines.append(
                f"- {edge['from']} -> {edge['to']}{branch}"
            )

    if not text_lines:
        raise ValueError(f"No readable image content or graph nodes returned for {image_name}.")

    return {
        "text": "\n".join(text_lines),
        "type": "IMAGE",
        "modality": "IMAGE",
        "flowchart_title": title,
        "flowchart_steps": nodes,
        "graph_nodes": nodes,
        "graph_edges": edges,
        "graph_triples": _build_graph_triples(
            title,
            nodes,
            edges,
            image_name,
        ),
        "image_name": image_name,
    }


def _fallback_image_segment(image_name: str, reason: object) -> dict:
    reason_text = _clean_text(reason)
    return {
        "text": (
            f"IMAGE: {image_name}\n"
            "Structured flowchart/image parsing failed, so no graph nodes or "
            "edges were extracted for this image.\n"
            f"Reason: {reason_text or 'unknown error'}"
        ),
        "type": "IMAGE",
        "modality": "IMAGE",
        "flowchart_title": "",
        "flowchart_steps": [],
        "graph_nodes": [],
        "graph_edges": [],
        "graph_triples": [],
        "image_name": image_name,
        "vision_error": reason_text,
    }


def _response_field(response, name):
    return response.get(name, "unknown") if isinstance(response, dict) else getattr(response, name, "unknown")


def _chat_vision(client, **kwargs):
    """Collect streamed JSON and stop runaway whitespace responses early."""
    stream = client.chat(stream=True, **kwargs)
    pieces = []
    length = 0
    last = {}
    try:
        for part in stream:
            last = part
            message = part.get("message", {}) if isinstance(part, dict) else part.message
            content = message.get("content", "") if isinstance(message, dict) else message.content
            content = content or ""
            pieces.append(content)
            length += len(content)
            if length > 512 and not "".join(pieces).strip():
                raise RuntimeError("Vision generated only whitespace instead of JSON.")
        return {"message": {"content": "".join(pieces)},
                "done_reason": _response_field(last, "done_reason"),
                "eval_count": _response_field(last, "eval_count")}
    finally:
        if hasattr(stream, "close"):
            stream.close()


def _vision_image_input(image_path=None, image_bytes=None):
    """Bound inference image size without modifying the uploaded source file."""
    from PIL import Image, ImageOps

    limit = int(os.environ.get("RAG_VISION_MAX_SIDE", "1024"))
    if limit < 256:
        raise ValueError("RAG_VISION_MAX_SIDE must be at least 256.")
    original = image_bytes if image_bytes is not None else os.fspath(image_path)
    with Image.open(BytesIO(image_bytes) if image_bytes is not None else image_path) as image:
        if max(image.size) <= limit:
            return original
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((limit, limit), Image.Resampling.LANCZOS)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


def read_image_segments(image_path=None, image_bytes=None, image_name=None):
    if image_path is None and image_bytes is None:
        raise ValueError("Provide image_path or image_bytes.")

    if image_name is None:
        image_name = (
            os.path.basename(os.fspath(image_path))
            if image_path is not None
            else "embedded_image"
        )

    image_input = _vision_image_input(image_path, image_bytes)
    model_name = _pick_vision_model()
    client = ollama.Client(timeout=1800)

    # Vision models can occasionally return malformed JSON even when JSON mode
    # is requested. Retry once with a more compact prompt and once without the
    # JSON formatter, then parse the response locally. We never fabricate graph
    # content when all attempts fail.
    prompts = [
        _image_prompt(),
        _image_prompt() + "\nReturn compact JSON only. Keep all node and edge text exact.",
    ]

    last_content = ""
    last_error = None

    for attempt, prompt in enumerate(prompts, start=1):
        try:
            kwargs = {
                "model": model_name,
                "think": False,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt + "\n/no_think",
                        "images": [image_input],
                    }
                ],
                "options": {
                    "temperature": 0,
                    "num_predict": 3200,
                    "num_ctx": 6144,
                },
            }

            # Ask for JSON in the prompt first. Forced formatting is a fallback
            # because it can produce empty/whitespace-only vision replies.
            if attempt == 2:
                kwargs["format"] = "json"
            response = _chat_vision(client, **kwargs)

            content = (
                response["message"]["content"]
                if isinstance(response, dict)
                else response.message.content
            )
            last_content = str(content or "")

            payload = _parse_json(last_content)
            if payload:
                return [_segment_from_payload(payload, image_name)]

            last_error = RuntimeError(
                f"invalid JSON on attempt {attempt}: "
                f"response length={len(last_content)}, "
                f"finish reason={_response_field(response, 'done_reason')}, "
                f"generated tokens={_response_field(response, 'eval_count')}"
            )

        except Exception as exc:
            last_error = exc

    # Final fallback: ask the model for plain text once. We only use the result
    # when it actually contains parseable JSON, so ingestion remains safe.
    try:
        response = _chat_vision(client,
            model=model_name,
            think=False,
            messages=[
                {
                    "role": "user",
                    "content": (
                        _image_prompt()
                        + "\nDo not use markdown. Return one compact JSON object and nothing else."
                        + "\n/no_think"
                    ),
                    "images": [image_input],
                }
            ],
            options={
                "temperature": 0,
                "num_predict": 3200,
                "num_ctx": 6144,
            },
        )

        content = (
            response["message"]["content"]
            if isinstance(response, dict)
            else response.message.content
        )
        last_content = str(content or "")

        payload = _parse_json(last_content)
        if payload:
            return [_segment_from_payload(payload, image_name)]

        last_error = RuntimeError(
            f"invalid JSON after fallback: response length={len(last_content)}, "
            f"finish reason={_response_field(response, 'done_reason')}, "
            f"generated tokens={_response_field(response, 'eval_count')}"
        )

    except Exception as exc:
        last_error = exc

    print(
        f"Warning: Ollama could not return valid structured JSON while reading "
        f"image '{image_name}' with model '{model_name}'. "
        f"Image enrichment failed. Last error: {last_error}",
        flush=True,
    )
    return [_fallback_image_segment(image_name, last_error)]

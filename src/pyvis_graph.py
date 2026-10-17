import html
import math
import os
import re
from collections import defaultdict, deque

try:
    from pyvis.network import Network
except ModuleNotFoundError:
    Network = None


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

TYPE_TO_SHAPE = {
    "start": "ellipse",
    "end": "ellipse",
    "decision": "diamond",
    "process": "box",
    "input": "box",
    "output": "box",
    "database": "database",
    "document": "box",
    "data": "box",
    "entity": "dot",
    "other": "dot",
}

TYPE_TO_SIZE = {
    "start": 32,
    "end": 30,
    "decision": 34,
    "process": 25,
    "input": 24,
    "output": 24,
    "database": 27,
    "document": 25,
    "data": 24,
    "entity": 22,
    "other": 22,
}

NODE_COLORS = [
    {"background": "#22c7e8", "border": "#0891b2"},
    {"background": "#dfc38f", "border": "#b9954e"},
    {"background": "#3ee179", "border": "#16a34a"},
    {"background": "#ff8fc2", "border": "#ec4899"},
    {"background": "#a78bfa", "border": "#7c3aed"},
    {"background": "#93c5fd", "border": "#2563eb"},
]


def _clean_label(value):
    value = re.sub(r"\s+", " ", str(value or "")).strip()
    return value


def _wrap_label(value, width=34):
    value = _clean_label(value)
    if len(value) <= width:
        return value

    words = value.split()
    lines = []
    current = ""

    for word in words:
        proposed = f"{current} {word}".strip()
        if current and len(proposed) > width:
            lines.append(current)
            current = word
        else:
            current = proposed

    if current:
        lines.append(current)

    return "\n".join(lines)


def _short_label(value, width=18):
    value = _clean_label(value)
    if len(value) <= width:
        return value
    return value[: width - 3].rstrip() + "..."


def _node_color(seed):
    index = sum(ord(character) for character in str(seed or "")) % len(NODE_COLORS)
    return NODE_COLORS[index]


def _node_key(label, node_id=None, source=None, merge_shared_entities=True):
    label = _clean_label(label)
    if node_id and not merge_shared_entities:
        return f"{source or 'source'}::{node_id}"

    if node_id and merge_shared_entities:
        # A repeated label with a unique node id inside a single source is kept
        # separate only when the exact label occurs more than once.
        return f"ENTITY::{label.casefold()}"

    return f"ENTITY::{label.casefold()}"


def _collect_graph_data(triples, merge_shared_entities=True):
    nodes = {}
    edges = []
    source_by_node = defaultdict(set)
    node_instances_by_source_label = defaultdict(list)
    visual_sources = {
        _clean_label(triple.get("source"))
        for triple in triples
        if str(triple.get("modality", "")).upper() == "IMAGE"
        and (
            triple.get("visual_evidence") is True
            or _clean_label(triple.get("relation")).casefold() in {"flows_to", "leads_to"}
        )
        and (
            triple.get("subject_id")
            or triple.get("object_id")
            or triple.get("subject_node_id")
            or triple.get("object_node_id")
        )
    }

    def should_skip_duplicate_semantic(triple):
        source = _clean_label(triple.get("source"))
        return (
            source in visual_sources
            and str(triple.get("modality", "")).upper() == "IMAGE"
            and triple.get("semantic_evidence") is True
            and triple.get("visual_evidence") is not True
        )

    # Count same labels within each source so duplicate visible boxes do not collapse.
    for triple in triples:
        if should_skip_duplicate_semantic(triple):
            continue

        relation = _clean_label(triple.get("relation"))
        if relation in HIDDEN_RELATIONS:
            continue

        source = _clean_label(triple.get("source"))
        for side in ("subject", "object"):
            label = _clean_label(triple.get(side))
            explicit_id = _clean_label(
                triple.get(f"{side}_id")
                or triple.get(f"{side}_node_id")
                or (
                    triple.get("node_id")
                    if _clean_label(triple.get("relation")) == "__node__"
                    and side == "subject"
                    else None
                )
            )
            if label:
                node_instances_by_source_label[
                    (source, label.casefold())
                ].append(explicit_id)

    duplicate_lookup = {
        key for key, ids in node_instances_by_source_label.items()
        if len({item for item in ids if item}) > 1
    }

    def resolve_node_id(triple, side):
        label = _clean_label(triple.get(side))
        source = _clean_label(triple.get("source"))
        explicit_id = _clean_label(
            triple.get(f"{side}_id")
            or triple.get(f"{side}_node_id")
            or (
                triple.get("node_id")
                if _clean_label(triple.get("relation")) == "__node__"
                and side == "subject"
                else None
            )
        )

        duplicate_key = (source, label.casefold())
        if explicit_id and duplicate_key in duplicate_lookup:
            return f"{source}::{explicit_id}"

        if (
            explicit_id
            and str(triple.get("modality", "")).upper() == "IMAGE"
            and not merge_shared_entities
        ):
            return f"{source}::{explicit_id}"

        return _node_key(
            label,
            explicit_id,
            source=source,
            merge_shared_entities=merge_shared_entities,
        )

    for triple in triples:
        if should_skip_duplicate_semantic(triple):
            continue

        relation = _clean_label(triple.get("relation")) or "RELATED_TO"
        if relation in HIDDEN_RELATIONS:
            # __node__ still carries useful node metadata.
            if relation != "__node__":
                continue

        subject = _clean_label(triple.get("subject"))
        object_name = _clean_label(triple.get("object"))
        source = _clean_label(triple.get("source")) or "unknown"

        if not subject:
            continue

        if relation == "__node__":
            object_name = subject

        subject_id = resolve_node_id(triple, "subject")
        object_id = (
            subject_id
            if relation == "__node__"
            else resolve_node_id(triple, "object")
        )

        subject_type = _clean_label(triple.get("subject_type")) or _clean_label(
            triple.get("node_type")
        ) or "entity"

        object_type = _clean_label(triple.get("object_type")) or subject_type

        subject_seq = triple.get("subject_sequence", triple.get("sequence", 0))
        object_seq = triple.get("object_sequence", triple.get("sequence", 0))

        try:
            subject_seq = int(subject_seq)
        except (TypeError, ValueError):
            subject_seq = 0

        try:
            object_seq = int(object_seq)
        except (TypeError, ValueError):
            object_seq = 0

        for node_id, label, node_type, sequence, side in (
            (subject_id, subject, subject_type, subject_seq, "subject"),
            (object_id, object_name, object_type, object_seq, "object"),
        ):
            if not label:
                continue

            current = nodes.setdefault(
                node_id,
                {
                    "label": label,
                    "type": node_type,
                    "sequence": sequence,
                    "source": source,
                    "sources": set(),
                    "center_x": None,
                    "center_y": None,
                },
            )

            if current["type"] in {"entity", "other"} and node_type not in {"entity", "other"}:
                current["type"] = node_type

            current["sequence"] = min(current["sequence"], sequence)
            current["sources"].add(source)

            coord_key = f"center_{'x' if side == 'subject' else 'y'}"
            if triple.get(coord_key) is not None and current.get(coord_key) is None:
                try:
                    current[coord_key] = float(triple[coord_key])
                except (TypeError, ValueError):
                    pass

            source_by_node[node_id].add(source)

        if relation == "__node__" or not object_name:
            continue

        edge_label = _clean_label(triple.get("details"))
        relation_upper = relation.upper()
        displayed_relation = (
            edge_label
            if edge_label
            else "" if relation_upper in {"FLOWS_TO", "LEADS_TO"}
            else relation_upper
        )

        edge_key = (
            subject_id,
            object_id,
            displayed_relation.casefold(),
            source.casefold(),
        )

        edges.append(
            {
                "key": edge_key,
                "from": subject_id,
                "to": object_id,
                "relation": displayed_relation,
                "raw_relation": relation,
                "source": source,
            }
        )

    unique_edges = []
    seen_edges = set()
    seen_pairs = set()
    for edge in edges:
        if edge["key"] in seen_edges:
            continue
        seen_edges.add(edge["key"])
        pair_key = (
            edge["from"],
            edge["to"],
            edge["source"].casefold(),
        )
        if pair_key in seen_pairs:
            continue
        seen_pairs.add(pair_key)
        unique_edges.append(edge)

    return nodes, unique_edges, source_by_node


def _compact_layout(nodes, edges):
    if not nodes:
        return

    adjacency = defaultdict(set)
    indegree = defaultdict(int)

    for node_id in nodes:
        adjacency[node_id]
        indegree[node_id] = 0

    for edge in edges:
        adjacency[edge["from"]].add(edge["to"])
        adjacency[edge["to"]].add(edge["from"])
        indegree[edge["to"]] += 1

    roots = [
        node_id
        for node_id, data in nodes.items()
        if data["type"] == "start"
    ]

    if not roots:
        roots = [
            node_id
            for node_id, degree in indegree.items()
            if degree == 0
        ]

    if not roots:
        roots = [
            min(nodes, key=lambda item: nodes[item]["sequence"])
        ]

    visited = set()
    depth = {}
    component_index = {}
    component_count = 0

    for root in roots:
        if root in visited:
            continue

        component_index[root] = component_count
        queue = deque([(root, 0)])
        visited.add(root)

        while queue:
            current, current_depth = queue.popleft()
            depth[current] = min(
                depth.get(current, current_depth),
                current_depth,
            )

            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    component_index[neighbor] = component_count
                    queue.append((neighbor, current_depth + 1))

        component_count += 1

    # Handle any isolated/unvisited node.
    for node_id in nodes:
        if node_id in visited:
            continue
        component_index[node_id] = component_count
        depth[node_id] = 0
        component_count += 1

    level_groups = defaultdict(list)
    for node_id, node_data in nodes.items():
        level = depth.get(node_id, node_data["sequence"])
        level_groups[(component_index[node_id], level)].append(node_id)

    max_width = max((len(group) for group in level_groups.values()), default=1)
    x_spacing = 270
    y_spacing = 180

    for (component, level), group in level_groups.items():
        group.sort(
            key=lambda node_id: (
                nodes[node_id]["sequence"],
                nodes[node_id]["label"].casefold(),
            )
        )

        group_width = (len(group) - 1) * x_spacing
        start_x = -group_width / 2

        for index, node_id in enumerate(group):
            x = start_x + index * x_spacing
            y = level * y_spacing

            # Start node is placed at the visual center of its component.
            if nodes[node_id]["type"] == "start":
                x = 0

            nodes[node_id]["x"] = x
            nodes[node_id]["y"] = y
            nodes[node_id]["fixed"] = True

    # Give each disconnected component a horizontal offset.
    offsets = {}
    for node_id in nodes:
        component = component_index[node_id]
        offsets.setdefault(component, len(offsets))
    component_spacing = max(900, (max_width + 1) * x_spacing)

    for node_id in nodes:
        component = component_index[node_id]
        nodes[node_id]["x"] += offsets[component] * component_spacing


def _render_static_graph(nodes, edges, source_by_node, output_path):
    # Static fallback keeps a knowledge-graph look when pyvis is not installed.
    # It lays nodes on a readable circle instead of forcing a flowchart stack.
    ordered_nodes = sorted(
        nodes,
        key=lambda node_id: (
            nodes[node_id].get("sequence", 0),
            nodes[node_id].get("label", "").casefold(),
            str(node_id),
        ),
    )
    radius = max(210, 70 * len(ordered_nodes))
    center_x = radius + 150
    center_y = radius + 150
    for index, node_id in enumerate(ordered_nodes):
        angle = (2 * math.pi * index / max(1, len(ordered_nodes))) - math.pi / 2
        nodes[node_id]["x"] = center_x + math.cos(angle) * radius
        nodes[node_id]["y"] = center_y + math.sin(angle) * radius

    x_values = [data.get("x", 0) for data in nodes.values()]
    y_values = [data.get("y", 0) for data in nodes.values()]
    min_x = min(x_values, default=0)
    min_y = min(y_values, default=0)
    max_x = max(x_values, default=0)
    max_y = max(y_values, default=0)
    padding = 150
    width = max(900, int(max_x - min_x + padding * 2))
    height = max(650, int(max_y - min_y + padding * 2))

    def point(node_id):
        data = nodes[node_id]
        return (
            float(data.get("x", 0)) - min_x + padding,
            float(data.get("y", 0)) - min_y + padding,
        )

    edge_parts = []
    for edge in edges:
        if edge["from"] not in nodes or edge["to"] not in nodes:
            continue
        x1, y1 = point(edge["from"])
        x2, y2 = point(edge["to"])
        dx = x2 - x1
        dy = y2 - y1
        length = math.hypot(dx, dy) or 1
        start_gap = 42
        end_gap = 42
        sx = x1 + dx / length * start_gap
        sy = y1 + dy / length * start_gap
        ex = x2 - dx / length * end_gap
        ey = y2 - dy / length * end_gap
        edge_parts.append(
            f'<line x1="{sx:.1f}" y1="{sy:.1f}" x2="{ex:.1f}" y2="{ey:.1f}" '
            'class="edge" marker-end="url(#arrow)" />'
        )

        if edge.get("relation"):
            mx = (sx + ex) / 2
            my = (sy + ey) / 2
            edge_parts.append(
                f'<text x="{mx:.1f}" y="{my - 8:.1f}" class="edge-label">'
                f'{html.escape(edge["relation"])}</text>'
            )

    node_parts = []
    for node_id, data in nodes.items():
        x, y = point(node_id)
        color = _node_color(data["label"])
        sources = ", ".join(sorted(source_by_node[node_id])) or "unknown"
        title = (
            f"{data['label']} | Type: {data['type']} | Source: {sources}"
        )
        node_parts.append(
            f'<g class="node"><title>{html.escape(title)}</title>'
            f'<circle cx="{x:.1f}" cy="{y:.1f}" r="35" '
            f'fill="{color["background"]}" stroke="{color["border"]}" '
            'stroke-width="3" />'
            f'<text x="{x:.1f}" y="{y + 58:.1f}" class="node-label">'
            f'{html.escape(_short_label(data["label"], 24))}</text></g>'
        )

    output_path = os.fspath(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Knowledge Graph</title>
<style>
body {{
    margin: 0;
    background: #ffffff;
    font-family: Arial, sans-serif;
    color: #111827;
}}
.notice {{
    padding: 10px 14px;
    background: #fff7ed;
    border-bottom: 1px solid #fed7aa;
    font-size: 14px;
}}
.canvas {{
    width: 100vw;
    height: calc(100vh - 39px);
    overflow: auto;
}}
svg {{
    min-width: 100%;
    min-height: 100%;
}}
.edge {{
    stroke: #808080;
    stroke-width: 2.6;
    opacity: 0.9;
}}
.edge-label {{
    font-size: 11px;
    fill: #4b5563;
    text-anchor: middle;
    paint-order: stroke;
    stroke: #ffffff;
    stroke-width: 4px;
}}
.node-label {{
    font-size: 15px;
    fill: #111827;
    text-anchor: middle;
}}
.node circle {{
    filter: drop-shadow(2px 3px 3px rgba(0,0,0,.25));
}}
</style>
</head>
<body>
<div class="notice">Static fallback view: install <b>pyvis</b> for draggable/zoomable graph.</div>
<div class="canvas">
<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}">
<defs>
<marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5"
    markerWidth="7" markerHeight="7" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#808080"></path>
</marker>
</defs>
{"".join(edge_parts)}
{"".join(node_parts)}
</svg>
</div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8", newline="\n") as file:
        file.write(html_text)

    return output_path


def render_graph(
    triples,
    output_path="storage/knowledge_graph.html",
    merge_shared_entities=True,
):
    """
    Render a compact, UTF-8-safe, semantic graph.

    The graph contains only real semantic nodes/relationships. Internal chunk IDs,
    segment IDs, source anchor nodes, and synthetic belongs_to edges are not shown.
    """
    triples = list(triples or [])
    nodes, edges, source_by_node = _collect_graph_data(
        triples,
        merge_shared_entities=merge_shared_entities,
    )

    if not nodes:
        output_path = os.fspath(output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        message = (
            "No displayable graph relationships were found for this view. "
            "This usually means the selected file only has internal metadata "
            "links, or image/flowchart parsing did not extract nodes and arrows. "
            "Reprocess the image after confirming your Ollama vision model is working."
        )
        html_text = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>No Graph Data</title>
<style>
body {{
    margin: 0;
    min-height: 100vh;
    display: grid;
    place-items: center;
    font-family: Arial, sans-serif;
    color: #111827;
    background: #ffffff;
}}
.message {{
    max-width: 720px;
    padding: 24px 28px;
    border: 1px solid #d1d5db;
    border-radius: 8px;
    background: #f9fafb;
    line-height: 1.5;
}}
h1 {{
    margin: 0 0 10px;
    font-size: 22px;
}}
</style>
</head>
<body>
<div class="message">
    <h1>No visible graph for this file</h1>
    <p>{html.escape(message)}</p>
</div>
</body>
</html>
"""
        with open(output_path, "w", encoding="utf-8", newline="\n") as file:
            file.write(html_text)
        return output_path

    if Network is None:
        return _render_static_graph(nodes, edges, source_by_node, output_path)

    network = Network(
        height="900px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#111827",
        cdn_resources="in_line",
        select_menu=False,
        filter_menu=False,
    )

    for node_id, data in nodes.items():
        node_type = data["type"]
        sources = ", ".join(sorted(source_by_node[node_id]))

        tooltip = (
            f"<b>{html.escape(data['label'])}</b><br>"
            f"Type: {html.escape(node_type)}<br>"
            f"Source: {html.escape(sources or 'unknown')}"
        )

        network.add_node(
            node_id,
            label=_short_label(data["label"], width=20),
            title=tooltip,
            shape="dot",
            size=max(20, TYPE_TO_SIZE.get(node_type, 20)),
            color=_node_color(data["label"]),
            margin=14,
            borderWidth=2,
        )

    for edge in edges:
        title = (
            f"<b>{html.escape(edge['raw_relation'])}</b><br>"
            f"Source: {html.escape(edge['source'])}"
        )

        network.add_edge(
            edge["from"],
            edge["to"],
            label=edge["relation"],
            title=title,
            arrows="to",
            width=2.2,
            smooth={"type": "cubicBezier"},
        )

    network.set_options(
        r"""
        {
          "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true,
            "dragNodes": true,
            "zoomView": true
          },
          "physics": {
            "enabled": true,
            "solver": "forceAtlas2Based",
            "forceAtlas2Based": {
              "gravitationalConstant": -180,
              "centralGravity": 0.015,
              "springLength": 360,
              "springConstant": 0.035,
              "damping": 0.78,
              "avoidOverlap": 1
            },
            "stabilization": {
              "enabled": true,
              "iterations": 1800,
              "updateInterval": 50,
              "fit": true
            }
          },
          "edges": {
            "smooth": {
              "enabled": true,
              "type": "continuous",
              "roundness": 0.25
            },
            "font": {
              "size": 9,
              "face": "Arial",
              "align": "middle",
              "strokeWidth": 3,
              "strokeColor": "#ffffff"
            },
            "color": {
              "inherit": false
            }
          },
          "nodes": {
            "font": {
              "size": 13,
              "face": "Arial",
              "multi": true
            },
            "shadow": {
              "enabled": true,
              "size": 8,
              "x": 2,
              "y": 2
            }
          }
        }
        """
    )

    output_path = os.fspath(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Never use network.write_html() here: some environments pick a cp1252/
    # charmaps default and fail on Unicode content. generate_html + UTF-8 avoids it.
    html_text = network.generate_html(
        name=os.path.basename(output_path),
        notebook=False,
    )

    with open(output_path, "w", encoding="utf-8", newline="\n") as file:
        file.write(html_text)

    return output_path

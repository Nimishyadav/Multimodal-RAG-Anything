from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
import os
import re


# =========================================================
# TEXT CLEANING
# =========================================================

def _clean_text(text):
    """
    Clean whitespace while preserving readable content.
    """

    return re.sub(
        r"\s+",
        " ",
        str(text or ""),
    ).strip()


# =========================================================
# ALIGNMENT
# =========================================================

def _get_alignment_name(paragraph):
    """
    Convert Word alignment enum into readable text.
    """

    alignment = paragraph.alignment

    if alignment is None:
        return "NONE"

    mapping = {
        WD_ALIGN_PARAGRAPH.LEFT: "LEFT",
        WD_ALIGN_PARAGRAPH.CENTER: "CENTER",
        WD_ALIGN_PARAGRAPH.RIGHT: "RIGHT",
        WD_ALIGN_PARAGRAPH.JUSTIFY: "JUSTIFY",
    }

    return mapping.get(
        alignment,
        "OTHER",
    )


# =========================================================
# LIST DETECTION
# =========================================================

def _detect_list_type(paragraph):
    """
    Detect common Word list styles.
    """

    style_name = ""

    try:
        style_name = (
            paragraph.style.name
            or ""
        )
    except Exception:
        pass

    style_lower = (
        style_name.lower()
    )

    if (
        "bullet"
        in style_lower
    ):
        return "BULLET"

    if (
        "number"
        in style_lower
    ):
        return "NUMBERED"

    # Fallback: inspect visible text
    text = str(
        paragraph.text
        or ""
    ).strip()

    if re.match(
        r"^[•●○◦▪\-]\s+",
        text,
    ):
        return "BULLET"

    if re.match(
        r"^\d+[\.\)]\s+",
        text,
    ):
        return "NUMBERED"

    return ""


# =========================================================
# FORMATTING DETECTION
# =========================================================

def _detect_run_formatting(paragraph):
    """
    Detect bold, italic, underline,
    strike and monospace-like formatting.
    """

    formats = []

    for run in paragraph.runs:

        text = str(
            run.text or ""
        )

        if not text.strip():
            continue

        if run.bold:
            formats.append("BOLD")

        if run.italic:
            formats.append("ITALIC")

        if run.underline:
            formats.append("UNDERLINE")

        if run.font.strike:
            formats.append("STRIKETHROUGH")

        font_name = (
            str(
                run.font.name
                or ""
            ).lower()
        )

        if (
            "courier"
            in font_name
            or
            "mono"
            in font_name
        ):
            formats.append("MONOSPACE")

    return sorted(
        set(formats)
    )


# =========================================================
# PARAGRAPH SEGMENT
# =========================================================

def _paragraph_segment(
    paragraph,
    document_name,
    paragraph_index,
):
    """
    Convert each Word paragraph into an individual
    structured segment.

    IMPORTANT:
    Every paragraph stays separate.
    """

    text = _clean_text(
        paragraph.text
    )

    if not text:
        return None

    # -----------------------------------------------------
    # STYLE
    # -----------------------------------------------------

    try:
        style_name = (
            paragraph.style.name
            or ""
        )
    except Exception:
        style_name = ""

    style_lower = (
        style_name.lower()
    )

    # -----------------------------------------------------
    # HEADING
    # -----------------------------------------------------

    is_heading = (
        style_lower.startswith(
            "heading"
        )
        or
        style_lower in {
            "title",
            "subtitle",
        }
    )

    # -----------------------------------------------------
    # LIST
    # -----------------------------------------------------

    list_type = _detect_list_type(
        paragraph
    )

    # -----------------------------------------------------
    # FORMATTING
    # -----------------------------------------------------

    formatting = (
        _detect_run_formatting(
            paragraph
        )
    )

    # -----------------------------------------------------
    # FONT SIZE
    # -----------------------------------------------------

    font_sizes = []

    for run in paragraph.runs:

        if (
            run.font.size
            is not None
        ):

            try:
                font_sizes.append(
                    round(
                        run.font.size.pt,
                        1,
                    )
                )
            except Exception:
                pass

    font_size = (
        max(font_sizes)
        if font_sizes
        else None
    )

    # -----------------------------------------------------
    # ALIGNMENT
    # -----------------------------------------------------

    alignment = (
        _get_alignment_name(
            paragraph
        )
    )

    # -----------------------------------------------------
    # ELEMENT TYPE
    # -----------------------------------------------------

    if is_heading:

        element_type = "HEADING"

    elif list_type:

        element_type = (
            "LIST_ITEM"
        )

    else:

        element_type = "PARAGRAPH"

    # -----------------------------------------------------
    # METADATA PREFIX
    # -----------------------------------------------------

    metadata_lines = [
        f"Word document: {document_name}",
        f"Element type: {element_type}",
    ]

    if style_name:
        metadata_lines.append(
            f"Word style: {style_name}"
        )

    if list_type:
        metadata_lines.append(
            f"List type: {list_type}"
        )

    metadata_lines.append(
        f"Alignment: {alignment}"
    )

    if font_size is not None:
        metadata_lines.append(
            f"Font size: {font_size} pt"
        )

    if formatting:
        metadata_lines.append(
            "Formatting: "
            + ", ".join(formatting)
        )

    metadata_lines.append(
        f"Content: {text}"
    )

    final_text = "\n".join(
        metadata_lines
    )

    # -----------------------------------------------------
    # RETURN SEGMENT
    # -----------------------------------------------------

    return {
        "id":
            f"{document_name}:"
            f"paragraph:"
            f"{paragraph_index}",

        "text":
            final_text,

        "type":
            "TEXT",

        "modality":
            "TEXT",

        "paragraph_index":
            paragraph_index,

        "style":
            style_name,

        "is_heading":
            is_heading,

        "element_type":
            element_type,

        "list_type":
            list_type,

        "formatting":
            formatting,

        "alignment":
            alignment,

        "font_size":
            font_size,

        "source":
            document_name,
    }


# =========================================================
# TABLE SEGMENT
# =========================================================

def _table_segment(
    table,
    document_name,
    table_index,
):
    """
    Convert Word tables into structured segments.

    Each table stays separate.
    """

    rows = []

    for row_index, row in enumerate(
        table.rows,
        start=1,
    ):

        values = [
            _clean_text(
                cell.text.replace(
                    "\n",
                    " ",
                )
            )
            for cell in row.cells
        ]

        if any(values):

            rows.append(
                {
                    "row_index":
                        row_index,

                    "values":
                        values,
                }
            )

    if not rows:
        return None

    lines = [
        f"Word document: {document_name}",
        f"Element type: TABLE",
        f"Table index: {table_index}",
        "",
        "TABLE CONTENT:",
    ]

    for row in rows:

        lines.append(
            f"Row {row['row_index']}: "
            + " | ".join(
                row["values"]
            )
        )

    return {
        "id":
            f"{document_name}:"
            f"table:"
            f"{table_index}",

        "text":
            "\n".join(lines),

        "type":
            "TABLE",

        "modality":
            "TABLE",

        "table_index":
            table_index,

        "table_rows":
            rows,

        "source":
            document_name,
    }


# =========================================================
# READ DOCX
# =========================================================

def read_docx(
    docx_path
):
    """
    Extract a DOCX while preserving document structure.

    Extracts:

        - title
        - headings
        - paragraphs
        - bullet lists
        - numbered lists
        - formatting information
        - alignment
        - font size
        - tables
    """

    document = Document(
        docx_path
    )

    document_name = os.path.basename(
        docx_path
    )

    sections = []

    # =====================================================
    # PARAGRAPHS
    # =====================================================

    for paragraph_index, paragraph in enumerate(
        document.paragraphs,
        start=1,
    ):

        segment = _paragraph_segment(
            paragraph,
            document_name,
            paragraph_index,
        )

        if segment:

            sections.append(
                segment
            )

    # =====================================================
    # TABLES
    # =====================================================

    for table_index, table in enumerate(
        document.tables,
        start=1,
    ):

        segment = _table_segment(
            table,
            document_name,
            table_index,
        )

        if segment:

            sections.append(
                segment
            )

    # =====================================================
    # STATISTICS
    # =====================================================

    paragraph_count = len(
        [
            p
            for p in document.paragraphs
            if _clean_text(p.text)
        ]
    )

    heading_count = len(
        [
            s
            for s in sections
            if s.get(
                "is_heading",
                False,
            )
        ]
    )

    bullet_count = len(
        [
            s
            for s in sections
            if s.get(
                "list_type"
            )
            == "BULLET"
        ]
    )

    numbered_count = len(
        [
            s
            for s in sections
            if s.get(
                "list_type"
            )
            == "NUMBERED"
        ]
    )

    stats = {
        "pages": 0,
        "tables":
            len(document.tables),
        "images":
            0,
        "paragraphs":
            paragraph_count,
        "headings":
            heading_count,
        "bullet_items":
            bullet_count,
        "numbered_items":
            numbered_count,
    }

    return sections, stats
"""PPTX reader compatibility wrapper.

MinerU is the primary and only document parser for PPTX in this project.
This module is kept as a compatibility import for older code that calls
read_presentation().
"""

from src.mineru_reader import read_mineru_document


def read_presentation(presentation_path):
    """Parse a PPTX file through MinerU."""
    return read_mineru_document(presentation_path)

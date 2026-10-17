"""DOCX reader compatibility wrapper.

MinerU is the primary and only document parser for DOCX in this project.
This module is kept as a compatibility import for older code that calls
read_word().
"""

from src.mineru_reader import read_mineru_document


def read_word(word_path):
    """Parse a DOCX file through MinerU."""
    return read_mineru_document(word_path)

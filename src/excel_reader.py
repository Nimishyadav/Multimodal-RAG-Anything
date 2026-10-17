"""Excel reader compatibility wrapper.

MinerU is the primary and only document parser for XLS/XLSX in this project.
This module is kept as a compatibility import for older code that calls
read_excel().
"""

from src.mineru_reader import read_mineru_document


def read_excel(excel_path):
    """Parse an XLS/XLSX file through MinerU."""
    return read_mineru_document(excel_path)

from src.mineru_reader import read_mineru_document


def read_pdf(pdf_path):
    """Keep the legacy PDF-reader entry point backed by local MinerU."""
    return read_mineru_document(pdf_path)

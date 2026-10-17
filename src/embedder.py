import sys

try:
    from sentence_transformers import SentenceTransformer
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The 'sentence-transformers' package is not installed in the Python "
        "environment running this project. The import name is "
        "'sentence_transformers', but the package you install is "
        "'sentence-transformers'.\n\n"
        f"Current Python: {sys.executable}\n\n"
        "Install all project dependencies with the same Python you use to run "
        "ingest.py:\n\n"
        "    python -m pip install -r requirements.txt\n\n"
        "Or install just this missing package:\n\n"
        "    python -m pip install sentence-transformers\n\n"
        "If VS Code is using C:\\msys64\\ucrt64\\bin\\python.exe, switch to a "
        "normal Windows Python interpreter first."
    ) from exc

_model = None  


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def create_embeddings(texts):
    """Converts a list of text strings into a list of vectors (numbers)."""
    model = get_model()
    return model.encode(texts)

from functools import lru_cache
from src.config import OLLAMA_HOST, TEXT_MODEL_NAME, VISION_MODEL_NAME

try:
    import ollama
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The 'ollama' Python package is not installed in the interpreter "
        "running this project. Install the project dependencies with the same "
        "Python you use to run ingest.py, for example:\n\n"
        "    py -3.13 -m pip install -r requirements.txt\n\n"
        "Then run ingestion with:\n\n"
        "    py -3.13 ingest.py\n\n"
        "If VS Code is running C:\\msys64\\ucrt64\\bin\\python.exe, switch the "
        "workspace interpreter to a standard Windows Python installation."
    ) from exc


# =========================================================
# LOCAL MODEL CONFIGURATION
# =========================================================

FAST_MODEL_NAME = TEXT_MODEL_NAME
OLLAMA_CONTEXT_TOKENS = 8192
VISION_CONTEXT_TOKENS = 8192  # allow room for image tokens, prompt and final JSON
VISION_MAX_ATTEMPTS = 3


# =========================================================
# OLLAMA CLIENT
# =========================================================

@lru_cache(maxsize=1)
def _get_client():
    """
    Reuse one Ollama client.
    Ollama server normally runs on localhost:11434.
    """

    return ollama.Client(
        host=OLLAMA_HOST,
        timeout=1800,
    )


def _add_no_think_instruction(messages):
    """Append Qwen's no-think control token to the most recent user message."""
    request_messages = [dict(message) for message in messages]
    for message in reversed(request_messages):
        if message.get("role") == "user":
            content = message.get("content")
            if not isinstance(content, str):
                raise TypeError("Ollama user message content must be text.")
            message["content"] = f"{content.rstrip()}\n/no_think"
            return request_messages
    raise ValueError("Ollama chat requests must include a user message.")


# =========================================================
# TEXT COMPLETION
# =========================================================

def chat_completion(
    messages,
    *,
    model=TEXT_MODEL_NAME,
):
    """
    Send a text-only request to local Ollama.
    """

    response = _get_client().chat(
        model=model,
        messages=_add_no_think_instruction(messages),
        think=False,
        options={
            "num_predict": -1,
            "num_ctx": OLLAMA_CONTEXT_TOKENS,
            "temperature": 0,
        },
    )

    content = response.message.content or ""
    if not content.strip():
        raise RuntimeError(
            f"Ollama model {model!r} returned no final response "
            f"(finish reason: {response.done_reason or 'unknown'}, "
            f"generated tokens: {response.eval_count or 0}, "
            f"context: {OLLAMA_CONTEXT_TOKENS})."
        )
    return content


def vision_completion(
    prompt,
    image_path,
    *,
    model=VISION_MODEL_NAME,
):
    """Send a prompt and local image to the configured multimodal Ollama model.

    Retries a few times, because local vision models can return an empty
    response under VRAM pressure. The first attempt asks for strict JSON
    mode; later attempts drop that constraint, since some models return
    nothing at all when forced into strict JSON under load."""

    last_error = None

    for attempt in range(1, VISION_MAX_ATTEMPTS + 1):
        use_json_mode = attempt == 1  # only force it on the first try

        request_kwargs = dict(
            model=model,
            think=False,
            messages=[
                {
                    "role": "user",
                    "content": prompt + "\n/no_think",
                    "images": [str(image_path)],
                }
            ],
            options={
                "num_predict": 2048,
                "num_ctx": VISION_CONTEXT_TOKENS,
                "temperature": 0,
            },
        )
        if use_json_mode:
            request_kwargs["format"] = "json"

        try:
            response = _get_client().chat(**request_kwargs)
        except Exception as exc:  # connection drop, model load failure, etc.
            last_error = exc
            continue

        content = response.message.content or ""
        if content.strip():
            return content

        last_error = RuntimeError(
            f"Ollama vision model {model!r} returned no final response on "
            f"attempt {attempt}/{VISION_MAX_ATTEMPTS} "
            f"(finish reason: {response.done_reason or 'unknown'}, "
            f"generated tokens: {response.eval_count or 0}, "
            f"json_mode: {use_json_mode})."
        )

    raise RuntimeError(
        f"Ollama vision model {model!r} returned no usable response after "
        f"{VISION_MAX_ATTEMPTS} attempts. Last error: {last_error}"
    )

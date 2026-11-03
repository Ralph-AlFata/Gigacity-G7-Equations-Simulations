"""Secrets from the environment or a `.env` file. Values are returned, never printed."""

import os
from pathlib import Path

HF_TOKEN_KEYS = ("HF_TOKEN", "HUGGING_FACE_KEY", "HUGGINGFACE_TOKEN")


def read_dotenv(path: Path) -> dict[str, str]:
    """Parses `KEY=VALUE` lines; ignores blanks, comments and surrounding quotes."""
    values = {}
    if not path.is_file():
        return values
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        values[key] = value.strip().strip("'\"")
    return values


def load_hf_token(dotenv: Path = Path(".env")) -> str | None:
    """Returns the Hugging Face token from the environment, then `.env`; None if absent.

    None lets `huggingface_hub` fall back to a token saved with `hf auth login`.
    """
    for key in HF_TOKEN_KEYS:
        if os.environ.get(key):
            return os.environ[key]
    file_values = read_dotenv(dotenv)
    for key in HF_TOKEN_KEYS:
        if file_values.get(key):
            return file_values[key]
    return None

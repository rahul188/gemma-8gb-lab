"""
Central model registry for gemma-8gb-lab.

Each entry describes a model that can be loaded in ≤8 GB VRAM (4-bit).
Keys are used by app.py, chat.py, and finetune_gemma4.py.
"""

from pathlib import Path

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# MODELS catalog
# ---------------------------------------------------------------------------
# Fields:
#   model_id       – HuggingFace / Unsloth model repo string
#   label          – human-readable name shown in the dashboard
#   max_seq        – recommended max_seq_length for 8 GB VRAM
#   chat_template  – unsloth chat-template name (None = use model default)
#   adapter_dir    – local LoRA adapter path (None = no local adapter)
#   use_exact      – pass use_exact_model_name=True to FastModel.from_pretrained
#   vram_note      – rough 4-bit VRAM estimate (informational)
# ---------------------------------------------------------------------------
MODELS: dict[str, dict] = {
    "gemma3-4b": {
        "model_id":      "unsloth/gemma-3-4b-it-bnb-4bit",
        "label":         "Gemma 3 4B · QLoRA fine-tuned · local inference",
        "max_seq":       4096,
        "chat_template": "gemma-3",
        "adapter_dir":   str(HERE / "gemma3-4b-lora"),
        "use_exact":     False,
        "vram_note":     "~3.0 GB (fine-tuned adapter)",
    },
    "gemma4-e2b": {
        "model_id":      "unsloth/gemma-4-E2B-it",
        "label":         "Gemma 4 E2B · base model · local inference",
        "max_seq":       2048,
        "chat_template": "gemma-4",
        "adapter_dir":   str(HERE / "gemma4-e2b-lora"),
        "use_exact":     True,
        "vram_note":     "~7.1 GB (barely fits, close all other apps)",
    },
    "llama32-3b": {
        "model_id":      "unsloth/Llama-3.2-3B-Instruct-bnb-4bit",
        "label":         "Llama 3.2 3B · base model · local inference",
        "max_seq":       4096,
        "chat_template": "llama-3.2",
        "adapter_dir":   str(HERE / "llama32-3b-lora"),
        "use_exact":     False,
        "vram_note":     "~1.7 GB",
    },
    "llama31-8b": {
        "model_id":      "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        "label":         "Llama 3.1 8B · base model · local inference",
        "max_seq":       4096,
        "chat_template": "llama-3.1",
        "adapter_dir":   str(HERE / "llama31-8b-lora"),
        "use_exact":     False,
        "vram_note":     "~4.6 GB",
    },
    "qwen25-7b": {
        "model_id":      "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "label":         "Qwen 2.5 7B · base model · local inference",
        "max_seq":       4096,
        "chat_template": "qwen-2.5",
        "adapter_dir":   str(HERE / "qwen25-7b-lora"),
        "use_exact":     False,
        "vram_note":     "~4.1 GB",
    },
    "mistral-7b": {
        "model_id":      "unsloth/mistral-7b-v0.3-bnb-4bit",
        "label":         "Mistral 7B v0.3 · base model · local inference",
        "max_seq":       4096,
        "chat_template": "mistral",
        "adapter_dir":   str(HERE / "mistral-7b-lora"),
        "use_exact":     False,
        "vram_note":     "~4.1 GB",
    },
    "phi3-mini": {
        "model_id":      "unsloth/Phi-3-mini-4k-instruct",
        "label":         "Phi-3 Mini 4K · base model · local inference",
        "max_seq":       4096,
        "chat_template": "phi-3",
        "adapter_dir":   str(HERE / "phi3-mini-lora"),
        "use_exact":     False,
        "vram_note":     "~2.2 GB",
    },
}

# ---------------------------------------------------------------------------
# Ordered chat-template candidates keyed by model-id substrings
# ---------------------------------------------------------------------------
_TEMPLATE_ORDER: list[tuple[str, tuple[str, ...]]] = [
    ("gemma-4",   ("gemma-4", "gemma-3")),
    ("gemma-3",   ("gemma-3", "gemma-4")),
    ("llama-3.2", ("llama-3.2", "llama-3.1", "llama-3")),
    ("llama-3.1", ("llama-3.1", "llama-3")),
    ("llama",     ("llama-3", "llama-3.1")),
    ("qwen2.5",   ("qwen-2.5",)),
    ("qwen",      ("qwen-2.5",)),
    ("mistral",   ("mistral",)),
    ("phi-3",     ("phi-3",)),
    ("phi",       ("phi-3",)),
]


def detect_template_order(model_id: str) -> tuple[str, ...]:
    """Return an ordered tuple of chat-template names to try for *model_id*."""
    low = model_id.lower()
    for keyword, templates in _TEMPLATE_ORDER:
        if keyword in low:
            return templates
    return ("gemma-3",)


def resolve(key_or_path: str) -> dict:
    """
    Return the MODELS entry for *key_or_path*.

    Lookup order:
    1. Exact key match in MODELS (e.g. ``"llama32-3b"``).
    2. ``model_id`` field match (e.g. the full HF repo string).
    3. Path on disk that looks like a local adapter directory.
    4. Unknown — return a synthetic minimal entry so the caller keeps working.
    """
    if key_or_path in MODELS:
        return MODELS[key_or_path]

    for entry in MODELS.values():
        if entry["model_id"] == key_or_path:
            return entry

    # Local adapter directory or arbitrary HF path
    is_hub = "/" in key_or_path and not Path(key_or_path).exists()
    return {
        "model_id":      key_or_path,
        "label":         key_or_path,
        "max_seq":       4096,
        "chat_template": None,
        "adapter_dir":   key_or_path if not is_hub else None,
        "use_exact":     is_hub,
        "vram_note":     "unknown",
    }

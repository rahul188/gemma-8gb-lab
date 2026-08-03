"""
Interactive chat with any supported model (or a local LoRA adapter).

Usage:
    python chat.py                      # chat with your fine-tuned Gemma 3 4B
    python chat.py --base               # chat with the base model (no adapter)
    python chat.py --model llama32-3b   # chat with Llama 3.2 3B
    python chat.py --model qwen25-7b    # chat with Qwen 2.5 7B
    python chat.py --list               # list all available model keys
"""

import sys
from pathlib import Path

from models import MODELS, resolve

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
if "--list" in sys.argv:
    print("Available model keys:")
    for key, v in MODELS.items():
        adapter_exists = v.get("adapter_dir") and Path(v["adapter_dir"]).exists()
        adapter_tag = "  [adapter ready]" if adapter_exists else ""
        print(f"  {key:<16}  {v['vram_note']:<24}  {v['label']}{adapter_tag}")
    sys.exit(0)

# Resolve which model to load
use_base = "--base" in sys.argv

model_key = "gemma3-4b"  # default
for i, arg in enumerate(sys.argv[1:], 1):
    if arg == "--model" and i < len(sys.argv):
        model_key = sys.argv[i + 1]
        break

entry = resolve(model_key)

if use_base:
    # Force the base HF model, skip any local adapter
    model_path = entry["model_id"]
else:
    adapter = entry.get("adapter_dir")
    model_path = adapter if adapter and Path(adapter).exists() else entry["model_id"]

from unsloth import FastModel  # noqa: E402  (import after path logic; slow)

print(f"Loading {model_path} ...")
model, tokenizer = FastModel.from_pretrained(
    model_name=model_path,
    max_seq_length=entry["max_seq"],
    load_in_4bit=True,
    **({"use_exact_model_name": True} if entry.get("use_exact") else {}),
)
FastModel.for_inference(model)
print(f"Ready ({entry['label']}). Type a message (or 'exit').\n")

while True:
    try:
        user = input("you > ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not user or user.lower() in {"exit", "quit"}:
        break
    messages = [{"role": "user", "content": [{"type": "text", "text": user}]}]
    try:
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        ).to("cuda")
    except Exception:
        # plain-string content fallback for text-only templates
        messages = [{"role": "user", "content": user}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        ).to("cuda")
    out = model.generate(
        input_ids=inputs, max_new_tokens=512, temperature=0.7, top_p=0.95
    )
    reply = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
    print(f"\n{model_key} > {reply}\n")

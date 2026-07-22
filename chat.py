"""
Interactive chat with the fine-tuned LoRA model (or the base model with --base).

Usage:
    python chat.py           # chat with your fine-tuned model
    python chat.py --base    # chat with the untouched base model
"""

import sys
from pathlib import Path

BASE_MODEL = "unsloth/gemma-3-4b-it-bnb-4bit"

here = Path(__file__).parent
adapter_dir = next(
    (d for d in (here / "gemma3-4b-lora", here / "gemma4-e2b-lora") if d.exists()),
    None,
)

use_base = "--base" in sys.argv
model_path = BASE_MODEL if use_base or adapter_dir is None else str(adapter_dir)

from unsloth import FastModel  # noqa: E402  (import after path logic; slow)

print(f"Loading {model_path} ...")
model, tokenizer = FastModel.from_pretrained(
    model_name=model_path,
    max_seq_length=2048,
    load_in_4bit=True,
)
FastModel.for_inference(model)
print("Ready. Type a message (or 'exit').\n")

while True:
    try:
        user = input("you > ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not user or user.lower() in {"exit", "quit"}:
        break
    messages = [{"role": "user", "content": [{"type": "text", "text": user}]}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
    ).to("cuda")
    out = model.generate(
        input_ids=inputs, max_new_tokens=512, temperature=0.7, top_p=0.95
    )
    print("\ngemma >", tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip(), "\n")

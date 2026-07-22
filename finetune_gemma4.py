"""
QLoRA fine-tune of Gemma 4 E2B on an 8GB RTX 4060 Ti (Windows, Unsloth).

Captures before/after generations and training metrics into ./results/
so they can be used as blog material.
"""

import json
import os
import time
from pathlib import Path

# Env overrides so the same script can run other models, e.g. the true
# Gemma 4 E2B attempt:  FT_MODEL=unsloth/gemma-4-E2B-it FT_EXACT=1
FT_MODEL = os.environ.get("FT_MODEL")
FT_SEQ = os.environ.get("FT_SEQ")
FT_EXACT = os.environ.get("FT_EXACT") == "1"
# Force the whole model onto GPU 0, bypassing accelerate's fit-check; on
# Windows WDDM small overshoot pages into shared memory instead of failing.
FT_FORCE_GPU = os.environ.get("FT_FORCE_GPU") == "1"
# Load + generate only, skip training — the quick "does it even run" test.
FT_INFER_ONLY = os.environ.get("FT_INFER_ONLY") == "1"

RESULTS = Path(__file__).parent / "results"
RESULTS.mkdir(exist_ok=True)

TEST_PROMPTS = [
    "Explain LoRA fine-tuning to a backend developer in 3 sentences.",
    "Write a Python one-liner to count word frequencies in a string.",
    "What is the capital of Telangana, and what is it known for in tech?",
]

# Gemma 4 E2B's pre-made dynamic 4-bit quant alone is 7.57GB of weights — more
# than this whole 8GB card. Default is Gemma 3 4B plain bnb-4bit (3.01GB); the
# E2B attempt instead loads the 16-bit repo with on-the-fly full 4-bit quant
# (FT_MODEL + FT_EXACT=1 so unsloth doesn't redirect to the dynamic quant repo).
MODEL_NAME = FT_MODEL or "unsloth/gemma-3-4b-it-bnb-4bit"
MAX_SEQ_LENGTH = int(FT_SEQ or 1024)  # keep activations small on 8GB
NUM_SAMPLES = 1000     # subset of FineTome-100k
MAX_STEPS = 60         # same as the official notebook


def generate(model, tokenizer, prompts, tag):
    from unsloth import FastModel
    FastModel.for_inference(model)
    outs = []
    for p in prompts:
        messages = [{"role": "user", "content": [{"type": "text", "text": p}]}]
        try:
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_tensors="pt",
            ).to("cuda")
        except Exception:
            # plain-string content fallback for text-only templates
            messages = [{"role": "user", "content": p}]
            inputs = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True,
                return_tensors="pt",
            ).to("cuda")
        out = model.generate(
            input_ids=inputs, max_new_tokens=200, temperature=0.7, top_p=0.95
        )
        text = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)
        outs.append({"prompt": p, "response": text.strip()})
        print(f"\n[{tag}] {p}\n{'-' * 60}\n{text.strip()[:500]}\n")
    return outs


def main():
    import torch
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template, standardize_sharegpt
    from datasets import load_dataset
    from trl import SFTConfig, SFTTrainer

    t0 = time.time()
    print(f"Loading {MODEL_NAME} in 4-bit ...")
    model, tokenizer = FastModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        load_in_8bit=False,
        full_finetuning=False,
        use_exact_model_name=FT_EXACT,
        **({"device_map": {"": 0}} if FT_FORCE_GPU else {}),
    )
    load_time = time.time() - t0
    load_vram_gb = torch.cuda.memory_reserved() / 1024**3
    print(f"Model loaded in {load_time:.0f}s, VRAM reserved: {load_vram_gb:.2f} GB")

    template_order = (
        ("gemma-4", "gemma-3") if "gemma-4" in MODEL_NAME.lower()
        else ("gemma-3", "gemma-4")
    )
    for name in template_order:
        try:
            tokenizer = get_chat_template(tokenizer, chat_template=name)
            chat_template_used = name
            break
        except Exception as e:
            print(f"chat template {name!r} failed: {e}")
    else:
        chat_template_used = "model default"

    # Baseline generations before any training
    import torch as _t
    _t.cuda.reset_peak_memory_stats()
    g0 = time.time()
    before = generate(model, tokenizer, TEST_PROMPTS, "BEFORE")
    gen_time = time.time() - g0
    gen_peak_gb = _t.cuda.max_memory_reserved() / 1024**3

    if FT_INFER_ONLY:
        info = {
            "model": MODEL_NAME,
            "mode": "inference-only",
            "model_load_seconds": round(load_time, 1),
            "vram_after_load_gb": round(load_vram_gb, 2),
            "generation_peak_vram_gb": round(gen_peak_gb, 2),
            "three_prompts_seconds": round(gen_time, 1),
        }
        tag = "-e2b" if "gemma-4" in MODEL_NAME.lower() else ""
        (RESULTS / f"metrics{tag}-infer.json").write_text(json.dumps(info, indent=2))
        (RESULTS / f"generations{tag}-infer.json").write_text(
            json.dumps({"before": before}, indent=2)
        )
        print("\nINFER-ONLY RESULT:", json.dumps(info, indent=2))
        return

    model = FastModel.get_peft_model(
        model,
        r=8,
        lora_alpha=8,
        lora_dropout=0,
        bias="none",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    print("Loading dataset (FineTome-100k subset) ...")
    dataset = load_dataset("mlabonne/FineTome-100k", split=f"train[:{NUM_SAMPLES}]")
    dataset = standardize_sharegpt(dataset)

    def formatting(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                c, tokenize=False, add_generation_prompt=False
            )
            for c in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting, batched=True)
    print("Sample formatted row:\n", dataset[0]["text"][:600])

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            dataset_text_field="text",
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            max_steps=MAX_STEPS,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=str(RESULTS / "checkpoints"),
            report_to="none",
            # no dataset_num_proc on Windows: any worker process fails to
            # unpickle Unsloth's compiled trainer module; main-process is fine
        ),
    )

    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    stats = trainer.train()
    train_time = time.time() - t1
    peak_vram_gb = torch.cuda.max_memory_reserved() / 1024**3

    losses = [
        {"step": h["step"], "loss": h["loss"]}
        for h in trainer.state.log_history
        if "loss" in h
    ]
    print(f"\nTraining done in {train_time / 60:.1f} min, peak VRAM {peak_vram_gb:.2f} GB")
    print(f"First loss: {losses[0]['loss']:.3f}  ->  last loss: {losses[-1]['loss']:.3f}")

    after = generate(model, tokenizer, TEST_PROMPTS, "AFTER")

    is_g4 = "gemma-4" in MODEL_NAME.lower()
    adapter_dir = Path(__file__).parent / ("gemma4-e2b-lora" if is_g4 else "gemma3-4b-lora")
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metrics = {
        "model": MODEL_NAME,
        "gpu": torch.cuda.get_device_name(0),
        "chat_template": chat_template_used,
        "max_seq_length": MAX_SEQ_LENGTH,
        "num_samples": NUM_SAMPLES,
        "max_steps": MAX_STEPS,
        "model_load_seconds": round(load_time, 1),
        "vram_after_load_gb": round(load_vram_gb, 2),
        "train_minutes": round(train_time / 60, 1),
        "peak_vram_gb": round(peak_vram_gb, 2),
        "train_runtime_s": stats.metrics.get("train_runtime"),
        "train_samples_per_second": stats.metrics.get("train_samples_per_second"),
        "losses": losses,
    }
    tag = "-e2b" if is_g4 else ""
    (RESULTS / f"metrics{tag}.json").write_text(json.dumps(metrics, indent=2))
    (RESULTS / f"generations{tag}.json").write_text(
        json.dumps({"before": before, "after": after}, indent=2)
    )
    print(f"\nSaved metrics + generations to {RESULTS}, adapters to {adapter_dir}")


if __name__ == "__main__":
    main()

# Blog notes: I actually tried "fine-tune an LLM on a laptop GPU" — here's what happened

Raw material from a real run on 2026-07-22. All numbers below were measured on
this machine, not quoted from anyone's docs.

## The claim being tested

A viral LinkedIn post (and the Unsloth docs) say you can fine-tune Gemma with
as little as 8GB VRAM — "the 2B model runs on 8GB, the 4B on around 10GB."

## My hardware

- NVIDIA GeForce RTX 4060 Ti, 8GB VRAM (7.996GB usable)
- Windows 11, Python 3.12, no WSL — native Windows all the way
- Reality check: a normal desktop (browser, Teams, WhatsApp) already eats
  ~2.6GB of VRAM, leaving ~5.3GB actually free

## Finding 1: "2B runs on 8GB" — not the Gemma 4 one, not on a desktop you're using

Gemma 4 E2B's 4-bit dynamic quant is **7.57GB of weights alone**
(measured via the HF API on `unsloth/gemma-4-E2B-it-unsloth-bnb-4bit`).
That's more than the entire card before a single activation is allocated.
Loading it threw:

> ValueError: Some modules are dispatched on the CPU or the disk. Make sure
> you have enough GPU RAM to fit the quantized model.

The "E2B = effective 2B" naming hides a much larger raw parameter count
(the MatFormer/per-layer-embedding trick). VRAM claims for it assume a
dedicated GPU and aggressive offloading — not a Windows desktop with apps open.

**What did fit:** `unsloth/gemma-3-4b-it-bnb-4bit` — plain 4-bit, 3.01GB of
weights. (The fancier "dynamic" quant of the same model is 4.25GB — the quant
*variant* matters as much as the model size.)

## Finding 2: it genuinely works, and it's fast

Fine-tuned Gemma 3 4B with QLoRA (r=8, all attention+MLP projections,
seq len 1024) on 1,000 samples of FineTome-100k:

| Metric | Value |
|---|---|
| Model load (4-bit, cached) | 14s, 3.09GB VRAM |
| 60 training steps | **6.7 minutes** |
| Peak VRAM during training | **5.22GB** |
| Training loss | **1.91 → 0.87** (first → last step) |
| Throughput | 0.6 samples/s (batch 1 × grad accum 4) |
| Adapter size on disk | 62.6MB |

So yes: a real 4B-parameter model, fine-tuned on a mid-range gaming GPU,
while the desktop stayed usable, in the time it takes to make chai.

## Finding 3: the Windows tax is real but small

Three things bit me that no Colab notebook mentions:

1. **`pip install unsloth` silently replaced CUDA PyTorch with the CPU build**
   (xformers pins an exact torch version; pip grabs the CPU wheel from PyPI).
   Fix: reinstall the same version with the CUDA suffix:
   `pip install torch==2.10.0+cu130 --index-url https://download.pytorch.org/whl/cu130`
2. **Any `dataset_num_proc` setting crashes on Windows** — datasets spawns a
   worker process that can't unpickle Unsloth's compiled trainer module
   (`ModuleNotFoundError: No module named 'UnslothSFTTrainer'`).
   Fix: omit it entirely; main-process tokenization of 1k samples takes seconds.
3. **transformers 5.x**: `apply_chat_template` needs explicit `tokenize=True`
   to return tensors.

Total setup time including all downloads and debugging: about an hour.

## Before/after (same prompts, same sampling)

**"Write a Python one-liner to count word frequencies in a string"**
- Before: gave `Counter(...).most_common(1)` — which returns only the single
  most common word, not the frequencies of all words, so it doesn't answer
  the question as asked.
- After: correct `Counter(word for word in string.split())`, concise
  numbered explanation, plus a caveat about whitespace splitting.

**"Capital of Telangana + tech scene"**
- Before: long structured answer with headers and bold, cut off mid-list at
  the 200-token cap.
- After: one crisp, complete, correct sentence.

**"Explain LoRA in 3 sentences"**
- Both correct; the tuned model followed the 3-sentence constraint with
  tighter phrasing.

The pattern after just 60 steps: answers became more direct and
instruction-following improved (FineTome is an instruction dataset — the
model learned its style). 60 steps is a smoke test, not a production tune;
the point is the loop works.

## Honest caveats for the blog

- 60 steps ≈ 0.24 epochs of 1k samples — proves the pipeline, not model quality.
- No eval set — before/after prompt vibes are illustrative, not a benchmark.
  (Several commenters on the original post made exactly this point.)
- Loss went from ~1.9 to ~0.9, but with a small dataset that's partly
  style-matching.

## Repo contents

- `finetune_gemma4.py` — the whole pipeline: load → baseline gen → train →
  after gen → save metrics/adapters
- `chat.py` — interactive chat with the tuned model (`--base` to compare)
- `results/metrics.json` — full loss curve + timings
- `results/generations.json` — before/after outputs verbatim
- `gemma3-4b-lora/` — the trained adapters (62.6MB)

## Possible blog angles

1. **"I fact-checked the '8GB is enough' claim"** — the E2B surprise is the
   hook; the 4B success is the payoff. Most differentiated angle.
2. **"Fine-tuning on native Windows in 2026"** — the three gotchas above are
   genuinely undocumented; strong SEO for `unsloth windows` searches.
3. **Straight tutorial** — most crowded space, but the real measured numbers
   (5.22GB peak, 6.7 min) make it credible.

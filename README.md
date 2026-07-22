# gemma-8gb-lab

Can you really fine-tune Gemma on an 8GB consumer GPU? I tested the claim on an
RTX 4060 Ti (8GB) on native Windows 11 — no cloud, no WSL, no Colab — and built
a small live dashboard to chat with the result.

<!-- Add your screenshot as docs/dashboard.png, then swap this line for:
![dashboard](docs/dashboard.png) -->
> 📸 Screenshot coming — run `python app.py` and see it live at `http://127.0.0.1:7860`.

## What actually happened

| Claim | Reality on my machine |
|---|---|
| "Gemma 4 2B (E2B) runs on 8GB" | **Inference: yes** — 7.05 GB peak, ~15 tok/s. **Fine-tuning: no** — the 4-bit model alone needs ~7 GB; there's no room left for gradients. |
| "4B needs ~10GB with LoRA" | Better than claimed: **Gemma 3 4B QLoRA peaked at 5.22 GB** (seq len 1024, batch 1, grad accum 4). |
| "8GB VRAM is enough" | Only if the GPU is *dedicated*. A normal desktop (browser, Teams, …) already holds ~2.6 GB. Budget against **free** VRAM, not card size. |

Fine-tune run (Gemma 3 4B, QLoRA r=8 via [Unsloth](https://github.com/unslothai/unsloth),
1,000 FineTome-100k samples):

- 60 steps in **6.7 minutes**
- peak VRAM **5.22 GB**
- train loss **1.91 → 0.87**
- adapter size **62 MB**

Full numbers in [`results/metrics.json`](results/metrics.json), before/after
sample generations in [`results/generations.json`](results/generations.json),
the Gemma 4 E2B inference test in
[`results/metrics-e2b-infer.json`](results/metrics-e2b-infer.json).

## The dashboard

`app.py` serves a claymorphic single-page dashboard (`dashboard.html`):
chat with the model on one side; live tokens/sec, latency, and VRAM charts
on the other. Replies stream token-by-token and every message updates the
charts in real time. Conversations are threaded and persist across
refreshes/restarts (`results/chat_state.json`, gitignored).

## Quickstart

Needs an NVIDIA GPU, Python 3.12, and recent drivers.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1

# CUDA torch FIRST, then unsloth (see gotcha #1 below)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install unsloth triton-windows fastapi uvicorn

# unsloth's resolver may have swapped torch for the CPU build — check:
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# if it prints "+cpu" / False, reinstall the SAME version as a CUDA wheel, e.g.:
pip install torch==2.10.0+cu130 torchvision==0.25.0+cu130 --index-url https://download.pytorch.org/whl/cu130

python finetune_gemma4.py      # fine-tune Gemma 3 4B (~10 min on a 4060 Ti)
python app.py                  # dashboard at http://127.0.0.1:7860
python chat.py                 # or plain terminal chat
```

To run the dashboard with base Gemma 4 E2B instead of the fine-tuned 4B
(close your other apps first — it needs ~7.6 GB):

```powershell
$env:APP_MODEL='unsloth/gemma-4-E2B-it'; $env:APP_FORCE_GPU='1'; $env:APP_SEQ='2048'
python app.py
```

Env knobs for `finetune_gemma4.py`: `FT_MODEL` (HF id), `FT_SEQ` (seq len),
`FT_EXACT=1` (don't redirect to unsloth's dynamic quant), `FT_FORCE_GPU=1`
(bypass accelerate's fit check; Windows pages small overshoot to RAM),
`FT_INFER_ONLY=1` (load + generate only, skip training).

## Windows gotchas nobody mentions

1. **`pip install unsloth` silently replaces CUDA torch with the CPU build**
   (xformers pins an exact torch version and pip resolves it from PyPI).
   Symptom: `Unsloth cannot find any torch accelerator`. Fix: reinstall the
   same version with the `+cuXXX` local tag from the PyTorch index.
2. **Any `dataset_num_proc` value crashes tokenization** — the spawned worker
   can't unpickle Unsloth's compiled trainer module
   (`ModuleNotFoundError: UnslothSFTTrainer`). Leave it unset.
3. **transformers 5.x** returns a *string* from `apply_chat_template` unless
   you pass `tokenize=True`.
4. Model downloads default to `C:\Users\<you>\.cache\huggingface` — that's
   where your C drive quietly went. Set `HF_HOME` to another drive.

## Honest caveats

- 60 steps ≈ 0.24 epochs. This proves the *pipeline*, not model quality.
- No eval set; the before/after generations are illustrative, not a benchmark.
- Numbers are from one machine (4060 Ti, Windows 11, driver 591.86,
  torch 2.10 cu130, unsloth 2026.7.4). Your VRAM ceiling will differ with
  different drivers/apps.

## Layout

```
finetune_gemma4.py   full pipeline: load → baseline gen → QLoRA train → after gen → metrics
app.py               FastAPI server: streaming chat (SSE), threads, GPU polling
dashboard.html       the dashboard (Chart.js, vanilla JS, no build step)
chat.py              minimal terminal chat with the tuned adapter
BLOG_NOTES.md        raw notes and measurements from the runs
results/             metrics + before/after generations (JSON)
```

MIT licensed. Not affiliated with Google or Unsloth — just a weekend
measurement of a LinkedIn claim.

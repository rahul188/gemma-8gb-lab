"""
Local dashboard server: chat with any supported model and watch inference
metrics update in real time.

Chats are organized into threads with full conversation context, and all
threads + per-message metrics persist to results/chat_state.json — so a
browser refresh (or server restart) restores everything.

Run:  python app.py              # start with Gemma 3 4B (default)
      APP_MODEL=llama32-3b python app.py   # start with Llama 3.2 3B
      APP_MODEL=qwen25-7b  python app.py   # start with Qwen 2.5 7B

Then open http://127.0.0.1:7860 — the dashboard has a model switcher that
hot-swaps models without restarting the server.

Supported APP_MODEL keys: gemma3-4b, gemma4-e2b, llama32-3b, llama31-8b,
                           qwen25-7b, mistral-7b, phi3-mini
(or pass any HuggingFace repo path / local adapter directory directly)
"""

import gc
import json
import subprocess
import threading
import time
import uuid
from pathlib import Path

from unsloth import FastModel  # must be imported before transformers

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
from transformers import TextIteratorStreamer

import os

from models import MODELS, resolve

HERE = Path(__file__).parent
STATE_FILE = HERE / "results" / "chat_state.json"

# ---------------------------------------------------------------------------
# Mutable model state — swapped atomically under gen_lock
# ---------------------------------------------------------------------------
_ms_lock = threading.Lock()   # guards the _ms_* variables below

_ms_model     = None
_ms_tokenizer = None
_ms_max_seq   = 4096
_ms_budget    = int(4096 * 0.8)
_ms_entry: dict = {}


def _do_load(entry: dict, seq_override: int | None = None) -> None:
    """Load *entry* into the global model state (caller holds _ms_lock)."""
    global _ms_model, _ms_tokenizer, _ms_max_seq, _ms_budget, _ms_entry
    import torch

    # Free previous model from GPU memory
    if _ms_model is not None:
        del _ms_model, _ms_tokenizer
        _ms_model = _ms_tokenizer = None
        torch.cuda.empty_cache()
        gc.collect()

    adapter = entry.get("adapter_dir")
    path = adapter if adapter and Path(adapter).exists() else entry["model_id"]
    seq = seq_override or entry["max_seq"]

    print(f"Loading {path} (max_seq={seq}) …")
    extra: dict = {}
    if os.environ.get("APP_FORCE_GPU") == "1":
        extra["device_map"] = {"": 0}
    if entry.get("use_exact") or (
        not Path(path).exists()
        and (path.startswith("unsloth/") or path.startswith("google/"))
    ):
        extra["use_exact_model_name"] = True

    m, tok = FastModel.from_pretrained(
        model_name=path,
        max_seq_length=seq,
        load_in_4bit=True,
        **extra,
    )
    FastModel.for_inference(m)

    _ms_model     = m
    _ms_tokenizer = tok
    _ms_max_seq   = seq
    _ms_budget    = int(seq * 0.8)
    _ms_entry     = entry


# Bootstrap: load the startup model
_initial_entry = resolve(os.environ.get("APP_MODEL", "gemma3-4b"))
_seq_override  = int(os.environ.get("APP_SEQ", 0)) or None
with _ms_lock:
    _do_load(_initial_entry, _seq_override)

print("Warming up (first generation compiles GPU kernels) …")
with _ms_lock:
    _warm_inputs = _ms_tokenizer.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        add_generation_prompt=True, tokenize=True, return_tensors="pt",
    ).to("cuda")
    _ms_model.generate(input_ids=_warm_inputs, max_new_tokens=2)
print("Model ready.")

app = FastAPI()
gen_lock = threading.Lock()   # held during active generation or model switch
state_lock = threading.Lock()

# state: threads carry the conversation; metrics_log is the global,
# chronological per-message metrics history that feeds the charts
if STATE_FILE.exists():
    state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
else:
    state = {"threads": [], "metrics_log": []}


def save_state():
    with state_lock:
        STATE_FILE.write_text(
            json.dumps(state, ensure_ascii=False, indent=1), encoding="utf-8"
        )


def find_thread(tid):
    return next((t for t in state["threads"] if t["id"] == tid), None)


def gpu_stats():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().split(", ")
    return {
        "vram_used_gb": round(int(out[0]) / 1024, 2),
        "vram_total_gb": round(int(out[1]) / 1024, 2),
        "gpu_util_pct": int(out[2]),
    }


def build_inputs(msgs):
    """Chat-template the thread history, dropping oldest turns until it fits."""
    with _ms_lock:
        tok = _ms_tokenizer
        budget = _ms_budget
    msgs = list(msgs)
    while True:
        conv = [
            {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
            for m in msgs
        ]
        inputs = tok.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        )
        if inputs.shape[1] <= budget or len(msgs) <= 1:
            return inputs.to("cuda")
        msgs = msgs[2:]  # drop the oldest user+assistant pair


@app.get("/")
def index():
    return FileResponse(HERE / "dashboard.html")


@app.get("/api/bootstrap")
def bootstrap():
    metrics_path = HERE / "results" / "metrics.json"
    training: dict = {}
    if metrics_path.exists():
        try:
            training = json.loads(metrics_path.read_text())
        except Exception:
            pass

    with _ms_lock:
        entry = _ms_entry

    adapter_path = entry.get("adapter_dir")
    adapter_mb = 0.0
    if adapter_path and Path(adapter_path).exists():
        adapter_mb = round(
            sum(f.stat().st_size for f in Path(adapter_path).glob("*")) / 1024**2, 1
        )

    return {
        "training": training,
        "adapter_mb": adapter_mb,
        "gpu": gpu_stats(),
        "model": entry.get("model_id", ""),
        "model_key": next(
            (k for k, v in MODELS.items() if v is entry), entry.get("model_id", "")
        ),
        "model_label": entry.get("label", entry.get("model_id", "")),
    }


@app.get("/api/models")
def list_models():
    """Return all supported models with their metadata."""
    return [
        {
            "key":       key,
            "model_id":  v["model_id"],
            "label":     v["label"],
            "max_seq":   v["max_seq"],
            "vram_note": v.get("vram_note", ""),
            "has_adapter": bool(
                v.get("adapter_dir") and Path(v["adapter_dir"]).exists()
            ),
        }
        for key, v in MODELS.items()
    ]


class ModelIn(BaseModel):
    key: str


@app.post("/api/model")
def switch_model(inp: ModelIn):
    """Hot-swap the running model. Blocks until any in-flight generation finishes."""
    if inp.key not in MODELS:
        raise HTTPException(
            400, f"unknown model key {inp.key!r}; known: {sorted(MODELS)}"
        )
    entry = MODELS[inp.key]   # use the hardcoded registry entry, not user-supplied paths
    with gen_lock:             # wait for active generation to finish
        with _ms_lock:
            _do_load(entry)
    return {"ok": True, "model_key": inp.key, "label": entry.get("label", inp.key)}


@app.get("/api/state")
def get_state():
    return state


@app.get("/api/gpu")
def gpu():
    return gpu_stats()


class ThreadIn(BaseModel):
    title: str = "New chat"


@app.post("/api/threads")
def create_thread(inp: ThreadIn):
    thread = {"id": uuid.uuid4().hex[:8], "title": inp.title[:48], "messages": []}
    state["threads"].append(thread)
    save_state()
    return thread


class ChatIn(BaseModel):
    thread_id: str
    message: str


@app.post("/api/chat")
def chat(inp: ChatIn):
    thread = find_thread(inp.thread_id)
    if thread is None:
        raise HTTPException(404, "unknown thread")

    def sse():
        with gen_lock:
            thread["messages"].append({"role": "user", "content": inp.message})
            if thread["title"] == "New chat":
                thread["title"] = inp.message[:48]
            inputs = build_inputs(thread["messages"])

            with _ms_lock:
                m = _ms_model
                tok = _ms_tokenizer

            streamer = TextIteratorStreamer(
                tok, skip_prompt=True, skip_special_tokens=True
            )
            result = {}

            def run():
                result["out"] = m.generate(
                    input_ids=inputs, streamer=streamer,
                    max_new_tokens=400, temperature=0.7, top_p=0.95,
                )

            t0 = time.perf_counter()
            thread_ = threading.Thread(target=run)
            thread_.start()
            first_token_s = None
            text = ""
            for piece in streamer:
                if not piece:
                    continue
                if first_token_s is None:
                    first_token_s = time.perf_counter() - t0
                text += piece
                yield "data: " + json.dumps({"token": piece}) + "\n\n"
            thread_.join()
            total_s = time.perf_counter() - t0

            n_tokens = int(result["out"].shape[1] - inputs.shape[1])
            decode_s = max(total_s - (first_token_s or 0), 1e-6)
            metrics = {
                "latency_s": round(total_s, 2),
                "first_token_s": round(first_token_s or total_s, 2),
                "tokens": n_tokens,
                "tokens_per_s": round(n_tokens / decode_s, 1),
                **gpu_stats(),
            }
            thread["messages"].append(
                {"role": "assistant", "content": text.strip(), "metrics": metrics}
            )
            state["metrics_log"].append({"thread_id": thread["id"], **metrics})
            save_state()
            yield "data: " + json.dumps({"done": True, **metrics}) + "\n\n"

    return StreamingResponse(sse(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=7860, log_level="warning")

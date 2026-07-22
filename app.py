"""
Local dashboard server: chat with the fine-tuned Gemma 3 4B LoRA and watch
inference metrics update in real time.

Chats are organized into threads with full conversation context, and all
threads + per-message metrics persist to results/chat_state.json — so a
browser refresh (or server restart) restores everything.

Run:  python app.py   then open http://127.0.0.1:7860
"""

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

HERE = Path(__file__).parent
ADAPTER_DIR = HERE / "gemma3-4b-lora"
STATE_FILE = HERE / "results" / "chat_state.json"

# Model is switchable: default = our fine-tuned Gemma 3 4B LoRA; set
# APP_MODEL=unsloth/gemma-4-E2B-it APP_FORCE_GPU=1 APP_SEQ=2048 for Gemma 4.
MODEL_PATH = os.environ.get("APP_MODEL", str(ADAPTER_DIR))
MAX_SEQ = int(os.environ.get("APP_SEQ", "4096"))
CONTEXT_BUDGET = int(MAX_SEQ * 0.8)  # prompt budget; older turns dropped past this

print(f"Loading {MODEL_PATH} (this takes a minute) ...")
extra = {}
if os.environ.get("APP_FORCE_GPU") == "1":
    extra["device_map"] = {"": 0}
if MODEL_PATH.startswith("unsloth/") or MODEL_PATH.startswith("google/"):
    extra["use_exact_model_name"] = True
model, tokenizer = FastModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=MAX_SEQ,
    load_in_4bit=True,
    **extra,
)
FastModel.for_inference(model)
print("Warming up (first generation compiles GPU kernels, takes a few minutes) ...")
_warm = tokenizer.apply_chat_template(
    [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    add_generation_prompt=True, tokenize=True, return_tensors="pt",
).to("cuda")
model.generate(input_ids=_warm, max_new_tokens=2)
print("Model ready.")

app = FastAPI()
gen_lock = threading.Lock()
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
    msgs = list(msgs)
    while True:
        conv = [
            {"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
            for m in msgs
        ]
        inputs = tokenizer.apply_chat_template(
            conv, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        )
        if inputs.shape[1] <= CONTEXT_BUDGET or len(msgs) <= 1:
            return inputs.to("cuda")
        msgs = msgs[2:]  # drop the oldest user+assistant pair


@app.get("/")
def index():
    return FileResponse(HERE / "dashboard.html")


@app.get("/api/bootstrap")
def bootstrap():
    metrics = json.loads((HERE / "results" / "metrics.json").read_text())
    adapter_mb = sum(f.stat().st_size for f in ADAPTER_DIR.glob("*")) / 1024**2
    label = (
        "Gemma 4 E2B · base model · local inference"
        if "gemma-4" in MODEL_PATH.lower()
        else "Gemma 3 4B · QLoRA adapters · local inference"
    )
    return {
        "training": metrics, "adapter_mb": round(adapter_mb, 1),
        "gpu": gpu_stats(), "model": MODEL_PATH, "model_label": label,
    }


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
            streamer = TextIteratorStreamer(
                tokenizer, skip_prompt=True, skip_special_tokens=True
            )
            result = {}

            def run():
                result["out"] = model.generate(
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

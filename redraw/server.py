"""SUPIR generative redraw HTTP service.

Loads the SUPIR (SDXL-based) diffusion model once at startup and exposes:

  GET  /health          -> {"ok", "loaded", "error", "vram_mb"}
  POST /redraw          -> PNG (multipart: file, scale=2|4, steps=10..50)

Designed to sit behind a Go gateway (see the parent project) that handles
auth, rate limiting, and fallback to lighter engines.
"""

import io
import os
import sys
import threading
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image

ROOT = Path(__file__).resolve().parent
SUPIR_ROOT = ROOT / "SUPIR"
sys.path.insert(0, str(SUPIR_ROOT))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from SUPIR.util import PIL2Tensor, Tensor2PIL, convert_dtype, create_SUPIR_model  # noqa: E402

DEVICE = "cuda:0"
MAX_INPUT = {2: 1536, 4: 768}

A_PROMPT = (
    "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, "
    "hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, "
    "extreme meticulous detailing, skin pore detailing, hyper sharpness, "
    "perfect without deformations."
)
N_PROMPT = (
    "painting, oil painting, illustration, drawing, art, sketch, cartoon, CG Style, "
    "3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, "
    "frames, watermark, signature, jpeg artifacts, deformed, lowres, over-smooth"
)

_lock = threading.Lock()
_model = None
_load_error: str | None = None


def _load() -> None:
    global _model, _load_error
    if _model is not None or _load_error is not None:
        return
    with _lock:
        if _model is not None or _load_error is not None:
            return
        try:
            m = create_SUPIR_model(str(SUPIR_ROOT / "options" / "SUPIR_v0.yaml"), SUPIR_sign="F")
            m = m.half()
            m.init_tile_vae(encoder_tile_size=1536, decoder_tile_size=128)
            m.ae_dtype = convert_dtype("bf16")
            m.model.dtype = convert_dtype("fp16")
            m = m.to(DEVICE)
            m.eval()
            _model = m
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[redraw] model loaded", flush=True)
        except Exception as exc:  # noqa: BLE001
            _load_error = f"{type(exc).__name__}: {exc}"
            traceback.print_exc()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    threading.Thread(target=_load, daemon=True).start()
    yield


app = FastAPI(lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    vram_mb = 0
    if torch.cuda.is_available():
        vram_mb = round(torch.cuda.memory_allocated() / (1024 * 1024))
    return {
        "ok": True,
        "loaded": _model is not None,
        "error": _load_error,
        "vram_mb": vram_mb,
    }


@app.post("/redraw")
def redraw(
    file: UploadFile = File(...),
    scale: int = Form(2),
    steps: int = Form(15),
) -> Response:
    if scale not in (2, 4):
        raise HTTPException(400, "invalid scale")
    steps = max(10, min(50, steps))
    raw = file.file.read(24 << 20)
    if not raw:
        raise HTTPException(400, "empty file")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(400, "cannot decode image") from exc
    img = img.convert("RGB")
    w, h = img.size
    if min(w, h) < 32 or max(w, h) > MAX_INPUT[scale]:
        raise HTTPException(413, f"input too large for redraw (max {MAX_INPUT[scale]}px)")
    if _load_error is not None:
        raise HTTPException(503, "model failed to load")
    _load()
    if _model is None:
        raise HTTPException(503, "model loading, retry later")
    try:
        with _lock, torch.no_grad():
            x, h0, w0 = PIL2Tensor(img, upsacle=scale, min_size=1024)
            x = x.unsqueeze(0).to(DEVICE)[:, :3, :, :]
            samples = _model.batchify_sample(
                x,
                [""],
                num_steps=steps,
                restoration_scale=-1,
                s_churn=5,
                s_noise=1.01,
                cfg_scale=4.0,
                control_scale=1.0,
                seed=1234,
                num_samples=1,
                p_p=A_PROMPT,
                n_p=N_PROMPT,
                color_fix_type="Wavelet",
                use_linear_CFG=True,
                use_linear_control_scale=False,
                cfg_scale_start=1.0,
                control_scale_start=0.0,
            )
            out = Tensor2PIL(samples[0], h0, w0)
    except torch.cuda.OutOfMemoryError as exc:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(507, "out of vram") from exc
    buf = io.BytesIO()
    out.save(buf, "PNG")
    return Response(buf.getvalue(), media_type="image/png")


if __name__ == "__main__":
    import uvicorn

    addr = os.environ.get("REDRAW_ADDR", "127.0.0.1:8082")
    uvicorn.run(app, host=addr.split(":")[0], port=int(addr.split(":")[1]), log_level="info")

# SUPIR redraw service

Optional Python microservice that runs [SUPIR](https://github.com/Kijai/SUPIR) — an SDXL-based diffusion model for true generative image redraw (not just GAN upscaling). Sits on `127.0.0.1:8082` and is meant to be called by the Go gateway in the parent directory.

> **Warning:** requires an NVIDIA GPU with ≥ 10 GB VRAM (tested on RTX 5070 12 GB). Occupies ~9–10 GB VRAM and ~6 GB system RAM while loaded. Generation takes roughly 1–3 minutes per image at 15 steps. This is a heavy, optional engine — the default `local` / `ai` engines in the Go server need none of this.

## What it does

`POST /redraw` accepts a JPEG/PNG (max 1536 px long side for ×2, 768 px for ×4) and returns a redrawn PNG at the requested scale. Internally it:

1. upscales with Lanczos to the diffusion working size,
2. runs SUPIR's ControlNet-style conditioning for `steps` sampling steps,
3. applies wavelet color correction,
4. tiles the VAE encode/decode to stay inside VRAM.

## Setup (Windows)

```powershell
cd redraw
powershell -ExecutionPolicy Bypass -File setup.ps1
```

This will:

1. clone SUPIR into `redraw/SUPIR/`,
2. create `redraw/.venv/` (Python 3.11),
3. install PyTorch + SUPIR requirements,
4. download three checkpoints (~20 GB total) into `redraw/models/`,
5. apply compatibility patches (`apply_patches.py`) so SUPIR runs on modern PyTorch / open_clip ≥ 3 / GPUs without xformers kernels (e.g. Blackwell sm_120).

Re-running is safe — every step is skipped if already done.

## Run

```powershell
run.bat
```

Health check:

```bash
curl http://127.0.0.1:8082/health
# {"ok":true,"loaded":true,"error":null,"vram_mb":9308}
```

Manual redraw:

```bash
curl -F "file=@photo.jpg" -F "scale=2" -F "steps=15" \
  http://127.0.0.1:8082/redraw -o redrawn.png
```

## Environment

| variable         | default             | purpose                          |
|------------------|---------------------|----------------------------------|
| `REDRAW_ADDR`    | `127.0.0.1:8082`    | listen address                   |
| `HF_HUB_DISABLE_XET` | `1` (set in `run.bat`) | avoids huggingface xet backend hangs |

## Patches explained

Vanilla SUPIR assumes xformers and older open_clip. On modern stacks (PyTorch ≥ 2.5, open_clip ≥ 3, or GPUs without xformers builds like RTX 50-series) three things break:

1. **Tiled VAE attention** — `SUPIR/utils/tilevae.py` originally routed attention through xformers. We add `sdpa_attn_forward` using `torch.nn.functional.scaled_dot_product_attention` and prefer it whenever available.
2. **CLIP text encoder layout** — `FrozenOpenCLIPEmbedder(2)` permuted tensors to sequence-first (LND) for old open_clip. open_clip ≥ 3 expects batch-first (NLD), so we drop the permutes.
3. **Attention masks** — torch ≥ 2.5 stopped broadcasting 2D masks inside `nn.MultiheadAttention`. `open_clip/transformer.py` is patched to expand masks to `(batch*heads, L, S)` before the call.

`options/SUPIR_v0.yaml` is also edited to use plain `softmax` / `vanilla` attention types (instead of `*-xformers`) and to point at the locally downloaded checkpoints. `CKPT_PTH.py` points the bigG CLIP2 text encoder at the local `models/CLIP2/` weights.

All patches are applied by `apply_patches.py` and are idempotent.

## Integrating with the Go gateway

The parent `main.go` currently does not call this service by default. To wire it in, add a `redrawClient` that `POST`s to `http://127.0.0.1:8082/redraw` and falls back to the `ai` (Real-ESRGAN) engine on any error — the same pattern used in the original XDownload integration.

## License

The code in this directory (server, setup, patches) is MIT, same as the parent project. SUPIR itself is under its own [license](SUPIR/LICENSE) — review it before commercial use. Model weights are subject to their own licenses (SDXL, OpenCLIP).

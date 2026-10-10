"""Download the three large checkpoints required by SUPIR.

Weights end up in ./models/ next to this script. Total ~20 GB.
Re-run safe: already-downloaded files are skipped by huggingface_hub
and by the curl fallback for the two biggest files.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)


def curl_download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  skip (exists): {dest.name}")
        return
    print(f"  curl {dest.name} ...")
    subprocess.run(
        ["curl.exe", "-L", "--fail", "--retry", "3", "-o", str(dest), url],
        check=True,
    )


def main() -> int:
    print("SDXL base 1.0 (~6.6 GB)...")
    curl_download(
        "https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0/resolve/main/sd_xl_base_1.0_0.9vae.safetensors",
        MODELS / "sd_xl_base_1.0_0.9vae.safetensors",
    )

    print("SUPIR-v0F fp16 (~2.5 GB)...")
    curl_download(
        "https://huggingface.co/Kijai/SUPIR_pruned/resolve/main/SUPIR-v0F_fp16.safetensors",
        MODELS / "SUPIR-v0F_fp16.safetensors",
    )

    print("CLIP2 bigG (~9.5 GB)...")
    clip2_dir = MODELS / "CLIP2"
    clip2_dir.mkdir(exist_ok=True)
    curl_download(
        "https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k/resolve/main/open_clip_pytorch_model.bin",
        clip2_dir / "open_clip_pytorch_model.bin",
    )

    print("done — all checkpoints in", MODELS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Apply compatibility patches to a freshly cloned SUPIR checkout.

Run once after `git clone https://github.com/Kijai/SUPIR` and after
`pip install -r requirements.txt` inside the venv. Safe to re-run.

Patches:
  1. SUPIR/utils/tilevae.py      — add SDPA attention forward, prefer it over xformers
  2. sgm/modules/encoders/modules.py — FrozenOpenCLIPEmbedder(2): keep batch-first
                                      layout expected by open_clip >= 3
  3. open_clip/transformer.py    — expand 2D attn masks to (bsz*heads, L, S)
                                   (torch >= 2.5 removed broadcasting)
  4. options/SUPIR_v0.yaml       — local checkpoint paths, softmax/vanilla attn
  5. CKPT_PTH.py                 — point CLIP2 at the locally downloaded weights
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUPIR = ROOT / "SUPIR"
MODELS = ROOT / "models"


def patch_tilevae(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if "def sdpa_attn_forward" in text and "if False: # disabled" in text:
        # already has SDPA fn; ensure else branch prefers SDPA over xformers
        old_else = """    else:
        task_queue.append(('store_res', lambda x: x))
        task_queue.append(('pre_norm', net.norm))
        if is_xformers_available:
            # task_queue.append(('attn', lambda x, net=net: attn_forward_new_xformers(net, x)))
            task_queue.append(
                ('attn', lambda x, net=net: xformer_attn_forward(net, x)))
        elif hasattr(F, "scaled_dot_product_attention"):
            task_queue.append(('attn', lambda x, net=net: attn_forward_new_pt2_0(net, x)))
        else:
            task_queue.append(('attn', lambda x, net=net: attn_forward_new(net, x)))
        task_queue.append(['add_res', None])"""
        new_else = """    else:
        task_queue.append(('store_res', lambda x: x))
        task_queue.append(('pre_norm', net.norm))
        if hasattr(F, "scaled_dot_product_attention"):
            task_queue.append(('attn', lambda x, net=net: sdpa_attn_forward(net, x)))
        elif is_xformers_available:
            task_queue.append(
                ('attn', lambda x, net=net: xformer_attn_forward(net, x)))
        else:
            task_queue.append(('attn', lambda x, net=net: attn_forward_new(net, x)))
        task_queue.append(['add_res', None])"""
        if old_else in text:
            path.write_text(text.replace(old_else, new_else, 1), encoding="utf-8")
            return True
        return False
    fn = '''def sdpa_attn_forward(self, h_):
    q = self.q(h_)
    k = self.k(h_)
    v = self.v(h_)

    B, C, H, W = q.shape
    q, k, v = map(lambda x: rearrange(x, 'b c h w -> b (h w) c'), (q, k, v))

    out = F.scaled_dot_product_attention(
        q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
    ).squeeze(1)
    out = rearrange(out, 'b (h w) c -> b c h w', h=H, w=W)
    out = self.proj_out(out)
    return h_ + out

'''
    anchor = "def attn2task(task_queue, net):"
    text = text.replace(anchor, fn + anchor, 1)
    text = text.replace(
        "task_queue.append(('attn', lambda x, net=net: attn_forward(net, x)))",
        "task_queue.append(('attn', lambda x, net=net: sdpa_attn_forward(net, x)))",
        1,
    )
    path.write_text(text, encoding="utf-8")
    return True


def patch_clip_embedders(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if "open_clip>=3 expects batch-first" in text:
        return False
    text = text.replace(
        "x = x.permute(1, 0, 2)  # NLD -> LND\n        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)\n        if self.legacy:",
        "# open_clip>=3 expects batch-first (NLD); skip legacy NLD->LND permute\n        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)\n        if self.legacy:",
        1,
    )
    text = text.replace(
        "outputs[\"penultimate\"] = x.permute(1, 0, 2)  # LND -> NLD",
        "outputs[\"penultimate\"] = x  # already NLD (batch-first)",
        1,
    )
    text = text.replace(
        "outputs[\"last\"] = x.permute(1, 0, 2)  # LND -> NLD",
        "outputs[\"last\"] = x  # already NLD (batch-first)",
        1,
    )
    text = text.replace(
        "x = x.permute(1, 0, 2)  # NLD -> LND\n        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)\n        x = x.permute(1, 0, 2)  # LND -> NLD\n        x = self.model.ln_final(x)",
        "# open_clip>=3 expects batch-first (NLD); skip legacy permutes\n        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)\n        x = self.model.ln_final(x)",
        1,
    )
    path.write_text(text, encoding="utf-8")
    return True


def patch_open_clip(site_packages: Path) -> bool:
    path = site_packages / "open_clip" / "transformer.py"
    if not path.exists():
        return False
    text = path.read_text(encoding="utf-8")
    if "expand(bsz * nh, *attn_mask.shape)" in text:
        return False
    old = """        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(
            q_x, k_x, v_x,
            need_weights=False,
            attn_mask=attn_mask
        )[0]"""
    new = """        attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        if attn_mask is not None and attn_mask.ndim == 2:
            bsz = q_x.shape[0]
            nh = self.attn.num_heads
            attn_mask = attn_mask.unsqueeze(0).expand(bsz * nh, *attn_mask.shape).contiguous()
        return self.attn(
            q_x, k_x, v_x,
            need_weights=False,
            attn_mask=attn_mask
        )[0]"""
    if old not in text:
        return False
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return True


def patch_yaml(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if "XImage-Enhancer" in text:
        return False
    sdxl = (MODELS / "sd_xl_base_1.0_0.9vae.safetensors").as_posix()
    supir = (MODELS / "SUPIR-v0F_fp16.safetensors").as_posix()
    text = re.sub(
        r"^SDXL_CKPT:.*$",
        f"SDXL_CKPT: {sdxl}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^SUPIR_CKPT_F:.*$",
        f"SUPIR_CKPT_F: {supir}",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^SUPIR_CKPT_Q:.*$",
        "SUPIR_CKPT_Q: ~",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = text.replace(
        "spatial_transformer_attn_type: softmax-xformers",
        "spatial_transformer_attn_type: softmax",
    )
    text = text.replace(
        "spatial_transformer_attn_type: vanilla-xformers",
        "spatial_transformer_attn_type: vanilla",
    )
    text = text.replace(
        "attn_type: softmax-xformers",
        "attn_type: softmax",
    )
    text = text.replace(
        "attn_type: vanilla-xformers",
        "attn_type: vanilla",
    )
    path.write_text(text, encoding="utf-8")
    return True


def patch_ckpt_pth(path: Path) -> bool:
    clip2 = (MODELS / "CLIP2" / "open_clip_pytorch_model.bin").as_posix()
    path.write_text(
        "LLAVA_CLIP_PATH = None\n"
        "LLAVA_MODEL_PATH = None\n"
        "SDXL_CLIP1_PATH = None\n"
        f"SDXL_CLIP2_CKPT_PTH = r\"{clip2}\"\n",
        encoding="utf-8",
    )
    return True


def main() -> int:
    if not SUPIR.is_dir():
        print("SUPIR clone not found — run setup first", file=sys.stderr)
        return 1
    changed = []
    if patch_tilevae(SUPIR / "SUPIR" / "utils" / "tilevae.py"):
        changed.append("tilevae.py")
    if patch_clip_embedders(SUPIR / "sgm" / "modules" / "encoders" / "modules.py"):
        changed.append("modules.py")
    sp = ROOT / ".venv" / "Lib" / "site-packages"
    if patch_open_clip(sp):
        changed.append("open_clip/transformer.py")
    yaml = SUPIR / "options" / "SUPIR_v0.yaml"
    if yaml.exists() and patch_yaml(yaml):
        changed.append("SUPIR_v0.yaml")
    if patch_ckpt_pth(SUPIR / "CKPT_PTH.py"):
        changed.append("CKPT_PTH.py")
    print("patched:", ", ".join(changed) if changed else "(nothing — already applied)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

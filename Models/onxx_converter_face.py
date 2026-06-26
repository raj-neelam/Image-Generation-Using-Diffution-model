"""
Export the trained face DDPM model to a single-file ONNX for browser use.

Usage:
    python face_onnx_converter.py

Outputs (place both next to index.html):
    ddpm_face.onnx          — the model weights
    face_noise_schedule.json — betas / alphas / alphas_cumprod arrays

Edit CKPT_PATH below to point at whichever checkpoint you want to export.
"""

import json
import torch
import onnx
from onnx.external_data_helper import load_external_data_for_model

# ── adjust this path to your latest checkpoint ───────────────────────────────
CKPT_PATH = "Models/face_model_150.pt"   # e.g. face_model_60.pt
OUT_ONNX  = "ddpm_face.onnx"
OUT_JSON  = "face_noise_schedule.json"
# ─────────────────────────────────────────────────────────────────────────────

from Model_architecture.DDPM_unet_face import DDPM_model


class Config:
    T               = 1000
    image_size      = 128
    in_channels     = 3
    # Must match what you trained with — check ddpm_face_script.py
    hidden_channels = 64


def export():
    config = Config()
    model  = DDPM_model(config)

    print(f"📂  Loading checkpoint: {CKPT_PATH}")
    state = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # ── STEP 1: torch.onnx.export ─────────────────────────────────────────────
    # Inputs:  x [1,3,128,128] float32,  t [1] int64
    # Output: noise_pred [1,3,128,128] float32
    dummy_x = torch.randn(1, config.in_channels, config.image_size, config.image_size)
    dummy_t = torch.tensor([999], dtype=torch.long)

    print("🔄  Exporting to ONNX …")
    torch.onnx.export(
        model,
        (dummy_x, dummy_t),
        OUT_ONNX,
        input_names   = ["x", "t"],
        output_names  = ["noise_pred"],
        dynamic_axes  = {
            "x":          {0: "batch"},
            "t":          {0: "batch"},
            "noise_pred": {0: "batch"},
        },
        opset_version       = 14,
        do_constant_folding = True,
        export_params       = True,
    )
    print(f"✅  Exported to {OUT_ONNX}")

    # ── STEP 2: merge .data sidecar into single file ──────────────────────────
    # onnxruntime-web in the browser cannot load a separate .data sidecar.
    print("🔄  Merging into single-file ONNX (no .data sidecar) …")
    proto = onnx.load(OUT_ONNX)
    load_external_data_for_model(proto, ".")
    onnx.save(proto, OUT_ONNX, save_as_external_data=False)
    print(f"✅  Single-file ONNX saved — safe to delete {OUT_ONNX}.data if it exists")

    # ── STEP 3: export noise schedule ─────────────────────────────────────────
    # Identical linear schedule used in ddpm_face_script.py
    betas          = torch.linspace(1e-4, 0.02, config.T)
    alphas         = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    schedule = {
        "betas":          betas.tolist(),
        "alphas":         alphas.tolist(),
        "alphas_cumprod": alphas_cumprod.tolist(),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(schedule, f)
    print(f"✅  Exported {OUT_JSON}")

    print(
        "\nDone! Place these two files next to index.html:\n"
        f"  • {OUT_ONNX}\n"
        f"  • {OUT_JSON}\n"
        "Then open index.html in a browser — face model loads by default."
    )


if __name__ == "__main__":
    export()
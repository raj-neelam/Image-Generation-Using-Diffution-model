"""
Run this script once to export your trained DDPM model to ONNX format.
Place this file in your DDPM project root (same level as DDPM_model files).

Usage:
    python onxx_converter.py

Output:
    ddpm_mnist.onnx  (~12MB, place this next to your HTML file)
"""

import torch
import onnx
from onnx.external_data_helper import load_external_data_for_model

from Model_architecture.DDPM_unnet_attntion import DDPM_model

# ── config (must match what you trained with) ─────────────────────────────────

class Config:
    T               = 1000
    num_classes     = 10
    image_size      = 28
    in_channels     = 1
    hidden_channels = 64


# ── load weights & export ─────────────────────────────────────────────────────

def export():
    config = Config()
    model  = DDPM_model(config)

    ckpt_path = "Models/attention_model_100.pt"
    state     = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    dummy_x     = torch.randn(1, 1, 28, 28)
    dummy_t     = torch.tensor([999], dtype=torch.long)
    dummy_label = torch.tensor([0],   dtype=torch.long)

    out_path = "ddpm_mnist.onnx"
    torch.onnx.export(
        model,
        (dummy_x, dummy_t, dummy_label),
        out_path,
        input_names  = ["x", "t", "label"],
        output_names = ["noise_pred"],
        dynamic_axes = {
            "x":          {0: "batch"},
            "t":          {0: "batch"},
            "label":      {0: "batch"},
            "noise_pred": {0: "batch"},
        },
        opset_version       = 14,
        do_constant_folding = True,
        export_params       = True,   # make sure weights are embedded
    )
    print(f"✅  Exported to {out_path}")

    # ── STEP 2: reload and force single-file (no .data sidecar) ──────────────
    # torch.onnx.export sometimes creates ddpm_mnist.onnx.data for large models.
    # onnxruntime-web in the browser CANNOT load that sidecar file.
    # This block pulls all weights back into the .onnx file itself.
    print("🔄  Merging into single-file ONNX (removing .data sidecar)...")
    model_proto = onnx.load(out_path)                    # load (may reference .data)
    load_external_data_for_model(model_proto, ".")       # pull .data contents into memory
    onnx.save(model_proto, out_path,
              save_as_external_data=False)               # save everything inline
    print("✅  Single-file ONNX saved — you can delete ddpm_mnist.onnx.data now")

    # ── STEP 3: export noise schedule ────────────────────────────────────────
    betas          = torch.linspace(1e-4, 0.02, config.T)
    alphas         = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    import json
    schedule = {
        "betas":          betas.tolist(),
        "alphas":         alphas.tolist(),
        "alphas_cumprod": alphas_cumprod.tolist(),
    }
    with open("noise_schedule.json", "w") as f:
        json.dump(schedule, f)
    print("✅  Exported noise_schedule.json")
    print("\nPlace ddpm_mnist.onnx and noise_schedule.json next to index.html and open in browser.")


if __name__ == "__main__":
    export()
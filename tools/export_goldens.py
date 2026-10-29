"""Pack the Python rung's parity goldens (anima-mlx/tests/goldens/*/*.npy, dumped by the
PyTorch oracle in anima-oracle/) into the single-file safetensors the Swift gates read.

The goldens are the POST-final-RMSNorm TE tap (comfy layer="last"; 2026-07-22 fix) and the
oracle's rope_scale=(1,4,4) DiT. AB-T-0020: the pre-norm-era Resources/*_golden.safetensors
were removed in 569ae17; this regenerates them from the corrected reference.

    <anima-mlx>/.venv/bin/python -I tools/export_goldens.py <anima-mlx>/tests/goldens Resources
"""
import hashlib
import os
import sys

import mlx.core as mx
import numpy as np

# golden file → the npy keys each Swift gate reads (Sources/AnimaCLI/main.swift)
LAYOUT = {
    "vae": ["in_latent", "out_image"],                                    # S1a --vae-gate
    "qwen3": ["in_ids", "hidden_postnorm"],                               # S1b --te-gate
    "adapter": ["in_source", "in_ids", "out_final"],                      # S1c --adapter-gate
    "dit": ["in_hidden", "in_timestep", "in_encoder", "in_padding", "out_final"],  # S1d --dit-gate
    "pipeline": ["cond_qwen_ids", "cond_t5_ids", "uncond_qwen_ids", "uncond_t5_ids",  # S2 --e2e-gate
                 "sigmas", "noise", "cond_context", "uncond_context", "v0_cfg", "final_latent"],
}


def main(src, dst):
    os.makedirs(dst, exist_ok=True)
    for name, keys in LAYOUT.items():
        arrays = {}
        for k in keys:
            a = np.load(os.path.join(src, name, k + ".npy"))
            a = a.astype(np.int32) if np.issubdtype(a.dtype, np.integer) else a.astype(np.float32)
            arrays[k] = mx.array(a)
        out = os.path.join(dst, f"{name}_golden.safetensors")
        mx.save_safetensors(out, arrays, metadata={"source": f"anima-mlx/tests/goldens/{name}",
                                                   "te_tap": "post-final-rmsnorm"})
        sha = hashlib.sha256(open(out, "rb").read()).hexdigest()[:12]
        print(f"{out}  sha256:{sha}  " + ", ".join(f"{k}{tuple(arrays[k].shape)}" for k in keys))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])

# anima-mlx-swift

Swift/MLX port of **[circlestone-labs/Anima](https://huggingface.co/circlestone-labs/Anima)** —
an anime/illustration **text-to-image** model: NVIDIA **Cosmos-Predict2-2B** DiT +
**Qwen3-0.6B** text encoder → `llm_adapter` conditioning + the Qwen-Image / **Wan 16-channel
VAE**. It ships as one engine-conformant MLXEngine `ModelPackage` exposing the **`textToImage`**
capability.

Reference = the parity-locked Python-MLX port
[`xocialize/anima-mlx`](https://github.com/xocialize/anima-mlx) (`python/`) + its PyTorch
goldens. The Swift port is **bit-for-bit parity-locked** to that rung.

> **Non-Commercial.** The Anima weights are licensed Non-Commercial (CircleStone Labs); the base
> denoiser is "Built on NVIDIA Cosmos" (NVIDIA Cosmos Open Model License). Personal / research use
> only. The port **code** here is MIT. See [LICENSE](LICENSE). The package enforces this at
> registration via a two-layer license gate (the NC weight is `rejectedWeight` under
> `.permissiveOnly`).

Weights (bf16 + int4, NC-flagged): **https://huggingface.co/xocialize/anima-mlx**

## Products

- **`Anima`** — the model core (Cosmos DiT + `llm_adapter` + Qwen3-0.6B TE + Wan VAE + FLOW-CONST
  pipeline + tokenizer). No `MLXToolKit` dependency.
- **`MLXAnima`** — the `AnimaT2IPackage` MLXEngine wrapper (capability **`.textToImage`**,
  measured `QuantFootprint`, C0–C13 conformant).

```swift
.package(url: "https://github.com/xocialize/anima-mlx-swift", from: "0.1.0")
```

## Parity (Swift, vs Python-MLX / PT goldens)

| component | cosine | max_abs |
|---|---|---|
| Cosmos DiT | 1.000000 | 3.1e-5 |
| llm_adapter | 1.000000 | 2.9e-6 |
| Qwen3-0.6B TE | 1.000000 | 6.1e-4 |
| Wan VAE | 1.000000 | 6.7e-6 |
| **e2e pipeline** | step-0 v0_cfg 0.9999996 · final latent **0.999105** | (== Python bit-for-bit) |
| int4 transformer | per-pass 0.99619 | |

Footprints (measured Swift peak @512²): **bf16 8.0 GB / int4 6.5 GB**.

Note: the TE / e2e rows were gated against PRE-norm TE goldens; after the 2026-07-22 post-norm
tap fix (see Sampling below) those goldens are stale until the Python rung regenerates them —
the numbers still establish that the transpose matched the then-reference.

## Sampling

ComfyUI `ModelType.FLOW`: `CONST` prediction + `ModelSamplingDiscreteFlow(shift=3, multiplier=1)`
→ `sigma(t) = 3t/(1+2t)`, **DiT timestep == sigma ∈ [0,1]**, Wan21 latent denorm before decode.
CFG 4–5. Tokenizers: Qwen2.5 (raw BPE, pad 151643) + T5-v1.1 SentencePiece (trailing eos).

**Sampler = `er_sde` (default, REQUIRED).** Root-caused 2026-07-22 (see anima-mlx `PIPELINE.md`):
the deterministic euler placeholder produces structural blob garbage at **every** size for this
model — the earlier "coherent 512²" eyeball gate was mis-calibrated and passed the blob class.
`sampleErSde` is a 1:1 transpose of the Python rung's `sample_er_sde` (ComfyUI k_diffusion
VP ER-SDE-Solver-3, arXiv:2309.06169); per-step noise comes from a deterministic **CPU-side**
gaussian stream (`GaussianRNG`) — MLX random ops built lazily inside the denoise loop yield
correlated draws → structured artifacts. euler is kept for A/B only.

**Negative prompt defaults to the reference quality-tag negative** (`worst quality, low quality,
score_1, score_2, score_3, blurry, jpeg artifacts`) — an empty negative collapses 1024² into
confetti.

**Text encoder tap = POST-final-RMSNorm** (fixed 2026-07-22). Comfy `layer="last"` returns
llama.py `outputs[0]`, which is post-`model.norm`; `layer_norm_hidden_state=False` only affects
intermediate-layer taps. The original pre-norm tap (hidden std ~25 vs ~3.8) made long positive
prompts (~30+ tokens) wash out and left short-prompt renders merely "coherent" instead of
ComfyUI-class. With the post-norm tap, long danbooru-style prompts render correctly.

## Quick start

**CLI:**

```bash
swift run anima-cli --generate "1girl, anime, masterpiece" <weights-dir> out.png
```

Component / e2e parity gates (need the goldens in `Resources/`, committed):
`swift run anima-cli --vae-gate | --te-gate | --adapter-gate | --dit-gate | --e2e-gate | --tok-gate`.

**Engine integration:**

```swift
MLXServeEngine.register(.of(AnimaT2IPackage.self), configuration: AnimaConfiguration())
```

## Credits

Anima — CircleStone Labs (NC) · Cosmos-Predict2 — NVIDIA · Qwen3 / Wan VAE — Alibaba ·
MLX port — xocialize.

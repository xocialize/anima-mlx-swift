# anima-swift — Stage-1 porting spec (phase table)

Swift-MLX port of Anima (Cosmos-Predict2-2B anime T2I, Non-Commercial). Path B (novel
architecture). Gated against the parity-locked **Python-MLX rung** (`../../anima-mlx`) + its
PT goldens, and the **published canonical-MLX weights** (`xocialize/anima-mlx`). CPU stream for
component parity; cosine + image-validity for the denoise loop (chaotic ODE — see anima-mlx PIPELINE.md).

## Reuse decisions (lift vs translate)
| component | decision | basis |
|---|---|---|
| Wan/Qwen-Image VAE | **LIFT** `qwen-image-edit-swift/QwenVAE.swift` (decoder-only) | same VAE; published weights canonical-MLX → remap-free load |
| Qwen3-0.6B TE | translate 1:1 from `qwen3_te.py` (idioms from mlx-swift-lm `Qwen3.swift`) | non-standard TE (pre-final-norm hidden, no lm_head) |
| Cosmos DiT | translate 1:1 from `cosmos_dit.py` | novel, no donor |
| llm_adapter | translate 1:1 from `llm_adapter.py` | novel, no donor |

## Phase table
| phase | what | gate | status |
|---|---|---|---|
| S0  | scaffold (Package.swift, Anima core, anima-cli); key contract | builds; strict-key load | **PASS 2026-06-26** |
| S1a | **AnimaVAE** (decoder) | `--vae-gate` vs Python golden | **PASS — cos 1.000000, maxabs 6.68e-6** · re-run 2026-10-29 on the 256² golden: 5.95e-6 |
| S1b | Qwen3-0.6B TE | `--te-gate` vs `goldens/qwen3` (`hidden_postnorm`) | **PASS 2026-10-29 (post-norm golden, AB-T-0020) — cos 1.000000, maxabs 1.24e-4** (was 6.10e-4 vs the retired pre-norm golden) |
| S1c | llm_adapter | `--adapter-gate` vs `goldens/adapter` | **PASS — cos 1.000000, maxabs 2.92e-6** |
| S1d | Cosmos DiT | `--dit-gate` vs `goldens/dit` | **PASS — cos 1.000000, maxabs 3.11e-5** · re-run 2026-10-29 on the rope-(1,4,4) golden: 3.15e-5 |
| S2  | Pipeline (flow CONST sampler, CFG, denorm) | `--e2e-gate` vs `goldens/pipeline` (inj noise+ids) | **PASS 2026-10-29 (post-norm goldens, AB-T-0020) — ctx 2.3e-6 / 4.2e-7, v0_cfg cos 1.0000000 maxabs 1.99e-3, final cos 0.985471 (== Python rung 0.9854712; bar ≥ 0.98 as the rung's)** · was final 0.999105 vs the pre-norm June goldens |
| S2b | tokenizer (Qwen2.5+T5 via swift-transformers) + 1 real GPU gen | id parity + eyeball | todo |
| S3  | int4 variant load + per-pass cosine | vs published int4 | **PASS — per-pass cos 0.99619** (selective quantize attn+ff g64; loadStrict leaves uint32-packed weights uncast) |
| S2b | tokenizer (swift-transformers Qwen2.5+T5) | `--tok-gate` id parity | **PASS — all ids exact** |
| S2b | real Swift GPU generation (euler) | `--generate` eyeball | ~~PASS~~ **RETRACTED 2026-07-22 — mis-calibrated**: the "coherent" 512²/24step euler render was the structural-blob class (== `anima-oracle/anima_bf16_gpu.png`), not a real image. euler produces blob garbage at every size. |
| S2c | **er_sde sampler** (`sampleErSde`, 1:1 of Python `sample_er_sde`; CPU-side per-step `GaussianRNG`) + quality-tag default negative + **TE post-norm fix** | `--generate` eyeball at 512² AND 1024² (30 step, cfg 5, short prompt, quality neg) vs the REAL-class references | **PASS 2026-07-22 — see "S2c gate results"** |
| S7  | engine wrap — `MLXAnima` target + `AnimaT2IPackage` (capability **.textToImage**, NC license, measured QuantFootprint, C0–C13) | conformance | **PASS** |

## S2c gate results (2026-07-22, honest re-calibrated eyeball)

Judged against the REAL class (`/tmp/anima_realneg.png`, `/tmp/anima_1024_best.png` — actual
characters with faces/limbs/readable composition) vs the BLOB class
(`anima-oracle/anima_bf16_gpu.png`) that the retracted S2b gate mistakenly passed.

- **512² / 30 step / cfg 5 / seed 1234** ("1girl, anime, chibi, smile, simple background",
  quality-tag neg, er_sde) → `/tmp/anima_swift_512.png`: **PASS** — crisp ComfyUI-class chibi
  (sharp lineart, detailed eyes, hair ornament, frilled dress). Peak 7.06 GB bf16.
- **1024² / 30 step / cfg 5 / seed 1234** (same prompts) → `/tmp/anima_swift_1024.png`:
  **PASS** — fully coherent chibi schoolgirl, detailed face/hands/clothing, clean background.
  Peak 9.13 GB bf16.
- **Long-prompt check** (24-token danbooru witch prompt, 512²) → `/tmp/anima_swift_longprompt.png`:
  **PASS** — all tags honored; confirms the TE post-norm fix resolves the washout.

Both fixes were REQUIRED to pass: with the pre-norm TE tap, 1024² was unstable (~1/3 real
character across seeds — one off-prompt-but-clean render, one abstract-garbage, one real —
reproduced identically in the Python rung with the same buggy tap), and long prompts washed out.

### Goldens regenerated (S1b / S2) — AB-T-0020, 2026-10-29

The pre-norm-era fixtures were removed in 569ae17. `tools/export_goldens.py` now packs
`Resources/*_golden.safetensors` from the Python rung's `tests/goldens`, which `anima-oracle/`
(PyTorch, CPU fp32) dumps with the post-final-RMSNorm TE tap and the rope_scale=(1,4,4) DiT.
Provenance check: re-running `qwen3_oracle.py` and `pipeline_oracle.py` against a fresh
`circlestone-labs/Anima` download (snapshot f973fc4, on DEV_VOL1) reproduces every array bit-for-bit
(max |Δ| = 0.0). `--te-gate` reads `hidden_postnorm`. The final-latent bar moves 0.999 → 0.98 to match
the Python rung's gate: the chaotic 6-step cfg loop makes that cosine track the oracle's torch build
(the rung itself reads 0.9855 on these goldens), and Swift matches the rung to six digits. All of
S1a–S1d, S2, S3 (int4 per-pass 0.99622) and `--tok-gate` pass on mlx-swift 0.32.3 / engine 0.73.0.

## S7 — conformance (PASS)
`Sources/MLXAnima/AnimaT2IPackage.swift` + `Tests/MLXAnimaTests`. `@InferenceActor final class AnimaT2IPackage:
ModelPackage`, `AnimaConfiguration: PackageConfiguration, ModelStorable, QuantConfigured`.
- Manifest: cap **.textToImage** (`T2IContract`), `LicenseDeclaration(weight: LicenseRef-CircleStone-NonCommercial,
  portCode: .mit)`, provenance circlestone-labs/Anima tier 3, footprints **bf16 8.0 GB / int4 6.5 GB** (measured Swift
  peak @512²), backends `.metalGPU`, OS macOS 26, chipFloor `.pro`.
- Tests PASS: `testManifest`, `testConfigurationDefaults`, **`testNonCommercialWeightGated`** (C7 — NC weight
  `rejectedWeight` under `.permissiveOnly`), full `testPackageRun` env-gated.
- **`--package-gate` PASS**: full ModelPackage contract load(2.6s)→run(8.4s)→unload → valid 512² PNG (460 KB) via
  the `.textToImage` surface, both bf16 + int4.
- C0 contractVersion (default), C1 one package + `registration .of(Self)`, C7/C8 two-layer license, C9 Configuration,
  C10 eligibility declared, C13 host-owned lifecycle (idempotent `load`, no singleton).

## Definition of done
Stage-1 COMPLETE. Remaining (out of band): push code to `github.com/xocialize/anima-mlx`; optional Phase-B er_sde
sampler for beauty parity; Stage-2 app integration (`MLXServeEngine.register` — single call, since the package is conformant).

All component/e2e gates match their Python-MLX floor exactly ⇒ Swift ≡ Python-MLX. The one non-remap-free
spot: `mlp.0/mlp.2` & `ff.net.0/2` (numeric-leaf list with a param-less GELU hole) → tiny loader rename to
`fc0/fc2` named slots (MLX `unflattened` makes numeric leaf keys arrays-with-gaps that can't be a clean Module).

## Key facts
- Capability = **`.textToImage`** (`T2IContract`/`T2IRequest`/`T2IResponse`). MLXToolKit via
  `github.com/xocialize/mlx-engine-swift` (added at S7).
- Published weights are **canonical MLX-key layout** (gamma 1-D, conv (O,kt,kh,kw,I), `.conv` wrapping)
  → Swift `loadArrays → ModuleParameters.unflattened → update`, **zero remap**. `AnimaWeights.loadStrict`
  enforces 0-missing/0-unused.
- Footprints (measured, anima-mlx): DiT resident bf16 3.91 / int8 2.21 / int4 1.38 GB; e2e peak ~14 GB(bf16)/~6 GB(int4)@512².
- Sampler: FLOW CONST, shift 3, multiplier 1 → `sigma(t)=3t/(1+2t)`, DiT timestep == sigma ∈ [0,1], CFG 4–5.

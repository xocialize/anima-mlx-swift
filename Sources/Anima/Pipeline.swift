// Anima T2I pipeline — 1:1 transpose of anima_mlx/pipeline.py.
// Sampler = comfy ModelType.FLOW: CONST prediction + ModelSamplingDiscreteFlow(shift=3, mult=1)
// → sigma(t)=3t/(1+2t), DiT timestep == sigma ∈ [0,1]. CFG: v_unc + cfg*(v_cond-v_unc).
// DEFAULT sampler = er_sde (stochastic VP ER-SDE-Solver-3); deterministic euler kept for A/B
// only — it produces structural blob garbage at every size (root-caused 2026-07-22).
import Foundation
import MLX
import MLXNN
import MLXProfiling

public let PAD_TO = 512
public let VAE_SPATIAL = 8

/// Reference workflow quality-tag negative. Load-bearing: an EMPTY negative collapses 1024²
/// into confetti (root-caused 2026-07-22, anima-mlx PIPELINE.md). Default everywhere the
/// caller does not supply a negative prompt.
public let DEFAULT_NEGATIVE_PROMPT =
    "worst quality, low quality, score_1, score_2, score_3, blurry, jpeg artifacts"

/// Sampler selection — er_sde is REQUIRED for real images (the euler placeholder produces
/// structural blob garbage at every size for this model); euler is kept for A/B only.
public enum AnimaSampler: String, Sendable {
    case erSde = "er_sde"
    case euler
}

/// Deterministic CPU-side standard-normal stream (SplitMix64 + Box-Muller, Double math).
/// The er_sde per-step noise MUST be iid per step: in the Python port, MLX random ops built
/// lazily inside the denoise loop yielded correlated draws → structured artifacts; the fix
/// was a seeded CPU generator (numpy there, this here) producing each step's gaussian
/// eagerly, then converting to an MLXArray.
public struct GaussianRNG {
    private var state: UInt64
    public init(seed: UInt64) { state = seed }

    private mutating func nextUInt64() -> UInt64 {
        state = state &+ 0x9E37_79B9_7F4A_7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
        z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
        return z ^ (z >> 31)
    }

    /// uniform in [0, 1)
    private mutating func nextUniform() -> Double {
        Double(nextUInt64() >> 11) * (1.0 / 9_007_199_254_740_992.0)
    }

    public mutating func standardNormal(count: Int) -> [Float] {
        var out = [Float]()
        out.reserveCapacity(count + 1)
        while out.count < count {
            let u1 = Swift.max(nextUniform(), 1e-300)
            let u2 = nextUniform()
            let r = (-2.0 * Foundation.log(u1)).squareRoot()
            let th = 2.0 * Double.pi * u2
            out.append(Float(r * Foundation.cos(th)))
            if out.count < count { out.append(Float(r * Foundation.sin(th))) }
        }
        return out
    }
}

public func timeSnrShift(_ a: Double, _ t: Double) -> Double { a * t / (1 + (a - 1) * t) }

/// comfy 'normal' scheduler for ModelSamplingDiscreteFlow (Double = numpy float64). sigma_max=1.0.
public func flowSigmas(_ steps: Int, shift: Double = 3.0, mult: Double = 1.0) -> [Double] {
    let sMax = timeSnrShift(shift, mult)
    let sMin = timeSnrShift(shift, (1.0 / 1000.0) * mult)
    let start = sMax * mult, end = sMin * mult
    var sigs = (0 ..< steps).map { i -> Double in
        let t = start + (end - start) * Double(i) / Double(steps - 1)
        return timeSnrShift(shift, t / mult)
    }
    sigs.append(0.0)
    return sigs
}

public final class AnimaPipeline {
    public let dit: CosmosTransformer3DModel
    public let adapter: LLMAdapter
    public let qwen: Qwen3TextEncoder
    public let vae: AnimaVAE

    public init(dit: CosmosTransformer3DModel, adapter: LLMAdapter, qwen: Qwen3TextEncoder, vae: AnimaVAE) {
        self.dit = dit; self.adapter = adapter; self.qwen = qwen; self.vae = vae
    }

    public static func load(transformer: URL, textEncoder: URL, adapter: URL, vae: URL,
                            dtype: DType = .bfloat16) throws -> AnimaPipeline {
        AnimaPipeline(
            dit: try AnimaWeights.loadDiT(file: transformer, dtype: dtype),
            adapter: try AnimaWeights.loadAdapter(file: adapter, dtype: dtype),
            qwen: try AnimaWeights.loadTextEncoder(file: textEncoder, dtype: dtype),
            vae: { let v = try! AnimaWeights.loadVAE(file: vae, dtype: dtype); v.weightDtype = dtype; return v }())
    }

    /// qwenIds [1,Lq], t5Ids [1,Lt5] -> DiT cross-attn context [1,512,1024].
    public func encodeContext(_ qwenIds: MLXArray, _ t5Ids: MLXArray) -> MLXArray {
        // Coarse span (MLX_PROFILE=1). The encode graph has no eval of its own — its compute
        // realizes at denoise step 0's eval, so this row times graph build only.
        MLXProfiler.shared.region("encode", "context") {
            let src = qwen(qwenIds)
            return adapter(src, t5Ids, padTo: PAD_TO)
        }
    }

    /// Deterministic Euler over the flow CONST schedule. noise/contexts on the model stream.
    /// A/B ONLY — NOT the default: euler produces structural blob garbage at every size for
    /// this model (the June "coherent 512²" eyeball was mis-calibrated). Use sampleErSde.
    public func sample(noise: MLXArray, condCtx: MLXArray, uncondCtx: MLXArray, sigmas: [Double],
                       cfg: Float) -> MLXArray {
        var x = Float(sigmas[0]) * noise
        let ctx = concatenated([condCtx, uncondCtx], axis: 0)
        for i in 0 ..< (sigmas.count - 1) {
            // Cooperative cancellation: bail per denoise step (non-throwing core API — the
            // MLXAnima wrapper's post-sample `try Task.checkCancellation()` rethrows).
            if Task.isCancelled { break }
            let span = MLXProfiler.shared.begin("denoise", "step", index: i,
                note: String(format: "σ=%.3f", sigmas[i]))
            let s = Float(sigmas[i])
            let xb = concatenated([x, x], axis: 0)
            let t = MLXArray([s, s])
            let v = dit(xb, t, ctx)
            let vCond = v[0 ..< 1]
            let vUnc = v[1 ..< 2]
            let vCfg = vUnc + cfg * (vCond - vUnc)
            x = x + vCfg * (Float(sigmas[i + 1]) - s)
            eval(x)
            MLXProfiler.shared.end(span)
        }
        return x
    }

    /// CONST prediction: x0 = x - sigma * v_cfg (timestep == sigma). 1:1 of pipeline.py `_denoised`.
    func denoised(_ x: MLXArray, _ sigma: Double, _ ctx: MLXArray, _ cfg: Float) -> MLXArray {
        let xb = concatenated([x, x], axis: 0)
        let t = MLXArray([Float(sigma), Float(sigma)])
        let v = dit(xb, t, ctx)
        let vCfg = v[1 ..< 2] + cfg * (v[0 ..< 1] - v[1 ..< 2])
        return x - Float(sigma) * vCfg
    }

    /// ComfyUI ModelType.FLOW default sampler: Extended Reverse-Time SDE (VP ER-SDE-Solver-3,
    /// arXiv:2309.06169), CONST model_sampling. 1:1 transpose of anima_mlx/pipeline.py
    /// `sample_er_sde` (itself a 1:1 transpose of comfy/k_diffusion `sample_er_sde`).
    /// Stochastic — the euler placeholder collapses at the model's native resolutions; this is
    /// the sampler the Anima card names as default. Scalar schedule math is Double on the CPU
    /// (== the numpy float64 reference), applied as Float scalars.
    public func sampleErSde(noise: MLXArray, condCtx: MLXArray, uncondCtx: MLXArray, sigmas: [Double],
                            cfg: Float, sNoise: Double = 1.0, maxStage: Int = 3,
                            seed: UInt64 = 0) -> MLXArray {
        var rng = GaussianRNG(seed: seed)               // iid per-step noise, CPU-side (see GaussianRNG)
        let ctx = concatenated([condCtx, uncondCtx], axis: 0)
        var sig = sigmas
        if sig[0] >= 1.0 {                              // offset_first_sigma_for_snr (CONST)
            sig[0] = timeSnrShift(3.0, 1.0 - 1e-4)
        }
        let er = sig.map { $0 / (1.0 - $0) }            // er_lambda = sigma/(1-sigma) = exp(-half_log_snr)
        func nscale(_ a: Double) -> Double {            // default_er_sde_noise_scaler
            a * (Foundation.exp(Foundation.pow(a, 0.3)) + 10.0)
        }
        let NPTS = 200.0
        var x = (Float(sig[0]) * noise).asType(.float32)
        var oldD: MLXArray?
        var oldDD: MLXArray?
        for i in 0 ..< (sig.count - 1) {
            if Task.isCancelled { break }
            let span = MLXProfiler.shared.begin("denoise", "er_sde step", index: i,
                note: String(format: "σ=%.3f", sig[i]))
            let d = denoised(x, sig[i], ctx, cfg)
            if sig[i + 1] == 0 {
                x = d
            } else {
                let ls = er[i], lt = er[i + 1]
                let aT = sig[i + 1] / lt
                let rA = aT / (sig[i] / ls)
                let r = nscale(lt) / nscale(ls)
                x = Float(rA * r) * x + Float(aT * (1 - r)) * d          // Stage 1 (Euler)
                let stage = min(maxStage, i + 1)
                if stage >= 2 {
                    let dt = lt - ls
                    let lss = -dt / NPTS
                    // 200-point sums over lpos = lt + pidx*lss, pidx = 0..<200 (== np.arange(0, NPTS))
                    var s = 0.0
                    var sU = 0.0
                    for p in 0 ..< Int(NPTS) {
                        let lpos = lt + Double(p) * lss
                        let spos = nscale(lpos)
                        s += 1.0 / spos
                        sU += (lpos - ls) / spos
                    }
                    s *= lss
                    sU *= lss
                    let dd = (d - oldD!) * Float(1.0 / (ls - er[i - 1]))
                    x = x + Float(aT * (dt + s * nscale(lt))) * dd
                    if stage >= 3 {
                        let du = (dd - oldDD!) * Float(1.0 / ((ls - er[i - 2]) / 2))
                        x = x + Float(aT * ((dt * dt) / 2 + sU * nscale(lt))) * du
                    }
                    oldDD = dd
                }
                if sNoise > 0 {
                    let coef = Float(aT * sNoise * Foundation.sqrt(Swift.max(lt * lt - ls * ls * r * r, 0.0)))
                    let nz = MLXArray(rng.standardNormal(count: x.size), x.shape)
                    x = x + coef * nz
                }
            }
            oldD = d
            eval(x)
            MLXProfiler.shared.end(span)
        }
        return x
    }

    /// model-space x0 [B,16,1,Hl,Wl] -> image [B,H,W,3] in [0,1].
    public func decode(_ x0: MLXArray) -> MLXArray {
        let lat = AnimaVAE.deNormalize(x0)
        let img = vae.decode(lat)                       // [B,3,1,H,W] in [-1,1]
        let im = (img[0..., 0..., 0].transposed(0, 2, 3, 1) + 1.0) * 0.5
        return clip(im, min: MLXArray(Float(0)), max: MLXArray(Float(1)))
    }
}

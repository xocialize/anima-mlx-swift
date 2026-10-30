//
//  main.swift
//  anima-mlx-swift / anima-validate
//
//  Drives AnimaT2IPackage through the **real `MLXServeEngine`** and reports the authoritative
//  split footprint (`MLXEngineTestKit.ValidationHarness`: `phys_footprint` sampling + the
//  kernel's lifetime high-water, floor read post-load/pre-run) — the numbers the manifest
//  declares (AB-L-0113: declare from phys, never from MLX's allocator view). `--package-gate`
//  bypasses register/prepare and the governor; this lane does not.
//
//  The engine is a research host: `.permissiveOrAcknowledged` + `.blocking`, so the NC
//  CircleStone weight must be admitted by the eval-acknowledged list — not by advisory slack.
//
//  Usage:  anima-validate <snapshotDir> [bf16|int4] [size] [steps] [out.png]
//

import Foundation
import MLX
import MLXToolKit
import MLXServeCore
import MLXEngineTestKit
import MLXAnima

setvbuf(stdout, nil, _IONBF, 0)

let args = Array(CommandLine.arguments.dropFirst())
guard let snapshot = args.first else {
    print("usage: anima-validate <snapshotDir> [bf16|int4] [size] [steps] [out.png]")
    exit(2)
}
let quant: Quant = args.count > 1 && args[1] == "int4" ? .int4 : .bf16
let size = args.count > 2 ? Int(args[2])! : 1024
let steps = args.count > 3 ? Int(args[3])! : 30
let outPath = args.count > 4 ? args[4] : nil

@MainActor
func main() async {
    let engine = MLXServeEngine(policy: .permissiveOrAcknowledged, licenseEnforcement: .blocking)
    let config = AnimaConfiguration(snapshotPath: snapshot, quant: quant)
    let request = T2IRequest(prompt: "1girl, anime, masterpiece, detailed background, soft lighting",
                             width: size, height: size, steps: steps, guidanceScale: 5.0, seed: 1234)
    do {
        let result = try await ValidationHarness.run(
            engine: engine,
            registration: AnimaT2IPackage.registration,
            configuration: config,
            capability: .textToImage,
            request: request,
            isolate: true,
            clearCache: { MLX.Memory.clearCache() },
            inputSummary: "\(size)x\(size) \(steps) steps cfg 5",
            heartbeatLabel: "anima")

        print("")
        print(result.run.splitLogLine("anima-\(quant)-\(size)"))
        print("")
        print("  DECLARE  residentBytes        = \(result.run.residentFloorBytes)")
        print("           peakActivationBytes  = \(result.run.activationBytes)")
        if result.run.retainedAfterRunBytes > 200_000_000 {
            print("  ⚠️ retains \(result.run.retainedAfterRunBytes) B after run+clearCache — "
                + "a live model holding intermediates. Belongs in the transient, not residency.")
        }
        if let out = result.response as? T2IResponse {
            print("  output: \(out.image.data.count) bytes, \(out.image.width ?? 0)x\(out.image.height ?? 0)")
            if let outPath {
                try? out.image.data.write(to: URL(fileURLWithPath: outPath))
                print("  wrote \(outPath) — eyeball it")
            }
        }
    } catch {
        print("❌ validation failed: \(error)")
        exit(1)
    }
}

await main()

// CancellationTests.swift — Anima (Cosmos-Predict2-2B anime T2I) through the engine's CAN
// gate (offline, no MLX kernels, no weights). CAN-1/2 drive the real run() pre-cancelled:
// the entry checkpoint (`try Task.checkCancellation()` as the FIRST act of run(), before
// notLoaded validation) fires before weights are touched, so a stub configuration suffices.
// CAN-3 is the document of record for the checkpoint cadence:
//   - post-encode seam — `try Task.checkCancellation()` in the wrapper's generate() after
//     encodeContext (AnimaT2IPackage.swift), before the denoise loop;
//   - denoise/step — `if Task.isCancelled { break }` at the top of the denoise loop in
//     AnimaPipeline.sampleErSde (the default sampler) and .sample (euler, A/B only)
//     (Sources/Anima/Pipeline.swift; non-throwing core — sanctioned break shape);
//   - pre-decode seam — `try Task.checkCancellation()` before the monolithic Wan-VAE decode
//     (ONE MLX eval, no chunk loop, so no per-chunk decode cadence is claimed).
// No catch blocks on the run() path — nothing to launder (CAN-2).
//
// longRunImplied: the split footprint (v0.4.0) declares peakActivationBytes 7.0 GB on both
// tiers — above the gate's 2 GB long-run threshold — so the envelope now implies long runs
// and the per-step cadence below is required, not merely volunteered.

import Foundation
import MLXServeConformance
import MLXToolKit
import XCTest

@testable import MLXAnima

final class CancellationTests: XCTestCase {

    // MARK: - CAN-1 / CAN-2 — pre-cancelled run() propagation + classification

    func testCANGatePreCancelledRun() async {
        // Stub config (empty snapshotPath — never touched: the pre-cancelled run() throws at
        // the entry checkpoint before load-state validation); construction is cheap (C13).
        let package = AnimaT2IPackage(configuration: AnimaConfiguration())
        let report = await CancellationConformance.checkRun(
            package: package,
            request: T2IRequest(prompt: "probe"))
        XCTAssertTrue(report.passed, report.summary)
    }

    // MARK: - CAN-3 — checkpoint-cadence declaration (the document of record)

    func testCANCadenceDeclaration() {
        // Split footprint: 7.0 GB declared activation ⇒ long-run implied (see note above).
        XCTAssertTrue(CancellationConformance.longRunImplied(by: AnimaT2IPackage.manifest))

        let report = CancellationConformance.checkCadence(
            manifest: AnimaT2IPackage.manifest,
            posture: .cadence([
                // Per-denoise-step Task.isCancelled break in AnimaPipeline.sample; the
                // wrapper's post-encode + pre-decode seams bracket it (single forwards —
                // seams, not recurring units).
                .init(phase: .denoise, unit: .step),
            ]))
        XCTAssertTrue(report.passed, report.summary)
    }
}

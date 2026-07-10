// CancellationTests.swift — Anima (Cosmos-Predict2-2B anime T2I) through the engine's CAN
// gate (offline, no MLX kernels, no weights). CAN-1/2 drive the real run() pre-cancelled:
// the entry checkpoint (`try Task.checkCancellation()` as the FIRST act of run(), before
// notLoaded validation) fires before weights are touched, so a stub configuration suffices.
// CAN-3 is the document of record for the checkpoint cadence:
//   - post-encode seam — `try Task.checkCancellation()` in the wrapper's generate() after
//     encodeContext (AnimaT2IPackage.swift), before the denoise loop;
//   - denoise/step — `if Task.isCancelled { break }` at the top of the Euler loop in
//     AnimaPipeline.sample (Sources/Anima/Pipeline.swift; non-throwing core — sanctioned
//     break shape);
//   - pre-decode seam — `try Task.checkCancellation()` before the monolithic Wan-VAE decode
//     (ONE MLX eval, no chunk loop, so no per-chunk decode cadence is claimed).
// No catch blocks on the run() path — nothing to launder (CAN-2).
//
// NOTE on longRunImplied: the manifest still carries the pre-1.14 FLAT footprint
// (residentBytes = measured peak; peakActivationBytes undeclared/0) and textToImage is not
// a long-run capability, so the gate's envelope heuristic reads short-run. The 30-step CFG
// denoise is nonetheless a long run in practice, so the cadence is declared anyway (the
// stricter posture); revisit the assertion if/when the footprint is split.

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
        // Flat pre-split footprint: no peakActivationBytes declared and t2i is not a
        // long-run capability, so the envelope heuristic reads short-run (see NOTE above).
        XCTAssertFalse(CancellationConformance.longRunImplied(by: AnimaT2IPackage.manifest))

        // Declared anyway — the 30-step denoise is minutes-scale on the pro-tier floor.
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

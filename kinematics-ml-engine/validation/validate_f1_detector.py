"""
F1 detector validation: how much gas pre-charge loss does the pipeline catch?

The stock scenarios (nominal / overload / fault) are separated perfectly, which
mostly shows that the synthetic fault is large. This script asks a harder
question: starting from a *nominal* F1 lap, lower the reservoir gas pressure
step by step (``pressure_scale``) and measure, over 30 random seeds per step,
how often the pipeline flags the damper as degraded and when it names the
fault mode.

Run from the repository root:

    python kinematics-ml-engine/validation/validate_f1_detector.py

Outputs (repository-relative):
    docs/validation/f1_detector_sensitivity.json
    docs/images/f1_detector_sensitivity.png
    docs/images/f1_nominal_vs_fault.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ENGINE_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = ENGINE_DIR.parent
sys.path.insert(0, str(ENGINE_DIR))

from kinematics_ml_engine import (  # noqa: E402
    OperationalProfile,
    StrutAnomalyDetector,
    generate_f1_telemetry,
    generate_telemetry,
    get_profile_config,
    run_telemetry_pipeline,
)

PRESSURE_SCALES = [1.0, 0.98, 0.96, 0.94, 0.92, 0.90, 0.88, 0.85, 0.80, 0.75, 0.70]
SEEDS = range(200, 230)
GAS_LOSS_MODE = "F1_RESERVOIR_GAS_LOSS_BLADDER_LEAK"


def fit_detector():
    cfg = get_profile_config(OperationalProfile.FORMULA1_TRACKSIDE)
    detector = StrutAnomalyDetector(cfg).fit(
        [
            generate_telemetry(cfg.profile, scenario="nominal", duration_sec=10.0, seed=11),
            generate_telemetry(cfg.profile, scenario="nominal", duration_sec=10.0, seed=29),
        ]
    )
    return detector


def sensitivity_sweep(detector):
    rows = []
    for scale in PRESSURE_SCALES:
        flagged, named, fractions = [], [], []
        for seed in SEEDS:
            batch = generate_f1_telemetry(
                "nominal", duration_sec=10.0, seed=seed, pressure_scale=scale
            )
            _, _, anomaly, _ = run_telemetry_pipeline(batch, detector=detector)
            flagged.append(anomaly.is_degraded)
            named.append(GAS_LOSS_MODE in anomaly.detected_fault_modes)
            fractions.append(anomaly.anomalous_fraction)
        rows.append(
            {
                "pressure_loss_pct": round((1.0 - scale) * 100, 1),
                "runs": len(flagged),
                "flagged_degraded_rate": float(np.mean(flagged)),
                "gas_loss_named_rate": float(np.mean(named)),
                "median_anomalous_fraction": float(np.median(fractions)),
            }
        )
        print(rows[-1])
    return rows


def plot_sensitivity(rows, path):
    loss = [r["pressure_loss_pct"] for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(loss, [r["flagged_degraded_rate"] * 100 for r in rows], "o-", lw=2,
            label="Damper flagged as degraded")
    ax.plot(loss, [r["gas_loss_named_rate"] * 100 for r in rows], "s--", lw=2,
            label="Gas-loss fault mode named")
    ax.set_xlabel("Reservoir gas pre-charge loss (%)")
    ax.set_ylabel("Runs (%) out of 30 seeds")
    ax.set_title("F1 profile: detection rate vs. gas pre-charge loss")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_nominal_vs_fault(detector, path):
    fig, axes = plt.subplots(3, 1, figsize=(10, 7.5), sharex=True)
    for scenario, color in (("nominal", "tab:blue"), ("fault", "tab:red")):
        batch = generate_f1_telemetry(scenario, duration_sec=10.0, seed=7)
        kin, _, anomaly, _ = run_telemetry_pipeline(batch, detector=detector)
        t = batch.timestamp_ms / 1000.0
        axes[0].plot(t, batch.strut_stroke_mm, color=color, lw=1, label=scenario)
        axes[1].plot(t, kin.damper_velocity_mm_s, color=color, lw=0.8, label=scenario)
        axes[2].plot(t, anomaly.anomaly_scores, color=color, lw=1, label=scenario)
    axes[0].set_ylabel("Damper stroke (mm)")
    axes[1].set_ylabel("Damper velocity (mm/s)")
    axes[2].set_ylabel("Anomaly score")
    axes[2].set_xlabel("Time (s)")
    axes[0].set_title("F1 profile, 100 Hz: nominal vs. gas-loss fault (synthetic)")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    images = REPO_ROOT / "docs" / "images"
    results = REPO_ROOT / "docs" / "validation"
    images.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    detector = fit_detector()
    rows = sensitivity_sweep(detector)
    (results / "f1_detector_sensitivity.json").write_text(json.dumps(rows, indent=2) + "\n")
    plot_sensitivity(rows, images / "f1_detector_sensitivity.png")
    plot_nominal_vs_fault(detector, images / "f1_nominal_vs_fault.png")
    print(f"Written to {images} and {results}")


if __name__ == "__main__":
    main()

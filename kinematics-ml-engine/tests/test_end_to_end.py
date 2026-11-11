"""
End-to-end integration tests for the 100 Hz telemetry kinematics, overload, and ML pipeline.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from kinematics_ml_engine import (
    InspectionSeverity,
    OperationalProfile,
    generate_telemetry,
    load_telemetry_from_csv,
    load_telemetry_from_json,
    run_telemetry_pipeline,
)


def test_full_pipeline_all_profiles_and_scenarios():
    """Verify end-to-end execution across both profiles and all three scenarios."""
    expected_severities = {
        (OperationalProfile.AEROSPACE_ATA32, "nominal"): InspectionSeverity.NOMINAL,
        (OperationalProfile.AEROSPACE_ATA32, "overload"): InspectionSeverity.CRITICAL_ACTION_REQUIRED,
        (OperationalProfile.AEROSPACE_ATA32, "fault"): InspectionSeverity.MANDATORY_INSPECTION,
        (OperationalProfile.FORMULA1_TRACKSIDE, "nominal"): InspectionSeverity.NOMINAL,
        (OperationalProfile.FORMULA1_TRACKSIDE, "overload"): InspectionSeverity.CRITICAL_ACTION_REQUIRED,
        (OperationalProfile.FORMULA1_TRACKSIDE, "fault"): InspectionSeverity.MANDATORY_INSPECTION,
    }

    for (profile, scenario), expected_sev in expected_severities.items():
        batch = generate_telemetry(profile, scenario=scenario, duration_sec=6.0, seed=42)
        kin, ovr, anom, report = run_telemetry_pipeline(batch)

        assert batch.num_samples == 600
        assert len(kin.damper_velocity_mm_s) == 600
        assert len(anom.anomaly_scores) == 600
        assert report.severity == expected_sev
        assert len(report.action_items) >= 1
        assert "Automated Structural Inspection Action Report" in report.markdown_report


def test_csv_and_json_roundtrip_ingestion():
    """Verify CSV and JSON files formatted in the C# streamer schema load cleanly into TelemetryBatch."""
    batch = generate_telemetry("f1", scenario="overload", duration_sec=2.0, seed=77)

    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "sample_f1.csv"
        json_path = Path(tmpdir) / "sample_f1.json"

        # Write CSV matching C# TelemetryPacket.CsvHeader
        lines = [
            "timestamp_ms,vertical_g,roll_deg,pitch_deg,strut_stroke_mm,internal_pressure_bar,wheel_speed_kph,profile,scenario"
        ]
        records = []
        for i in range(batch.num_samples):
            lines.append(
                f"{int(batch.timestamp_ms[i])},{batch.vertical_g[i]:.4f},{batch.roll_deg[i]:.4f},"
                f"{batch.pitch_deg[i]:.4f},{batch.strut_stroke_mm[i]:.4f},"
                f"{batch.internal_pressure_bar[i]:.4f},{batch.wheel_speed_kph[i]:.4f},f1,overload"
            )
            records.append(
                {
                    "timestamp_ms": int(batch.timestamp_ms[i]),
                    "vertical_g": float(batch.vertical_g[i]),
                    "roll_deg": float(batch.roll_deg[i]),
                    "pitch_deg": float(batch.pitch_deg[i]),
                    "strut_stroke_mm": float(batch.strut_stroke_mm[i]),
                    "internal_pressure_bar": float(batch.internal_pressure_bar[i]),
                    "wheel_speed_kph": float(batch.wheel_speed_kph[i]),
                    "profile": "f1",
                    "scenario": "overload",
                }
            )

        csv_path.write_text("\n".join(lines), encoding="utf-8")
        json_path.write_text(json.dumps(records), encoding="utf-8")

        loaded_csv = load_telemetry_from_csv(csv_path)
        loaded_json = load_telemetry_from_json(json_path)

        assert loaded_csv.num_samples == batch.num_samples
        assert loaded_csv.profile == OperationalProfile.FORMULA1_TRACKSIDE
        assert loaded_json.num_samples == batch.num_samples
        assert loaded_json.scenario == "overload"


def test_dashboard_figure_builders():
    """Verify dashboard Matplotlib figure builders execute cleanly for both profiles."""
    import sys
    import matplotlib.pyplot as plt

    repo_root = Path(__file__).resolve().parents[2]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from dashboard.app import (
        PROFILE_OPTIONS,
        build_anomaly_manifold_figure,
        build_time_series_figure,
        build_velocity_histogram_figure,
    )
    from kinematics_ml_engine import get_profile_config

    assert "Aerospace MRO Mode (ATA 32 Landing Gear)" in PROFILE_OPTIONS
    assert "Formula 1 Trackside Mode (Suspension & Kerb Loads)" in PROFILE_OPTIONS

    for label, profile in PROFILE_OPTIONS.items():
        cfg = get_profile_config(profile)
        batch = generate_telemetry(profile, scenario="overload", duration_sec=5.0, seed=42)
        kin, ovr, anom, _ = run_telemetry_pipeline(batch)

        fig1 = build_time_series_figure(batch, kin, ovr, anom, cfg)
        fig2 = build_velocity_histogram_figure(kin, cfg)
        fig3 = build_anomaly_manifold_figure(batch, anom, cfg)

        assert fig1 is not None and len(fig1.axes) == 4
        assert fig2 is not None and len(fig2.axes) == 1
        assert fig3 is not None and len(fig3.axes) == 2

        plt.close(fig1)
        plt.close(fig2)
        plt.close(fig3)


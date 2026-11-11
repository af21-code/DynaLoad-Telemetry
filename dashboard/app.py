"""
DynaLoad-Telemetry Interactive Engineering Dashboard (Streamlit).

Supports dual operational profiles:
- "Aerospace MRO Mode (ATA 32 Landing Gear)"
- "Formula 1 Trackside Mode (Suspension & Kerb Loads)"

Displays:
1. 100 Hz Time-Series Telemetry Traces (Vertical G, Strut Stroke, Damper Velocity, Pressure, Attitude, Wheel Speed)
2. Damper Velocity Histogram (Bump vs Rebound with 4-Regime Low/High-Speed breakdown)
3. Isolation Forest Anomaly Scores & Polytropic Stroke-vs-Pressure Hysteresis Manifold
4. Automated Structural Inspection Action Report (EASA AMM 05-51-11 / FIA Trackside Directives)
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Ensure kinematics_ml_engine is importable from sibling directory
REPO_ROOT = Path(__file__).resolve().parents[1]
ENGINE_DIR = REPO_ROOT / "kinematics-ml-engine"
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from kinematics_ml_engine import (
    AEROSPACE_ATA32_CONFIG,
    F1_TRACKSIDE_CONFIG,
    AnomalyResult,
    InspectionReport,
    InspectionSeverity,
    KinematicsResult,
    OperationalProfile,
    OverloadEvaluation,
    ProfileConfig,
    StrutAnomalyDetector,
    TelemetryBatch,
    VelocityRegime,
    capture_udp_telemetry,
    generate_telemetry,
    get_profile_config,
    load_telemetry_from_csv,
    load_telemetry_from_json,
    run_telemetry_pipeline,
)

PROFILE_OPTIONS: Dict[str, OperationalProfile] = {
    "Aerospace MRO Mode (ATA 32 Landing Gear)": OperationalProfile.AEROSPACE_ATA32,
    "Formula 1 Trackside Mode (Suspension & Kerb Loads)": OperationalProfile.FORMULA1_TRACKSIDE,
}

SCENARIO_OPTIONS: Dict[str, str] = {
    "Nominal Run (Within Structural & Thermo-Hydraulic Envelope)": "nominal",
    "Structural Overload (Hard Landing / High-Speed Kerb & Plank Strike)": "overload",
    "Thermo-Hydraulic Fault (N2 Gas Pressure Loss & Damping Cavitation)": "fault",
}


def build_time_series_figure(
    batch: TelemetryBatch,
    kinematics: KinematicsResult,
    overload: OverloadEvaluation,
    anomaly: AnomalyResult,
    config: ProfileConfig,
) -> plt.Figure:
    """Create a 4-panel 100 Hz engineering telemetry time-series chart."""
    t_sec = batch.time_seconds
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.patch.set_facecolor("#0e1117")
    for ax in axes:
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.grid(True, linestyle="--", alpha=0.25, color="#8b949e")

    # Panel 1: Vertical G-Load & Structural Thresholds
    ax0 = axes[0]
    ax0.plot(t_sec, batch.vertical_g, color="#58a6ff", linewidth=1.6, label="Vertical Load (G)")
    ax0.axhline(
        config.phase1_overload_vertical_g,
        color="#d29922",
        linestyle="--",
        linewidth=1.2,
        label=f"Phase 1 Trigger ({config.phase1_overload_vertical_g:.2f} G)",
    )
    ax0.axhline(
        config.phase2_critical_vertical_g,
        color="#f85149",
        linestyle="-.",
        linewidth=1.4,
        label=f"Phase 2 Critical ({config.phase2_critical_vertical_g:.2f} G)",
    )
    ax0.set_ylabel("Vertical Load [G]", color="#e6edf3", fontweight="bold")
    ax0.set_title(
        f"100 Hz Structural & Kinematic Telemetry — {config.display_name}",
        color="#f0f6fc",
        fontsize=12,
        fontweight="bold",
    )
    ax0.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)

    # Panel 2: Strut/Damper Stroke (mm) & Mechanical Bottoming Limit
    ax1 = axes[1]
    ax1.plot(t_sec, batch.strut_stroke_mm, color="#3fb950", linewidth=1.6, label="Strut/Damper Stroke (mm)")
    ax1.axhline(
        config.bottoming_stroke_threshold_mm,
        color="#f85149",
        linestyle="--",
        linewidth=1.3,
        label=f"Bottoming Threshold ({config.bottoming_stroke_threshold_mm:.1f} mm)",
    )
    ax1.axhline(
        config.max_stroke_mm,
        color="#ff7b72",
        linestyle=":",
        linewidth=1.1,
        label=f"Max Mechanical Stop ({config.max_stroke_mm:.1f} mm)",
    )
    ax1.set_ylabel("Stroke [mm]", color="#e6edf3", fontweight="bold")
    ax1.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)

    # Panel 3: Numerical Damper Velocity (mm/s) & High/Low-Speed Knees
    ax2 = axes[2]
    ax2.plot(
        t_sec,
        kinematics.damper_velocity_mm_s,
        color="#d2a8ff",
        linewidth=1.4,
        label="Damper Velocity dx/dt (mm/s)",
    )
    ax2.axhline(
        config.bump_knee_velocity_mm_s,
        color="#d29922",
        linestyle="--",
        linewidth=1.0,
        label=f"+HS Bump Knee (+{config.bump_knee_velocity_mm_s:.0f} mm/s)",
    )
    ax2.axhline(
        -config.rebound_knee_velocity_mm_s,
        color="#79c0ff",
        linestyle="--",
        linewidth=1.0,
        label=f"-HS Rebound Knee (-{config.rebound_knee_velocity_mm_s:.0f} mm/s)",
    )
    ax2.axhline(0.0, color="#8b949e", linewidth=0.8)
    ax2.set_ylabel("Velocity [mm/s]", color="#e6edf3", fontweight="bold")
    ax2.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)

    # Panel 4: Internal Chamber Pressure (bar) vs Expected Polytropic Curve
    ax3 = axes[3]
    ax3.plot(
        t_sec,
        batch.internal_pressure_bar,
        color="#ffa657",
        linewidth=1.6,
        label="Measured Internal Pressure (bar)",
    )
    ax3.plot(
        t_sec,
        anomaly.expected_polytropic_pressure_bar,
        color="#7ee787",
        linestyle="--",
        linewidth=1.3,
        alpha=0.85,
        label="Nominal Polytropic Gas Curve P(x)",
    )
    ax3.set_ylabel("Pressure [bar]", color="#e6edf3", fontweight="bold")
    ax3.set_xlabel("Time [s] (100 Hz / 10 ms frame interval)", color="#e6edf3", fontweight="bold")
    ax3.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)

    fig.tight_layout()
    return fig


def build_velocity_histogram_figure(
    kinematics: KinematicsResult,
    config: ProfileConfig,
) -> plt.Figure:
    """Create the Damper Velocity Histogram (Bump vs Rebound) colored by 4-quadrant regime."""
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b22")
    ax.tick_params(colors="#c9d1d9")
    for spine in ax.spines.values():
        spine.set_color("#30363d")
    ax.grid(True, linestyle="--", alpha=0.25, color="#8b949e")

    edges = kinematics.histogram_bin_edges_mm_s
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges) * 0.90
    densities = kinematics.histogram_densities

    colors = []
    for c in centers:
        if c > config.bump_knee_velocity_mm_s:
            colors.append("#f85149")  # High-Speed Bump
        elif c > config.deadband_velocity_mm_s:
            colors.append("#ffa657")  # Low-Speed Bump
        elif c < -config.rebound_knee_velocity_mm_s:
            colors.append("#1f6feb")  # High-Speed Rebound
        elif c < -config.deadband_velocity_mm_s:
            colors.append("#58a6ff")  # Low-Speed Rebound
        else:
            colors.append("#8b949e")  # Static deadband

    ax.bar(centers, densities, width=widths, color=colors, edgecolor="#0d1117", alpha=0.9)
    ax.axvline(
        config.bump_knee_velocity_mm_s,
        color="#f85149",
        linestyle="--",
        linewidth=1.3,
        label=f"HS Bump Knee (+{config.bump_knee_velocity_mm_s:.0f} mm/s)",
    )
    ax.axvline(
        -config.rebound_knee_velocity_mm_s,
        color="#58a6ff",
        linestyle="--",
        linewidth=1.3,
        label=f"HS Rebound Knee (-{config.rebound_knee_velocity_mm_s:.0f} mm/s)",
    )
    ax.axvline(0.0, color="#f0f6fc", linewidth=1.0)

    ax.set_title(
        "Damper Velocity Histogram — Rebound (< 0) vs. Bump (> 0)",
        color="#f0f6fc",
        fontsize=11,
        fontweight="bold",
    )
    ax.set_xlabel("Damper Velocity dx/dt [mm/s]", color="#e6edf3", fontweight="bold")
    ax.set_ylabel("Sample Occupancy [%]", color="#e6edf3", fontweight="bold")
    ax.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)
    fig.tight_layout()
    return fig


def build_anomaly_manifold_figure(
    batch: TelemetryBatch,
    anomaly: AnomalyResult,
    config: ProfileConfig,
) -> plt.Figure:
    """Create a 2-panel chart: (Left) Polytropic Stroke-vs-Pressure Manifold, (Right) Anomaly Score Timeline."""
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 4.6))
    fig.patch.set_facecolor("#0e1117")
    for ax in (ax_left, ax_right):
        ax.set_facecolor("#161b22")
        ax.tick_params(colors="#c9d1d9")
        for spine in ax.spines.values():
            spine.set_color("#30363d")
        ax.grid(True, linestyle="--", alpha=0.25, color="#8b949e")

    # Left: Polytropic Stroke vs Pressure Phase Space
    stroke_grid = np.linspace(0.0, config.max_stroke_mm * 0.96, 200)
    l0 = config.effective_gas_length_mm
    nominal_curve = config.nominal_static_pressure_bar * np.power(
        l0 / (l0 - np.minimum(stroke_grid, 0.94 * l0)), config.polytropic_gamma
    )
    ax_left.plot(
        stroke_grid,
        nominal_curve,
        color="#3fb950",
        linestyle="--",
        linewidth=2.0,
        label="Theoretical Polytropic Curve P(x)",
    )

    inlier_mask = ~anomaly.is_anomaly
    if np.any(inlier_mask):
        ax_left.scatter(
            batch.strut_stroke_mm[inlier_mask],
            batch.internal_pressure_bar[inlier_mask],
            c="#58a6ff",
            s=14,
            alpha=0.65,
            label="Nominal Manifold Samples",
        )
    if np.any(anomaly.is_anomaly):
        ax_left.scatter(
            batch.strut_stroke_mm[anomaly.is_anomaly],
            batch.internal_pressure_bar[anomaly.is_anomaly],
            c="#f85149",
            s=22,
            alpha=0.85,
            label="Isolation Forest Anomaly",
        )

    ax_left.set_title("Stroke vs. Internal Pressure Manifold", color="#f0f6fc", fontsize=11, fontweight="bold")
    ax_left.set_xlabel("Strut / Damper Stroke [mm]", color="#e6edf3", fontweight="bold")
    ax_left.set_ylabel("Internal Pressure [bar]", color="#e6edf3", fontweight="bold")
    ax_left.legend(loc="upper left", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)

    # Right: Isolation Forest Anomaly Score Timeline
    t_sec = batch.time_seconds
    ax_right.plot(
        t_sec,
        anomaly.anomaly_scores,
        color="#ffa657",
        linewidth=1.5,
        label="Anomaly Severity Score [0-1]",
    )
    ax_right.axhline(0.55, color="#f85149", linestyle="--", linewidth=1.2, label="Anomaly Trigger Threshold (0.55)")
    ax_right.fill_between(
        t_sec,
        0.0,
        anomaly.anomaly_scores,
        where=anomaly.is_anomaly,
        color="#f85149",
        alpha=0.28,
        label="Flagged Anomaly Window",
    )
    ax_right.set_ylim(0.0, 1.05)
    ax_right.set_title("Isolation Forest Anomaly Score Timeline", color="#f0f6fc", fontsize=11, fontweight="bold")
    ax_right.set_xlabel("Time [s]", color="#e6edf3", fontweight="bold")
    ax_right.set_ylabel("Anomaly Score", color="#e6edf3", fontweight="bold")
    ax_right.legend(loc="upper right", facecolor="#161b22", edgecolor="#30363d", labelcolor="#c9d1d9", fontsize=8)

    fig.tight_layout()
    return fig


def render_streamlit_app() -> None:
    """Main entrypoint for the Streamlit interactive application."""
    import streamlit as st

    st.set_page_config(
        page_title="DynaLoad-Telemetry | 100 Hz Structural & Damper Analytics",
        page_icon="🛠️",
        layout="wide",
    )

    st.title("DynaLoad-Telemetry — 100 Hz Structural Load, Damper Kinematics & Anomaly Platform")
    st.caption(
        "Unified Non-Linear Hydro-Pneumatic Kinematics & Unsupervised Isolation Forest Diagnostics for "
        "**Aerospace MRO (ATA Chapter 32)** and **Formula 1 Trackside Vehicle Dynamics**"
    )

    with st.sidebar:
        st.header("Operational Profile & Telemetry Source")
        selected_profile_label = st.selectbox(
            "Select Operational Profile",
            options=list(PROFILE_OPTIONS.keys()),
            index=0,
        )
        active_profile = PROFILE_OPTIONS[selected_profile_label]
        cfg = get_profile_config(active_profile)

        source_mode = st.radio(
            "Telemetry Ingestion Mode",
            options=[
                "100 Hz Physics Simulator",
                "Upload CSV / JSON (from .NET Streamer)",
                "Live 100 Hz UDP Listener (Port 5005)",
            ],
            index=0,
        )

        selected_scenario_label = st.selectbox(
            "Operational Scenario",
            options=list(SCENARIO_OPTIONS.keys()),
            index=0,
            disabled=(source_mode != "100 Hz Physics Simulator"),
        )
        scenario_key = SCENARIO_OPTIONS[selected_scenario_label]

        st.subheader("Signal & Physics Controls")
        duration_sec = st.slider("Capture Duration (s)", min_value=5.0, max_value=20.0, value=10.0, step=1.0)
        g_scale = st.slider("External Vertical Load Scale", min_value=0.70, max_value=1.50, value=1.00, step=0.05)
        pressure_scale = st.slider(
            "N2 Gas Pre-Charge Scale", min_value=0.45, max_value=1.20, value=1.00, step=0.05
        )
        smooth_savgol = st.checkbox("Apply Savitzky-Golay Derivative Filter", value=False)

        st.markdown("---")
        st.markdown(f"**Active Standard:**  \n`{cfg.reference_standard}`")
        st.markdown(
            f"- **Max Stroke:** `{cfg.max_stroke_mm:.0f} mm` (Bottoming: `{cfg.bottoming_stroke_threshold_mm:.0f} mm`)\n"
            f"- **Damper Knee:** `+{cfg.bump_knee_velocity_mm_s:.0f}` / `-{cfg.rebound_knee_velocity_mm_s:.0f} mm/s`\n"
            f"- **Nominal $N_2$ Pre-Charge:** `{cfg.nominal_static_pressure_bar:.1f} bar` ($\\gamma = {cfg.polytropic_gamma}$)"
        )

    batch = None
    if source_mode == "Upload CSV / JSON (from .NET Streamer)":
        uploaded = st.file_uploader("Upload CSV or JSON Telemetry File", type=["csv", "json"])
        if uploaded is not None:
            suffix = ".json" if uploaded.name.lower().endswith(".json") else ".csv"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = Path(tmp.name)
            batch = (
                load_telemetry_from_json(tmp_path)
                if suffix == ".json"
                else load_telemetry_from_csv(tmp_path)
            )
            cfg = get_profile_config(batch.profile)
        else:
            st.info("Awaiting file upload — displaying synthetic 100 Hz baseline in the meantime.")
    elif source_mode == "Live 100 Hz UDP Listener (Port 5005)":
        col_udp1, col_udp2 = st.columns([1, 3])
        udp_port = col_udp1.number_input("UDP Port", min_value=1024, max_value=65535, value=5005)
        if col_udp2.button("Capture 100 Hz UDP Burst (3.0 s Timeout)"):
            with st.spinner(f"Listening on udp://127.0.0.1:{int(udp_port)} ..."):
                batch = capture_udp_telemetry(
                    host="127.0.0.1",
                    port=int(udp_port),
                    max_packets=int(duration_sec * 100),
                    timeout_sec=3.0,
                    default_profile=active_profile,
                )
            if batch is None:
                st.warning(
                    "No UDP packets detected on port 5005. Start `/telemetry-streamer-dotnet` with "
                    "`dotnet run --project telemetry-streamer-dotnet` or inspect the simulated run below."
                )

    if batch is None:
        batch = generate_telemetry(
            profile=active_profile,
            scenario=scenario_key,
            duration_sec=duration_sec,
            sample_rate_hz=100,
            seed=42,
            g_scale=g_scale,
            pressure_scale=pressure_scale,
        )

    # Train Isolation Forest on nominal baseline for the active profile and evaluate current batch
    nom_train_1 = generate_telemetry(cfg.profile, scenario="nominal", duration_sec=10.0, seed=11)
    nom_train_2 = generate_telemetry(cfg.profile, scenario="nominal", duration_sec=10.0, seed=29)
    detector = StrutAnomalyDetector(cfg).fit([nom_train_1, nom_train_2])

    kinematics, overload, anomaly, report = run_telemetry_pipeline(
        batch,
        detector=detector,
        smooth_savgol=smooth_savgol,
    )

    # Top KPI Banner
    severity_colors = {
        InspectionSeverity.NOMINAL: "🟢 NOMINAL",
        InspectionSeverity.ADVISORY: "🟡 ADVISORY",
        InspectionSeverity.MANDATORY_INSPECTION: "🟠 MANDATORY INSPECTION",
        InspectionSeverity.CRITICAL_ACTION_REQUIRED: "🔴 CRITICAL OVERLOAD / AOG",
    }
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Disposition Status", severity_colors[report.severity])
    k2.metric(
        "Peak Vertical Load",
        f"{overload.peak_vertical_g:.2f} G",
        delta=f"Limit: {cfg.phase1_overload_vertical_g:.2f} G",
        delta_color="inverse",
    )
    k3.metric(
        "Max Strut Stroke",
        f"{kinematics.max_stroke_mm:.1f} mm",
        delta=f"{kinematics.stroke_utilization_pct:.1f}% of {cfg.max_stroke_mm:.0f} mm",
        delta_color="inverse",
    )
    k4.metric(
        "Peak Bump / Rebound",
        f"+{kinematics.peak_bump_velocity_mm_s:.0f} / {kinematics.peak_rebound_velocity_mm_s:.0f} mm/s",
        delta=f"RMS: {kinematics.rms_velocity_mm_s:.1f} mm/s",
    )
    k5.metric(
        "Polytropic Pressure Ratio",
        f"{anomaly.mean_pressure_ratio:.2f}",
        delta=f"Peak P: {overload.peak_internal_pressure_bar:.1f} bar",
    )
    k6.metric(
        "Isolation Forest Anomalies",
        f"{anomaly.anomalous_fraction * 100.0:.1f}%",
        delta=f"Peak Score: {anomaly.peak_anomaly_score:.2f}",
        delta_color="inverse",
    )

    # Section 1: 100 Hz Time-Series Telemetry Traces
    st.subheader("1. 100 Hz Time-Series Structural & Kinematic Telemetry Traces")
    fig_ts = build_time_series_figure(batch, kinematics, overload, anomaly, cfg)
    st.pyplot(fig_ts, use_container_width=True)
    plt.close(fig_ts)

    # Section 2: Damper Velocity Histogram & Regime Breakdown
    st.subheader("2. Damper Velocity Histogram (Bump vs. Rebound) & Regime Classification")
    col_hist, col_regimes = st.columns([3, 2])
    with col_hist:
        fig_hist = build_velocity_histogram_figure(kinematics, cfg)
        st.pyplot(fig_hist, use_container_width=True)
        plt.close(fig_hist)

    with col_regimes:
        st.markdown("##### 4-Quadrant Damper Velocity Regime Occupancy")
        regime_df = pd.DataFrame(
            [
                {
                    "Velocity Regime": VelocityRegime.HIGH_SPEED_BUMP.value,
                    "Threshold Window": f"> +{cfg.bump_knee_velocity_mm_s:.0f} mm/s",
                    "Occupancy (%)": kinematics.regime_fractions[VelocityRegime.HIGH_SPEED_BUMP.value],
                },
                {
                    "Velocity Regime": VelocityRegime.LOW_SPEED_BUMP.value,
                    "Threshold Window": f"+{cfg.deadband_velocity_mm_s:.0f} to +{cfg.bump_knee_velocity_mm_s:.0f} mm/s",
                    "Occupancy (%)": kinematics.regime_fractions[VelocityRegime.LOW_SPEED_BUMP.value],
                },
                {
                    "Velocity Regime": VelocityRegime.STATIC.value,
                    "Threshold Window": f"±{cfg.deadband_velocity_mm_s:.0f} mm/s",
                    "Occupancy (%)": kinematics.regime_fractions[VelocityRegime.STATIC.value],
                },
                {
                    "Velocity Regime": VelocityRegime.LOW_SPEED_REBOUND.value,
                    "Threshold Window": f"-{cfg.rebound_knee_velocity_mm_s:.0f} to -{cfg.deadband_velocity_mm_s:.0f} mm/s",
                    "Occupancy (%)": kinematics.regime_fractions[VelocityRegime.LOW_SPEED_REBOUND.value],
                },
                {
                    "Velocity Regime": VelocityRegime.HIGH_SPEED_REBOUND.value,
                    "Threshold Window": f"< -{cfg.rebound_knee_velocity_mm_s:.0f} mm/s",
                    "Occupancy (%)": kinematics.regime_fractions[VelocityRegime.HIGH_SPEED_REBOUND.value],
                },
            ]
        )
        st.dataframe(regime_df, use_container_width=True, hide_index=True)
        st.markdown(
            f"- **Bump / Rebound Symmetry Ratio:** `{kinematics.bump_rebound_symmetry_ratio:.3f}`\n"
            f"- **Bottoming Samples ($\\ge {cfg.bottoming_stroke_threshold_mm:.0f}\\text{{ mm}}$):** "
            f"`{kinematics.bottoming_sample_count}` (`{len(kinematics.bottoming_events)}` contiguous events)\n"
            f"- **High-Speed Impact Strikes ($\\ge {cfg.extreme_bump_velocity_mm_s:.0f}\\text{{ mm/s}}$):** "
            f"`{kinematics.high_speed_strike_count}`"
        )

    # Section 3: Isolation Forest Anomaly Detection & Thermo-Hydraulic Manifold
    st.subheader("3. Unsupervised Isolation Forest Anomaly Detection (Stroke-vs-Pressure Manifold)")
    fig_anom = build_anomaly_manifold_figure(batch, anomaly, cfg)
    st.pyplot(fig_anom, use_container_width=True)
    plt.close(fig_anom)

    # Section 4: Automated Structural Inspection Action Report
    st.subheader("4. Automated Structural Inspection Action Report")
    if report.severity == InspectionSeverity.CRITICAL_ACTION_REQUIRED:
        st.error(f"**{report.headline}** — {report.summary_rationale}")
    elif report.severity == InspectionSeverity.MANDATORY_INSPECTION:
        st.warning(f"**{report.headline}** — {report.summary_rationale}")
    elif report.severity == InspectionSeverity.ADVISORY:
        st.info(f"**{report.headline}** — {report.summary_rationale}")
    else:
        st.success(f"**{report.headline}** — {report.summary_rationale}")

    st.markdown(report.markdown_report)

    col_dl1, col_dl2 = st.columns(2)
    col_dl1.download_button(
        label="Download Action Report (Markdown)",
        data=report.markdown_report,
        file_name=f"dynaload_inspection_report_{cfg.profile.value}_{batch.scenario}.md",
        mime="text/markdown",
    )
    json_payload = json.dumps(
        {
            "profile": report.profile.value,
            "profile_display_name": report.profile_display_name,
            "reference_standard": report.reference_standard,
            "severity": report.severity.value,
            "headline": report.headline,
            "summary_rationale": report.summary_rationale,
            "telemetry_kpis": report.telemetry_kpis,
            "detected_fault_modes": anomaly.detected_fault_modes,
            "triggered_codes": overload.triggered_codes,
            "action_items": [
                {
                    "code": a.code,
                    "component": a.component,
                    "directive": a.directive,
                    "reference": a.reference,
                    "priority": a.priority,
                }
                for a in report.action_items
            ],
        },
        indent=2,
    )
    col_dl2.download_button(
        label="Download Action Report (JSON)",
        data=json_payload,
        file_name=f"dynaload_inspection_report_{cfg.profile.value}_{batch.scenario}.json",
        mime="application/json",
    )


if __name__ == "__main__":
    render_streamlit_app()

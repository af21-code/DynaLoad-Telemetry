"""
Structural Load Envelope & Automated Inspection Trigger Engine.

Evaluates 100 Hz telemetry and damper kinematics against:
- Profile A: EASA CS 25.473 & Aircraft Maintenance Manual (AMM 05-51-11 / ATA Chapter 32)
  Hard-Landing Phase 1 & Phase 2 inspection triggers.
- Profile B: Formula 1 Trackside Suspension Pushrod Overload, High-Speed Kerb Strike,
  and FIA Art. 3.5.9 Titanium Skid-Block / Plank Bottoming limits.
"""

from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from .models import (
    AnomalyResult,
    InspectionActionItem,
    InspectionReport,
    InspectionSeverity,
    KinematicsResult,
    OperationalProfile,
    OverloadEvaluation,
    ProfileConfig,
    TelemetryBatch,
    get_profile_config,
)


def evaluate_structural_loads(
    batch: TelemetryBatch,
    kinematics: KinematicsResult,
    config: Optional[ProfileConfig] = None,
) -> OverloadEvaluation:
    """
    Evaluate peak vertical G-loads, roll attitude coupling, damper bottoming,
    and high-speed impact velocity triggers.
    """
    cfg = config or get_profile_config(batch.profile)

    if batch.num_samples == 0:
        return OverloadEvaluation(
            peak_vertical_g=0.0,
            min_vertical_g=0.0,
            peak_vertical_g_timestamp_ms=0.0,
            peak_roll_deg=0.0,
            peak_pitch_deg=0.0,
            peak_internal_pressure_bar=0.0,
            min_internal_pressure_bar=0.0,
            is_advisory_triggered=False,
            is_phase1_overload_triggered=False,
            is_phase2_critical_triggered=False,
            is_bottoming_triggered=False,
            is_asymmetric_roll_triggered=False,
            is_kerb_strike_overload=False,
            triggered_codes=[],
        )

    peak_g_idx = int(np.argmax(batch.vertical_g))
    peak_g = float(batch.vertical_g[peak_g_idx])
    min_g = float(np.min(batch.vertical_g))
    peak_g_ts = float(batch.timestamp_ms[peak_g_idx])

    peak_roll = float(np.max(np.abs(batch.roll_deg)))
    peak_pitch = float(np.max(np.abs(batch.pitch_deg)))
    peak_pressure = float(np.max(batch.internal_pressure_bar))
    min_pressure = float(np.min(batch.internal_pressure_bar))

    is_advisory = peak_g >= cfg.advisory_vertical_g
    is_phase1 = peak_g >= cfg.phase1_overload_vertical_g
    is_phase2 = peak_g >= cfg.phase2_critical_vertical_g
    is_bottoming = kinematics.bottoming_sample_count > 0
    is_asymmetric_roll = peak_roll >= cfg.max_touchdown_or_corner_roll_deg
    is_kerb_or_impact = (
        kinematics.high_speed_strike_count > 0
        or kinematics.peak_bump_velocity_mm_s >= cfg.extreme_bump_velocity_mm_s
    )

    codes: List[str] = []
    if cfg.profile == OperationalProfile.AEROSPACE_ATA32:
        if is_phase2:
            codes.append("ATA32-HL-PHASE2-SEVERE-OVERLOAD")
        elif is_phase1:
            codes.append("ATA32-HL-PHASE1-MANDATORY-INSPECTION")
        elif is_advisory:
            codes.append("ATA32-ROUGH-TOUCHDOWN-ADVISORY")

        if is_bottoming:
            codes.append("ATA32-OLEO-STRUT-MECHANICAL-BOTTOMING")
        if is_asymmetric_roll:
            codes.append("ATA32-HIGH-BANK-ASYMMETRIC-SIDESTAY-LOAD")
        if is_kerb_or_impact:
            codes.append("ATA32-HIGH-SINK-RATE-ORIFICE-VELOCITY-EXCEEDANCE")
    else:
        if is_phase2:
            codes.append("F1-PUSHROD-BUCKLING-LOAD-EXCEEDANCE")
        elif is_phase1:
            codes.append("F1-HEAVY-KERB-LOAD-WARNING")
        elif is_advisory:
            codes.append("F1-ELEVATED-VERTICAL-AERO-KERB-LOAD")

        if is_bottoming:
            codes.append("F1-PLANK-SKIDBLOCK-BOTTOMING-STRIKE")
        if is_kerb_or_impact:
            codes.append("F1-SAWTOOTH-KERB-HIGH-SPEED-BUMP-SPIKE")
        if is_asymmetric_roll:
            codes.append("F1-EXCESSIVE-ROLL-ANTIROLLBAR-SATURATION")

    return OverloadEvaluation(
        peak_vertical_g=round(peak_g, 3),
        min_vertical_g=round(min_g, 3),
        peak_vertical_g_timestamp_ms=round(peak_g_ts, 1),
        peak_roll_deg=round(peak_roll, 3),
        peak_pitch_deg=round(peak_pitch, 3),
        peak_internal_pressure_bar=round(peak_pressure, 2),
        min_internal_pressure_bar=round(min_pressure, 2),
        is_advisory_triggered=is_advisory,
        is_phase1_overload_triggered=is_phase1,
        is_phase2_critical_triggered=is_phase2,
        is_bottoming_triggered=is_bottoming,
        is_asymmetric_roll_triggered=is_asymmetric_roll,
        is_kerb_strike_overload=is_kerb_or_impact,
        triggered_codes=codes,
    )


def generate_inspection_report(
    batch: TelemetryBatch,
    kinematics: KinematicsResult,
    overload: OverloadEvaluation,
    anomaly: AnomalyResult,
    config: Optional[ProfileConfig] = None,
) -> InspectionReport:
    """
    Synthesize kinematics, structural G-load triggers, and Isolation Forest anomaly diagnostics
    into an automated Structural Inspection Action Report.
    """
    cfg = config or get_profile_config(batch.profile)
    actions: List[InspectionActionItem] = []

    if cfg.profile == OperationalProfile.AEROSPACE_ATA32:
        severity, headline, summary, actions = _build_aerospace_report(
            kinematics, overload, anomaly, cfg
        )
    else:
        severity, headline, summary, actions = _build_f1_report(
            kinematics, overload, anomaly, cfg
        )

    kpis: Dict[str, float] = {
        "peak_vertical_g": overload.peak_vertical_g,
        "max_stroke_mm": kinematics.max_stroke_mm,
        "stroke_utilization_pct": kinematics.stroke_utilization_pct,
        "peak_bump_velocity_mm_s": kinematics.peak_bump_velocity_mm_s,
        "peak_rebound_velocity_mm_s": kinematics.peak_rebound_velocity_mm_s,
        "peak_internal_pressure_bar": overload.peak_internal_pressure_bar,
        "min_internal_pressure_bar": overload.min_internal_pressure_bar,
        "mean_polytropic_pressure_ratio": anomaly.mean_pressure_ratio,
        "anomalous_frame_pct": round(anomaly.anomalous_fraction * 100.0, 2),
        "bottoming_samples": float(kinematics.bottoming_sample_count),
    }

    md_lines = [
        f"### Automated Structural Inspection Action Report",
        f"- **Operational Profile:** {cfg.display_name}",
        f"- **Governing Standard:** `{cfg.reference_standard}`",
        f"- **Disposition Severity:** `{severity.value}`",
        f"- **Assessment Headline:** **{headline}**",
        "",
        f"#### Engineering Rationale",
        f"{summary}",
        "",
        f"#### Key Structural & Kinematic Telemetry Metrics",
        f"| Metric | Value | Limit / Reference |",
        f"| :--- | :--- | :--- |",
        f"| Peak Vertical Load ($n_z$) | `{overload.peak_vertical_g:.2f} G` @ `{overload.peak_vertical_g_timestamp_ms:.0f} ms` | Phase 1: `{cfg.phase1_overload_vertical_g:.2f} G` / Phase 2: `{cfg.phase2_critical_vertical_g:.2f} G` |",
        f"| Max Strut/Damper Stroke | `{kinematics.max_stroke_mm:.1f} mm` (`{kinematics.stroke_utilization_pct:.1f}%`) | Bottoming Threshold: `{cfg.bottoming_stroke_threshold_mm:.1f} mm` (Max `{cfg.max_stroke_mm:.1f} mm`) |",
        f"| Peak Bump / Rebound Velocity | `+{kinematics.peak_bump_velocity_mm_s:.1f}` / `{kinematics.peak_rebound_velocity_mm_s:.1f} mm/s` | Knee: `±{cfg.bump_knee_velocity_mm_s:.0f} mm/s` / Extreme: `{cfg.extreme_bump_velocity_mm_s:.0f} mm/s` |",
        f"| Chamber Pressure Range | `{overload.min_internal_pressure_bar:.1f}` – `{overload.peak_internal_pressure_bar:.1f} bar` | Polytropic Ratio: `{anomaly.mean_pressure_ratio:.2f}` (Min `{cfg.min_healthy_pressure_ratio:.2f}`) |",
        f"| ML Anomaly Rate (Isolation Forest) | `{anomaly.anomalous_fraction * 100.0:.1f}%` (Peak Score `{anomaly.peak_anomaly_score:.2f}`) | Fault Modes: `{', '.join(anomaly.detected_fault_modes) if anomaly.detected_fault_modes else 'NONE'}` |",
        "",
        f"#### Mandated MRO / Race Engineering Action Directives",
    ]

    for idx, item in enumerate(actions, start=1):
        md_lines.append(
            f"{idx}. **[{item.priority}] `{item.code}` — {item.component}**: {item.directive} *(Ref: {item.reference})*"
        )

    return InspectionReport(
        profile=cfg.profile,
        profile_display_name=cfg.display_name,
        reference_standard=cfg.reference_standard,
        severity=severity,
        headline=headline,
        summary_rationale=summary,
        telemetry_kpis=kpis,
        action_items=actions,
        markdown_report="\n".join(md_lines),
    )


def _build_aerospace_report(
    kinematics: KinematicsResult,
    overload: OverloadEvaluation,
    anomaly: AnomalyResult,
    cfg: ProfileConfig,
) -> tuple[InspectionSeverity, str, str, List[InspectionActionItem]]:
    actions: List[InspectionActionItem] = []

    has_gas_loss = "ATA32_N2_PRECHARGE_LOSS_OR_SEAL_BLOWBY" in anomaly.detected_fault_modes
    has_damping_loss = "ATA32_HYDRAULIC_ORIFICE_DAMPING_DEGRADATION" in anomaly.detected_fault_modes

    if overload.is_phase2_critical_triggered or overload.is_bottoming_triggered:
        severity = InspectionSeverity.CRITICAL_ACTION_REQUIRED
        headline = "AOG GROUNDING: EASA AMM 05-51-11 Phase 2 Hard-Landing & Oleo Bottoming Inspection Required"
        summary = (
            f"Touchdown vertical load factor reached {overload.peak_vertical_g:.2f} G (exceeding EASA Phase 2 critical threshold "
            f"of {cfg.phase2_critical_vertical_g:.2f} G) with maximum oleo-strut compression of {kinematics.max_stroke_mm:.1f} mm "
            f"({kinematics.stroke_utilization_pct:.1f}% stroke utilization). Aircraft must be placed on AOG status prior to next dispatch."
        )
        actions.append(
            InspectionActionItem(
                code="ATA32-WO-001",
                component="Main Landing Gear (MLG) Shock Strut & Pintle Pins",
                directive="Perform Phase 1 & Phase 2 Hard-Landing NDT (Magnetic Particle / Ultrasonic) on MLG outer cylinder, trunnion pins, and wing/fuselage gear beam attachments.",
                reference="AMM 05-51-11 Phase 2 / ATA 32-11-00",
                priority="CRITICAL - AOG",
            )
        )
        if overload.is_bottoming_triggered:
            actions.append(
                InspectionActionItem(
                    code="ATA32-WO-002",
                    component="MLG Oleo Internal Stop & Metering Pin",
                    directive=f"Oleo compressed to {kinematics.max_stroke_mm:.1f} mm ({kinematics.bottoming_sample_count} samples >= {cfg.bottoming_stroke_threshold_mm:.0f} mm). Borescope internal stop ring, orifice plate, and lower bearing bronzeland for impact brinelling.",
                    reference="CMM 32-11-14",
                    priority="CRITICAL - AOG",
                )
            )
        if overload.is_asymmetric_roll_triggered:
            actions.append(
                InspectionActionItem(
                    code="ATA32-WO-003",
                    component="MLG Side-Stay, Drag Brace & Torque Links",
                    directive=f"Touchdown bank angle reached {overload.peak_roll_deg:.1f} deg (> {cfg.max_touchdown_or_corner_roll_deg:.1f} deg limit). Inspect side-stay cardan joint and upper attachment lug for lateral shear overload.",
                    reference="AMM 05-51-11 Para 4.B (High-Roll Touchdown)",
                    priority="HIGH",
                )
            )
    elif overload.is_phase1_overload_triggered or has_gas_loss or has_damping_loss:
        severity = InspectionSeverity.MANDATORY_INSPECTION
        if has_gas_loss or has_damping_loss:
            headline = "MANDATORY MRO ACTION: ATA 32 Oleo-Pneumatic Gas Charge / Damping Degradation Detected"
            summary = (
                f"While peak vertical touchdown load was {overload.peak_vertical_g:.2f} G, the Isolation Forest manifold detector "
                f"flagged {anomaly.anomalous_fraction * 100.0:.1f}% anomalous frames (mean polytropic pressure ratio {anomaly.mean_pressure_ratio:.2f}). "
                f"Strut compressed to {kinematics.max_stroke_mm:.1f} mm under sub-nominal gas spring stiffness and under-damped recoil."
            )
        else:
            headline = "MANDATORY MRO ACTION: EASA AMM 05-51-11 Phase 1 Hard-Landing Inspection Triggered"
            summary = (
                f"Peak vertical load factor of {overload.peak_vertical_g:.2f} G exceeded the EASA Phase 1 mandatory inspection threshold "
                f"({cfg.phase1_overload_vertical_g:.2f} G). Visual and dimensional checks of the landing gear installation are required before release."
            )

        if overload.is_phase1_overload_triggered:
            actions.append(
                InspectionActionItem(
                    code="ATA32-WO-010",
                    component="MLG Strut Assembly, Tires & Wheels",
                    directive="Execute AMM 05-51-11 Phase 1 visual inspection for fluid leakage, chrome piston scoring, tire carcass ply separation, and fuselage skin wrinkling.",
                    reference="AMM 05-51-11 Phase 1",
                    priority="MANDATORY",
                )
            )
        if has_gas_loss:
            actions.append(
                InspectionActionItem(
                    code="ATA32-WO-011",
                    component="MLG Nitrogen (N2) Charging Valve & Upper Floating Piston Seal",
                    directive=f"Measured chamber pressure ratio ({anomaly.mean_pressure_ratio:.2f}) is significantly below nominal polytropic curve. Jack aircraft, verify H-dimension vs pressure chart, inspect T-seals for N2 blow-by, and recharge with dry nitrogen per AMM 12-14-32.",
                    reference="AMM 12-14-32 / ATA 32-11",
                    priority="MANDATORY",
                )
            )
        if has_damping_loss:
            actions.append(
                InspectionActionItem(
                    code="ATA32-WO-012",
                    component="MLG Hydraulic Chamber & Metering Orifice Valve",
                    directive="Excessive post-impact heave oscillation and damping admittance detected. Sample MIL-PRF-87257/5606 hydraulic fluid for foaming/gas entrainment and verify recoil orifice check-valve seating.",
                    reference="AMM 32-11-00",
                    priority="MANDATORY",
                )
            )
    elif overload.is_advisory_triggered:
        severity = InspectionSeverity.ADVISORY
        headline = "LINE MAINTENANCE ADVISORY: Elevated Touchdown Load Factor Logged"
        summary = (
            f"Touchdown vertical load reached {overload.peak_vertical_g:.2f} G (Advisory range {cfg.advisory_vertical_g:.2f}–{cfg.phase1_overload_vertical_g:.2f} G). "
            f"Oleo-strut thermo-hydraulic signature remained nominal."
        )
        actions.append(
            InspectionActionItem(
                code="ATA32-WO-020",
                component="MLG Chrome Piston & Tire Tread",
                directive="Perform walk-around visual check of exposed oleo chrome extension and check hydraulic seal wiper ring during turnaround.",
                reference="AMM 32-00-00 Turnaround Check",
                priority="ADVISORY",
            )
        )
    else:
        severity = InspectionSeverity.NOMINAL
        headline = "SERVICEABLE: Nominal ATA 32 Touchdown Kinematics & Polytropic Signature"
        summary = (
            f"Touchdown peak vertical load ({overload.peak_vertical_g:.2f} G), oleo stroke utilization ({kinematics.stroke_utilization_pct:.1f}%), "
            f"and polytropic nitrogen pressure ratio ({anomaly.mean_pressure_ratio:.2f}) are all within nominal EASA CS-25 structural envelopes."
        )
        actions.append(
            InspectionActionItem(
                code="ATA32-OK-000",
                component="ATA 32 Landing Gear System",
                directive="No unscheduled maintenance action required. Archive 100 Hz QAR touchdown trace to Fleet Health Monitoring database.",
                reference="ATA 32 Continuous Airworthiness",
                priority="ROUTINE",
            )
        )

    return severity, headline, summary, actions


def _build_f1_report(
    kinematics: KinematicsResult,
    overload: OverloadEvaluation,
    anomaly: AnomalyResult,
    cfg: ProfileConfig,
) -> tuple[InspectionSeverity, str, str, List[InspectionActionItem]]:
    actions: List[InspectionActionItem] = []

    has_gas_loss = "F1_RESERVOIR_GAS_LOSS_BLADDER_LEAK" in anomaly.detected_fault_modes
    has_cavitation = "F1_DAMPER_HYDRAULIC_CAVITATION_SHIM_FATIGUE" in anomaly.detected_fault_modes

    if overload.is_phase2_critical_triggered or overload.is_bottoming_triggered:
        severity = InspectionSeverity.CRITICAL_ACTION_REQUIRED
        headline = "TRACKSIDE RED ALERT: Pushrod Overload & FIA Skid-Block Bottoming Strike Detected"
        summary = (
            f"Suspension recorded a peak vertical transient of {overload.peak_vertical_g:.2f} G and peak high-speed bump velocity of "
            f"+{kinematics.peak_bump_velocity_mm_s:.1f} mm/s, reaching {kinematics.max_stroke_mm:.1f} mm travel "
            f"({kinematics.stroke_utilization_pct:.1f}% of {cfg.max_stroke_mm:.0f} mm max limit). Immediate garage inspection required."
        )
        actions.append(
            InspectionActionItem(
                code="F1-NDT-001",
                component="Carbon Pushrod Assembly, Flexures & Upright Clevis",
                directive=f"Peak load ({overload.peak_vertical_g:.2f} G) exceeded pushrod buckling margin ({cfg.phase2_critical_vertical_g:.2f} G). Perform ultrasonic phased-array NDT on carbon pushrod bondlines and titanium upright bracket.",
                reference="Trackside Structural Safety Protocol SSP-04",
                priority="CRITICAL - GARAGE STOP",
            )
        )
        if overload.is_bottoming_triggered:
            actions.append(
                InspectionActionItem(
                    code="F1-FIA-002",
                    component="Underfloor Plank / Titanium Skid Blocks & Packers",
                    directive=f"Damper bottomed for {kinematics.bottoming_sample_count} samples (>= {cfg.bottoming_stroke_threshold_mm:.1f} mm). Laser-scan FIA skid-block wear holes (Art. 3.5.9 max 1.0 mm wear) and add +2.0 mm bump-rubber packer or raise rear ride height by +1.5 mm.",
                    reference="FIA F1 Technical Regulations Art. 3.5.9",
                    priority="CRITICAL - LEGALITY",
                )
            )
        if overload.is_kerb_strike_overload:
            actions.append(
                InspectionActionItem(
                    code="F1-DMP-003",
                    component="High-Speed Bump Blow-Off Valve & Heave Belleville Stack",
                    directive=f"High-speed bump velocity reached +{kinematics.peak_bump_velocity_mm_s:.1f} mm/s across sawtooth kerbs. Stiffen high-speed bump blow-off threshold by +2 clicks and inspect bump rubber for splitting.",
                    reference="Race Engineering Setup Note RE-12",
                    priority="HIGH",
                )
            )
    elif overload.is_phase1_overload_triggered or has_gas_loss or has_cavitation:
        severity = InspectionSeverity.MANDATORY_INSPECTION
        if has_gas_loss or has_cavitation:
            headline = "GARAGE ACTION: Damper Reservoir Gas Loss & Hydraulic Cavitation Detected"
            summary = (
                f"Isolation Forest anomaly detector flagged {anomaly.anomalous_fraction * 100.0:.1f}% of the lap sector "
                f"(mean gas reservoir pressure ratio {anomaly.mean_pressure_ratio:.2f}). Damper exhibited loss of nitrogen pre-charge "
                f"and high-frequency wheel-hop oscillation."
            )
        else:
            headline = "GARAGE ACTION: Elevated Kerb Strike Load & High-Speed Damper Excursion"
            summary = (
                f"Peak vertical load reached {overload.peak_vertical_g:.2f} G with high-speed bump velocity of +{kinematics.peak_bump_velocity_mm_s:.1f} mm/s. "
                f"Suspension inspection required at end of run."
            )

        if has_gas_loss:
            actions.append(
                InspectionActionItem(
                    code="F1-DMP-010",
                    component="Damper Nitrogen (N2) Piggyback Reservoir & Bladder",
                    directive=f"Reservoir pressure dropped to {overload.min_internal_pressure_bar:.1f} bar (ratio {anomaly.mean_pressure_ratio:.2f} vs nominal {cfg.nominal_static_pressure_bar:.1f} bar). Replace corner damper module and check floating piston quad-ring seal.",
                    reference="Hydropneumatic Damper Manual HDM-07",
                    priority="MANDATORY - BOX NOW",
                )
            )
        if has_cavitation:
            actions.append(
                InspectionActionItem(
                    code="F1-DMP-011",
                    component="Damper Hydraulic Circuit & Shim Stack",
                    directive="Cavitation / oil aeration signature detected in velocity-vs-load phase space. Vacuum-bleed damper assembly on Roehrig dyno and verify high-speed rebound check shim integrity.",
                    reference="Hydropneumatic Damper Manual HDM-09",
                    priority="MANDATORY",
                )
            )
        if overload.is_phase1_overload_triggered:
            actions.append(
                InspectionActionItem(
                    code="F1-SUS-012",
                    component="Wishbone Inboard Flexures & Rocker Bearings",
                    directive="Check wishbone strain-gauge zero offsets and inspect inboard rocker needle bearings after heavy kerb strike.",
                    reference="Trackside Reliability Checklist TRC-02",
                    priority="MANDATORY",
                )
            )
    elif overload.is_advisory_triggered:
        severity = InspectionSeverity.ADVISORY
        headline = "RACE ENGINEERING ADVISORY: Moderate Kerb Compression & Aero Load"
        summary = (
            f"Peak vertical load of {overload.peak_vertical_g:.2f} G and max travel of {kinematics.max_stroke_mm:.1f} mm approached "
            f"upper setup window, while damper reservoir pressure remained healthy."
        )
        actions.append(
            InspectionActionItem(
                code="F1-SET-020",
                component="Corner & Heave Damper Setup",
                directive="Monitor skid-block laser ride height sensor at Vmax on main straight; advise driver to reduce apex sausage-kerb overlap in Sector 3.",
                reference="Run Plan Advisory",
                priority="ADVISORY",
            )
        )
    else:
        severity = InspectionSeverity.NOMINAL
        headline = "GREEN TRACK STATUS: Nominal F1 Suspension Loads & Balanced Damper Histogram"
        summary = (
            f"Peak vertical load ({overload.peak_vertical_g:.2f} G), damper travel ({kinematics.max_stroke_mm:.1f} mm / {kinematics.stroke_utilization_pct:.1f}%), "
            f"bump/rebound symmetry ({kinematics.bump_rebound_symmetry_ratio:.2f}), and reservoir gas curve ({anomaly.mean_pressure_ratio:.2f}) are all optimal."
        )
        actions.append(
            InspectionActionItem(
                code="F1-OK-000",
                component="Corner & Heave Suspension Assembly",
                directive="Car cleared for continuous high-fuel and qualifying runs. Damper velocity histogram centered within target window.",
                reference="Race Engineering Sign-Off",
                priority="ROUTINE",
            )
        )

    return severity, headline, summary, actions

"""
Unit tests for EASA ATA Chapter 32 Hard-Landing triggers and F1 Trackside Pushrod/Plank overload rules.
"""

from __future__ import annotations

from kinematics_ml_engine import (
    AEROSPACE_ATA32_CONFIG,
    F1_TRACKSIDE_CONFIG,
    InspectionSeverity,
    analyze_kinematics,
    evaluate_structural_loads,
    generate_aerospace_telemetry,
    generate_inspection_report,
)


def test_aerospace_nominal_vs_overload_triggers(
    aero_nominal_batch,
    aero_overload_batch,
    aero_detector,
):
    """Verify Aerospace Nominal run passes and Overload run triggers EASA Phase 2 Hard-Landing AOG."""
    kin_nom = analyze_kinematics(aero_nominal_batch, AEROSPACE_ATA32_CONFIG)
    ovr_nom = evaluate_structural_loads(aero_nominal_batch, kin_nom, AEROSPACE_ATA32_CONFIG)
    anom_nom = aero_detector.evaluate(aero_nominal_batch, kin_nom.damper_velocity_mm_s)
    rep_nom = generate_inspection_report(
        aero_nominal_batch, kin_nom, ovr_nom, anom_nom, AEROSPACE_ATA32_CONFIG
    )

    assert not ovr_nom.is_phase1_overload_triggered
    assert not ovr_nom.is_phase2_critical_triggered
    assert not ovr_nom.is_bottoming_triggered
    assert rep_nom.severity == InspectionSeverity.NOMINAL

    kin_hl = analyze_kinematics(aero_overload_batch, AEROSPACE_ATA32_CONFIG)
    ovr_hl = evaluate_structural_loads(aero_overload_batch, kin_hl, AEROSPACE_ATA32_CONFIG)
    anom_hl = aero_detector.evaluate(aero_overload_batch, kin_hl.damper_velocity_mm_s)
    rep_hl = generate_inspection_report(
        aero_overload_batch, kin_hl, ovr_hl, anom_hl, AEROSPACE_ATA32_CONFIG
    )

    assert ovr_hl.peak_vertical_g >= AEROSPACE_ATA32_CONFIG.phase2_critical_vertical_g
    assert ovr_hl.is_phase2_critical_triggered
    assert ovr_hl.is_bottoming_triggered
    assert ovr_hl.is_asymmetric_roll_triggered
    assert "ATA32-HL-PHASE2-SEVERE-OVERLOAD" in ovr_hl.triggered_codes
    assert "ATA32-OLEO-STRUT-MECHANICAL-BOTTOMING" in ovr_hl.triggered_codes
    assert rep_hl.severity == InspectionSeverity.CRITICAL_ACTION_REQUIRED
    assert any(item.code == "ATA32-WO-001" for item in rep_hl.action_items)


def test_aerospace_phase1_hard_landing_trigger(aero_detector):
    """Verify moderate hard landing (2.10 G <= n_z < 2.60 G) triggers EASA Phase 1 mandatory inspection."""
    batch_p1 = generate_aerospace_telemetry(scenario="nominal", duration_sec=6.0, seed=19, g_scale=2.25)
    kin = analyze_kinematics(batch_p1, AEROSPACE_ATA32_CONFIG)
    ovr = evaluate_structural_loads(batch_p1, kin, AEROSPACE_ATA32_CONFIG)
    anom = aero_detector.evaluate(batch_p1, kin.damper_velocity_mm_s)
    rep = generate_inspection_report(batch_p1, kin, ovr, anom, AEROSPACE_ATA32_CONFIG)

    assert AEROSPACE_ATA32_CONFIG.phase1_overload_vertical_g <= ovr.peak_vertical_g < AEROSPACE_ATA32_CONFIG.phase2_critical_vertical_g
    assert ovr.is_phase1_overload_triggered
    assert not ovr.is_phase2_critical_triggered
    assert "ATA32-HL-PHASE1-MANDATORY-INSPECTION" in ovr.triggered_codes
    assert rep.severity == InspectionSeverity.MANDATORY_INSPECTION


def test_f1_nominal_vs_overload_triggers(
    f1_nominal_batch,
    f1_overload_batch,
    f1_detector,
):
    """Verify F1 Nominal lap is cleared and Overload lap triggers Plank Bottoming & Pushrod NDT."""
    kin_nom = analyze_kinematics(f1_nominal_batch, F1_TRACKSIDE_CONFIG)
    ovr_nom = evaluate_structural_loads(f1_nominal_batch, kin_nom, F1_TRACKSIDE_CONFIG)
    anom_nom = f1_detector.evaluate(f1_nominal_batch, kin_nom.damper_velocity_mm_s)
    rep_nom = generate_inspection_report(
        f1_nominal_batch, kin_nom, ovr_nom, anom_nom, F1_TRACKSIDE_CONFIG
    )

    assert not ovr_nom.is_phase1_overload_triggered
    assert not ovr_nom.is_bottoming_triggered
    assert rep_nom.severity == InspectionSeverity.NOMINAL

    kin_ovr = analyze_kinematics(f1_overload_batch, F1_TRACKSIDE_CONFIG)
    ovr_ovr = evaluate_structural_loads(f1_overload_batch, kin_ovr, F1_TRACKSIDE_CONFIG)
    anom_ovr = f1_detector.evaluate(f1_overload_batch, kin_ovr.damper_velocity_mm_s)
    rep_ovr = generate_inspection_report(
        f1_overload_batch, kin_ovr, ovr_ovr, anom_ovr, F1_TRACKSIDE_CONFIG
    )

    assert ovr_ovr.is_phase2_critical_triggered
    assert ovr_ovr.is_bottoming_triggered
    assert ovr_ovr.is_kerb_strike_overload
    assert "F1-PLANK-SKIDBLOCK-BOTTOMING-STRIKE" in ovr_ovr.triggered_codes
    assert "F1-PUSHROD-BUCKLING-LOAD-EXCEEDANCE" in ovr_ovr.triggered_codes
    assert rep_ovr.severity == InspectionSeverity.CRITICAL_ACTION_REQUIRED
    assert any(item.code == "F1-FIA-002" for item in rep_ovr.action_items)

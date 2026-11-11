"""
Unit tests for Isolation Forest thermo-hydraulic manifold anomaly detection.
"""

from __future__ import annotations

import numpy as np

from kinematics_ml_engine import (
    AEROSPACE_ATA32_CONFIG,
    F1_TRACKSIDE_CONFIG,
    StrutAnomalyDetector,
    compute_expected_polytropic_pressure,
)


def test_polytropic_gas_spring_law_monotonicity():
    """Expected gas spring pressure P(x) = P0 * (L0 / (L0 - x))^gamma must increase nonlinearly with stroke."""
    strokes = np.array([0.0, 100.0, 200.0, 300.0, 400.0])
    pressures = compute_expected_polytropic_pressure(strokes, AEROSPACE_ATA32_CONFIG)

    np.testing.assert_allclose(pressures[0], AEROSPACE_ATA32_CONFIG.nominal_static_pressure_bar, atol=1e-6)
    assert np.all(np.diff(pressures) > 0.0)
    # Second derivative must also be positive (convex stiffening gas spring)
    assert np.all(np.diff(pressures, n=2) > 0.0)


def test_aerospace_fault_detection_n2_leak_and_damping(
    aero_nominal_batch,
    aero_fault_batch,
    aero_detector,
):
    """Isolation Forest must distinguish nominal ATA 32 oleo compression from N2 gas leak + orifice wear."""
    res_nom = aero_detector.evaluate(aero_nominal_batch)
    res_flt = aero_detector.evaluate(aero_fault_batch)

    assert not res_nom.is_degraded
    assert res_nom.mean_pressure_ratio > 0.90
    assert res_flt.is_degraded
    assert res_flt.anomalous_fraction > res_nom.anomalous_fraction
    assert res_flt.mean_pressure_ratio < AEROSPACE_ATA32_CONFIG.min_healthy_pressure_ratio
    assert "ATA32_N2_PRECHARGE_LOSS_OR_SEAL_BLOWBY" in res_flt.detected_fault_modes


def test_f1_fault_detection_reservoir_loss_and_cavitation(
    f1_nominal_batch,
    f1_fault_batch,
    f1_detector,
):
    """Isolation Forest must detect F1 damper nitrogen bladder loss and high-frequency cavitation."""
    res_nom = f1_detector.evaluate(f1_nominal_batch)
    res_flt = f1_detector.evaluate(f1_fault_batch)

    assert not res_nom.is_degraded
    assert res_flt.is_degraded
    assert res_flt.anomalous_fraction > res_nom.anomalous_fraction
    assert "F1_RESERVOIR_GAS_LOSS_BLADDER_LEAK" in res_flt.detected_fault_modes
    assert "F1_DAMPER_HYDRAULIC_CAVITATION_SHIM_FATIGUE" in res_flt.detected_fault_modes


def test_unfitted_detector_raises_runtime_error(aero_nominal_batch):
    """Calling evaluate() before fit() must raise RuntimeError."""
    det = StrutAnomalyDetector(AEROSPACE_ATA32_CONFIG)
    try:
        det.evaluate(aero_nominal_batch)
        assert False, "Expected RuntimeError when evaluating unfitted detector"
    except RuntimeError as exc:
        assert "must be fitted" in str(exc)

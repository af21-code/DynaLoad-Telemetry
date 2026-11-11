"""
Unit tests for 100 Hz numerical damper velocity differentiation and regime classification.
"""

from __future__ import annotations

import numpy as np

from kinematics_ml_engine import (
    AEROSPACE_ATA32_CONFIG,
    F1_TRACKSIDE_CONFIG,
    VelocityRegime,
    analyze_kinematics,
    classify_velocity_regimes,
    compute_damper_velocity,
    compute_velocity_histogram,
)


def test_compute_damper_velocity_constant_slope():
    """Constant stroke ramp x(t) = 125.0 * t should yield exact 125.0 mm/s velocity."""
    t_ms = np.arange(0, 1000, 10, dtype=np.float64)  # 100 Hz for 1.0 s
    stroke_mm = 125.0 * (t_ms / 1000.0)
    vel = compute_damper_velocity(stroke_mm, t_ms, smooth_savgol=False)

    assert vel.shape == stroke_mm.shape
    np.testing.assert_allclose(vel, 125.0, atol=1e-6)


def test_compute_damper_velocity_harmonic_sine_wave():
    """Harmonic stroke x(t) = A * sin(2*pi*f*t) at 100 Hz should match analytical derivative."""
    t_ms = np.arange(0, 2000, 10, dtype=np.float64)
    t_sec = t_ms / 1000.0
    amplitude_mm = 20.0
    freq_hz = 2.0
    omega = 2.0 * np.pi * freq_hz

    stroke_mm = 30.0 + amplitude_mm * np.sin(omega * t_sec)
    expected_vel = amplitude_mm * omega * np.cos(omega * t_sec)

    computed_vel = compute_damper_velocity(stroke_mm, t_ms, smooth_savgol=False)
    # Central finite difference error is O(dt^2); for f=2 Hz, dt=0.01s, relative error < 0.3%
    np.testing.assert_allclose(computed_vel[2:-2], expected_vel[2:-2], rtol=5e-3, atol=0.5)


def test_compute_damper_velocity_savgol_smoothing():
    """Savitzky-Golay derivative should accurately recover ramp velocity in presence of noise."""
    t_ms = np.arange(0, 1000, 10, dtype=np.float64)
    stroke_mm = 80.0 * (t_ms / 1000.0)
    vel_sg = compute_damper_velocity(stroke_mm, t_ms, smooth_savgol=True, savgol_window=7, savgol_polyorder=2)
    np.testing.assert_allclose(vel_sg[5:-5], 80.0, atol=1e-4)


def test_compute_damper_velocity_invalid_timestamps_raises():
    """Non-monotonic timestamps must raise ValueError."""
    t_ms = np.array([0.0, 10.0, 10.0, 20.0])
    stroke = np.array([1.0, 2.0, 3.0, 4.0])
    try:
        compute_damper_velocity(stroke, t_ms)
        assert False, "Expected ValueError for non-monotonic timestamps"
    except ValueError as exc:
        assert "monotonically increasing" in str(exc)


def test_classify_velocity_regimes_f1_thresholds():
    """Verify F1 4-regime damper classification around ±5 mm/s deadband and ±150 mm/s knee."""
    velocities = np.array([-320.0, -45.0, 0.0, 3.2, 65.0, 290.0])
    labels, fractions = classify_velocity_regimes(velocities, F1_TRACKSIDE_CONFIG)

    assert labels[0] == VelocityRegime.HIGH_SPEED_REBOUND.value
    assert labels[1] == VelocityRegime.LOW_SPEED_REBOUND.value
    assert labels[2] == VelocityRegime.STATIC.value
    assert labels[3] == VelocityRegime.STATIC.value
    assert labels[4] == VelocityRegime.LOW_SPEED_BUMP.value
    assert labels[5] == VelocityRegime.HIGH_SPEED_BUMP.value
    assert abs(sum(fractions.values()) - 100.0) < 1e-2


def test_classify_velocity_regimes_aerospace_thresholds():
    """Verify Aerospace ATA 32 classification around +350/-300 mm/s orifice blow-off knees."""
    velocities = np.array([-450.0, -120.0, 2.0, 180.0, 620.0])
    labels, fractions = classify_velocity_regimes(velocities, AEROSPACE_ATA32_CONFIG)

    assert labels[0] == VelocityRegime.HIGH_SPEED_REBOUND.value
    assert labels[1] == VelocityRegime.LOW_SPEED_REBOUND.value
    assert labels[2] == VelocityRegime.STATIC.value
    assert labels[3] == VelocityRegime.LOW_SPEED_BUMP.value
    assert labels[4] == VelocityRegime.HIGH_SPEED_BUMP.value
    assert abs(sum(fractions.values()) - 100.0) < 1e-2


def test_velocity_histogram_conservation():
    """Histogram bin density percentages must sum to 100%."""
    rng = np.random.default_rng(7)
    vel = rng.normal(0.0, 110.0, size=1000)
    edges, counts, pct = compute_velocity_histogram(vel, num_bins=31)

    assert len(edges) == 32
    assert len(counts) == 31
    assert int(np.sum(counts)) == 1000
    assert abs(float(np.sum(pct)) - 100.0) < 1e-6


def test_analyze_kinematics_detects_bottoming_and_kerb_strikes(f1_nominal_batch, f1_overload_batch):
    """Overload F1 batch must exhibit bottoming events and higher peak bump velocity than nominal."""
    kin_nom = analyze_kinematics(f1_nominal_batch)
    kin_ovr = analyze_kinematics(f1_overload_batch)

    assert kin_nom.bottoming_sample_count == 0
    assert kin_ovr.bottoming_sample_count > 0
    assert len(kin_ovr.bottoming_events) >= 1
    assert kin_ovr.peak_bump_velocity_mm_s > kin_nom.peak_bump_velocity_mm_s
    assert kin_ovr.max_stroke_mm >= F1_TRACKSIDE_CONFIG.bottoming_stroke_threshold_mm

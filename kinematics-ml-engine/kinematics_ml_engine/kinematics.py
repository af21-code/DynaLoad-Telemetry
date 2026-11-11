"""
Damper Kinematics & Numerical Differentiation Engine (100 Hz).

Computes instantaneous strut/damper velocity v(t) = dx/dt (mm/s) and acceleration a(t) = d^2x/dt^2 (mm/s^2)
from discrete 100 Hz stroke measurements, classifies 4-quadrant bump/rebound regimes, and evaluates
damper velocity histograms, stroke utilization, and mechanical bottoming events.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import numpy as np
from scipy.signal import savgol_filter

from .models import (
    KinematicsResult,
    ProfileConfig,
    TelemetryBatch,
    VelocityRegime,
    get_profile_config,
)


def compute_damper_velocity(
    strut_stroke_mm: np.ndarray,
    timestamp_ms: np.ndarray,
    smooth_savgol: bool = False,
    savgol_window: int = 5,
    savgol_polyorder: int = 2,
) -> np.ndarray:
    """
    Compute the numerical time derivative of strut_stroke_mm to obtain damper velocity (mm/s).

    Convention:
    - Positive velocity (v > 0): Bump / Compression (strut shortening under load)
    - Negative velocity (v < 0): Rebound / Extension (strut extending during recoil/unloading)

    Parameters
    ----------
    strut_stroke_mm : np.ndarray
        1D array of damper/oleo stroke values in millimeters (mm).
    timestamp_ms : np.ndarray
        1D array of strictly increasing timestamps in milliseconds (ms).
    smooth_savgol : bool
        If True, applies a Savitzky-Golay differentiating filter for potentiometer noise rejection.
    savgol_window : int
        Odd window length for Savitzky-Golay filter.
    savgol_polyorder : int
        Polynomial order for Savitzky-Golay filter.

    Returns
    -------
    np.ndarray
        1D array of damper velocities in mm/s.
    """
    stroke = np.asarray(strut_stroke_mm, dtype=np.float64)
    t_ms = np.asarray(timestamp_ms, dtype=np.float64)

    if stroke.ndim != 1 or t_ms.ndim != 1:
        raise ValueError("strut_stroke_mm and timestamp_ms must be 1D arrays.")
    if len(stroke) != len(t_ms):
        raise ValueError("strut_stroke_mm and timestamp_ms must have the same length.")
    if len(stroke) < 2:
        return np.zeros_like(stroke, dtype=np.float64)

    time_sec = t_ms / 1000.0
    dt_array = np.diff(time_sec)
    if np.any(dt_array <= 0.0):
        raise ValueError("timestamp_ms must be strictly monotonically increasing.")

    if smooth_savgol and len(stroke) >= savgol_window and savgol_window >= 3:
        median_dt = float(np.median(dt_array))
        win = savgol_window if savgol_window % 2 == 1 else savgol_window + 1
        return savgol_filter(
            stroke,
            window_length=win,
            polyorder=min(savgol_polyorder, win - 1),
            deriv=1,
            delta=median_dt,
        )

    edge_order = 2 if len(stroke) >= 3 else 1
    return np.gradient(stroke, time_sec, edge_order=edge_order)


def classify_velocity_regimes(
    velocity_mm_s: np.ndarray,
    config: ProfileConfig,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Classify each 100 Hz velocity sample into Low/High-Speed Bump, Low/High-Speed Rebound, or Static.

    Returns
    -------
    regime_labels : np.ndarray of str
    regime_fractions : Dict[str, float]
        Percentage (0.0 to 100.0) of samples in each regime.
    """
    v = np.asarray(velocity_mm_s, dtype=np.float64)
    n = len(v)
    if n == 0:
        empty_fracs = {r.value: 0.0 for r in VelocityRegime}
        return np.array([], dtype=object), empty_fracs

    deadband = config.deadband_velocity_mm_s
    bump_knee = config.bump_knee_velocity_mm_s
    reb_knee = config.rebound_knee_velocity_mm_s

    labels = np.full(n, VelocityRegime.STATIC.value, dtype=object)

    hs_bump_mask = v > bump_knee
    ls_bump_mask = (v > deadband) & (v <= bump_knee)
    ls_reb_mask = (v < -deadband) & (v >= -reb_knee)
    hs_reb_mask = v < -reb_knee

    labels[hs_bump_mask] = VelocityRegime.HIGH_SPEED_BUMP.value
    labels[ls_bump_mask] = VelocityRegime.LOW_SPEED_BUMP.value
    labels[ls_reb_mask] = VelocityRegime.LOW_SPEED_REBOUND.value
    labels[hs_reb_mask] = VelocityRegime.HIGH_SPEED_REBOUND.value

    fractions: Dict[str, float] = {}
    for regime in VelocityRegime:
        count = int(np.sum(labels == regime.value))
        fractions[regime.value] = round(100.0 * count / n, 3)

    return labels, fractions


def compute_velocity_histogram(
    velocity_mm_s: np.ndarray,
    num_bins: int = 41,
    max_abs_velocity_mm_s: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute symmetric damper velocity histogram bins, sample counts, and normalized probability density.
    """
    v = np.asarray(velocity_mm_s, dtype=np.float64)
    if len(v) == 0:
        edges = np.linspace(-100.0, 100.0, num_bins + 1)
        zeros = np.zeros(num_bins, dtype=np.float64)
        return edges, zeros, zeros

    bound = max_abs_velocity_mm_s
    if bound is None or bound <= 0.0:
        observed_max = float(np.max(np.abs(v)))
        bound = max(50.0, observed_max * 1.05)

    bin_edges = np.linspace(-bound, bound, num_bins + 1)
    counts, _ = np.histogram(v, bins=bin_edges)
    total = max(1, int(np.sum(counts)))
    percentages = (counts.astype(np.float64) / total) * 100.0

    return bin_edges, counts.astype(np.float64), percentages


def _extract_contiguous_events(
    mask: np.ndarray,
    timestamp_ms: np.ndarray,
    signal: np.ndarray,
    min_gap_samples: int = 3,
) -> List[Tuple[float, float, float]]:
    """
    Group boolean mask into contiguous events (start_ms, end_ms, peak_abs_signal),
    merging micro-gaps shorter than `min_gap_samples`.
    """
    if len(mask) == 0 or not np.any(mask):
        return []

    idx = np.flatnonzero(mask)
    events: List[Tuple[float, float, float]] = []
    start_idx = int(idx[0])
    prev_idx = int(idx[0])

    for current in idx[1:]:
        curr_idx = int(current)
        if curr_idx - prev_idx > min_gap_samples:
            seg = signal[start_idx : prev_idx + 1]
            peak_val = float(np.max(np.abs(seg)))
            events.append((float(timestamp_ms[start_idx]), float(timestamp_ms[prev_idx]), peak_val))
            start_idx = curr_idx
        prev_idx = curr_idx

    seg = signal[start_idx : prev_idx + 1]
    peak_val = float(np.max(np.abs(seg)))
    events.append((float(timestamp_ms[start_idx]), float(timestamp_ms[prev_idx]), peak_val))
    return events


def analyze_kinematics(
    batch: TelemetryBatch,
    config: Optional[ProfileConfig] = None,
    smooth_savgol: bool = False,
    num_bins: int = 41,
) -> KinematicsResult:
    """
    Execute complete 100 Hz damper kinematics pipeline on a TelemetryBatch.
    """
    cfg = config or get_profile_config(batch.profile)

    velocity = compute_damper_velocity(
        batch.strut_stroke_mm,
        batch.timestamp_ms,
        smooth_savgol=smooth_savgol,
    )

    acceleration = compute_damper_velocity(
        velocity,
        batch.timestamp_ms,
        smooth_savgol=False,
    )

    regime_labels, regime_fractions = classify_velocity_regimes(velocity, cfg)
    bin_edges, counts, percentages = compute_velocity_histogram(velocity, num_bins=num_bins)

    bump_samples = velocity[velocity > cfg.deadband_velocity_mm_s]
    rebound_samples = velocity[velocity < -cfg.deadband_velocity_mm_s]

    peak_bump = float(np.max(velocity)) if len(velocity) > 0 else 0.0
    peak_rebound = float(np.min(velocity)) if len(velocity) > 0 else 0.0
    rms_vel = float(np.sqrt(np.mean(np.square(velocity)))) if len(velocity) > 0 else 0.0

    mean_bump = float(np.mean(bump_samples)) if len(bump_samples) > 0 else 0.0
    mean_rebound = float(np.mean(np.abs(rebound_samples))) if len(rebound_samples) > 0 else 0.0
    symmetry_ratio = mean_bump / mean_rebound if mean_rebound > 1e-6 else 1.0

    max_stroke = float(np.max(batch.strut_stroke_mm)) if batch.num_samples > 0 else 0.0
    min_stroke = float(np.min(batch.strut_stroke_mm)) if batch.num_samples > 0 else 0.0
    stroke_utilization_pct = float(np.clip((max_stroke / cfg.max_stroke_mm) * 100.0, 0.0, 100.0))

    bottoming_mask = batch.strut_stroke_mm >= cfg.bottoming_stroke_threshold_mm
    bottoming_count = int(np.sum(bottoming_mask))
    bottoming_events = _extract_contiguous_events(
        bottoming_mask,
        batch.timestamp_ms,
        batch.strut_stroke_mm,
        min_gap_samples=5,
    )

    high_speed_strike_mask = velocity >= cfg.extreme_bump_velocity_mm_s
    strike_events = _extract_contiguous_events(
        high_speed_strike_mask,
        batch.timestamp_ms,
        velocity,
        min_gap_samples=8,
    )

    return KinematicsResult(
        damper_velocity_mm_s=velocity,
        damper_acceleration_mm_s2=acceleration,
        regime_labels=regime_labels,
        regime_fractions=regime_fractions,
        histogram_bin_edges_mm_s=bin_edges,
        histogram_counts=counts,
        histogram_densities=percentages,
        peak_bump_velocity_mm_s=round(peak_bump, 3),
        peak_rebound_velocity_mm_s=round(peak_rebound, 3),
        rms_velocity_mm_s=round(rms_vel, 3),
        bump_rebound_symmetry_ratio=round(symmetry_ratio, 4),
        max_stroke_mm=round(max_stroke, 3),
        min_stroke_mm=round(min_stroke, 3),
        stroke_utilization_pct=round(stroke_utilization_pct, 2),
        bottoming_sample_count=bottoming_count,
        bottoming_events=bottoming_events,
        high_speed_strike_count=len(strike_events),
    )

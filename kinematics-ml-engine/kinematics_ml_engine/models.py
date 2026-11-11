"""
Domain models, operational profile configurations, and result dataclasses for DynaLoad-Telemetry.

Unifies:
- Profile A: Aerospace MRO (ATA Chapter 32 Landing Gear Oleo-Pneumatic Strut)
- Profile B: Motorsport Formula 1 Trackside (Suspension Pushrod & Hydropneumatic Damper)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Tuple
import numpy as np


class OperationalProfile(str, Enum):
    """Supported high-frequency (100 Hz) structural dynamics profiles."""

    AEROSPACE_ATA32 = "aero"
    FORMULA1_TRACKSIDE = "f1"


class VelocityRegime(str, Enum):
    """Damper velocity regime classification along the stroke-rate curve."""

    HIGH_SPEED_BUMP = "HIGH_SPEED_BUMP"
    LOW_SPEED_BUMP = "LOW_SPEED_BUMP"
    STATIC = "STATIC"
    LOW_SPEED_REBOUND = "LOW_SPEED_REBOUND"
    HIGH_SPEED_REBOUND = "HIGH_SPEED_REBOUND"


class InspectionSeverity(str, Enum):
    """Engineering inspection urgency tier."""

    NOMINAL = "NOMINAL"
    ADVISORY = "ADVISORY"
    MANDATORY_INSPECTION = "MANDATORY_INSPECTION"
    CRITICAL_ACTION_REQUIRED = "CRITICAL_ACTION_REQUIRED"


@dataclass(frozen=True)
class ProfileConfig:
    """
    Physical, thermodynamic, and structural threshold parameters for an operational profile.
    """

    profile: OperationalProfile
    display_name: str
    reference_standard: str
    sample_rate_hz: float
    # Kinematics & stroke geometry
    max_stroke_mm: float
    bottoming_stroke_threshold_mm: float
    deadband_velocity_mm_s: float
    bump_knee_velocity_mm_s: float
    rebound_knee_velocity_mm_s: float
    extreme_bump_velocity_mm_s: float
    # Polytropic gas spring model: P(x) = P0 * (L0 / (L0 - x))^gamma
    nominal_static_pressure_bar: float
    effective_gas_length_mm: float
    polytropic_gamma: float
    min_healthy_pressure_ratio: float
    # Structural G-load thresholds
    advisory_vertical_g: float
    phase1_overload_vertical_g: float
    phase2_critical_vertical_g: float
    max_touchdown_or_corner_roll_deg: float


AEROSPACE_ATA32_CONFIG = ProfileConfig(
    profile=OperationalProfile.AEROSPACE_ATA32,
    display_name="Aerospace MRO Mode (ATA 32 Landing Gear)",
    reference_standard="EASA CS 25.473 / Airbus & Boeing AMM 05-51-11 (ATA Chapter 32)",
    sample_rate_hz=100.0,
    max_stroke_mm=450.0,
    bottoming_stroke_threshold_mm=430.0,
    deadband_velocity_mm_s=8.0,
    bump_knee_velocity_mm_s=350.0,
    rebound_knee_velocity_mm_s=300.0,
    extreme_bump_velocity_mm_s=950.0,
    nominal_static_pressure_bar=30.0,
    effective_gas_length_mm=480.0,
    polytropic_gamma=1.35,
    min_healthy_pressure_ratio=0.76,
    advisory_vertical_g=1.85,
    phase1_overload_vertical_g=2.10,
    phase2_critical_vertical_g=2.60,
    max_touchdown_or_corner_roll_deg=5.0,
)

F1_TRACKSIDE_CONFIG = ProfileConfig(
    profile=OperationalProfile.FORMULA1_TRACKSIDE,
    display_name="Formula 1 Trackside Mode (Suspension & Kerb Loads)",
    reference_standard="FIA F1 Technical Regulations Art. 3.5.9 (Plank) & Trackside Damper Directive",
    sample_rate_hz=100.0,
    max_stroke_mm=65.0,
    bottoming_stroke_threshold_mm=60.0,
    deadband_velocity_mm_s=5.0,
    bump_knee_velocity_mm_s=150.0,
    rebound_knee_velocity_mm_s=150.0,
    extreme_bump_velocity_mm_s=480.0,
    nominal_static_pressure_bar=16.0,
    effective_gas_length_mm=82.0,
    polytropic_gamma=1.38,
    min_healthy_pressure_ratio=0.74,
    advisory_vertical_g=3.60,
    phase1_overload_vertical_g=4.30,
    phase2_critical_vertical_g=4.90,
    max_touchdown_or_corner_roll_deg=2.4,
)


def get_profile_config(profile: OperationalProfile | str) -> ProfileConfig:
    """Resolve a ProfileConfig from an enum or string identifier."""
    if isinstance(profile, str):
        norm = profile.strip().lower()
        if norm in ("aero", "aerospace", "ata32", "aerospace_ata32"):
            return AEROSPACE_ATA32_CONFIG
        if norm in ("f1", "formula1", "motorsport", "formula1_trackside"):
            return F1_TRACKSIDE_CONFIG
        raise ValueError(f"Unknown operational profile: {profile!r}")
    return AEROSPACE_ATA32_CONFIG if profile == OperationalProfile.AEROSPACE_ATA32 else F1_TRACKSIDE_CONFIG


@dataclass
class TelemetryBatch:
    """
    Vectorized 100 Hz telemetry batch containing the 7 canonical channels.
    """

    timestamp_ms: np.ndarray
    vertical_g: np.ndarray
    roll_deg: np.ndarray
    pitch_deg: np.ndarray
    strut_stroke_mm: np.ndarray
    internal_pressure_bar: np.ndarray
    wheel_speed_kph: np.ndarray
    profile: OperationalProfile = OperationalProfile.AEROSPACE_ATA32
    scenario: str = "nominal"

    def __post_init__(self) -> None:
        self.timestamp_ms = np.asarray(self.timestamp_ms, dtype=np.float64)
        self.vertical_g = np.asarray(self.vertical_g, dtype=np.float64)
        self.roll_deg = np.asarray(self.roll_deg, dtype=np.float64)
        self.pitch_deg = np.asarray(self.pitch_deg, dtype=np.float64)
        self.strut_stroke_mm = np.asarray(self.strut_stroke_mm, dtype=np.float64)
        self.internal_pressure_bar = np.asarray(self.internal_pressure_bar, dtype=np.float64)
        self.wheel_speed_kph = np.asarray(self.wheel_speed_kph, dtype=np.float64)

        n = len(self.timestamp_ms)
        for name, arr in (
            ("vertical_g", self.vertical_g),
            ("roll_deg", self.roll_deg),
            ("pitch_deg", self.pitch_deg),
            ("strut_stroke_mm", self.strut_stroke_mm),
            ("internal_pressure_bar", self.internal_pressure_bar),
            ("wheel_speed_kph", self.wheel_speed_kph),
        ):
            if len(arr) != n:
                raise ValueError(f"Channel '{name}' length ({len(arr)}) does not match timestamp_ms ({n}).")

    @property
    def num_samples(self) -> int:
        return int(len(self.timestamp_ms))

    @property
    def time_seconds(self) -> np.ndarray:
        if len(self.timestamp_ms) == 0:
            return np.array([], dtype=np.float64)
        return (self.timestamp_ms - self.timestamp_ms[0]) / 1000.0


@dataclass
class KinematicsResult:
    """
    Output of the 100 Hz numerical damper kinematics and regime classification stage.
    """

    damper_velocity_mm_s: np.ndarray
    damper_acceleration_mm_s2: np.ndarray
    regime_labels: np.ndarray
    regime_fractions: Dict[str, float]
    histogram_bin_edges_mm_s: np.ndarray
    histogram_counts: np.ndarray
    histogram_densities: np.ndarray
    peak_bump_velocity_mm_s: float
    peak_rebound_velocity_mm_s: float
    rms_velocity_mm_s: float
    bump_rebound_symmetry_ratio: float
    max_stroke_mm: float
    min_stroke_mm: float
    stroke_utilization_pct: float
    bottoming_sample_count: int
    bottoming_events: List[Tuple[float, float, float]]
    high_speed_strike_count: int


@dataclass
class OverloadEvaluation:
    """
    Structural load envelope and regulatory/engineering trigger evaluation.
    """

    peak_vertical_g: float
    min_vertical_g: float
    peak_vertical_g_timestamp_ms: float
    peak_roll_deg: float
    peak_pitch_deg: float
    peak_internal_pressure_bar: float
    min_internal_pressure_bar: float
    is_advisory_triggered: bool
    is_phase1_overload_triggered: bool
    is_phase2_critical_triggered: bool
    is_bottoming_triggered: bool
    is_asymmetric_roll_triggered: bool
    is_kerb_strike_overload: bool
    triggered_codes: List[str] = field(default_factory=list)


@dataclass
class AnomalyResult:
    """
    Output of the Isolation Forest thermo-hydraulic manifold anomaly detector.
    """

    anomaly_scores: np.ndarray
    raw_decision_scores: np.ndarray
    is_anomaly: np.ndarray
    expected_polytropic_pressure_bar: np.ndarray
    pressure_ratio: np.ndarray
    anomalous_fraction: float
    peak_anomaly_score: float
    mean_pressure_ratio: float
    detected_fault_modes: List[str]
    is_degraded: bool


@dataclass
class InspectionActionItem:
    """Single actionable MRO or Race Engineering inspection directive."""

    code: str
    component: str
    directive: str
    reference: str
    priority: str


@dataclass
class InspectionReport:
    """
    Automated Structural Inspection Action Report combining kinematics, G-loads, and ML diagnostics.
    """

    profile: OperationalProfile
    profile_display_name: str
    reference_standard: str
    severity: InspectionSeverity
    headline: str
    summary_rationale: str
    telemetry_kpis: Dict[str, float]
    action_items: List[InspectionActionItem]
    markdown_report: str

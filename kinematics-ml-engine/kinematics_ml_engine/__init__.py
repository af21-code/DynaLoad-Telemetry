"""
DynaLoad-Telemetry: High-Frequency (100 Hz) Structural Load, Damper Kinematics,
and Isolation Forest Anomaly Detection Engine.
"""

from __future__ import annotations

from typing import Optional, Tuple

from .anomaly_detector import (
    HAS_SKLEARN,
    StrutAnomalyDetector,
    compute_expected_polytropic_pressure,
    extract_feature_matrix,
)
from .kinematics import (
    analyze_kinematics,
    classify_velocity_regimes,
    compute_damper_velocity,
    compute_velocity_histogram,
)
from .models import (
    AEROSPACE_ATA32_CONFIG,
    F1_TRACKSIDE_CONFIG,
    AnomalyResult,
    InspectionActionItem,
    InspectionReport,
    InspectionSeverity,
    KinematicsResult,
    OperationalProfile,
    OverloadEvaluation,
    ProfileConfig,
    TelemetryBatch,
    VelocityRegime,
    get_profile_config,
)
from .simulator import (
    generate_aerospace_telemetry,
    generate_f1_telemetry,
    generate_telemetry,
    load_telemetry_from_csv,
    load_telemetry_from_json,
)
from .structural_loads import (
    evaluate_structural_loads,
    generate_inspection_report,
)
from .udp_listener import capture_udp_telemetry
from .f1_game_udp import GameSession, analyze_session, build_session, load_session, record_udp


def run_telemetry_pipeline(
    batch: TelemetryBatch,
    detector: Optional[StrutAnomalyDetector] = None,
    smooth_savgol: bool = False,
) -> Tuple[KinematicsResult, OverloadEvaluation, AnomalyResult, InspectionReport]:
    """
    Execute the complete end-to-end 100 Hz telemetry analysis pipeline:
    1. Numerical damper kinematics & 4-regime bump/rebound histogram
    2. Peak G-load & structural overload evaluation
    3. Isolation Forest thermo-hydraulic anomaly detection
    4. Automated Structural Inspection Action Report synthesis
    """
    cfg = get_profile_config(batch.profile)
    kinematics = analyze_kinematics(batch, config=cfg, smooth_savgol=smooth_savgol)
    overload = evaluate_structural_loads(batch, kinematics, config=cfg)

    if detector is None:
        baseline_nom_1 = generate_telemetry(cfg.profile, scenario="nominal", duration_sec=10.0, seed=11)
        baseline_nom_2 = generate_telemetry(cfg.profile, scenario="nominal", duration_sec=10.0, seed=29)
        detector = StrutAnomalyDetector(cfg).fit([baseline_nom_1, baseline_nom_2])

    anomaly = detector.evaluate(batch, velocity_mm_s=kinematics.damper_velocity_mm_s)
    report = generate_inspection_report(batch, kinematics, overload, anomaly, config=cfg)
    return kinematics, overload, anomaly, report


__all__ = [
    "AEROSPACE_ATA32_CONFIG",
    "F1_TRACKSIDE_CONFIG",
    "GameSession",
    "AnomalyResult",
    "HAS_SKLEARN",
    "InspectionActionItem",
    "InspectionReport",
    "InspectionSeverity",
    "KinematicsResult",
    "OperationalProfile",
    "OverloadEvaluation",
    "ProfileConfig",
    "StrutAnomalyDetector",
    "TelemetryBatch",
    "VelocityRegime",
    "analyze_kinematics",
    "analyze_session",
    "build_session",
    "capture_udp_telemetry",
    "classify_velocity_regimes",
    "compute_damper_velocity",
    "compute_expected_polytropic_pressure",
    "compute_velocity_histogram",
    "evaluate_structural_loads",
    "extract_feature_matrix",
    "generate_aerospace_telemetry",
    "generate_f1_telemetry",
    "generate_inspection_report",
    "generate_telemetry",
    "get_profile_config",
    "load_session",
    "load_telemetry_from_csv",
    "load_telemetry_from_json",
    "record_udp",
    "run_telemetry_pipeline",
]

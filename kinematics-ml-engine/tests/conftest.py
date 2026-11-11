"""
Pytest configuration and shared telemetry fixtures for kinematics-ml-engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure package root is importable
PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

try:
    import pytest
except ImportError:  # pragma: no cover
    pytest = None  # type: ignore

from kinematics_ml_engine import (
    AEROSPACE_ATA32_CONFIG,
    F1_TRACKSIDE_CONFIG,
    StrutAnomalyDetector,
    generate_aerospace_telemetry,
    generate_f1_telemetry,
)


if pytest is not None:

    @pytest.fixture(scope="session")
    def aero_nominal_batch():
        return generate_aerospace_telemetry(scenario="nominal", duration_sec=8.0, seed=42)

    @pytest.fixture(scope="session")
    def aero_overload_batch():
        return generate_aerospace_telemetry(scenario="overload", duration_sec=8.0, seed=42)

    @pytest.fixture(scope="session")
    def aero_fault_batch():
        return generate_aerospace_telemetry(scenario="fault", duration_sec=8.0, seed=42)

    @pytest.fixture(scope="session")
    def f1_nominal_batch():
        return generate_f1_telemetry(scenario="nominal", duration_sec=8.0, seed=42)

    @pytest.fixture(scope="session")
    def f1_overload_batch():
        return generate_f1_telemetry(scenario="overload", duration_sec=8.0, seed=42)

    @pytest.fixture(scope="session")
    def f1_fault_batch():
        return generate_f1_telemetry(scenario="fault", duration_sec=8.0, seed=42)

    @pytest.fixture(scope="session")
    def aero_detector(aero_nominal_batch):
        b2 = generate_aerospace_telemetry(scenario="nominal", duration_sec=8.0, seed=99)
        return StrutAnomalyDetector(AEROSPACE_ATA32_CONFIG).fit([aero_nominal_batch, b2])

    @pytest.fixture(scope="session")
    def f1_detector(f1_nominal_batch):
        b2 = generate_f1_telemetry(scenario="nominal", duration_sec=8.0, seed=99)
        return StrutAnomalyDetector(F1_TRACKSIDE_CONFIG).fit([f1_nominal_batch, b2])

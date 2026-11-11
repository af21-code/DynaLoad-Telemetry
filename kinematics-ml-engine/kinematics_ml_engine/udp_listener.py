"""
UDP 100 Hz Telemetry Receiver.

Listens for JSON telemetry datagrams broadcasted by `/telemetry-streamer-dotnet`
and assembles them into a vectorized `TelemetryBatch`.
"""

from __future__ import annotations

import json
import socket
import time
from typing import List, Optional
import numpy as np

from .models import OperationalProfile, TelemetryBatch, get_profile_config


def capture_udp_telemetry(
    host: str = "127.0.0.1",
    port: int = 5005,
    max_packets: int = 500,
    timeout_sec: float = 3.0,
    default_profile: OperationalProfile = OperationalProfile.AEROSPACE_ATA32,
) -> Optional[TelemetryBatch]:
    """
    Capture up to `max_packets` 100 Hz UDP telemetry frames within `timeout_sec`.
    Returns None if no valid packets are received before timeout.
    """
    records: List[dict] = []
    deadline = time.monotonic() + timeout_sec

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(0.25)
        sock.bind((host, port))

        while len(records) < max_packets and time.monotonic() < deadline:
            try:
                data, _ = sock.recvfrom(4096)
                payload = json.loads(data.decode("utf-8"))
                if "timestamp_ms" in payload and "strut_stroke_mm" in payload:
                    records.append(payload)
            except socket.timeout:
                continue
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue

    if len(records) < 3:
        return None

    # Sort by timestamp_ms and deduplicate
    records.sort(key=lambda r: float(r["timestamp_ms"]))
    deduped: List[dict] = [records[0]]
    for rec in records[1:]:
        if float(rec["timestamp_ms"]) > float(deduped[-1]["timestamp_ms"]):
            deduped.append(rec)

    if len(deduped) < 3:
        return None

    profile_str = deduped[0].get("profile", default_profile.value)
    scenario_str = deduped[0].get("scenario", "live_udp")
    cfg = get_profile_config(profile_str)

    return TelemetryBatch(
        timestamp_ms=np.array([float(r["timestamp_ms"]) for r in deduped], dtype=np.float64),
        vertical_g=np.array([float(r["vertical_g"]) for r in deduped], dtype=np.float64),
        roll_deg=np.array([float(r["roll_deg"]) for r in deduped], dtype=np.float64),
        pitch_deg=np.array([float(r["pitch_deg"]) for r in deduped], dtype=np.float64),
        strut_stroke_mm=np.array([float(r["strut_stroke_mm"]) for r in deduped], dtype=np.float64),
        internal_pressure_bar=np.array([float(r["internal_pressure_bar"]) for r in deduped], dtype=np.float64),
        wheel_speed_kph=np.array([float(r["wheel_speed_kph"]) for r in deduped], dtype=np.float64),
        profile=cfg.profile,
        scenario=scenario_str,
    )

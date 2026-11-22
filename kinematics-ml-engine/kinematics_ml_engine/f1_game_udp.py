"""
Real suspension telemetry from the EA SPORTS F1 game UDP feed.

The F1 game (F1 23, F1 24, F1 25) can stream telemetry over UDP. Two packets matter here:

- Motion (id 0): g-forces, yaw, pitch and roll for every car.
- Motion Ex (id 13): player car only. Suspension position, velocity and acceleration for each
  wheel, wheel speeds and forces, aero heights and more.

This module records those packets to a capture file, reads them back, rebuilds a time series per
wheel and runs the DynaLoad kinematics on it. Because the game also sends its own suspension
velocity, the velocity that DynaLoad derives from position can be checked against an independent
channel on real data.

The UDP specification does not state units for suspension position, velocity or acceleration.
They are kept as sent ("game units") and treated as millimetres only where noted.

Wheel order in every array follows the game: RL, RR, FL, FR.

Usage:
    python -m kinematics_ml_engine.f1_game_udp record --out session.dlcap --duration 300
    python -m kinematics_ml_engine.f1_game_udp analyze session.dlcap --json summary.json
"""

from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from .kinematics import classify_velocity_regimes, compute_damper_velocity
from .models import F1_TRACKSIDE_CONFIG, OperationalProfile, ProfileConfig, TelemetryBatch

# --------------------------------------------------------------------------------------------
# Packet layout (little-endian, packed)
# --------------------------------------------------------------------------------------------

WHEELS: Tuple[str, ...] = ("RL", "RR", "FL", "FR")

PACKET_ID_MOTION = 0
PACKET_ID_MOTION_EX = 13

HEADER_STRUCT = struct.Struct("<HBBBBBQfIIBB")
HEADER_SIZE = HEADER_STRUCT.size  # 29 bytes

CAR_MOTION_STRUCT = struct.Struct("<6f6h6f")
CAR_MOTION_SIZE = CAR_MOTION_STRUCT.size  # 60 bytes
NUM_CARS = 22

# Number of float32 values after the header in Motion Ex, per UDP format.
#   2023: 8 wheel arrays + COG height + 3 local vel + 3 ang vel + 3 ang acc + front wheels angle
#   2024: + wheel vertical force[4] + front/rear aero height + front/rear roll angle + chassis yaw
#   2025: + chassis pitch + wheel camber[4] + wheel camber gain[4]
MOTION_EX_FLOATS: Dict[int, int] = {2023: 47, 2024: 52, 2025: 61}
MOTION_EX_SIZES: Dict[int, int] = {fmt: HEADER_SIZE + 4 * n for fmt, n in MOTION_EX_FLOATS.items()}

CAPTURE_MAGIC = b"DLCAP1\n"
CAPTURE_RECORD = struct.Struct("<Id")  # payload length, receive time (s since epoch)


@dataclass(frozen=True)
class PacketHeader:
    packet_format: int
    game_year: int
    game_major_version: int
    game_minor_version: int
    packet_version: int
    packet_id: int
    session_uid: int
    session_time: float
    frame_identifier: int
    overall_frame_identifier: int
    player_car_index: int
    secondary_player_car_index: int


@dataclass(frozen=True)
class CarMotion:
    """Motion data for one car (Motion packet, id 0)."""

    world_position: Tuple[float, float, float]
    world_velocity: Tuple[float, float, float]
    g_force_lateral: float
    g_force_longitudinal: float
    g_force_vertical: float
    yaw_rad: float
    pitch_rad: float
    roll_rad: float


@dataclass(frozen=True)
class MotionExFrame:
    """Player-car Motion Ex packet (id 13). Wheel arrays are ordered RL, RR, FL, FR."""

    header: PacketHeader
    suspension_position: np.ndarray
    suspension_velocity: np.ndarray
    suspension_acceleration: np.ndarray
    wheel_speed: np.ndarray
    wheel_slip_ratio: np.ndarray
    wheel_slip_angle: np.ndarray
    wheel_lat_force: np.ndarray
    wheel_long_force: np.ndarray
    height_of_cog_above_ground: float
    local_velocity: Tuple[float, float, float]
    angular_velocity: Tuple[float, float, float]
    angular_acceleration: Tuple[float, float, float]
    front_wheels_angle: float
    wheel_vert_force: Optional[np.ndarray] = None
    front_aero_height: Optional[float] = None
    rear_aero_height: Optional[float] = None
    front_roll_angle: Optional[float] = None
    rear_roll_angle: Optional[float] = None
    chassis_yaw: Optional[float] = None
    chassis_pitch: Optional[float] = None
    wheel_camber: Optional[np.ndarray] = None
    wheel_camber_gain: Optional[np.ndarray] = None


# --------------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------------


def parse_header(data: bytes) -> PacketHeader:
    """Parse the 29-byte header shared by every packet."""
    if len(data) < HEADER_SIZE:
        raise ValueError(f"Packet too short for a header: {len(data)} bytes")
    return PacketHeader(*HEADER_STRUCT.unpack_from(data, 0))


def parse_motion_ex(data: bytes) -> MotionExFrame:
    """Parse a Motion Ex packet (id 13) in UDP format 2023, 2024 or 2025."""
    header = parse_header(data)
    if header.packet_id != PACKET_ID_MOTION_EX:
        raise ValueError(f"Not a Motion Ex packet (id {header.packet_id})")

    fmt = header.packet_format
    if fmt not in MOTION_EX_FLOATS:
        # Unknown year: pick the layout by size, newest first.
        fmt = next((f for f, size in sorted(MOTION_EX_SIZES.items(), reverse=True) if len(data) >= size), 0)
        if fmt == 0:
            raise ValueError(f"Unsupported Motion Ex packet: format {header.packet_format}, {len(data)} bytes")

    n_floats = MOTION_EX_FLOATS[fmt]
    if len(data) < MOTION_EX_SIZES[fmt]:
        raise ValueError(f"Motion Ex packet too short for format {fmt}: {len(data)} bytes")

    v = np.frombuffer(data, dtype="<f4", count=n_floats, offset=HEADER_SIZE).astype(np.float64)

    def arr(i: int) -> np.ndarray:
        return v[i : i + 4].copy()

    kwargs = dict(
        header=header,
        suspension_position=arr(0),
        suspension_velocity=arr(4),
        suspension_acceleration=arr(8),
        wheel_speed=arr(12),
        wheel_slip_ratio=arr(16),
        wheel_slip_angle=arr(20),
        wheel_lat_force=arr(24),
        wheel_long_force=arr(28),
        height_of_cog_above_ground=float(v[32]),
        local_velocity=(float(v[33]), float(v[34]), float(v[35])),
        angular_velocity=(float(v[36]), float(v[37]), float(v[38])),
        angular_acceleration=(float(v[39]), float(v[40]), float(v[41])),
        front_wheels_angle=float(v[42]),
    )
    if n_floats >= MOTION_EX_FLOATS[2024]:
        kwargs.update(
            wheel_vert_force=arr(43),
            front_aero_height=float(v[47]),
            rear_aero_height=float(v[48]),
            front_roll_angle=float(v[49]),
            rear_roll_angle=float(v[50]),
            chassis_yaw=float(v[51]),
        )
    if n_floats >= MOTION_EX_FLOATS[2025]:
        kwargs.update(
            chassis_pitch=float(v[52]),
            wheel_camber=arr(53),
            wheel_camber_gain=arr(57),
        )
    return MotionExFrame(**kwargs)


def parse_car_motion(data: bytes, car_index: Optional[int] = None) -> Tuple[PacketHeader, CarMotion]:
    """Parse one car from a Motion packet (id 0). Defaults to the player's car."""
    header = parse_header(data)
    if header.packet_id != PACKET_ID_MOTION:
        raise ValueError(f"Not a Motion packet (id {header.packet_id})")
    idx = header.player_car_index if car_index is None else car_index
    if not 0 <= idx < NUM_CARS:
        raise ValueError(f"Car index out of range: {idx}")
    offset = HEADER_SIZE + idx * CAR_MOTION_SIZE
    if len(data) < offset + CAR_MOTION_SIZE:
        raise ValueError(f"Motion packet too short: {len(data)} bytes")
    f = CAR_MOTION_STRUCT.unpack_from(data, offset)
    motion = CarMotion(
        world_position=(f[0], f[1], f[2]),
        world_velocity=(f[3], f[4], f[5]),
        g_force_lateral=f[12],
        g_force_longitudinal=f[13],
        g_force_vertical=f[14],
        yaw_rad=f[15],
        pitch_rad=f[16],
        roll_rad=f[17],
    )
    return header, motion


# --------------------------------------------------------------------------------------------
# Capture files
# --------------------------------------------------------------------------------------------


def write_capture(path: Path | str, packets: Iterable[Tuple[float, bytes]]) -> int:
    """Write (receive_time, payload) pairs to a capture file. Returns the number of packets."""
    count = 0
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(CAPTURE_MAGIC)
        for recv_time, payload in packets:
            fh.write(CAPTURE_RECORD.pack(len(payload), recv_time))
            fh.write(payload)
            count += 1
    return count


def read_capture(path: Path | str) -> Iterator[Tuple[float, bytes]]:
    """Yield (receive_time, payload) pairs from a capture file."""
    with open(path, "rb") as fh:
        if fh.read(len(CAPTURE_MAGIC)) != CAPTURE_MAGIC:
            raise ValueError(f"{path} is not a DynaLoad capture file")
        while True:
            head = fh.read(CAPTURE_RECORD.size)
            if not head:
                return
            if len(head) < CAPTURE_RECORD.size:
                raise ValueError(f"{path} is truncated")
            length, recv_time = CAPTURE_RECORD.unpack(head)
            payload = fh.read(length)
            if len(payload) < length:
                raise ValueError(f"{path} is truncated")
            yield recv_time, payload


def record_udp(
    out_path: Path | str,
    duration_sec: float,
    host: str = "0.0.0.0",
    port: int = 20777,
    packet_ids: Optional[Sequence[int]] = (PACKET_ID_MOTION, PACKET_ID_MOTION_EX),
    max_packets: Optional[int] = None,
    on_packet=None,
) -> Counter:
    """
    Listen on UDP and write packets to a capture file for `duration_sec` seconds.

    Only Motion and Motion Ex are kept by default; pass packet_ids=None to keep everything.
    Returns a Counter of (packet_format, packet_id) for every packet received.
    """
    seen: Counter = Counter()
    keep = None if packet_ids is None else set(packet_ids)
    deadline = time.monotonic() + duration_sec
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock, open(out_path, "wb") as fh:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.settimeout(0.25)
        sock.bind((host, port))
        fh.write(CAPTURE_MAGIC)
        kept = 0
        while time.monotonic() < deadline:
            try:
                payload, _ = sock.recvfrom(2048)
            except socket.timeout:
                continue
            try:
                header = parse_header(payload)
            except ValueError:
                continue
            seen[(header.packet_format, header.packet_id)] += 1
            if keep is not None and header.packet_id not in keep:
                continue
            fh.write(CAPTURE_RECORD.pack(len(payload), time.time()))
            fh.write(payload)
            kept += 1
            if on_packet is not None:
                on_packet(header)
            if max_packets is not None and kept >= max_packets:
                break
    return seen


# --------------------------------------------------------------------------------------------
# Session assembly
# --------------------------------------------------------------------------------------------


@dataclass
class GameSession:
    """
    Player-car time series rebuilt from Motion Ex (and, when present, Motion) packets.

    Rows are frames in arrival order after removing duplicates; columns follow WHEELS.
    `segment_id` changes wherever session time jumps backwards (flashback) or stalls (pause).
    """

    session_uid: int
    packet_format: int
    session_time_s: np.ndarray
    overall_frame: np.ndarray
    segment_id: np.ndarray
    suspension_position: np.ndarray
    suspension_velocity: np.ndarray
    suspension_acceleration: np.ndarray
    wheel_speed: np.ndarray
    wheel_vert_force: Optional[np.ndarray]
    front_aero_height: Optional[np.ndarray]
    rear_aero_height: Optional[np.ndarray]
    vertical_g: np.ndarray
    pitch_deg: np.ndarray
    roll_deg: np.ndarray
    dropped_frames: int = 0
    other_sessions: int = 0

    @property
    def num_samples(self) -> int:
        return int(len(self.session_time_s))

    @property
    def num_segments(self) -> int:
        return int(len(np.unique(self.segment_id))) if self.num_samples else 0

    @property
    def sample_rate_hz(self) -> float:
        rates = []
        for seg in np.unique(self.segment_id):
            t = self.session_time_s[self.segment_id == seg]
            if len(t) > 1:
                rates.append(1.0 / float(np.median(np.diff(t))))
        return float(np.median(rates)) if rates else 0.0

    @property
    def duration_s(self) -> float:
        total = 0.0
        for seg in np.unique(self.segment_id):
            t = self.session_time_s[self.segment_id == seg]
            if len(t) > 1:
                total += float(t[-1] - t[0])
        return total

    def to_corner_batch(self, wheel: str, segment: Optional[int] = None) -> TelemetryBatch:
        """
        Build a DynaLoad TelemetryBatch for one wheel. Suspension position is used as stroke
        (game units, treated as mm). The game sends no damper pressure, so that channel is NaN:
        use the kinematics stage only, not the pressure-based anomaly detector.
        """
        i = WHEELS.index(wheel)
        mask = np.ones(self.num_samples, dtype=bool) if segment is None else self.segment_id == segment
        t = self.session_time_s[mask]
        return TelemetryBatch(
            timestamp_ms=(t - t[0]) * 1000.0 if len(t) else t,
            vertical_g=self.vertical_g[mask],
            roll_deg=self.roll_deg[mask],
            pitch_deg=self.pitch_deg[mask],
            strut_stroke_mm=self.suspension_position[mask, i],
            internal_pressure_bar=np.full(int(mask.sum()), np.nan),
            wheel_speed_kph=self.wheel_speed[mask, i] * 3.6,
            profile=OperationalProfile.FORMULA1_TRACKSIDE,
            scenario=f"f1_game_{wheel}",
        )


def build_session(
    packets: Iterable[Tuple[float, bytes]],
    max_gap_s: float = 0.5,
) -> GameSession:
    """
    Rebuild the player-car time series from raw packets.

    - Keeps the session with the most Motion Ex frames when a capture spans several sessions.
    - Drops repeated frames (same overall frame identifier or same session time).
    - Starts a new segment when session time goes backwards (flashback) or jumps by more than
      `max_gap_s` (pause, menu, garage), so derivatives are never taken across a discontinuity.
    - Pairs each Motion Ex frame with the Motion packet of the same overall frame for g-force,
      pitch and roll (NaN when no Motion packet was received for that frame).
    """
    ex_frames: List[MotionExFrame] = []
    motion_by_frame: Dict[Tuple[int, int], CarMotion] = {}

    for _, payload in packets:
        try:
            header = parse_header(payload)
        except ValueError:
            continue
        try:
            if header.packet_id == PACKET_ID_MOTION_EX:
                ex_frames.append(parse_motion_ex(payload))
            elif header.packet_id == PACKET_ID_MOTION:
                _, motion = parse_car_motion(payload)
                motion_by_frame[(header.session_uid, header.overall_frame_identifier)] = motion
        except ValueError:
            continue

    if not ex_frames:
        raise ValueError("No Motion Ex packets found. Check that UDP telemetry is on and that you are driving.")

    uid_counts = Counter(f.header.session_uid for f in ex_frames)
    session_uid, _ = uid_counts.most_common(1)[0]
    frames = [f for f in ex_frames if f.header.session_uid == session_uid]

    kept: List[MotionExFrame] = []
    seen_frames = set()
    dropped = 0
    for f in frames:
        key = f.header.overall_frame_identifier
        if key in seen_frames or (kept and f.header.session_time == kept[-1].header.session_time):
            dropped += 1
            continue
        seen_frames.add(key)
        kept.append(f)

    n = len(kept)
    t = np.array([f.header.session_time for f in kept], dtype=np.float64)
    segment = np.zeros(n, dtype=np.int64)
    for k in range(1, n):
        step = t[k] - t[k - 1]
        segment[k] = segment[k - 1] + (1 if (step <= 0.0 or step > max_gap_s) else 0)

    def stack(attr: str) -> np.ndarray:
        return np.vstack([getattr(f, attr) for f in kept])

    has_2024 = kept[0].wheel_vert_force is not None
    vertical_g = np.full(n, np.nan)
    pitch = np.full(n, np.nan)
    roll = np.full(n, np.nan)
    for k, f in enumerate(kept):
        m = motion_by_frame.get((session_uid, f.header.overall_frame_identifier))
        if m is not None:
            vertical_g[k] = m.g_force_vertical
            pitch[k] = np.degrees(m.pitch_rad)
            roll[k] = np.degrees(m.roll_rad)

    return GameSession(
        session_uid=int(session_uid),
        packet_format=int(kept[0].header.packet_format),
        session_time_s=t,
        overall_frame=np.array([f.header.overall_frame_identifier for f in kept], dtype=np.int64),
        segment_id=segment,
        suspension_position=stack("suspension_position"),
        suspension_velocity=stack("suspension_velocity"),
        suspension_acceleration=stack("suspension_acceleration"),
        wheel_speed=stack("wheel_speed"),
        wheel_vert_force=stack("wheel_vert_force") if has_2024 else None,
        front_aero_height=np.array([f.front_aero_height for f in kept]) if has_2024 else None,
        rear_aero_height=np.array([f.rear_aero_height for f in kept]) if has_2024 else None,
        vertical_g=vertical_g,
        pitch_deg=pitch,
        roll_deg=roll,
        dropped_frames=dropped,
        other_sessions=len(uid_counts) - 1,
    )


def load_session(path: Path | str, max_gap_s: float = 0.5) -> GameSession:
    """Read a capture file and rebuild the session."""
    return build_session(read_capture(path), max_gap_s=max_gap_s)


# --------------------------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------------------------


@dataclass
class VelocityCheck:
    """How well DynaLoad's derived velocity matches the game's own suspension velocity."""

    samples: int
    pearson_r: float
    slope: float
    rmse: float
    best_lag_samples: int


def derived_velocity(session: GameSession, wheel_index: int, smooth_savgol: bool = False) -> np.ndarray:
    """Differentiate suspension position within each continuous segment (NaN where too short)."""
    out = np.full(session.num_samples, np.nan)
    for seg in np.unique(session.segment_id):
        mask = session.segment_id == seg
        if int(mask.sum()) < 3:
            continue
        out[mask] = compute_damper_velocity(
            session.suspension_position[mask, wheel_index],
            session.session_time_s[mask] * 1000.0,
            smooth_savgol=smooth_savgol,
        )
    return out


def compare_velocity(derived: np.ndarray, reference: np.ndarray, max_lag: int = 5) -> VelocityCheck:
    """Correlation, least-squares slope (reference ≈ slope * derived), RMSE and best lag."""
    ok = np.isfinite(derived) & np.isfinite(reference)
    d = derived[ok]
    r = reference[ok]
    if len(d) < 3 or np.std(d) == 0.0 or np.std(r) == 0.0:
        return VelocityCheck(samples=int(len(d)), pearson_r=float("nan"), slope=float("nan"),
                             rmse=float("nan"), best_lag_samples=0)
    pearson = float(np.corrcoef(d, r)[0, 1])
    slope = float(np.dot(d, r) / np.dot(d, d))
    rmse = float(np.sqrt(np.mean((r - d) ** 2)))

    best_lag, best_corr = 0, -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag >= 0:
            a, b = d[: len(d) - lag], r[lag:]
        else:
            a, b = d[-lag:], r[: len(r) + lag]
        if len(a) > 2 and np.std(a) > 0 and np.std(b) > 0:
            c = float(np.corrcoef(a, b)[0, 1])
            if c > best_corr:
                best_lag, best_corr = lag, c
    return VelocityCheck(samples=int(len(d)), pearson_r=pearson, slope=slope, rmse=rmse,
                         best_lag_samples=best_lag)


def analyze_session(
    session: GameSession,
    config: ProfileConfig = F1_TRACKSIDE_CONFIG,
    smooth_savgol: bool = False,
) -> Dict[str, object]:
    """
    Per-wheel summary: position range, derived-velocity regimes and the check against the
    game's velocity channel. Velocity thresholds come from the DynaLoad F1 profile and are
    applied to game units, so regime shares are indicative.
    """
    wheels: Dict[str, object] = {}
    for i, name in enumerate(WHEELS):
        vel = derived_velocity(session, i, smooth_savgol=smooth_savgol)
        finite = vel[np.isfinite(vel)]
        _, fractions = classify_velocity_regimes(finite, config)
        check = compare_velocity(vel, session.suspension_velocity[:, i])
        pos = session.suspension_position[:, i]
        wheels[name] = {
            "position_min": round(float(np.min(pos)), 3),
            "position_max": round(float(np.max(pos)), 3),
            "position_range": round(float(np.ptp(pos)), 3),
            "derived_velocity_peak_positive": round(float(np.max(finite)), 3) if len(finite) else None,
            "derived_velocity_peak_negative": round(float(np.min(finite)), 3) if len(finite) else None,
            "derived_velocity_rms": round(float(np.sqrt(np.mean(finite ** 2))), 3) if len(finite) else None,
            "regime_share_pct": fractions,
            "check_vs_game_velocity": {
                "samples": check.samples,
                "pearson_r": round(check.pearson_r, 4),
                "slope": round(check.slope, 4),
                "rmse": round(check.rmse, 4),
                "best_lag_samples": check.best_lag_samples,
            },
        }

    summary: Dict[str, object] = {
        "session_uid": str(session.session_uid),
        "udp_format": session.packet_format,
        "samples": session.num_samples,
        "segments": session.num_segments,
        "duration_s": round(session.duration_s, 2),
        "sample_rate_hz": round(session.sample_rate_hz, 2),
        "dropped_duplicate_frames": session.dropped_frames,
        "motion_packets_matched_pct": round(100.0 * float(np.mean(np.isfinite(session.vertical_g))), 1),
        "wheels": wheels,
    }
    if np.any(np.isfinite(session.vertical_g)):
        summary["vertical_g_peak"] = round(float(np.nanmax(session.vertical_g)), 3)
    if session.front_aero_height is not None:
        summary["front_aero_height_min"] = round(float(np.min(session.front_aero_height)), 4)
        summary["rear_aero_height_min"] = round(float(np.min(session.rear_aero_height)), 4)
    return summary


# --------------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------------


def _print_summary(summary: Dict[str, object]) -> None:
    print(f"UDP format {summary['udp_format']} | {summary['samples']} frames | "
          f"{summary['duration_s']} s | {summary['sample_rate_hz']} Hz | {summary['segments']} segment(s)")
    print(f"Motion packets matched: {summary['motion_packets_matched_pct']}% | "
          f"duplicates dropped: {summary['dropped_duplicate_frames']}")
    print()
    print(f"{'wheel':<6}{'range':>10}{'v peak +':>11}{'v peak -':>11}{'v rms':>9}{'r vs game':>11}{'slope':>8}{'lag':>5}")
    for name, w in summary["wheels"].items():  # type: ignore[union-attr]
        c = w["check_vs_game_velocity"]
        print(f"{name:<6}{w['position_range']:>10.3f}{w['derived_velocity_peak_positive']:>11.1f}"
              f"{w['derived_velocity_peak_negative']:>11.1f}{w['derived_velocity_rms']:>9.1f}"
              f"{c['pearson_r']:>11.4f}{c['slope']:>8.3f}{c['best_lag_samples']:>5d}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Record and analyse F1 game suspension telemetry.")
    sub = parser.add_subparsers(dest="command", required=True)

    rec = sub.add_parser("record", help="record Motion and Motion Ex packets to a capture file")
    rec.add_argument("--out", required=True, help="capture file to write (.dlcap)")
    rec.add_argument("--duration", type=float, default=300.0, help="seconds to record (default 300)")
    rec.add_argument("--host", default="0.0.0.0", help="address to listen on (default all interfaces)")
    rec.add_argument("--port", type=int, default=20777, help="UDP port set in the game (default 20777)")
    rec.add_argument("--all-packets", action="store_true", help="keep every packet type, not only motion")

    ana = sub.add_parser("analyze", help="analyse a capture file")
    ana.add_argument("capture", help="capture file (.dlcap)")
    ana.add_argument("--json", help="also write the summary to this JSON file")
    ana.add_argument("--savgol", action="store_true", help="smooth with a Savitzky-Golay derivative")

    args = parser.parse_args(argv)

    if args.command == "record":
        print(f"Listening on {args.host}:{args.port} for {args.duration:.0f} s. Drive a few laps...")
        ex_count = [0]

        def progress(header: PacketHeader) -> None:
            if header.packet_id == PACKET_ID_MOTION_EX:
                ex_count[0] += 1
                if ex_count[0] % 300 == 0:
                    print(f"  {ex_count[0]} Motion Ex frames, session time {header.session_time:.1f} s")

        seen = record_udp(args.out, args.duration, host=args.host, port=args.port,
                          packet_ids=None if args.all_packets else (PACKET_ID_MOTION, PACKET_ID_MOTION_EX),
                          on_packet=progress)
        if not seen:
            print("No packets received. Check UDP Telemetry = On, the IP address and the port in the game.")
            return 1
        print("Packets received (format, id): " + ", ".join(f"{k}: {v}" for k, v in sorted(seen.items())))
        if not any(pid == PACKET_ID_MOTION_EX for _, pid in seen):
            print("No Motion Ex packets: set UDP Format to 2023 or later and make sure you are driving.")
            return 1
        print(f"Saved to {args.out}")
        return 0

    summary = analyze_session(load_session(args.capture), smooth_savgol=args.savgol)
    _print_summary(summary)
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=2))
        print(f"\nSummary written to {args.json}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())

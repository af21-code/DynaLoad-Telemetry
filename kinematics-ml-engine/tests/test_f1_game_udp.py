"""
Tests for the F1 game UDP reader, using packets built byte by byte from the published
UDP specification (F1 23, F1 24 and F1 25 Motion Ex layouts).
"""

from __future__ import annotations

import socket
import struct
import threading
import time

import numpy as np
import pytest

from kinematics_ml_engine.f1_game_udp import (
    CAR_MOTION_SIZE,
    HEADER_SIZE,
    MOTION_EX_FLOATS,
    MOTION_EX_SIZES,
    NUM_CARS,
    PACKET_ID_MOTION,
    PACKET_ID_MOTION_EX,
    WHEELS,
    analyze_session,
    build_session,
    compare_velocity,
    derived_velocity,
    load_session,
    main,
    parse_car_motion,
    parse_header,
    parse_motion_ex,
    read_capture,
    record_udp,
    write_capture,
)

SESSION_UID = 0x1234_5678_9ABC_DEF0


def make_header(packet_format: int, packet_id: int, session_time: float, frame: int,
                overall: int | None = None, player_index: int = 0, uid: int = SESSION_UID) -> bytes:
    return struct.pack("<HBBBBBQfIIBB", packet_format, packet_format % 100, 1, 2, 1, packet_id,
                       uid, session_time, frame, frame if overall is None else overall, player_index, 255)


def make_motion_ex(packet_format: int, session_time: float, frame: int, position=None, velocity=None,
                   overall: int | None = None, uid: int = SESSION_UID) -> bytes:
    n = MOTION_EX_FLOATS[packet_format]
    values = np.arange(n, dtype=np.float32) * 0.5  # distinctive filler so offsets can be checked
    if position is not None:
        values[0:4] = position
    if velocity is not None:
        values[4:8] = velocity
    return make_header(packet_format, PACKET_ID_MOTION_EX, session_time, frame, overall, uid=uid) + values.astype("<f4").tobytes()


def make_motion(packet_format: int, session_time: float, frame: int, player_index: int = 0,
                g_vertical: float = 1.0, pitch: float = 0.01, roll: float = -0.02,
                overall: int | None = None) -> bytes:
    cars = b""
    for i in range(NUM_CARS):
        gv = g_vertical if i == player_index else 9.0
        cars += struct.pack("<6f6h6f", 1, 2, 3, 4, 5, 6, 0, 0, 32767, 32767, 0, 0, 0.1, 0.2, gv, 0.3, pitch, roll)
    return make_header(packet_format, PACKET_ID_MOTION, session_time, frame, overall, player_index) + cars


# ---------------------------------------------------------------------------------------------


def test_header_round_trip():
    data = make_header(2025, PACKET_ID_MOTION_EX, 12.5, 700, overall=900, player_index=3)
    assert len(data) == HEADER_SIZE == 29
    h = parse_header(data)
    assert (h.packet_format, h.packet_id, h.session_uid) == (2025, 13, SESSION_UID)
    assert h.session_time == pytest.approx(12.5)
    assert (h.frame_identifier, h.overall_frame_identifier, h.player_car_index) == (700, 900, 3)


def test_header_too_short():
    with pytest.raises(ValueError):
        parse_header(b"\x00" * 10)


@pytest.mark.parametrize("fmt,size", [(2023, 217), (2024, 237), (2025, 273)])
def test_motion_ex_sizes_match_specification(fmt, size):
    assert MOTION_EX_SIZES[fmt] == size
    assert len(make_motion_ex(fmt, 1.0, 1)) == size


@pytest.mark.parametrize("fmt", [2023, 2024, 2025])
def test_motion_ex_fields_and_wheel_order(fmt):
    pos = [1.0, 2.0, 3.0, 4.0]  # RL, RR, FL, FR
    vel = [-10.0, 20.0, -30.0, 40.0]
    f = parse_motion_ex(make_motion_ex(fmt, 3.0, 42, position=pos, velocity=vel))
    np.testing.assert_allclose(f.suspension_position, pos)
    np.testing.assert_allclose(f.suspension_velocity, vel)
    np.testing.assert_allclose(f.wheel_speed, [6.0, 6.5, 7.0, 7.5])  # floats 12..15 of the filler
    assert f.front_wheels_angle == pytest.approx(42 * 0.5)
    if fmt >= 2024:
        np.testing.assert_allclose(f.wheel_vert_force, [21.5, 22.0, 22.5, 23.0])
        assert f.front_aero_height == pytest.approx(47 * 0.5)
        assert f.chassis_yaw == pytest.approx(51 * 0.5)
    else:
        assert f.wheel_vert_force is None and f.front_aero_height is None
    if fmt >= 2025:
        assert f.chassis_pitch == pytest.approx(52 * 0.5)
        np.testing.assert_allclose(f.wheel_camber_gain, [28.5, 29.0, 29.5, 30.0])
    else:
        assert f.wheel_camber is None


def test_motion_ex_rejects_other_packets_and_short_data():
    with pytest.raises(ValueError):
        parse_motion_ex(make_motion(2025, 1.0, 1))
    with pytest.raises(ValueError):
        parse_motion_ex(make_motion_ex(2025, 1.0, 1)[:200])


def test_car_motion_picks_player_car():
    data = make_motion(2025, 1.0, 1, player_index=5, g_vertical=2.5)
    assert len(data) == HEADER_SIZE + NUM_CARS * CAR_MOTION_SIZE == 1349
    header, m = parse_car_motion(data)
    assert header.player_car_index == 5
    assert m.g_force_vertical == pytest.approx(2.5)
    assert m.roll_rad == pytest.approx(-0.02)
    _, other = parse_car_motion(data, car_index=0)
    assert other.g_force_vertical == pytest.approx(9.0)


def test_capture_round_trip(tmp_path):
    packets = [(1000.0 + i, make_motion_ex(2025, i / 60, i)) for i in range(5)]
    path = tmp_path / "s.dlcap"
    assert write_capture(path, packets) == 5
    back = list(read_capture(path))
    assert [p for _, p in back] == [p for _, p in packets]
    assert back[0][0] == pytest.approx(1000.0)


def test_capture_rejects_foreign_file(tmp_path):
    bad = tmp_path / "x.bin"
    bad.write_bytes(b"not a capture")
    with pytest.raises(ValueError):
        list(read_capture(bad))


def _sine_session_packets(fmt=2025, rate=60.0, seconds=4.0, freq=2.0, amp=8.0, start_frame=0, t0=0.0):
    packets = []
    for k in range(int(rate * seconds)):
        t = t0 + k / rate
        pos = [amp * np.sin(2 * np.pi * freq * t + w) for w in range(4)]
        vel = [amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t + w) for w in range(4)]
        frame = start_frame + k
        packets.append((t, make_motion_ex(fmt, t, frame, position=pos, velocity=vel)))
        packets.append((t, make_motion(fmt, t, frame, g_vertical=1.0 + 0.1 * k / rate)))
    return packets


def test_session_pairs_motion_and_derives_velocity():
    session = build_session(_sine_session_packets())
    assert session.num_samples == 240 and session.num_segments == 1
    assert session.sample_rate_hz == pytest.approx(60.0, rel=1e-3)
    assert np.all(np.isfinite(session.vertical_g))

    for i in range(4):
        check = compare_velocity(derived_velocity(session, i), session.suspension_velocity[:, i])
        assert check.pearson_r > 0.99
        assert check.slope == pytest.approx(1.0, abs=0.05)
        assert check.best_lag_samples == 0


def test_flashback_and_duplicates_start_new_segments():
    first = _sine_session_packets(seconds=2.0)
    duplicate = [first[-2]]  # same Motion Ex frame again
    # Flashback: session time goes back to 1.0 s while the overall frame keeps counting up.
    second = []
    for k in range(60):
        t = 1.0 + k / 60
        second.append((t, make_motion_ex(2025, t, 500 + k, position=[0, 0, 0, 0], velocity=[0, 0, 0, 0])))
    session = build_session(first + duplicate + second)
    assert session.dropped_frames == 1
    assert session.num_segments == 2
    assert session.num_samples == 120 + 60
    assert session.duration_s == pytest.approx(119 / 60 + 59 / 60, rel=1e-3)


def test_majority_session_is_kept():
    packets = _sine_session_packets(seconds=1.0)
    packets += [(0.0, make_motion_ex(2025, 0.1 * k, 9000 + k, uid=7)) for k in range(5)]
    session = build_session(packets)
    assert session.session_uid == SESSION_UID
    assert session.other_sessions == 1


def test_no_motion_ex_raises():
    with pytest.raises(ValueError):
        build_session([(0.0, make_motion(2025, 0.0, 0))])


def test_corner_batch_and_analysis(tmp_path):
    path = tmp_path / "lap.dlcap"
    write_capture(path, _sine_session_packets(fmt=2024))
    session = load_session(path)
    batch = session.to_corner_batch("FL")
    assert batch.num_samples == session.num_samples
    np.testing.assert_allclose(batch.strut_stroke_mm, session.suspension_position[:, WHEELS.index("FL")])
    assert np.all(np.isnan(batch.internal_pressure_bar))

    summary = analyze_session(session)
    assert summary["udp_format"] == 2024
    assert set(summary["wheels"]) == set(WHEELS)
    for w in summary["wheels"].values():
        assert w["check_vs_game_velocity"]["pearson_r"] > 0.99
        assert sum(w["regime_share_pct"].values()) == pytest.approx(100.0, abs=0.01)
    assert "front_aero_height_min" in summary


def test_cli_analyze_writes_json(tmp_path, capsys):
    cap = tmp_path / "lap.dlcap"
    out = tmp_path / "summary.json"
    write_capture(cap, _sine_session_packets())
    assert main(["analyze", str(cap), "--json", str(out)]) == 0
    assert out.exists()
    assert "r vs game" in capsys.readouterr().out


def test_record_udp_over_loopback(tmp_path):
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    packets = [p for _, p in _sine_session_packets(seconds=0.5)]
    packets.append(make_header(2025, 6, 0.0, 0) + b"\x00" * 100)  # car telemetry, filtered out

    def send():
        time.sleep(0.3)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as tx:
            for p in packets:
                tx.sendto(p, ("127.0.0.1", port))

    sender = threading.Thread(target=send)
    sender.start()
    out = tmp_path / "live.dlcap"
    seen = record_udp(out, duration_sec=1.5, host="127.0.0.1", port=port)
    sender.join()

    assert seen[(2025, PACKET_ID_MOTION_EX)] == 30
    assert seen[(2025, 6)] == 1
    kept_ids = {parse_header(p).packet_id for _, p in read_capture(out)}
    assert kept_ids == {PACKET_ID_MOTION, PACKET_ID_MOTION_EX}

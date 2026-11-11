"""
100 Hz Telemetry Physics Simulator & Dataset Loader.

Mirrors the differential equations and polytropic gas-spring thermodynamics of the
C# .NET 8 `/telemetry-streamer-dotnet` application, and provides CSV/JSON ingestion
utilities for seamless cross-language interoperability.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List
import numpy as np

from .models import (
    AEROSPACE_ATA32_CONFIG,
    F1_TRACKSIDE_CONFIG,
    OperationalProfile,
    TelemetryBatch,
    get_profile_config,
)


def generate_aerospace_telemetry(
    scenario: str = "nominal",
    duration_sec: float = 10.0,
    sample_rate_hz: int = 100,
    seed: int = 42,
    g_scale: float = 1.0,
    pressure_scale: float = 1.0,
) -> TelemetryBatch:
    """
    Generate 100 Hz Profile A (Aerospace MRO - ATA Chapter 32) touchdown QAR telemetry.
    Scenarios: 'nominal', 'overload' (Hard Landing), 'fault' (N2 gas loss + orifice wear).
    """
    scen = scenario.strip().lower()
    total_samples = max(10, int(round(duration_sec * sample_rate_hz)))
    dt = 1.0 / sample_rate_hz
    rng = np.random.default_rng(seed)

    cfg = AEROSPACE_ATA32_CONFIG
    max_stroke = cfg.max_stroke_mm
    l0 = cfg.effective_gas_length_mm
    gamma = cfg.polytropic_gamma

    if scen == "overload":
        static_p0 = 30.0 * pressure_scale
        peak_impact_g_delta = 1.82 * g_scale
        settled_stroke_mm = 255.0
        dynamic_overshoot_mm = 210.0
        zeta = 0.48
        bank_deg = 5.9
    elif scen == "fault":
        static_p0 = 15.5 * pressure_scale  # Severe N2 pre-charge loss
        peak_impact_g_delta = 0.65 * g_scale
        settled_stroke_mm = 295.0
        dynamic_overshoot_mm = 62.0
        zeta = 0.14  # Degraded orifice hydraulic damping
        bank_deg = 1.8
    else:
        scen = "nominal"
        static_p0 = 30.0 * pressure_scale
        peak_impact_g_delta = 0.48 * g_scale
        settled_stroke_mm = 225.0
        dynamic_overshoot_mm = 68.0
        zeta = 0.62
        bank_deg = 0.9

    t = np.arange(total_samples, dtype=np.float64) * dt
    timestamp_ms = np.round(t * 1000.0)

    touchdown_sec = 2.0
    wn = 2.0 * np.pi * 1.85
    wd = wn * np.sqrt(1.0 - zeta * zeta)

    vertical_g = np.empty(total_samples, dtype=np.float64)
    roll_deg = np.empty(total_samples, dtype=np.float64)
    pitch_deg = np.empty(total_samples, dtype=np.float64)
    stroke_mm = np.empty(total_samples, dtype=np.float64)
    wheel_speed_kph = np.empty(total_samples, dtype=np.float64)

    pre_mask = t < touchdown_sec
    post_mask = ~pre_mask

    # Phase 1: Approach & Flare
    t_pre = t[pre_mask]
    flare_prog = t_pre / touchdown_sec
    vertical_g[pre_mask] = 1.00 + 0.04 * np.sin(2.0 * np.pi * 0.7 * t_pre) + rng.normal(0.0, 0.008, len(t_pre))
    roll_deg[pre_mask] = bank_deg * (0.6 + 0.4 * flare_prog) + rng.normal(0.0, 0.04, len(t_pre))
    pitch_deg[pre_mask] = 4.2 + 1.6 * flare_prog + rng.normal(0.0, 0.04, len(t_pre))
    stroke_mm[pre_mask] = np.maximum(0.0, rng.normal(0.0, 0.12, len(t_pre)))
    wheel_speed_kph[pre_mask] = 0.0

    # Phase 2–4: Weight-on-Wheels Impact, Derotation, and Rollout
    tau = t[post_mask] - touchdown_sec
    impact_pulse = peak_impact_g_delta * np.exp(-np.square((tau - 0.18) / 0.11))
    damped_ring_g = (
        0.28 * peak_impact_g_delta * np.exp(-zeta * wn * tau) * np.cos(wd * (tau - 0.18))
    )
    nose_gear_pulse = 0.16 * np.exp(-np.square((tau - 1.80) / 0.14))
    vertical_g[post_mask] = (
        1.00 + impact_pulse + damped_ring_g + nose_gear_pulse + rng.normal(0.0, 0.012, len(tau))
    )

    ramp_up = 1.0 - (1.0 + tau / 0.055) * np.exp(-tau / 0.055)
    primary_impact_stroke = dynamic_overshoot_mm * np.exp(-np.square((tau - 0.16) / 0.09))
    damped_osc = (0.55 * dynamic_overshoot_mm) * np.exp(-zeta * wn * tau) * np.sin(wd * tau)
    rollout_vib = 2.8 * np.sin(2.0 * np.pi * 4.2 * tau) * np.exp(-0.15 * tau)
    raw_stroke = (
        settled_stroke_mm * ramp_up
        + primary_impact_stroke
        + np.maximum(-45.0, damped_osc)
        + rollout_vib
        + rng.normal(0.0, 0.22, len(tau))
    )
    stroke_mm[post_mask] = np.clip(raw_stroke, 0.0, max_stroke - 3.0)

    roll_deg[post_mask] = bank_deg * np.exp(-tau / 0.65) * np.cos(3.5 * tau) + rng.normal(0.0, 0.04, len(tau))
    derotation = 1.0 / (1.0 + np.exp((tau - 1.45) / 0.28))
    pitch_deg[post_mask] = 0.2 + 5.6 * derotation + rng.normal(0.0, 0.04, len(tau))

    spin_up = 1.0 - np.exp(-tau / 0.06)
    autobrake = np.maximum(0.0, tau - 0.5) * 16.5
    wheel_speed_kph[post_mask] = np.maximum(
        60.0, (252.0 - autobrake) * spin_up + rng.normal(0.0, 0.3, len(tau))
    )

    vel_mm_s = np.gradient(stroke_mm, dt)
    clamped_stroke = np.minimum(stroke_mm, 0.94 * l0)
    gas_pressure_bar = static_p0 * np.power(l0 / (l0 - clamped_stroke), gamma)
    orifice_coeff = 0.000004 if scen == "fault" else 0.000018
    hyd_transient = np.clip(orifice_coeff * vel_mm_s * np.abs(vel_mm_s), -25.0, 55.0)
    internal_pressure_bar = np.maximum(
        static_p0 * 0.85,
        gas_pressure_bar + hyd_transient + rng.normal(0.0, 0.22, total_samples),
    )

    return TelemetryBatch(
        timestamp_ms=timestamp_ms,
        vertical_g=np.round(vertical_g, 4),
        roll_deg=np.round(roll_deg, 4),
        pitch_deg=np.round(pitch_deg, 4),
        strut_stroke_mm=np.round(stroke_mm, 4),
        internal_pressure_bar=np.round(internal_pressure_bar, 4),
        wheel_speed_kph=np.round(wheel_speed_kph, 4),
        profile=OperationalProfile.AEROSPACE_ATA32,
        scenario=scen,
    )


def generate_f1_telemetry(
    scenario: str = "nominal",
    duration_sec: float = 10.0,
    sample_rate_hz: int = 100,
    seed: int = 42,
    g_scale: float = 1.0,
    pressure_scale: float = 1.0,
) -> TelemetryBatch:
    """
    Generate 100 Hz Profile B (Motorsport F1 Trackside) suspension & kerb telemetry.
    Scenarios: 'nominal', 'overload' (Kerb Strike & Plank Bottoming), 'fault' (Reservoir gas loss & cavitation).
    """
    scen = scenario.strip().lower()
    total_samples = max(10, int(round(duration_sec * sample_rate_hz)))
    dt = 1.0 / sample_rate_hz
    rng = np.random.default_rng(seed)

    cfg = F1_TRACKSIDE_CONFIG
    max_stroke = cfg.max_stroke_mm
    l0 = cfg.effective_gas_length_mm
    gamma = cfg.polytropic_gamma

    if scen == "fault":
        static_gas_bar = 8.2 * pressure_scale
    else:
        static_gas_bar = 16.0 * pressure_scale
        if scen != "overload":
            scen = "nominal"

    t = np.arange(total_samples, dtype=np.float64) * dt
    timestamp_ms = np.round(t * 1000.0)

    vertical_g = np.empty(total_samples, dtype=np.float64)
    roll_deg = np.empty(total_samples, dtype=np.float64)
    pitch_deg = np.empty(total_samples, dtype=np.float64)
    stroke_mm = np.empty(total_samples, dtype=np.float64)
    wheel_speed_kph = np.empty(total_samples, dtype=np.float64)

    for i in range(total_samples):
        ti = t[i]
        s = ti % 10.0

        if s < 2.5:
            frac = s / 2.5
            ws = 290.0 + 44.0 * frac
            aero_g = 1.15 * ((ws / 300.0) ** 2)
            porp_mm = (11.5 if scen == "overload" else 2.8) * np.sin(2.0 * np.pi * 6.2 * s) * frac
            porp_g = (1.45 if scen == "overload" else 0.22) * np.sin(2.0 * np.pi * 6.2 * s) * frac
            bg = 1.0 + aero_g + porp_g
            pitch = -0.35 - 0.30 * frac
            roll = 0.08 * np.sin(2.0 * np.pi * 1.2 * s)
            bst = 16.0 + 24.5 * ((ws / 300.0) ** 2) + porp_mm

            if scen == "overload" and 1.7 < s < 2.35:
                bp = np.exp(-(((s - 2.05) / 0.08) ** 2))
                bst = max(bst, 63.6 * bp + bst * (1.0 - bp))
                bg += 2.15 * bp
        elif s < 4.5:
            frac = (s - 2.5) / 2.0
            ws = 334.0 - 202.0 * (frac**0.75)
            pitch = -1.45 * np.sin(np.pi * frac)
            roll = -0.45 * frac
            bg = 1.65 + 0.55 * (1.0 - frac) + 0.20 * np.sin(2.0 * np.pi * 8.5 * s)
            bst = 38.0 - 11.0 * frac + 2.2 * np.sin(2.0 * np.pi * 8.5 * s)
        elif s < 7.0:
            frac = (s - 4.5) / 2.5
            ws = 132.0 + 92.0 * frac
            pitch = -0.25 + 0.45 * np.sin(2.0 * np.pi * 0.8 * (s - 4.5))
            roll = 1.85 * np.sin(2.0 * np.pi * 0.45 * (s - 4.5))

            kerb_amp_mm = 28.5 if scen == "overload" else (16.5 if scen == "fault" else 14.5)
            kerb_amp_g = 3.45 if scen == "overload" else (1.55 if scen == "fault" else 1.35)

            k1 = np.exp(-(((s - 4.95) / 0.032) ** 2)) * np.cos(2.0 * np.pi * 14.0 * (s - 4.95))
            k2 = np.exp(-(((s - 5.85) / 0.035) ** 2)) * np.cos(2.0 * np.pi * 13.0 * (s - 5.85))

            bg = 1.85 + kerb_amp_g * (abs(k1) + 0.85 * abs(k2))
            bst = 33.5 + kerb_amp_mm * (k1 + 0.9 * k2)
        else:
            frac = (s - 7.0) / 3.0
            ws = 224.0 + 81.0 * frac
            pitch = -0.40 - 0.15 * frac
            roll = 2.05 * np.sin(np.pi * min(1.0, frac * 1.3))
            bg = 2.25 + 0.78 * np.sin(np.pi * frac) + 0.18 * np.sin(2.0 * np.pi * 9.5 * s)
            bst = 36.0 + 11.5 * np.sin(np.pi * frac) + 1.8 * np.sin(2.0 * np.pi * 9.5 * s)

        if scen == "fault":
            bst += 4.5 + 4.8 * np.sin(2.0 * np.pi * 11.5 * ti)
            bg += 0.25 * np.cos(2.0 * np.pi * 11.5 * ti)

        stroke_mm[i] = np.clip(bst + rng.normal(0.0, 0.16), 1.5, max_stroke - 0.5)
        vertical_g[i] = bg * g_scale + rng.normal(0.0, 0.025)
        roll_deg[i] = roll + rng.normal(0.0, 0.02)
        pitch_deg[i] = pitch + rng.normal(0.0, 0.02)
        wheel_speed_kph[i] = ws + rng.normal(0.0, 0.25)

    vel_mm_s = np.gradient(stroke_mm, dt)
    clamped_stroke = np.minimum(stroke_mm, 0.92 * l0)
    gas_pressure_bar = static_gas_bar * np.power(l0 / (l0 - clamped_stroke), gamma)
    shim_coeff = 0.0015 if scen == "fault" else 0.0115
    hyd_delta = shim_coeff * vel_mm_s
    internal_pressure_bar = np.maximum(
        static_gas_bar * 0.75,
        gas_pressure_bar + hyd_delta + rng.normal(0.0, 0.14, total_samples),
    )

    return TelemetryBatch(
        timestamp_ms=timestamp_ms,
        vertical_g=np.round(vertical_g, 4),
        roll_deg=np.round(roll_deg, 4),
        pitch_deg=np.round(pitch_deg, 4),
        strut_stroke_mm=np.round(stroke_mm, 4),
        internal_pressure_bar=np.round(internal_pressure_bar, 4),
        wheel_speed_kph=np.round(wheel_speed_kph, 4),
        profile=OperationalProfile.FORMULA1_TRACKSIDE,
        scenario=scen,
    )


def generate_telemetry(
    profile: OperationalProfile | str,
    scenario: str = "nominal",
    duration_sec: float = 10.0,
    sample_rate_hz: int = 100,
    seed: int = 42,
    g_scale: float = 1.0,
    pressure_scale: float = 1.0,
) -> TelemetryBatch:
    """Dispatch generator for either Profile A (Aerospace) or Profile B (F1)."""
    cfg = get_profile_config(profile)
    if cfg.profile == OperationalProfile.AEROSPACE_ATA32:
        return generate_aerospace_telemetry(
            scenario=scenario,
            duration_sec=duration_sec,
            sample_rate_hz=sample_rate_hz,
            seed=seed,
            g_scale=g_scale,
            pressure_scale=pressure_scale,
        )
    return generate_f1_telemetry(
        scenario=scenario,
        duration_sec=duration_sec,
        sample_rate_hz=sample_rate_hz,
        seed=seed,
        g_scale=g_scale,
        pressure_scale=pressure_scale,
    )


def load_telemetry_from_csv(filepath: str | Path) -> TelemetryBatch:
    """Load a TelemetryBatch from a CSV file exported by the C# .NET 8 streamer."""
    path = Path(filepath)
    ts, vg, roll, pitch, stroke, press, ws = [], [], [], [], [], [], []
    profile_str = "aero"
    scenario_str = "nominal"

    with path.open("r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts.append(float(row["timestamp_ms"]))
            vg.append(float(row["vertical_g"]))
            roll.append(float(row["roll_deg"]))
            pitch.append(float(row["pitch_deg"]))
            stroke.append(float(row["strut_stroke_mm"]))
            press.append(float(row["internal_pressure_bar"]))
            ws.append(float(row["wheel_speed_kph"]))
            if "profile" in row and row["profile"]:
                profile_str = row["profile"]
            if "scenario" in row and row["scenario"]:
                scenario_str = row["scenario"]

    cfg = get_profile_config(profile_str)
    return TelemetryBatch(
        timestamp_ms=np.array(ts, dtype=np.float64),
        vertical_g=np.array(vg, dtype=np.float64),
        roll_deg=np.array(roll, dtype=np.float64),
        pitch_deg=np.array(pitch, dtype=np.float64),
        strut_stroke_mm=np.array(stroke, dtype=np.float64),
        internal_pressure_bar=np.array(press, dtype=np.float64),
        wheel_speed_kph=np.array(ws, dtype=np.float64),
        profile=cfg.profile,
        scenario=scenario_str,
    )


def load_telemetry_from_json(filepath: str | Path) -> TelemetryBatch:
    """Load a TelemetryBatch from a JSON array exported by the C# .NET 8 streamer."""
    path = Path(filepath)
    with path.open("r", encoding="utf-8-sig") as f:
        records: List[dict] = json.load(f)

    if not records:
        raise ValueError(f"JSON telemetry file {filepath} contains no records.")

    profile_str = records[0].get("profile", "aero")
    scenario_str = records[0].get("scenario", "nominal")
    cfg = get_profile_config(profile_str)

    return TelemetryBatch(
        timestamp_ms=np.array([float(r["timestamp_ms"]) for r in records], dtype=np.float64),
        vertical_g=np.array([float(r["vertical_g"]) for r in records], dtype=np.float64),
        roll_deg=np.array([float(r["roll_deg"]) for r in records], dtype=np.float64),
        pitch_deg=np.array([float(r["pitch_deg"]) for r in records], dtype=np.float64),
        strut_stroke_mm=np.array([float(r["strut_stroke_mm"]) for r in records], dtype=np.float64),
        internal_pressure_bar=np.array([float(r["internal_pressure_bar"]) for r in records], dtype=np.float64),
        wheel_speed_kph=np.array([float(r["wheel_speed_kph"]) for r in records], dtype=np.float64),
        profile=cfg.profile,
        scenario=scenario_str,
    )

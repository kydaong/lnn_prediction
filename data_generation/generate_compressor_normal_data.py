"""Generate synthetic normal-operation data for the centrifugal compressor train.

Anchors baseline mean/std for each of the 45 sensor tags from the sample
sheets in raw_data_file/KBC_AOM_c_compressor_data.xlsx, then simulates a
long time series driven by shared latent factors (load, ambient temperature)
so tags co-vary the way real correlated sensors would. Intended as training
data for an LNN autoencoder that learns the compressor's normal operating
manifold.

Usage:
    .venv/Scripts/python.exe scripts/generate_compressor_normal_data.py
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_XLSX = Path("raw_data_file/KBC_AOM_c_compressor_data.xlsx")

# DGS sample sheet columns -> matching Ops sheet tag names.
DGS_TAG_MAP = {
    "filter_dp": "dgs_filter_dp_bar",
    "seal_gas_flow": "dgs_seal_gas_flow_nm3h",
    "seal_gas_diff_pressure": "dgs_seal_gas_diff_press_bar",
    "seal_gas_temp": "dgs_seal_gas_temp_c",
    "primary_vent_flow": "dgs_primary_vent_flow_nm3h",
    "primary_vent_pressure": "dgs_primary_vent_press_barg",
    "secondary_seal_gas_flow": "dgs_secondary_seal_gas_flow_nm3h",
    "separation_seal_gas_flow": "dgs_separation_seal_gas_flow_nm3h",
    "separation_seal_gas_pressure": "dgs_separation_seal_gas_press_barg",
    "seal_gas_to_vent_diff_pressure": "dgs_seal_gas_to_vent_diff_press_bar",
}


def load_baseline_stats() -> dict[str, dict[str, float]]:
    """Mean/std per tag, preferring the larger DGS sample where it overlaps."""
    xls = pd.ExcelFile(RAW_XLSX)
    ops = xls.parse("Compressor Ops Data (sample)", header=1)
    ops["timestamp"] = pd.to_datetime(ops["timestamp"], errors="coerce")
    ops = ops.drop(columns=["timestamp"])

    stats = {
        col: {"mean": float(ops[col].mean()), "std": float(ops[col].std(ddof=0))}
        for col in ops.columns
    }

    dgs = xls.parse("DGS Data points (Sample)")
    for dgs_col, tag in DGS_TAG_MAP.items():
        stats[tag] = {
            "mean": float(dgs[dgs_col].mean()),
            "std": float(dgs[dgs_col].std(ddof=0)),
        }

    # Floor std so tiny-sample noise doesn't collapse to zero variance.
    for tag, s in stats.items():
        s["std"] = max(s["std"], 0.02 * abs(s["mean"]), 1e-6)
    return stats


def ou_like_noise(n: int, tau_minutes: float, std: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth mean-reverting noise via EWM-filtered white noise, scaled to `std`.

    Prepends a burn-in of ~5 correlation times so the EWM reaches its
    steady-state variance before the returned window starts — without it,
    slow (large-tau) series carry a full-variance transient at t=0 that
    the steady-state scaling then wildly over-amplifies.
    """
    alpha = min(max(1.0 / tau_minutes, 1e-4), 1.0)
    burn_in = int(min(max(5 * tau_minutes, 50), 5 * n))
    eps = rng.normal(0.0, 1.0, n + burn_in)
    y = pd.Series(eps).ewm(alpha=alpha, adjust=False).mean().to_numpy()
    var = alpha / (2 - alpha) if alpha < 1 else 1.0
    return y[burn_in:] / np.sqrt(var) * std


def fouling_sawtooth(n: int, low: float, high: float, cycle_days: tuple[float, float], rng: np.random.Generator) -> np.ndarray:
    """Gradual creep from `low` to `high` with periodic resets (filter service events)."""
    minutes_per_day = 1440
    values = np.empty(n)
    idx = 0
    while idx < n:
        cycle_len = int(rng.uniform(*cycle_days) * minutes_per_day)
        cycle_len = max(1, min(cycle_len, n - idx))
        values[idx : idx + cycle_len] = np.linspace(low, high, cycle_len)
        idx += cycle_len
    return values


def build_load_series(
    n: int, rng: np.random.Generator, load_center: np.ndarray, load_std: np.ndarray
) -> np.ndarray:
    """Piecewise target load (operator setpoint changes) smoothed by control-loop lag.

    `load_center`/`load_std` are per-minute arrays (from the active process
    grade) so short-term load wander happens around the current grade's
    typical throughput, not a single fixed setpoint.
    """
    minutes_per_day = 1440
    targets = np.empty(n)
    idx = 0
    while idx < n:
        seg_len = int(rng.uniform(0.5, 4.0) * minutes_per_day)
        seg_len = max(1, min(seg_len, n - idx))
        target = np.clip(rng.normal(load_center[idx], load_std[idx]), 0.5, 1.05)
        targets[idx : idx + seg_len] = target
        idx += seg_len
    smoothed = pd.Series(targets).ewm(span=30, adjust=False).mean().to_numpy().copy()
    smoothed += rng.normal(0.0, 0.004, n)
    return np.clip(smoothed, 0.45, 1.08)


# Three process grades this compressor normally runs: different feed gas
# composition (MW/Z-factor) and correspondingly different discharge pressure
# and flow targets. All still "normal" — the autoencoder needs to learn all
# three as valid operating regions, not just noise around one point.
GRADES = [
    {"name": "grade_A", "mw_factor": 1.00, "z_offset": 0.000, "discharge_press_factor": 1.00, "flow_factor": 1.00, "load_center": 0.90, "load_std": 0.06},
    {"name": "grade_B", "mw_factor": 1.15, "z_offset": -0.018, "discharge_press_factor": 1.07, "flow_factor": 0.90, "load_center": 0.80, "load_std": 0.06},
    {"name": "grade_C", "mw_factor": 0.88, "z_offset": 0.022, "discharge_press_factor": 0.94, "flow_factor": 1.08, "load_center": 0.85, "load_std": 0.06},
]


def build_grade_series(
    n: int, rng: np.random.Generator, campaign_days: tuple[float, float] = (10, 35), ramp_hours: float = 8
) -> dict[str, np.ndarray]:
    """Discrete grade campaigns (days-to-weeks long) with a smooth ramp between them.

    Returns per-minute arrays: a `grade_name` label (nominal grade, ignoring
    the ramp) plus smoothly-transitioning `mw_factor`/`z_offset`/
    `discharge_press_factor`/`flow_factor`/`load_center`/`load_std`.
    """
    minutes_per_day = 1440
    grade_idx = np.empty(n, dtype=int)
    idx = 0
    last_grade = -1
    while idx < n:
        choices = [i for i in range(len(GRADES)) if i != last_grade]
        g = rng.choice(choices) if last_grade != -1 else int(rng.integers(0, len(GRADES)))
        seg_len = int(rng.uniform(*campaign_days) * minutes_per_day)
        seg_len = max(1, min(seg_len, n - idx))
        grade_idx[idx : idx + seg_len] = g
        idx += seg_len
        last_grade = g

    span = ramp_hours * 60

    def smooth(key: str) -> np.ndarray:
        raw = np.array([GRADES[g][key] for g in grade_idx])
        return pd.Series(raw).ewm(span=span, adjust=False).mean().to_numpy()

    return {
        "grade_name": np.array([GRADES[g]["name"] for g in grade_idx]),
        "mw_factor": smooth("mw_factor"),
        "z_offset": smooth("z_offset"),
        "discharge_press_factor": smooth("discharge_press_factor"),
        "flow_factor": smooth("flow_factor"),
        "load_center": smooth("load_center"),
        "load_std": smooth("load_std"),
    }


def build_ambient_series(timestamps: pd.DatetimeIndex, rng: np.random.Generator) -> np.ndarray:
    day_of_year = timestamps.dayofyear.to_numpy()
    hour = timestamps.hour.to_numpy() + timestamps.minute.to_numpy() / 60.0
    seasonal = 10.0 * np.sin(2 * np.pi * (day_of_year - 80) / 365.0)
    diurnal = 5.0 * np.sin(2 * np.pi * (hour - 9) / 24.0)
    weather_noise = ou_like_noise(len(timestamps), tau_minutes=360, std=1.5, rng=rng)
    return 15.0 + seasonal + diurnal + weather_noise


def generate(n_days: int, start: str, freq: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    periods = int(n_days * 24 * 60)
    timestamps = pd.date_range(start=start, periods=periods, freq=freq)
    n = len(timestamps)

    stats = load_baseline_stats()

    def m(tag: str) -> float:
        return stats[tag]["mean"]

    def s(tag: str) -> float:
        return stats[tag]["std"]

    grade = build_grade_series(n, rng)
    load = build_load_series(n, rng, grade["load_center"], grade["load_std"])
    load_ref = load.mean()
    load_frac = load / load_ref
    load_dev = load_frac - 1.0

    ambient = build_ambient_series(timestamps, rng)
    ambient_dev = ambient - ambient.mean()

    df = pd.DataFrame(index=timestamps)
    df.index.name = "timestamp"

    # --- Flow-like tags: scale with load fraction ---
    flow_tags = {
        "comp_flow_nm3h": 15,
        "dgs_seal_gas_flow_nm3h": 10,
        "dgs_primary_vent_flow_nm3h": 10,
        "dgs_secondary_seal_gas_flow_nm3h": 10,
        "dgs_separation_seal_gas_flow_nm3h": 10,
        "hx_water_flow_m3h": 20,
        "motor_current_a": 10,
        "motor_power_kw": 10,
    }
    for tag, tau in flow_tags.items():
        noise = ou_like_noise(n, tau, max(s(tag) * 0.4, m(tag) * 0.01), rng)
        grade_factor = grade["flow_factor"] if tag == "comp_flow_nm3h" else 1.0
        df[tag] = m(tag) * grade_factor * (0.35 + 0.65 * load_frac) + noise

    # --- Pressure tags: mild positive sensitivity to load ---
    pressure_tags = {
        "comp_suction_press_barg": 30,
        "comp_discharge_press_barg": 30,
        "lube_supply_press_barg": 30,
        "dgs_seal_gas_diff_press_bar": 20,
        "dgs_primary_vent_press_barg": 20,
        "dgs_separation_seal_gas_press_barg": 20,
        "dgs_seal_gas_to_vent_diff_press_bar": 20,
    }
    for tag, tau in pressure_tags.items():
        noise = ou_like_noise(n, tau, max(s(tag) * 0.5, m(tag) * 0.015), rng)
        grade_factor = grade["discharge_press_factor"] if tag == "comp_discharge_press_barg" else 1.0
        df[tag] = m(tag) * grade_factor * (1 + 0.35 * load_dev) + noise

    # --- Compressor speed: fixed-speed drive, near-constant ---
    df["comp_speed_rpm"] = m("comp_speed_rpm") * (1 + 0.01 * load_dev) + ou_like_noise(
        n, 5, s("comp_speed_rpm"), rng
    )

    # --- Vibration: stationary, weak load sensitivity ---
    for tag in ["comp_vib_x_de_rms_mms", "comp_vib_y_de_rms_mms"]:
        df[tag] = np.clip(
            m(tag) * (1 + 0.05 * load_dev) + ou_like_noise(n, 5, max(s(tag), 0.03), rng),
            0.05,
            None,
        )

    # --- Thrust position ---
    df["comp_thrust_pos_mm"] = np.clip(
        m("comp_thrust_pos_mm") * (1 + 0.1 * load_dev)
        + ou_like_noise(n, 10, max(s("comp_thrust_pos_mm"), 0.005), rng),
        0.0,
        None,
    )

    # --- Temperatures: ambient-linked with load self-heating ---
    df["comp_suction_temp_c"] = (
        m("comp_suction_temp_c")
        + 0.4 * ambient_dev
        + 0.05 * m("comp_suction_temp_c") * load_dev
        + ou_like_noise(n, 60, s("comp_suction_temp_c"), rng)
    )
    df["comp_discharge_temp_c"] = (
        df["comp_suction_temp_c"]
        + (m("comp_discharge_temp_c") - m("comp_suction_temp_c"))
        * grade["discharge_press_factor"]
        * (1 + 0.4 * load_dev)
        + ou_like_noise(n, 30, s("comp_discharge_temp_c") * 0.6, rng)
    )
    df["coupling_temp_c"] = (
        m("coupling_temp_c") * (1 + 0.08 * load_dev)
        + 0.2 * ambient_dev
        + ou_like_noise(n, 45, s("coupling_temp_c"), rng)
    )
    df["motor_stator_temp_u_c"] = (
        m("motor_stator_temp_u_c") * (1 + 0.12 * load_dev)
        + 0.25 * ambient_dev
        + ou_like_noise(n, 45, s("motor_stator_temp_u_c") * 0.6, rng)
    )
    df["motor_de_brg_temp_c"] = (
        m("motor_de_brg_temp_c") * (1 + 0.08 * load_dev)
        + 0.2 * ambient_dev
        + ou_like_noise(n, 45, s("motor_de_brg_temp_c"), rng)
    )
    df["comp_radial_brg_temp_de_c"] = (
        m("comp_radial_brg_temp_de_c") * (1 + 0.08 * load_dev)
        + 0.15 * ambient_dev
        + ou_like_noise(n, 45, s("comp_radial_brg_temp_de_c"), rng)
    )
    df["lube_supply_temp_c"] = (
        m("lube_supply_temp_c")
        + 0.3 * ambient_dev
        + 0.1 * m("lube_supply_temp_c") * load_dev
        + ou_like_noise(n, 60, s("lube_supply_temp_c"), rng)
    )
    df["dgs_seal_gas_temp_c"] = (
        m("dgs_seal_gas_temp_c")
        + 0.3 * (df["comp_discharge_temp_c"] - m("comp_discharge_temp_c"))
        + ou_like_noise(n, 30, s("dgs_seal_gas_temp_c"), rng)
    )
    df["cooler_water_outlet_temp_c"] = (
        m("cooler_water_outlet_temp_c")
        + 0.5 * ambient_dev
        + 0.15 * m("cooler_water_outlet_temp_c") * load_dev
        + ou_like_noise(n, 45, s("cooler_water_outlet_temp_c"), rng)
    )
    df["hx_gas_inlet_temp_c"] = (
        m("hx_gas_inlet_temp_c")
        + 0.5 * (df["comp_discharge_temp_c"] - m("comp_discharge_temp_c"))
        + ou_like_noise(n, 30, s("hx_gas_inlet_temp_c") * 0.6, rng)
    )
    df["hx_water_inlet_temp_c"] = (
        m("hx_water_inlet_temp_c")
        + 0.6 * ambient_dev
        + ou_like_noise(n, 45, s("hx_water_inlet_temp_c"), rng)
    )
    df["hx_gas_outlet_temp_c"] = (
        m("hx_gas_outlet_temp_c")
        + 0.3 * (df["hx_gas_inlet_temp_c"] - m("hx_gas_inlet_temp_c"))
        + 0.3 * ambient_dev
        + ou_like_noise(n, 30, s("hx_gas_outlet_temp_c") * 0.6, rng)
    )
    df["hx_water_outlet_temp_c"] = (
        df["hx_water_inlet_temp_c"]
        + (m("hx_water_outlet_temp_c") - m("hx_water_inlet_temp_c")) * (1 + 0.1 * load_dev)
        + ou_like_noise(n, 30, s("hx_water_outlet_temp_c") * 0.6, rng)
    )
    df["hx_approach_temp_c"] = np.clip(
        m("hx_approach_temp_c")
        + 0.1 * ambient_dev
        + ou_like_noise(n, 45, s("hx_approach_temp_c"), rng),
        0.2,
        None,
    )
    df["cooler_effectiveness_pct"] = np.clip(
        m("cooler_effectiveness_pct")
        - 0.5 * ambient_dev
        - 0.1 * m("cooler_effectiveness_pct") * load_dev
        + ou_like_noise(n, 60, s("cooler_effectiveness_pct"), rng),
        0.0,
        100.0,
    )

    # --- Coupling phase diff & HX residual: stationary noise ---
    df["coupling_phase_diff_deg"] = m("coupling_phase_diff_deg") + ou_like_noise(
        n, 10, s("coupling_phase_diff_deg"), rng
    )
    df["hx_energy_balance_residual_pct"] = m("hx_energy_balance_residual_pct") + ou_like_noise(
        n, 5, s("hx_energy_balance_residual_pct"), rng
    )

    # --- Process spec: grade-driven composition shift + slow drift within a grade ---
    df["proc_gas_mw_kg_kmol"] = m("proc_gas_mw_kg_kmol") * grade["mw_factor"] + ou_like_noise(
        n, 2880, s("proc_gas_mw_kg_kmol") * 0.5, rng
    )
    df["proc_gas_z_factor"] = (
        m("proc_gas_z_factor")
        + grade["z_offset"]
        + ou_like_noise(n, 2880, s("proc_gas_z_factor") * 0.5, rng)
    )
    df["proc_scrubber_level_pct"] = np.clip(
        m("proc_scrubber_level_pct") + ou_like_noise(n, 60, s("proc_scrubber_level_pct") * 1.5, rng),
        10.0,
        80.0,
    )

    # --- IGV position: derived from load, actual tracks setpoint with lag ---
    igv_sp = np.clip(m("igv_position_sp") + (load - load_ref) * 150.0, 5.0, 100.0)
    igv_sp += rng.normal(0.0, 0.05, n)
    igv_actual = pd.Series(igv_sp).ewm(span=5, adjust=False).mean().to_numpy().copy()
    igv_actual += rng.normal(0.0, max(s("igv_position_actual") * 0.3, 0.03), n)
    df["igv_position_sp"] = igv_sp
    df["igv_position_actual"] = igv_actual
    df["igv_position_diff"] = df["igv_position_actual"] - df["igv_position_sp"]
    df["igv_travel_rate_of_change"] = df["igv_position_actual"].diff().abs().fillna(0.0)

    # --- Filter differential pressures: fouling creep with periodic service resets ---
    df["dgs_filter_dp_bar"] = fouling_sawtooth(
        n, low=max(0.02, m("dgs_filter_dp_bar") * 0.5), high=m("dgs_filter_dp_bar") * 2.2,
        cycle_days=(30, 90), rng=rng,
    ) + ou_like_noise(n, 30, s("dgs_filter_dp_bar") * 0.3, rng)
    df["lube_filter_dp_bar"] = fouling_sawtooth(
        n, low=max(0.02, m("lube_filter_dp_bar") * 0.5), high=m("lube_filter_dp_bar") * 2.0,
        cycle_days=(45, 120), rng=rng,
    ) + ou_like_noise(n, 30, s("lube_filter_dp_bar") * 0.3, rng)

    # --- Metadata / labels ---
    df["equipment_id"] = "C-101"
    df["operating_mode"] = "normal_operation"
    df["process_grade"] = grade["grade_name"]

    # Reorder to match the source workbook's tag order.
    tag_order = [
        "comp_suction_press_barg", "comp_discharge_press_barg", "comp_suction_temp_c",
        "comp_discharge_temp_c", "comp_flow_nm3h", "comp_speed_rpm",
        "comp_vib_x_de_rms_mms", "comp_vib_y_de_rms_mms", "comp_thrust_pos_mm",
        "comp_radial_brg_temp_de_c", "coupling_temp_c", "coupling_phase_diff_deg",
        "proc_gas_mw_kg_kmol", "proc_gas_z_factor", "proc_scrubber_level_pct",
        "motor_current_a", "motor_power_kw", "motor_stator_temp_u_c", "motor_de_brg_temp_c",
        "lube_supply_press_barg", "lube_supply_temp_c", "lube_filter_dp_bar",
        "igv_position_sp", "igv_position_actual", "igv_position_diff", "igv_travel_rate_of_change",
        "dgs_filter_dp_bar", "dgs_seal_gas_flow_nm3h", "dgs_seal_gas_diff_press_bar",
        "dgs_seal_gas_temp_c", "dgs_primary_vent_flow_nm3h", "dgs_primary_vent_press_barg",
        "dgs_secondary_seal_gas_flow_nm3h", "dgs_separation_seal_gas_flow_nm3h",
        "dgs_separation_seal_gas_press_barg", "dgs_seal_gas_to_vent_diff_press_bar",
        "cooler_water_outlet_temp_c", "cooler_effectiveness_pct", "hx_gas_inlet_temp_c",
        "hx_gas_outlet_temp_c", "hx_water_inlet_temp_c", "hx_water_outlet_temp_c",
        "hx_water_flow_m3h", "hx_energy_balance_residual_pct", "hx_approach_temp_c",
    ]
    df = df[["equipment_id", "operating_mode", "process_grade"] + tag_order]
    return df.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=float, default=182, help="Duration in days (default ~6 months)")
    parser.add_argument("--start", type=str, default="2025-01-01", help="Start timestamp")
    parser.add_argument("--freq", type=str, default="1min", help="Sample interval")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--out", type=str, default="data/processed/compressor_normal_operation.parquet",
        help="Output path (.parquet or .csv)",
    )
    args = parser.parse_args()

    df = generate(n_days=args.days, start=args.start, freq=args.freq, seed=args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".csv":
        df.to_csv(out_path, index=False)
    else:
        df.to_parquet(out_path, index=False)

    preview_path = out_path.with_name(out_path.stem + "_preview.csv")
    df.head(2000).to_csv(preview_path, index=False)

    print(f"Generated {len(df):,} rows x {df.shape[1]} columns")
    print(f"Date range: {df['timestamp'].min()} -> {df['timestamp'].max()}")
    print(f"Saved: {out_path}")
    print(f"Preview (first 2000 rows): {preview_path}")


if __name__ == "__main__":
    main()

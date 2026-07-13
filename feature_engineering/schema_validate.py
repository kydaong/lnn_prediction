"""Pandera schema for compressor_normal_operation rows.

Generous physical bounds — this is a garbage/corruption catch (sensor
dropout, unit mix-ups, transmitter faults), not a tight statistical filter
on normal variation. Reused at both training-data prep and, later,
live-inference time so both paths reject the same malformed input.
"""
from __future__ import annotations

import pandera.pandas as pa
from pandera.pandas import Column, Check

# (min, max) physical bounds per sensor tag.
SENSOR_RANGES: dict[str, tuple[float, float]] = {
    "comp_suction_press_barg": (1.0, 8.0),
    "comp_discharge_press_barg": (6.0, 18.0),
    "comp_suction_temp_c": (5.0, 65.0),
    "comp_discharge_temp_c": (50.0, 150.0),
    "comp_flow_nm3h": (10000.0, 75000.0),
    "comp_speed_rpm": (8000.0, 14000.0),
    "comp_vib_x_de_rms_mms": (0.0, 12.0),
    "comp_vib_y_de_rms_mms": (0.0, 12.0),
    "comp_thrust_pos_mm": (0.0, 0.6),
    "comp_radial_brg_temp_de_c": (20.0, 120.0),
    "coupling_temp_c": (10.0, 100.0),
    "coupling_phase_diff_deg": (0.0, 25.0),
    "proc_gas_mw_kg_kmol": (8.0, 40.0),
    "proc_gas_z_factor": (0.4, 1.3),
    "proc_scrubber_level_pct": (0.0, 100.0),
    "motor_current_a": (50.0, 450.0),
    "motor_power_kw": (1000.0, 7000.0),
    "motor_stator_temp_u_c": (30.0, 130.0),
    "motor_de_brg_temp_c": (20.0, 110.0),
    "lube_supply_press_barg": (0.2, 5.0),
    "lube_supply_temp_c": (10.0, 90.0),
    "lube_filter_dp_bar": (0.0, 2.5),
    "igv_position_sp": (0.0, 100.0),
    "igv_position_actual": (0.0, 100.0),
    "igv_position_diff": (-15.0, 15.0),
    "igv_travel_rate_of_change": (0.0, 15.0),
    "dgs_filter_dp_bar": (0.0, 2.5),
    "dgs_seal_gas_flow_nm3h": (0.0, 35.0),
    "dgs_seal_gas_diff_press_bar": (0.0, 9.0),
    "dgs_seal_gas_temp_c": (0.0, 110.0),
    "dgs_primary_vent_flow_nm3h": (0.0, 6.0),
    "dgs_primary_vent_press_barg": (0.0, 3.5),
    "dgs_secondary_seal_gas_flow_nm3h": (0.0, 16.0),
    "dgs_separation_seal_gas_flow_nm3h": (0.0, 11.0),
    "dgs_separation_seal_gas_press_barg": (0.0, 3.5),
    "dgs_seal_gas_to_vent_diff_press_bar": (0.0, 2.5),
    "cooler_water_outlet_temp_c": (0.0, 65.0),
    "cooler_effectiveness_pct": (0.0, 100.0),
    "hx_gas_inlet_temp_c": (40.0, 170.0),
    "hx_gas_outlet_temp_c": (10.0, 90.0),
    "hx_water_inlet_temp_c": (-5.0, 60.0),
    "hx_water_outlet_temp_c": (0.0, 70.0),
    "hx_water_flow_m3h": (50.0, 650.0),
    "hx_energy_balance_residual_pct": (-15.0, 15.0),
    "hx_approach_temp_c": (-2.0, 30.0),
}

SENSOR_TAGS = list(SENSOR_RANGES.keys())

_sensor_columns = {
    tag: Column(float, Check.in_range(*bounds), nullable=False)
    for tag, bounds in SENSOR_RANGES.items()
}

CompressorSchema = pa.DataFrameSchema(
    {
        "timestamp": Column("datetime64[ns]", nullable=False),
        "equipment_id": Column(str, nullable=False),
        "operating_mode": Column(str, nullable=False),
        "process_grade": Column(str, nullable=True),
        **_sensor_columns,
    },
    strict=False,  # tolerate extra columns (e.g. future metadata) without failing
    coerce=True,
)


def validate(df):
    """Validate and return the (possibly type-coerced) DataFrame, or raise pandera.errors.SchemaErrors."""
    return CompressorSchema.validate(df, lazy=True)

"""Create the compressor_normal_operation table in LNN_data if it doesn't exist.
"""

from db_connection import get_connection

TABLE_NAME = "compressor_normal_operation"

# Must match the tag_order list in generate_compressor_normal_data.py.
SENSOR_TAGS = [
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

sensor_columns_sql = ",\n    ".join(f"[{tag}] FLOAT NULL" for tag in SENSOR_TAGS)


DDL = f"""
IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = '{TABLE_NAME}')
BEGIN
    CREATE TABLE dbo.{TABLE_NAME} (
        [timestamp] DATETIME2(0) NOT NULL,
        [equipment_id] VARCHAR(20) NOT NULL,
        [operating_mode] VARCHAR(30) NOT NULL,
        {sensor_columns_sql},
        CONSTRAINT PK_{TABLE_NAME} PRIMARY KEY CLUSTERED ([equipment_id], [timestamp])
    );
END
"""


def main() -> None:
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(DDL)
    conn.commit()
    print(f"Table dbo.{TABLE_NAME} is ready ({len(SENSOR_TAGS)} sensor columns).")
    conn.close()


if __name__ == "__main__":
    main()

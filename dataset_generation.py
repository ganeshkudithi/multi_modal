import pandas as pd
import numpy as np

df = pd.read_csv("matclip_phase1_cluster.csv")


def build_caption(row):
    parts = []


    if pd.notna(row["scale_text"]):
        parts.append(row["scale_text"])
    # Morphology description
    if pd.notna(row["morphology"]):
        parts.append(f"The morphology is {row['morphology']}.")
    


    # Process
    if pd.notna(row["param_process_type"]):
        parts.append(f"The sample was fabricated using {row['param_process_type']}.")
    
    

    # Processing parameters
    process = []

    if pd.notna(row["param_laser_power_W"]):
        process.append(f"laser power of {row['param_laser_power_W']} W")

    if pd.notna(row["param_scan_speed_mm_s"]):
        process.append(f"scan speed of {row['param_scan_speed_mm_s']} mm/s")

    if pd.notna(row["param_hatch_spacing_um"]):
        process.append(f"hatch spacing of {row['param_hatch_spacing_um']} μm")

    if pd.notna(row["param_layer_height_um"]):
        process.append(f"layer height of {row['param_layer_height_um']} μm")

    if pd.notna(row["param_energy_density_J_mm3"]):
        process.append(f"energy density of {row['param_energy_density_J_mm3']} J/mm³")

    if pd.notna(row["param_travel_speed_mm_min"]):
        process.append(f"travel speed of {row['param_travel_speed_mm_min']} mm/min")

    if pd.notna(row["param_wire_feed_speed_m_min"]):
        process.append(f"wire feed speed of {row['param_wire_feed_speed_m_min']} m/min")

    if pd.notna(row["param_current_A"]):
        process.append(f"current of {row['param_current_A']} A")

    if pd.notna(row["param_voltage_V"]):
        process.append(f"voltage of {row['param_voltage_V']} V")

    if pd.notna(row["param_heat_input_J_mm"]):
        process.append(f"heat input of {row['param_heat_input_J_mm']} J/mm")

    if pd.notna(row["param_interlayer_temp_C"]):
        process.append(f"interlayer temperature of {row['param_interlayer_temp_C']} °C")

    if pd.notna(row["param_interlayer_delay_s"]):
        process.append(f"interlayer delay of {row['param_interlayer_delay_s']} s")

    if process:
        parts.append("Processing parameters include " + ", ".join(process) + ".")

    # Shielding gas
    if pd.notna(row["param_shielding_gas"]):
        parts.append(f"{row['param_shielding_gas']} was used as the shielding gas.")

    if pd.notna(row["param_material_grade"]):
        parts.append(f"The material grade was {row['param_material_grade']}.")

    # Mechanical properties
    mech = []

    if pd.notna(row["mech_UTS_MPa"]):
        mech.append(f"ultimate tensile strength of {row['mech_UTS_MPa']} MPa")

    if pd.notna(row["mech_YS_MPa"]):
        mech.append(f"yield strength of {row['mech_YS_MPa']} MPa")

    if pd.notna(row["mech_hardness_HV"]):
        mech.append(f"hardness of {row['mech_hardness_HV']} HV")

    if pd.notna(row["mech_elongation_pct"]):
        mech.append(f"elongation of {row['mech_elongation_pct']}%")

    if mech:
        parts.append("The material exhibited " + ", ".join(mech) + ".")

    if pd.notna(row["mech_grain_size_um"]):
        parts.append(f"The average grain size was {row['mech_grain_size_um']} μm.")

    return " ".join(parts)


df["generated_caption"] = df.apply(build_caption, axis=1)

df.to_csv("matclip_phase1_cluster_caption.csv", index=False)
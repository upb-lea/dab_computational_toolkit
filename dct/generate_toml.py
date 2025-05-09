"""Generate default toml files."""
# python libraries
import os

def check_for_missing_toml_files(working_directory: str) -> None:
    """
    Check for missing toml default files. Generate them, if missing.

    :param working_directory: working directory
    :type working_directory: str
    """
    if not os.path.exists(working_directory):
        os.makedirs(working_directory)
    # check for all the toml files
    if not os.path.isfile(os.path.join(working_directory, "progFlow.toml")):
        generate_flow_control_toml(working_directory)
    if not os.path.isfile(os.path.join(working_directory, "DabCircuitConf.toml")):
        generate_circuit_toml(working_directory)
    if not os.path.isfile(os.path.join(working_directory, "DabInductorConf.toml")):
        generate_inductor_toml(working_directory)
    if not os.path.isfile(os.path.join(working_directory, "DabTransformerConf.toml")):
        generate_transformer_toml(working_directory)
    if not os.path.isfile(os.path.join(working_directory, "DabHeatSinkConf.toml")):
        generate_heat_sink_toml(working_directory)


def generate_flow_control_toml(working_directory: str) -> None:
    """
    Generate the default progFlow.toml file.

    :param working_directory: working directory
    :type working_directory: str
    """
    toml_data = f'''
    # Path configuration 
    [general]
        project_directory = "{working_directory}"
    
    [breakpoints]
        # possible values: no/pause/stop
        circuit_pareto = "no"    # After Electrical Pareto front calculation
        circuit_filtered = "no"  # After Electrical filtered result calculation
        inductor = "no"          # After inductor Pareto front calculations of for all correspondent electrical points
        transformer = "no"       # After transformer Pareto front calculations of for all correspondent electrical points
        heat_sink = "no"         # After heat sink Pareto front calculation
        summary = "no"           # After heat sink Pareto front calculation
    
    [conditional_breakpoints] # conditional breakpoints in case of bad definition array (only for experts and currently not implemented)
        circuit = 1000           # Number of trials with ZVS less than 70%
        inductor = 1000          # Number of trials which exceed a limit value?
        transformer = 1000       # Number of trials which exceed a limit value?
        heat_sink = 1000         # Number of trials which exceed a limit value?
    
    [circuit]
        number_of_trials = 100
        calculation_mode = "skip" # (new,continue,skip)
        subdirectory = "01_circuit"
    
    [inductor]
        number_of_trials = 100
        calculation_mode = "skip" # (new,continue,skip)
        subdirectory = "02_inductor"
    
    [transformer]
        number_of_trials = 100
        calculation_mode = "skip" # (new,continue,skip)
        subdirectory = "03_transformer"
    
    [heat_sink]
        number_of_trials = 100
        calculation_mode = "skip" # (new,continue,skip)
        subdirectory = "04_heat_sink"
    
    [summary]
        calculation_mode = "skip" # (new,skip)
        subdirectory = "05_summary"
    
    [configuration_data_files]
        circuit_configuration_file = "DabCircuitConf.toml"
        inductor_configuration_file = "DabInductorConf.toml"
        transformer_configuration_file = "DabTransformerConf.toml"
        heat_sink_configuration_file = "DabHeatSinkConf.toml"
    '''
    with open(f"{working_directory}/progFlow.toml", 'w') as output:
        output.write(toml_data)

def generate_circuit_toml(working_directory: str) -> None:
    """
    Generate the default DabCircuitConf.toml file.

    :param working_directory: working directory
    :type working_directory: str
    """
    toml_data = '''
    [design_space]
        f_s_min_max_list=[50e3, 300e3]
        l_s_min_max_list=[20e-6, 900e-6]
        l_1_min_max_list=[10e-6, 10e-3]
        l_2__min_max_list=[10e-6, 1e-3]
        n_min_max_list=[3, 7]
        transistor_1_name_list=['CREE_C3M0065100J', 'CREE_C3M0120100J']
        transistor_2_name_list=['CREE_C3M0060065J', 'CREE_C3M0120065J']
        c_par_1=16e-12
        c_par_2=16e-12
    
    [output_range]
        v_1_min_nom_max_list=[690, 700, 710]
        v_2_min_nom_max_list=[175, 235, 295]
        p_min_nom_max_list=[-2000, 2000, 2200]
        steps_per_direction=3
    
    [filter_distance]
        number_filtered_designs = 1
        difference_percentage = 5
    '''
    with open(f"{working_directory}/DabCircuitConf.toml", 'w') as output:
        output.write(toml_data)

def generate_inductor_toml(working_directory: str) -> None:
    """
    Generate the default DabInductorConf.toml file.

    :param working_directory: working directory
    :type working_directory: str
    """
    toml_data = '''
    [design_space]
        core_name_list=["PQ 50/50", "PQ 50/40", "PQ 40/40", "PQ 40/30", "PQ 35/35", "PQ 32/30", "PQ 32/20", "PQ 26/25", "PQ 26/20", "PQ 20/20", "PQ 20/16"]
        material_name_list=["3C95"]
        litz_wire_list=["1.5x105x0.1", "1.4x200x0.071", "1.1x60x0.1"]
        core_inner_diameter_list=[]
        window_h_list=[]
        window_w_list=[]
    
    [boundary_conditions]
        temperature=100
    
    [material_data_sources]
        permeability_datasource=""
        permeability_datatype=""
        permeability_measurement_setup=""
        permittivity_datasource=""
        permittivity_datatype=""
        permittivity_measurement_setup=""
    
    [insulations]
        primary_to_primary=0.2e-3
        core_bot=1e-3
        core_top=1e-3
        core_right=1e-3
        core_left=1e-3
    
    [filter_distance]
        factor_min_dc_losses = 0.01
        factor_max_dc_losses = 100
    '''
    with open(f"{working_directory}/DabInductorConf.toml", 'w') as output:
        output.write(toml_data)

def generate_transformer_toml(working_directory: str) -> None:
    """
    Generate the default DabTransformerConf.toml file.

    :param working_directory: working directory
    :type working_directory: str
    """
    toml_data = '''
    [design_space]
        material_name_list=['3C95']
        core_name_list=["PQ 40/40", "PQ 40/30", "PQ 35/35", "PQ 32/30", "PQ 32/20", "PQ 26/25", "PQ 26/20", "PQ 20/20", "PQ 20/16"]
        core_inner_diameter_min_max_list=[15e-3, 30e-3]
        window_w_min_max_list=[10e-3, 40e-3]
        window_h_bot_min_max_list=[10e-3, 50e-3]
        primary_litz_wire_list=['1.1x60x0.1']
        secondary_litz_wire_list=['1.35x200x0.071', '1.1x60x0.1']
        # sweep parameters: geometry and materials
        n_p_top_min_max_list=[1, 30]
        n_p_bot_min_max_list=[10, 80]
    
    [boundary_conditions]
        # maximum limitation for transformer total height and core volume
        max_transformer_total_height=60e-3
        max_core_volume=0.007853982  # 50e-3 ** 2 * 3.141593
        temperature=100
    
    [insulation]
        # insulation for top core window
        iso_window_top_core_top=1.3e-3
        iso_window_top_core_bot=1.3e-3
        iso_window_top_core_left=1.3e-3
        iso_window_top_core_right=1.3e-3
        # insulation for bottom core window
        iso_window_bot_core_top=1.3e-3
        iso_window_bot_core_bot=1.3e-3
        iso_window_bot_core_left=1.3e-3
        iso_window_bot_core_right=1.3e-3
        # winding-to-winding insulation
        iso_primary_to_primary=0.2e-3
        iso_secondary_to_secondary=0.2e-3
        iso_primary_to_secondary=0.2e-3
    
    [settings]
        fft_filter_value_factor=0.01
        mesh_accuracy=0.8
    
    [filter_distance]
        factor_min_dc_losses=0.01
        factor_max_dc_losses=100
    '''
    with open(f"{working_directory}/DabTransformerConf.toml", 'w') as output:
        output.write(toml_data)

def generate_heat_sink_toml(working_directory: str) -> None:
    """
    Generate the default DabHeatSinkConf.toml file.

    :param working_directory: working directory
    :type working_directory: str
    """
    toml_data = '''
    [design_space]
        height_c_list=[0.02, 0.08]
        width_b_list=[0.02, 0.08]
        length_l_list=[0.08, 0.20]
        height_d_list=[0.001, 0.003]
        number_fins_n_list=[5, 20]
        thickness_fin_t_list=[1e-3, 5e-3]
    
    [boundary_conditions]
        t_ambient=40
        area_min=0.001
    
    [settings]
        number_directions=3
        factor_pcb_area_copper_coin = 1.42
        factor_bottom_area_copper_coin = 0.39
        # W/(m*K)
        thermal_conductivity_copper = 136
    
    [thermal_resistance_data]
        # [tim_thickness, tim_conductivity]
        transistor_b1_cooling = [1e-3,12.0]
        transistor_b2_cooling = [1e-3,12.0]
        inductor_cooling = [1e-3,12.0]
        transformer_cooling = [1e-3,12.0]
        # [t_ambient, t_hs_max] in °C
        heat_sink = [40.0, 90.0]

    '''
    with open(f"{working_directory}/DabHeatSinkConf.toml", 'w') as output:
        output.write(toml_data)

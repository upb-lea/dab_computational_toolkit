
# python libraries
import os
import pickle

#3rd party libraries
import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# import own libraries
from femmt.optimization.io import InductorOptimization
from femmt.optimization.sto import StackedTransformerOptimization
from femmt.optimization.io_dtos import InductorInsulationDTO, InductorOptimizationTargetAndFixedParameters, IoFemInput, IoReluctanceModelInput, IoReluctanceModelOutput
from femmt.optimization.sto_dtos import StoTargetAndFixedParameters, StoInsulation, StoFemInput, StoFemOutput, StoReluctanceModelInput, StoReluctanceModelOutput
import femmt.functions as ff
from dct.topology.dab.dab_datasets import HandleDabDto
from dct.components.component_dtos import InductorResults, StackedTransformerResults
from dct.constant_path import CIRCUIT_WAVEFORMS_FOLDER, FILTERED_RESULTS_PATH, GECKO_COMPONENT_MODELS_DIRECTORY
from dct.topology.dab.dab_circuit_topology_dtos import CircuitSampling
import magnethub as mh
import materialdatabase as mdb
import femmt.functions_reluctance as fr
import femmt.optimization.ito_functions as itof
from femmt import MaterialDataSources
from dct.generalplotsettings import colors

# file paths
#workspace_name = "workspace_vijay"
workspace_name = "workspace"
#project_name = "2026-02-20_design_vijay_meshgrid"
#project_name = "2026-03-13"
project_name = "2026-03-31"

DTO_FILEPATH = f"/home/nikolasf/Dokumente/01_git/30_Python/dab_computational_toolkit/{workspace_name}/{project_name}"
DTO_FILE = "0.pkl"
IND_RECALCULATION_FILEPATH = f"/home/nikolasf/Dokumente/01_git/30_Python/dab_computational_toolkit/{workspace_name}/recalculation/inductor"
STO_RECALCULATION_FILEPATH = f"/home/nikolasf/Dokumente/01_git/30_Python/dab_computational_toolkit/{workspace_name}/recalculation/transformer"
IND_FEMMT_WORKSPACE = f"/home/nikolasf/Dokumente/01_git/30_Python/dab_computational_toolkit/{workspace_name}/recalculation/inductor/00_femmt_simulation/process_1"
STO_FEMMT_WORKSPACE = f"/home/nikolasf/Dokumente/01_git/30_Python/dab_computational_toolkit/{workspace_name}/recalculation/transformer/00_femmt_simulation/process_1"
CIRCUIT_WORKSPACE = f"/home/nikolasf/Dokumente/01_git/30_Python/dab_computational_toolkit/{workspace_name}/recalculation/circuit"


OPERATING_POINT_NUMBER = 2
IS_IND_FEM = False
IS_STO_FEM = False
show_visual_outputs = False

# inductor parameters
ind_litz_wire_name = "1.1x60x0.1"
ind_material_name = mdb.Material._3C95
ind_core_inner_diameter = 14.35e-3
ind_window_w = 8.825e-3
ind_window_h = 18.9e-3
ind_turns = 52
ind_target_inductance = 553e-6
ind_temperature = 80
ind_insulations = InductorInsulationDTO(
    core_left=1e-3,
    core_right=1e-3,
    core_top=1e-3,
    core_bot=1e-3,
    primary_to_primary=0.2e-3)

# transformer parameters

# target parameters, which are the real parameters
sto_target_n = 3.974
sto_target_ls =137e-6
sto_target_lh =615e-6
# sto_target_n = 4.2
# sto_target_ls =125e-6
# sto_target_lh =670e-6

sto_target_inductance_matrix = [[sto_target_ls + sto_target_lh, sto_target_lh / sto_target_n], [sto_target_lh / sto_target_n, sto_target_lh / (sto_target_n ** 2)]]

sto_core_inner_diameter=14.9e-3
sto_window_w=11.05e-3
sto_window_h_bot=19.04e-3
sto_window_h_top=8.2e-3
sto_turns_1_top=24
sto_turns_1_bot=38
sto_turns_2_bot=9
sto_litz_wire_name_1="1.1x60x0.1"
sto_litz_wire_name_2="1.4x200x0.071"

sto_material_name=mdb.Material._3C95

sto_temperature=80

sto_insulations = StoInsulation(

    # insulation for top core window
    iso_window_top_core_top=1e-3,
    iso_window_top_core_bot=1e-3,
    iso_window_top_core_left=1e-3,
    iso_window_top_core_right=1e-3,
    # insulation for bottom core window
    iso_window_bot_core_top=1e-3,
    iso_window_bot_core_bot=1e-3,
    iso_window_bot_core_left=1e-3,
    iso_window_bot_core_right=1e-3,
    # winding-to-winding insulation
    iso_primary_to_primary=0.2e-3,
    iso_secondary_to_secondary=0.2e-3,
    iso_primary_to_secondary=0.4e-3,
)


# inductor calculate fix parameters
def calculate_ind_fix_parameters(time_current_vec, material_name_list, inductor_optimization_directory):

    """Calculate fix parameters what can be derived from the input configuration.

    return values are:

        i_rms
        i_rms_2
        time_extracted_vec
        current_extracted_vec
        current_extracted_2_vec
        material_mu_r_abs_list
        fundamental_frequency
        target_inductance_matrix
        fem_working_directory
        fem_simulation_results_directory
        reluctance_model_results_directory
        fem_thermal_simulation_results_directory

    :param config: configuration file
    :type config: ItoSingleInputConfig
    :return: calculated target and fix parameters
    :rtype: ItoTargetAndFixedParameters
    """
    # currents
    time_extracted, current_extracted_vec = fr.time_vec_current_vec_from_time_current_vec(time_current_vec)
    fundamental_frequency = 1 / time_extracted[-1]

    i_rms = fr.i_rms(time_current_vec)

    i_peak, = fr.max_value_from_value_vec(current_extracted_vec)

    (fft_frequencies, fft_amplitudes, fft_phases) = ff.fft(
        period_vector_t_i=time_current_vec, sample_factor=1000, plot='no', mode='time', filter_type='factor', filter_value_factor=0.03)

    # material properties
    material_db = mdb.Data()

    material_mu_r_abs_list = []
    magnet_model_list = []
    for material_name in material_name_list:
        small_signal_mu_real_over_f_at_T = material_db.get_datasheet_curve(material_name, mdb.DatasheetCurveType.small_signal_mu_real_over_f_at_T)
        small_signal_mu_imag_over_f_at_T = material_db.get_datasheet_curve(material_name, mdb.DatasheetCurveType.small_signal_mu_imag_over_f_at_T)

        mu_real_at_f = np.interp(10_000, small_signal_mu_real_over_f_at_T["f"], small_signal_mu_real_over_f_at_T["mu_real"])
        mu_imag_at_f = np.interp(10_000, small_signal_mu_imag_over_f_at_T["f"], small_signal_mu_imag_over_f_at_T["mu_imag"])

        material_mu_r_abs = np.abs(np.array([mu_real_at_f + 1j * mu_imag_at_f]))
        material_mu_r_abs_list.append(material_mu_r_abs)
        # instantiate material-specific model
        mdl: mh.loss.LossModel = mh.loss.LossModel(material=material_name, team="paderborn")
        magnet_model_list.append(mdl)

    # set up working directories
    working_directories = itof.set_up_folder_structure(inductor_optimization_directory)

    # finalize data to dto
    target_and_fix_parameters = InductorOptimizationTargetAndFixedParameters(
        i_rms=i_rms,
        i_peak=i_peak,
        time_extracted_vec=time_extracted,
        current_extracted_vec=current_extracted_vec,
        material_name_list=material_name_list,
        material_mu_r_abs_list=material_mu_r_abs_list,
        magnet_hub_model_list=magnet_model_list,
        fundamental_frequency=fundamental_frequency,
        working_directories=working_directories,
        fft_frequency_list=fft_frequencies,
        fft_amplitude_list=fft_amplitudes,
        fft_phases_list=fft_phases
    )

    return target_and_fix_parameters

# transformer calculate fix parameters
def calculate_sto_fix_parameters(time_current_1_vec, time_current_2_vec, l_s12_target, l_h_target, n_target, material_list, stacked_transformer_optimization_directory):
    """Calculate fix parameters what can be derived from the input configuration.

    return values are:

        i_rms_1
        i_rms_2
        time_extracted_vec
        current_extracted_1_vec
        current_extracted_2_vec
        material_mu_r_abs_list
        fundamental_frequency
        target_inductance_matrix
        fem_working_directory
        fem_simulation_results_directory
        reluctance_model_results_directory
        fem_thermal_simulation_results_directory

    :param config: configuration file
    :type config: ItoSingleInputConfig
    :return: calculated target and fix parameters
    :rtype: ItoTargetAndFixedParameters
    """
    # currents
    time_extracted, current_extracted_1_vec = fr.time_vec_current_vec_from_time_current_vec(
        time_current_1_vec)
    time_extracted, current_extracted_2_vec = fr.time_vec_current_vec_from_time_current_vec(
        time_current_2_vec)
    fundamental_frequency = 1 / time_extracted[-1]

    # plt.plot(time_extracted, current_extracted_2_vec)
    # plt.show()


    i_rms_1 = fr.i_rms(time_current_1_vec)
    i_rms_2 = fr.i_rms(time_current_2_vec)

    i_peak_1, i_peak_2 = fr.max_value_from_value_vec(current_extracted_1_vec, current_extracted_2_vec)
    phi_deg_1, phi_deg_2 = fr.phases_deg_from_time_current(time_extracted, current_extracted_1_vec, current_extracted_2_vec)
    # print(f"{i_peak_1=}")
    # print(f"{i_peak_2=}")
    # print(f"{phi_deg_1=}")
    # print(f"{phi_deg_2=}")
    time_sine = np.linspace(0, time_extracted[-1], 50)
    current_1_sine = i_peak_1 * np.cos(time_sine * 2 * np.pi / time_sine[-1] - phi_deg_1 * 2 * np.pi / 360)
    current_2_sine = i_peak_2 * np.cos(time_sine * 2 * np.pi / time_sine[-1] - phi_deg_2 * 2 * np.pi / 360)

    # plt.plot(time_extracted, current_extracted_1_vec, label="current 1", color="orange")
    # plt.plot(time_extracted, current_extracted_2_vec, label="current 2", color="blue")
    # plt.plot(time_sine, current_1_sine, label="current 1 sine", color="orange", linestyle="--")
    # plt.plot(time_sine, current_2_sine, label="current 2 sine", color="blue", linestyle="--")
    # plt.grid()
    # plt.show()

    # phi_deg_2 = phi_deg_2 - 180

    # target inductances
    target_inductance_matrix = fr.calculate_inductance_matrix_from_ls_lh_n(l_s12_target,
                                                                           l_h_target,
                                                                           n_target)

    (fft_frequencies_1, fft_amplitudes_1, fft_phases_1) = ff.fft(
        period_vector_t_i=time_current_1_vec, sample_factor=1000, plot='no', mode='time', filter_type='factor', filter_value_factor=0.03)

    (fft_frequencies_2, fft_amplitudes_2, fft_phases_2) = ff.fft(
        period_vector_t_i=time_current_2_vec, sample_factor=1000, plot='no', mode='time', filter_type='factor', filter_value_factor=0.03)

    # material properties
    material_db = mdb.Data()

    material_mu_r_abs_list = []
    magnet_model_list = []
    for material_name in material_list:
        small_signal_mu_real_over_f_at_T = material_db.get_datasheet_curve(material_name, mdb.DatasheetCurveType.small_signal_mu_real_over_f_at_T)
        small_signal_mu_imag_over_f_at_T = material_db.get_datasheet_curve(material_name, mdb.DatasheetCurveType.small_signal_mu_imag_over_f_at_T)

        mu_real_at_f = np.interp(10_000, small_signal_mu_real_over_f_at_T["f"], small_signal_mu_real_over_f_at_T["mu_real"])
        mu_imag_at_f = np.interp(10_000, small_signal_mu_imag_over_f_at_T["f"], small_signal_mu_imag_over_f_at_T["mu_imag"])

        material_mu_r_abs = np.abs(np.array([mu_real_at_f + 1j * mu_imag_at_f]))
        material_mu_r_abs_list.append(material_mu_r_abs)

        # instantiate material-specific model
        mdl: mh.loss.LossModel = mh.loss.LossModel(material=material_name, team="paderborn")
        magnet_model_list.append(mdl)

    # set up working directories
    working_directories = itof.set_up_folder_structure(stacked_transformer_optimization_directory)

    # finalize data to dto
    target_and_fix_parameters = StoTargetAndFixedParameters(
        i_rms_1=i_rms_1,
        i_rms_2=i_rms_2,
        i_peak_1=i_peak_1,
        i_peak_2=i_peak_2,
        i_phase_deg_1=phi_deg_1,
        i_phase_deg_2=phi_deg_2,
        time_extracted_vec=time_extracted,
        magnet_hub_model_list=magnet_model_list,
        current_extracted_1_vec=current_extracted_1_vec,
        current_extracted_2_vec=current_extracted_2_vec,
        material_name_list=material_list,
        material_complex_mu_r_list=material_mu_r_abs_list,
        fundamental_frequency=fundamental_frequency,
        target_inductance_matrix=target_inductance_matrix,
        working_directories=working_directories,
        # winding 1
        fft_frequency_list_1=fft_frequencies_1,
        fft_amplitude_list_1=fft_amplitudes_1,
        fft_phases_list_1=fft_phases_1,

        # winding 2
        fft_frequency_list_2=fft_frequencies_2,
        fft_amplitude_list_2=fft_amplitudes_2,
        fft_phases_list_2=fft_phases_2
    )

    return target_and_fix_parameters





def generate_inductor_data():
    operating_point_number = 0
    # iterate over operating points

    # load circuit calculation
    dto = HandleDabDto.load_from_file(f"{DTO_FILEPATH}/01_circuit/VijayCircuit/{FILTERED_RESULTS_PATH}/{DTO_FILE}")

    inductor_core_loss_reluctance = np.full_like(dto.input_config.mesh_v1, np.nan)
    inductor_core_loss_fem_sine = np.full_like(dto.input_config.mesh_v1, np.nan)
    inductor_core_loss_fem_magnet = np.full_like(dto.input_config.mesh_v1, np.nan)
    inductor_winding_loss_reluctance = np.full_like(dto.input_config.mesh_v1, np.nan)
    inductor_winding_loss_fem = np.full_like(dto.input_config.mesh_v1, np.nan)

    for vec_vvp in np.ndindex(dto.calc_modulation.phi.shape):
        print(f"{operating_point_number=}")
        print(f"{dto.input_config.mesh_v1[vec_vvp]=}")
        print(f"{dto.input_config.mesh_v2[vec_vvp]=}")
        print(f"{dto.input_config.mesh_p[vec_vvp]=}")

        #if operating_point_number == OPERATING_POINT_NUMBER:
        if True:
            ind_time = dto.component_requirements.inductor_requirements[0].time_array[vec_vvp]
            ind_current = dto.component_requirements.inductor_requirements[0].current_array[vec_vvp]
            ind_time_current_vec = [ind_time, ind_current]
            ind_target_and_fixed_parameters = calculate_ind_fix_parameters(ind_time_current_vec, [ind_material_name], IND_RECALCULATION_FILEPATH)

            for count, ind_material_mu_r_abs_value in enumerate(ind_target_and_fixed_parameters.material_mu_r_abs_list):
                if ind_target_and_fixed_parameters.material_name_list[count] == ind_material_name:
                    material_mu_r_abs = ind_material_mu_r_abs_value
                    magnet_material_model = ind_target_and_fixed_parameters.magnet_hub_model_list[count]

            ind_litz_wire = ff.litz_database()[ind_litz_wire_name]
            ind_litz_wire_diameter = 2 * ind_litz_wire["conductor_radii"]

            ind_reluctance_model_input = IoReluctanceModelInput(
                core_inner_diameter=ind_core_inner_diameter,
                window_w=ind_window_w,
                window_h=ind_window_h,
                turns=ind_turns,
                target_inductance=ind_target_inductance,
                litz_wire_name=ind_litz_wire_name,
                litz_wire_diameter=ind_litz_wire_diameter,
                insulations=ind_insulations,
                material_mu_r_abs=material_mu_r_abs,
                magnet_material_model=magnet_material_model,
                temperature=ind_temperature,
                time_extracted_vec=ind_target_and_fixed_parameters.time_extracted_vec,
                current_extracted_vec=ind_target_and_fixed_parameters.current_extracted_vec,
                fundamental_frequency=ind_target_and_fixed_parameters.fundamental_frequency,
                fft_frequency_list=ind_target_and_fixed_parameters.fft_frequency_list,
                fft_amplitude_list=ind_target_and_fixed_parameters.fft_amplitude_list,
                fft_phases_list=ind_target_and_fixed_parameters.fft_phases_list
            )

            ind_reluctance_output: IoReluctanceModelOutput = InductorOptimization.ReluctanceModel.single_reluctance_model_simulation(ind_reluctance_model_input)
            print(f"{ind_reluctance_output.l_air_gap=}")

            # collect inductor reluctance model results
            inductor_core_loss_reluctance[vec_vvp] = ind_reluctance_output.p_hyst
            inductor_winding_loss_reluctance[vec_vvp] = ind_reluctance_output.p_winding

            if not os.path.exists(IND_FEMMT_WORKSPACE):
                os.mkdir(IND_FEMMT_WORKSPACE)

            # material properties
            material_db = mdb.Data()
            small_signal_mu_real_over_f_at_T = material_db.get_datasheet_curve(ind_material_name, mdb.DatasheetCurveType.small_signal_mu_real_over_f_at_T)
            small_signal_mu_imag_over_f_at_T = material_db.get_datasheet_curve(ind_material_name, mdb.DatasheetCurveType.small_signal_mu_imag_over_f_at_T)

            mu_real_at_f = np.interp(10_000, small_signal_mu_real_over_f_at_T["f"], small_signal_mu_real_over_f_at_T["mu_real"])
            mu_imag_at_f = np.interp(10_000, small_signal_mu_imag_over_f_at_T["f"], small_signal_mu_imag_over_f_at_T["mu_imag"])

            material_mu_r_abs = np.abs(np.array([mu_real_at_f + 1j * mu_imag_at_f]))

            ind_fem_input = IoFemInput(
                # general parameters
                working_directory=IND_FEMMT_WORKSPACE,
                simulation_name='xx',

                # material and geometry parameters
                material_name=ind_material_name,
                litz_wire_name=ind_litz_wire_name,
                core_inner_diameter=ind_core_inner_diameter,
                window_w=ind_window_w,
                window_h=ind_window_h,
                air_gap_length=ind_reluctance_output.l_air_gap - 0.1e-3,
                turns=ind_turns,
                insulations=ind_insulations,

                # data sources
                material_data_sources=MaterialDataSources(permeability_datasource=mdb.DataSource.LEA_MTB, permittivity_datasource=mdb.DataSource.LEA_MTB),

                # operating point conditions
                temperature=ind_temperature,
                fundamental_frequency=ind_target_and_fixed_parameters.fundamental_frequency,
                fft_frequency_list=ind_target_and_fixed_parameters.fft_frequency_list,
                fft_amplitude_list=ind_target_and_fixed_parameters.fft_amplitude_list,
                fft_phases_list=ind_target_and_fixed_parameters.fft_phases_list,

                time_vec=ind_time,
                current_vec=ind_current
            )



            if IS_IND_FEM:
                ind_fem_output = InductorOptimization.FemSimulation.single_fem_simulation(ind_fem_input, False)

                inductor_core_loss_fem_sine[vec_vvp] = ind_fem_output.p_core_sine
                inductor_core_loss_fem_magnet[vec_vvp] = ind_fem_output.p_core_magnet
                inductor_winding_loss_fem[vec_vvp] = ind_fem_output.p_loss_winding

                print(f"Inductance reluctance: {ind_target_inductance}")
                print(f"Inductance FEM: {ind_fem_output.inductance}")
                print(f"Inductance derivation: "
                            f"{(ind_fem_output.inductance - ind_target_inductance) / ind_target_inductance * 100} %")
                print(f"Volume reluctance: {ind_reluctance_output.volume}")
                print(f"Volume FEM: {ind_fem_output.volume}")
                print(f"Volume derivation: {(ind_reluctance_output.volume - ind_fem_output.volume) / ind_reluctance_output.volume * 100} %")
                print(f"P_winding reluctance: {ind_reluctance_output.p_winding}")
                print(f"P_winding FEM: {ind_fem_output.p_loss_winding}")
                print(f"P_winding derivation: {(ind_fem_output.p_loss_winding - ind_reluctance_output.p_winding) / ind_fem_output.p_loss_winding * 100} %")
                print(f"P_hyst reluctance: {ind_reluctance_output.p_hyst}")
                print(f"P_hyst FEM (from magnet per cell in FEM simulation): {ind_fem_output.p_core_magnet}")
                print(f"P_hyst FEM (from sine wave): {ind_fem_output.p_core_sine}")
                print(f"P_hyst derivation (magnet): {(ind_reluctance_output.p_hyst - ind_fem_output.p_core_magnet) / ind_reluctance_output.p_hyst * 100} %")
                print(f"P_hyst derivation (sine): {(ind_reluctance_output.p_hyst - ind_fem_output.p_core_sine) / ind_reluctance_output.p_hyst * 100} %")

        operating_point_number += 1

    inductor_results_reluctance = InductorResults(
    circuit_id="", inductor_id="", inductor_number_in_circuit=0,
    loss_array=inductor_winding_loss_reluctance+inductor_core_loss_reluctance,
    winding_loss_array=inductor_winding_loss_reluctance,
    core_loss_array=inductor_core_loss_reluctance,
    volume=0, area_to_heat_sink=0, r_th_ind_heat_sink=0)
    inductor_results_fem = InductorResults(
    circuit_id="", inductor_id="", inductor_number_in_circuit=0,
    loss_array=inductor_winding_loss_fem+inductor_core_loss_fem_magnet,
    winding_loss_array=inductor_winding_loss_fem,
    core_loss_array=inductor_core_loss_fem_magnet,
    volume=0, area_to_heat_sink=0, r_th_ind_heat_sink=0)

    with open(f"{IND_RECALCULATION_FILEPATH}/inductor_results_reluctance.pkl", 'wb') as output:
        pickle.dump(inductor_results_reluctance, output, pickle.HIGHEST_PROTOCOL)
    if IS_IND_FEM:
        with open(f"{IND_RECALCULATION_FILEPATH}/inductor_results_fem.pkl", 'wb') as output:
            pickle.dump(inductor_results_fem, output, pickle.HIGHEST_PROTOCOL)


def generate_transformer_data():
    operating_point_number = 0
    # iterate over operating points

    # load circuit calculation
    dto = HandleDabDto.load_from_file(f"{DTO_FILEPATH}/01_circuit/VijayCircuit/{FILTERED_RESULTS_PATH}/{DTO_FILE}")

    # dto = HandleDabDto.init_config(name=name, mesh_v1=mesh_v1, mesh_v2=mesh_v2, mesh_p=mesh_p,
    #                 sampling=sampling, n=n, ls=ls, lc1=lc1, lc2=lc2, fs=fs,
    #                 transistor_dto_1=transistor_dto_1, transistor_dto_2=transistor_dto_2,
    #                 lossfilepath=lossfilepath, c_par_1=c_par_1, c_par_2=c_par_2, t_dead_1_max=t_dead_1_max, t_dead_2_max=t_dead_2_max)

    transformer_core_loss_reluctance = np.full_like(dto.input_config.mesh_v1, np.nan)
    transformer_core_loss_fem_sine = np.full_like(dto.input_config.mesh_v1, np.nan)
    transformer_winding_1_loss_reluctance = np.full_like(dto.input_config.mesh_v1, np.nan)
    transformer_winding_2_loss_reluctance = np.full_like(dto.input_config.mesh_v1, np.nan)
    transformer_winding_1_loss_fem = np.full_like(dto.input_config.mesh_v1, np.nan)
    transformer_winding_2_loss_fem = np.full_like(dto.input_config.mesh_v1, np.nan)
    transformer_current_amplitude_1 = np.full_like(dto.input_config.mesh_v1, np.nan)
    transformer_current_amplitude_2 = np.full_like(dto.input_config.mesh_v1, np.nan)
    transformer_phase_deg_1 = np.full_like(dto.input_config.mesh_v1, np.nan)
    transformer_phase_deg_2= np.full_like(dto.input_config.mesh_v1, np.nan)

    for vec_vvp in np.ndindex(dto.calc_modulation.phi.shape):
        print(f"{operating_point_number=}")
        print(f"{dto.input_config.mesh_v1[vec_vvp]=}")
        print(f"{dto.input_config.mesh_v2[vec_vvp]=}")
        print(f"{dto.input_config.mesh_p[vec_vvp]=}")

        # if operating_point_number == OPERATING_POINT_NUMBER:
        if True:

            #### transformer
            sto_time = dto.component_requirements.transformer_requirements[0].time_array[vec_vvp]
            sto_current_1 = dto.component_requirements.transformer_requirements[0].current_1_array[vec_vvp]
            sto_current_2 = dto.component_requirements.transformer_requirements[0].current_2_array[vec_vvp]
            sto_time_current_vec_1 = [sto_time, sto_current_1]
            sto_time_current_vec_2 = [sto_time, sto_current_2]
            sto_target_and_fixed_parameters = calculate_sto_fix_parameters(sto_time_current_vec_1, sto_time_current_vec_2, sto_target_ls, sto_target_lh,
                                                                           sto_target_n, [sto_material_name], STO_RECALCULATION_FILEPATH)

            for count, ind_material_mu_r_abs_value in enumerate(sto_target_and_fixed_parameters.material_complex_mu_r_list):
                if sto_target_and_fixed_parameters.material_name_list[count] == ind_material_name:
                    material_mu_r_abs = ind_material_mu_r_abs_value
                    magnet_material_model = sto_target_and_fixed_parameters.magnet_hub_model_list[count]

            sto_litz_wire_1 = ff.litz_database()[sto_litz_wire_name_1]
            sto_litz_wire_diameter_1 = 2 * sto_litz_wire_1["conductor_radii"]
            sto_litz_wire_2 = ff.litz_database()[sto_litz_wire_name_2]
            sto_litz_wire_diameter_2 = 2 * sto_litz_wire_2["conductor_radii"]

            # plt.plot(sto_time_current_vec_1[0], sto_time_current_vec_1[1])
            # plt.plot(sto_time_current_vec_2[0], sto_time_current_vec_2[1])
            # plt.show()

            sto_time_high_resolution=np.linspace(sto_time_current_vec_1[0][0], sto_time_current_vec_1[0][-1])
            sto_current_1_high_resolution = np.interp(sto_time_high_resolution, sto_time_current_vec_1[0], sto_time_current_vec_1[1])
            sto_current_2_high_resolution = np.interp(sto_time_high_resolution, sto_time_current_vec_2[0], sto_time_current_vec_2[1])

            sto_reluctance_model_input = StoReluctanceModelInput(
                target_inductance_matrix=sto_target_inductance_matrix,
                core_inner_diameter=sto_core_inner_diameter,
                window_w=sto_window_w,
                window_h_bot=sto_window_h_bot,
                window_h_top=sto_window_h_top,
                turns_1_top=sto_turns_1_top,
                turns_1_bot=sto_turns_1_bot,
                turns_2_bot=sto_turns_2_bot,
                litz_wire_name_1=sto_litz_wire_name_1,
                litz_wire_diameter_1=sto_litz_wire_diameter_1,
                litz_wire_name_2=sto_litz_wire_name_2,
                litz_wire_diameter_2=sto_litz_wire_diameter_2,

                insulations=sto_insulations,
                material_name=sto_material_name,
                material_mu_r_abs=material_mu_r_abs,
                magnet_material_model=magnet_material_model,

                temperature=sto_temperature,
                time_extracted_vec=sto_time_current_vec_1[0],
                current_extracted_vec_1=sto_time_current_vec_1[1],
                current_extracted_vec_2=sto_time_current_vec_2[1],
                fundamental_frequency=sto_target_and_fixed_parameters.fundamental_frequency,

                i_rms_1=sto_target_and_fixed_parameters.i_rms_1,
                i_rms_2=sto_target_and_fixed_parameters.i_rms_2,

                primary_litz_dict=sto_litz_wire_1,
                secondary_litz_dict=sto_litz_wire_2,
                # # winding 1
                fft_frequency_list_1=sto_target_and_fixed_parameters.fft_frequency_list_1,
                fft_amplitude_list_1=sto_target_and_fixed_parameters.fft_amplitude_list_1,
                fft_phases_list_1=sto_target_and_fixed_parameters.fft_phases_list_1,
                #
                # # winding 2
                fft_frequency_list_2=sto_target_and_fixed_parameters.fft_frequency_list_2,
                fft_amplitude_list_2=sto_target_and_fixed_parameters.fft_amplitude_list_2,
                fft_phases_list_2=sto_target_and_fixed_parameters.fft_phases_list_2,
            )
            print(f"{sto_reluctance_model_input=}")

            sto_reluctance_output: StoReluctanceModelOutput = StackedTransformerOptimization.ReluctanceModel.single_reluctance_model_simulation(sto_reluctance_model_input)
            print(f"{sto_reluctance_output=}")
            print(f"{sto_reluctance_output.l_top_air_gap=}")
            print(f"{sto_reluctance_output.l_bot_air_gap=}")
            transformer_core_loss_reluctance[vec_vvp] = sto_reluctance_output.p_hyst
            transformer_winding_1_loss_reluctance[vec_vvp] = sto_reluctance_output.winding_1_loss
            transformer_winding_2_loss_reluctance[vec_vvp] = sto_reluctance_output.winding_2_loss


            sto_fem_input=StoFemInput(

                # general parameters
                working_directory=STO_FEMMT_WORKSPACE,
                simulation_name="",

                # material and geometry parameters
                material_name=sto_material_name,
                primary_litz_wire_name=sto_litz_wire_name_1,
                secondary_litz_wire_name=sto_litz_wire_name_2,
                core_inner_diameter=sto_core_inner_diameter,
                window_w=sto_window_w,
                window_h_top=sto_window_h_top,
                window_h_bot=sto_window_h_bot,
                air_gap_length_top=sto_reluctance_output.l_top_air_gap,
                air_gap_length_bot=sto_reluctance_output.l_bot_air_gap,
                turns_primary_top=sto_turns_1_top,
                turns_primary_bot=sto_turns_1_bot,
                turns_secondary_bot=sto_turns_2_bot,
                insulations=sto_insulations,

                # data sources
                material_data_sources=MaterialDataSources(permeability_datasource=mdb.DataSource.LEA_MTB, permittivity_datasource=mdb.DataSource.LEA_MTB),

                # operating point conditions
                temperature=sto_temperature,
                fundamental_frequency=sto_target_and_fixed_parameters.fundamental_frequency,
                time_current_1_vec=[sto_time_high_resolution, sto_current_1_high_resolution],
                time_current_2_vec=[sto_time_high_resolution, sto_current_2_high_resolution]
            )

            if IS_STO_FEM:
                sto_fem_output: StoFemOutput = StackedTransformerOptimization.FemSimulation.single_fem_simulation(sto_fem_input, show_visual_outputs)

                transformer_core_loss_fem_sine[vec_vvp] = sto_fem_output.p_core_sine
                transformer_winding_1_loss_fem[vec_vvp] = sto_fem_output.p_loss_winding_1
                transformer_winding_2_loss_fem[vec_vvp] = sto_fem_output.p_loss_winding_2
                transformer_current_amplitude_1[vec_vvp] = sto_fem_output.study_excitation["hysteresis"]["transformer"]["current_amplitudes"][0]
                transformer_current_amplitude_2[vec_vvp] = sto_fem_output.study_excitation["hysteresis"]["transformer"]["current_amplitudes"][1]
                transformer_phase_deg_1[vec_vvp] = sto_fem_output.study_excitation["hysteresis"]["transformer"]["current_phases_deg"][0]
                transformer_phase_deg_2[vec_vvp] = sto_fem_output.study_excitation["hysteresis"]["transformer"]["current_phases_deg"][1]





                print(f"{sto_fem_output=}")
                print(f"Stacked Transformer target inductance matrix: {sto_target_inductance_matrix}")
                print(f"Stacked n target: {sto_target_n}")
                print(f"Stacked l_s target: {sto_target_ls}")
                print(f"Stacked l_h target: {sto_target_lh}")
                print(f"Stacked Transformer FEM n: {sto_fem_output.n_conc}")
                print(f"Stacked Transformer FEM l_s: {sto_fem_output.l_s_conc}")
                print(f"Stacked Transformer FEM l_h: {sto_fem_output.l_h_conc}")
                print(f"Stacked Transformer n derivation: {(sto_target_n - sto_fem_output.n_conc) / sto_target_n * 100} %")
                print(f"Stacked Transformer ls derivation: {(sto_target_ls - sto_fem_output.l_s_conc) / sto_target_ls * 100} %")
                print(f"Stacked Transformer lh derivation: {(sto_target_lh - sto_fem_output.l_h_conc) / sto_target_lh * 100} %")
                print(f"Volume reluctance: {sto_reluctance_output.volume}")
                print(f"Volume FEM: {sto_fem_output.volume}")
                print(f"Volume derivation: {(sto_reluctance_output.volume - sto_fem_output.volume) / sto_reluctance_output.volume * 100} %")
                print(f"P_winding_1 reluctance: {sto_reluctance_output.winding_1_loss} W")
                print(f"P_winding_2 reluctance: {sto_reluctance_output.winding_2_loss} W")
                print(f"P_winding_1 FEM: {sto_fem_output.p_loss_winding_1} W")
                print(f"P_winding_2 FEM: {sto_fem_output.p_loss_winding_2} W")
                print(f"P_winding_1 derivation: {(sto_fem_output.p_loss_winding_1 - sto_reluctance_output.winding_1_loss) / sto_fem_output.p_loss_winding_1 * 100} %")
                print(f"P_winding_2 derivation: {(sto_fem_output.p_loss_winding_2 - sto_reluctance_output.winding_2_loss) / sto_fem_output.p_loss_winding_2 * 100} %")
                print(f"P_hyst reluctance: {sto_reluctance_output.p_hyst} W")
                #print(f"P_hyst FEM (from magnet per cell in FEM simulation): {sto_fem_output.p_core_magnet}")
                print(f"P_hyst FEM (from sine wave): {sto_fem_output.p_core_sine} W")
                #print(f"P_hyst derivation (magnet): {(ind_reluctance_output.p_hyst - ind_fem_output.p_core_magnet) / ind_reluctance_output.p_hyst * 100} %")
                print(f"P_hyst derivation (sine): {(sto_reluctance_output.p_hyst - sto_fem_output.p_core_sine) / sto_reluctance_output.p_hyst * 100} %")

        operating_point_number += 1


    transformer_results_reluctance = StackedTransformerResults(
        circuit_id="", transformer_id="", transformer_number_in_circuit=0,
    loss_array=transformer_core_loss_reluctance+transformer_winding_1_loss_reluctance+transformer_winding_2_loss_reluctance,
    core_loss_array=transformer_core_loss_reluctance,
    winding_1_loss_array=transformer_winding_1_loss_reluctance,
    winding_2_loss_array=transformer_winding_2_loss_reluctance,
    volume=0, area_to_heat_sink=0, r_th_xfmr_heat_sink=0, transformer_current_amplitude_1=None,
                transformer_current_amplitude_2=None,
                transformer_phase_deg_1=None,
                transformer_phase_deg_2=None)
    transformer_results_fem= StackedTransformerResults(
        circuit_id="", transformer_id="", transformer_number_in_circuit=0,
    loss_array=transformer_core_loss_fem_sine+transformer_winding_1_loss_fem+transformer_winding_2_loss_reluctance,
    core_loss_array=transformer_core_loss_fem_sine,
    winding_1_loss_array=transformer_winding_1_loss_fem,
    winding_2_loss_array=transformer_winding_2_loss_fem,
    volume=0, area_to_heat_sink=0, r_th_xfmr_heat_sink=0, transformer_current_amplitude_1=transformer_current_amplitude_1,
                transformer_current_amplitude_2=transformer_current_amplitude_2,
                transformer_phase_deg_1=transformer_phase_deg_1,
                transformer_phase_deg_2=transformer_phase_deg_2)

    with open(f"{STO_RECALCULATION_FILEPATH}/transformer_results_reluctance.pkl", 'wb') as output:
        pickle.dump(transformer_results_reluctance, output, pickle.HIGHEST_PROTOCOL)
    if IS_STO_FEM:
        with open(f"{STO_RECALCULATION_FILEPATH}/transformer_results_fem.pkl", 'wb') as output:
            pickle.dump(transformer_results_fem, output, pickle.HIGHEST_PROTOCOL)


def show_results():
    # load circuit calculation
    dto = HandleDabDto.load_from_file(f"{DTO_FILEPATH}/01_circuit/VijayCircuit/{FILTERED_RESULTS_PATH}/{DTO_FILE}")

    with open(f"{IND_RECALCULATION_FILEPATH}/inductor_results_reluctance.pkl", 'rb') as pickle_file_data:
        loaded_inductor_reluctance = pickle.load(pickle_file_data)
    with open(f"{IND_RECALCULATION_FILEPATH}/inductor_results_fem.pkl", 'rb') as pickle_file_data:
        loaded_inductor_fem = pickle.load(pickle_file_data)

    with open(f"{STO_RECALCULATION_FILEPATH}/transformer_results_reluctance.pkl", 'rb') as pickle_file_data:
        loaded_transformer_reluctance = pickle.load(pickle_file_data)
    with open(f"{STO_RECALCULATION_FILEPATH}/transformer_results_fem.pkl", 'rb') as pickle_file_data:
        loaded_transformer_fem = pickle.load(pickle_file_data)


    switching_loss_b2 = [[4, 5, 7, 9], [5, 8, 10, 13]]

    print(f"{dto.calc_losses.p_m1_switching.flatten()=}")
    print(f"{dto.calc_losses.p_m2_switching.flatten()=}")

    data_reluctance = {
        "circuit b1 conduction": dto.calc_losses.p_m1_conduction.flatten(),
        "circuit b1 switching": dto.calc_losses.p_m1_switching.flatten(),
        "circuit b2 conduction": dto.calc_losses.p_m2_conduction.flatten(),
        "circuit b2 switching": dto.calc_losses.p_m2_switching.flatten(),
        "inductor winding": loaded_inductor_reluctance.winding_loss_array.flatten(),
        "inductor core": loaded_inductor_reluctance.core_loss_array.flatten(),
        "transformer winding 1": loaded_transformer_reluctance.winding_1_loss_array.flatten(),
        "transformer winding 2": loaded_transformer_reluctance.winding_2_loss_array.flatten(),
        "transformer core": loaded_transformer_reluctance.core_loss_array.flatten()
    }

    data_fem = {
        "circuit b1 conduction": dto.calc_losses.p_m1_conduction.flatten(),
        "circuit b1 switching": dto.calc_losses.p_m1_switching.flatten(),
        "circuit b2 conduction": dto.calc_losses.p_m2_conduction.flatten(),
        "circuit b2 switching": dto.calc_losses.p_m2_switching.flatten(),
        "inductor winding": loaded_inductor_fem.winding_loss_array.flatten(),
        "inductor core": loaded_inductor_fem.core_loss_array.flatten(),
        "transformer winding 1": loaded_transformer_fem.winding_1_loss_array.flatten(),
        "transformer winding 2": loaded_transformer_fem.winding_2_loss_array.flatten(),
        "transformer core": loaded_transformer_fem.core_loss_array.flatten()
    }

    bar_colors_reluctance = {
        "circuit b1 conduction": colors()["green"],
        "circuit b1 switching": colors()["green"],
        "circuit b2 conduction": colors()["blue"],
        "circuit b2 switching": colors()["blue"],
        "inductor winding": colors()["orange"],
        "inductor core": colors()["orange"],
        "transformer winding 1": colors()["red"],
        "transformer winding 2": colors()["red"],
        "transformer core": colors()["red"]
    }

    bar_colors_fem = {
        "circuit b1 conduction": colors()["green"],
        "circuit b1 switching": colors()["green"],
        "circuit b2 conduction": colors()["blue"],
        "circuit b2 switching": colors()["blue"],
        "inductor winding": colors()["orange"],
        "inductor core": colors()["orange"],
        "transformer winding 1": colors()["red"],
        "transformer winding 2": colors()["red"],
        "transformer core": colors()["red"]
    }

    textures_reluctance = {
        "circuit b1 conduction": "",
        "circuit b1 switching": "+",
        "circuit b2 conduction": "",
        "circuit b2 switching": "+",
        "inductor winding": "",
        "inductor core": ".",
        "transformer winding 1": "",
        "transformer winding 2": ".",
        "transformer core": "+"
    }

    textures_fem = {
        "circuit b1 conduction": "",
        "circuit b1 switching": "+",
        "circuit b2 conduction": "",
        "circuit b2 switching": "+",
        "inductor winding": "",
        "inductor core": ".",
        "transformer winding 1": "",
        "transformer winding 2": ".",
        "transformer core": "+"
    }

    data_measurement = [23.2, 34.4, 48.7, 72.8]

    # set up operating point x-labels
    x_labels = []
    for count, _ in enumerate(dto.input_config.mesh_v1.flatten()):
        v1 = int(dto.input_config.mesh_v1.flatten()[count])
        v2 = int(dto.input_config.mesh_v2.flatten()[count])
        power = int(dto.input_config.mesh_p.flatten()[count])
        operating_point_str = f"{v1} V,\n{v2} V,\n{power} W"
        x_labels.append(operating_point_str)

    for vec_vvp in np.ndindex(dto.calc_modulation.phi.shape):
        print(f"{dto.input_config.mesh_v1[vec_vvp]=}")
        print(f"{dto.input_config.mesh_v2[vec_vvp]=}")
        print(f"{dto.input_config.mesh_p[vec_vvp]=}")

        # generate current comparison graph
        amplitude_rebuild_1 = loaded_transformer_fem.transformer_current_amplitude_1[vec_vvp]
        amplitude_rebuild_2 = loaded_transformer_fem.transformer_current_amplitude_2[vec_vvp]
        phase_deg_rebuild_1 = loaded_transformer_fem.transformer_phase_deg_1[vec_vvp]
        phase_deg_rebuild_2 = loaded_transformer_fem.transformer_phase_deg_2[vec_vvp]

        time_period = 1 / dto.input_config.fs
        time = np.linspace(0, time_period)
        current_1_from_fem = amplitude_rebuild_1 * np.cos(2 * np.pi * time / time_period  - phase_deg_rebuild_1 / 360 * 2 * np.pi)
        current_2_from_fem = amplitude_rebuild_2 * np.cos(2 * np.pi * time / time_period  - phase_deg_rebuild_2 / 360 * 2 * np.pi)

        # fig, axs = plt.subplots(3,1, sharex=True)
        # axs[0].plot(time, current_1_from_fem, label='FEM primary')
        # axs[0].plot(dto.component_requirements.transformer_requirements[0].time_array[vec_vvp], dto.component_requirements.transformer_requirements[0].current_1_array[vec_vvp], label="i_prim requirement")
        # axs[0].legend()
        # axs[0].grid()
        #
        # axs[1].plot(time, current_2_from_fem, label='FEM secondary')
        # axs[1].plot(dto.component_requirements.transformer_requirements[0].time_array[vec_vvp], dto.component_requirements.transformer_requirements[0].current_2_array[vec_vvp], label="i_sec requirement")
        # axs[1].legend()
        # axs[1].grid()
        #
        # # calculate i_mag
        # i_mag_reluctance = dto.component_requirements.transformer_requirements[0].current_1_array[vec_vvp] + dto.component_requirements.transformer_requirements[0].current_2_array[vec_vvp] / sto_target_n
        # i_mag_fem = current_1_from_fem + current_2_from_fem / sto_target_n
        # axs[2].plot(time, i_mag_fem, label='FEM imag')
        # axs[2].plot(dto.component_requirements.transformer_requirements[0].time_array[vec_vvp], i_mag_reluctance, label='Reluctance imag')
        # axs[2].legend()
        # axs[2].grid()
        #
        # plt.show()






    # generate bar graph
    fig, ax = plt.subplots()
    width=0.3
    number_operating_points = len(np.array(dto.calc_modulation.phi).flatten())
    operating_point_list = np.linspace(1, number_operating_points, number_operating_points).tolist()
    bottom_reluctance = np.zeros(number_operating_points)
    bottom_fem = np.zeros(number_operating_points)
    for label, data_count_reluctance in data_reluctance.items():
        data_count_fem=data_fem[label]
        ax.bar(operating_point_list - np.full_like(operating_point_list, width), data_count_reluctance, width, bottom=bottom_reluctance, label=label, color=bar_colors_reluctance[label], hatch=textures_reluctance[label], edgecolor="black")
        ax.bar(operating_point_list, data_count_fem, width, bottom=bottom_fem, color=bar_colors_fem[label], hatch=textures_fem[label], edgecolor="black")
        bottom_reluctance += data_count_reluctance
        bottom_fem += data_count_fem
    ax.bar(operating_point_list + np.full_like(operating_point_list, width), data_measurement, width, label="measurement", color="gray",
           edgecolor="black")
    plt.title("Reluctance / FEM / Measurement")
    plt.grid()
    plt.xticks([1, 2, 3, 4], ["500 W", "1 kW", "1.5 kW", "2 kW"])
    plt.ylabel("Loss / W")
    plt.xlabel("Power / W")
    plt.legend()
    plt.show()

if __name__ == '__main__':
    generate_inductor_data()
    generate_transformer_data()
    show_results()


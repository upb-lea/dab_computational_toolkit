"""Stacked transformer optimization class."""

# python libraries
import os
import pickle
import logging
import copy

# 3rd party libraries
import numpy as np
import tqdm

# own libraries
import femmt as fmt

# configure root logger
logging.basicConfig(format='%(levelname)s,%(asctime)s:%(message)s', encoding='utf-8')
logging.getLogger().setLevel(logging.ERROR)



# python libraries
import logging
import os.path
import pickle
import sys


# own libraries
import dct
import femmt as fmt

# 3rd party libraries
import numpy as np
import pandas as pd
import tqdm

# configure root logger
logging.basicConfig(format='%(levelname)s,%(asctime)s:%(message)s', encoding='utf-8')
logging.getLogger().setLevel(logging.ERROR)


class Transfsim:

    # Configuration list
    sim_config_list = []

    @staticmethod
    def init_configuration(act_transf_config_name: str, act_ginfo: dct.GeneralInformation, act_designspace_dict: dict,
                           act_transformerdata_dict: dict) -> bool:
        """
        Initialize the configuration.

        :param act_ind_config_name : Name of the transformer study
        :type act_ind_config_name : str
        :param act_ginfo : General information about the study
        :type  act_ginfo : dct.GeneralInformation:
        :param act_designspace_dict : dict with data of the design space
        :type  act_designspace_dict : dict
        :param act_transformerdata_dict : dict with parameter of the transformer
        :type  act_transformerdata_dict : dict

        :return: True, if the configuration was sucessfull initialized
        :rtype: bool
        """
        # Variable declaration
        # Return variable initialized to True (ASA: Usage is to add later, currently not used
        ret_val = True

        # Insulation parameter
        act_insulation = fmt.StoInsulation( # insulation for top core window
                                             iso_window_top_core_top=act_transformerdata_dict["iso_window_top_core_top"],
                                             iso_window_top_core_bot=act_transformerdata_dict["iso_window_top_core_bot"],
                                             iso_window_top_core_left=act_transformerdata_dict["iso_window_top_core_left"],
                                             iso_window_top_core_right=act_transformerdata_dict["iso_window_top_core_right"],
                                             # insulation for bottom core window
                                             iso_window_bot_core_top=act_transformerdata_dict["iso_window_bot_core_top"],
                                             iso_window_bot_core_bot=act_transformerdata_dict["iso_window_bot_core_bot"],
                                             iso_window_bot_core_left=act_transformerdata_dict["iso_window_bot_core_left"],
                                             iso_window_bot_core_right=act_transformerdata_dict["iso_window_bot_core_right"],
                                             # winding-to-winding insulation
                                             iso_primary_to_primary=act_transformerdata_dict["iso_primary_to_primary"],
                                             iso_secondary_to_secondary=act_transformerdata_dict["iso_secondary_to_secondary"],
                                             iso_primary_to_secondary=act_transformerdata_dict["iso_primary_to_secondary"],
                                           )

        # Init the material data source       
        act_material_data_sources = fmt.StackedTransformerMaterialDataSources(
                                            permeability_datasource=fmt.MaterialDataSource.Measurement,
                                            permeability_datatype=fmt.MeasurementDataType.ComplexPermeability,
                                            permeability_measurement_setup=fmt.MeasurementSetup.MagNet,
                                            permittivity_datasource=fmt.MaterialDataSource.ManufacturerDatasheet,
                                            permittivity_datatype=fmt.MeasurementDataType.ComplexPermittivity,
                                            permittivity_measurement_setup=fmt.MeasurementSetup.LEA_LK
                                        )

        # Create fix part of io_config
        sto_config_gen = fmt.StoSingleInputConfig(
            stacked_transformer_study_name=act_transf_config_name,
            # target parameters  initialized with default values
            l_s12_target=0,
            l_h_target=0,
            n_target=0,
            # operating point: current waveforms and temperature initialized with default values
            time_current_1_vec=np.ndarray([]),
            time_current_2_vec=np.ndarray([]),
            temperature=100,   # ASA Later it becomes a dynamic value?
            # sweep parameters: geometry and materials
            n_p_top_min_max_list=act_transformerdata_dict["n_p_top_min_max_list"],
            n_p_bot_min_max_list=act_transformerdata_dict["n_p_bot_min_max_list"],
            material_list=act_designspace_dict["material_name_list"],
            core_name_list=act_designspace_dict["core_name_list"],
            core_inner_diameter_min_max_list=act_designspace_dict["core_inner_diameter_min_max_list"],
            window_w_min_max_list=act_designspace_dict["window_w_min_max_list"],
            window_h_bot_min_max_list=act_designspace_dict["window_h_bot_min_max_list"],
            primary_litz_wire_list=act_designspace_dict["primary_litz_wire_list"],
            secondary_litz_wire_list=act_designspace_dict["secondary_litz_wire_list"],
            # maximum limitation for transformer total height and core volume
            max_transformer_total_height=act_transformerdata_dict["max_transformer_total_height"],
            max_core_volume=act_transformerdata_dict["max_core_volume"],
            # fix parameters: insulations
            insulations=act_insulation,
            # misc
            stacked_transformer_optimization_directory="",

            fft_filter_value_factor=act_transformerdata_dict["fft_filter_value_factor"],
            mesh_accuracy=act_transformerdata_dict["mesh_accuracy"],

            # data sources
            material_data_sources=act_material_data_sources
        )

        # Empty the list
        Transfsim.sim_config_list = []

        # Create the sto_config_list for all trials
        for circuit_trial_number in act_ginfo.filtered_list_id:
            circuit_filepath = os.path.join(act_ginfo.circuit_study_path, "filtered_results", f"{circuit_trial_number}.pkl")

            # Check filename
            if os.path.isfile(circuit_filepath):
                # Read results from circuit optimization
                circuit_dto = dct.HandleDabDto.load_from_file(circuit_filepath)
                # get the peak current waveform
                sorted_max_angles, i_l_s_max_current_waveform, i_hf_2_max_current_waveform = dct.HandleDabDto.get_max_peak_waveform_transformer(
                    circuit_dto, False)
                time = sorted_max_angles / 2 / np.pi / circuit_dto.input_config.fs
                transformer_target_params = dct.HandleDabDto.export_transformer_target_parameters_dto(dab_dto=circuit_dto)

                # Generate new sto_config
                next_io_config = copy.deepcopy(sto_config_gen)
                # Add dynamic values to next_io_config
                # target parameters
                next_io_config.l_s12_target = float(transformer_target_params.l_s12_target)
                next_io_config.l_h_target = float(transformer_target_params.l_h_target)
                next_io_config.n_target = float(transformer_target_params.n_target)
                # operating point: current waveforms and temperature initialized with default values
                next_io_config.time_current_1_vec = transformer_target_params.time_current_1_vec
                next_io_config.time_current_2_vec = transformer_target_params.time_current_2_vec
                # misc
                next_io_config.stacked_transformer_optimization_directory \
                 = os.path.join(act_ginfo.transformer_study_path,circuit_trial_number,act_transf_config_name)
                Transfsim.sim_config_list.append([circuit_trial_number, next_io_config])
            else:
                print(f"Wrong path or file {circuit_filepath} does not exists!")

        return ret_val

    @staticmethod
    def _simulation(circuit_id: int, act_sto_config: fmt.StoSingleInputConfig, act_ginfo: dct.GeneralInformation,
                    act_target_number_trials: int, act_filter_factor: float, act_re_simulate: bool, debug: bool):

        """
        Simulate.

        :param circuit_trial_numbers: List of circuit trial numbers to perform transformer optimization
        :type circuit_trial_numbers: list
        :param process_number: Process number (in case of parallel computing)
        :type process_number: int
        :param target_number_trials: Number of trials for the reluctance model optimization
        :type target_number_trials: int
        :param filter_factor: Pareto filter, tolerance band = Multiplication of minimum losses
        :type filter_factor: flot
        :param re_simulate: True to re-simulate all waveforms
        :type re_simulate: bool
        :param debug: True to debug, defaults to False
        :type debug: bool
        """
        # Variable declaration
        # Process_number used in femmt
        process_number = 1

        # Load configuration
        circuit_dto = dct.HandleDabDto.load_from_file(os.path.join(act_ginfo.circuit_study_path, "filtered_results", f"{circuit_id}.pkl"))
        # Check number of trials
        if act_target_number_trials > 0:
            fmt.optimization.StackedTransformerOptimization.ReluctanceModel.start_proceed_study(
                act_sto_config, target_number_trials=act_target_number_trials)
        else:
            print(f"Target number of trials = {act_target_number_trials} which are less equal 0!. No simulation is performed")
            return

        # perform FEM simulations
        if act_filter_factor != 0:
            df = fmt.optimization.StackedTransformerOptimization.ReluctanceModel.study_to_df(act_sto_config)
            df_filtered = fmt.optimization.StackedTransformerOptimization.ReluctanceModel.filter_loss_list_df(
                df, factor_min_dc_losses=act_filter_factor, factor_max_dc_losses=100)
            if debug:
                # reduce dataset to the fist 5 entries
                df_filtered = df_filtered.iloc[:5]

            fmt.optimization.StackedTransformerOptimization.FemSimulation.fem_simulations_from_reluctance_df(
                df_filtered, act_sto_config, process_number=process_number)

        # plot options
        # fmt.optimization.StackedTransformerOptimization.ReluctanceModel.show_study_results(sto_config)
        # fmt.optimization.StackedTransformerOptimization.ReluctanceModel.df_plot_pareto_front(df, df_filtered, label_list=["all", "front"], interactive=False)

        if act_re_simulate:
            fem_results_folder_path = os.path.join(act_sto_config.stacked_transformer_optimization_directory,
                                                   "02_fem_simulation_results")
            df = fmt.optimization.StackedTransformerOptimization.ReluctanceModel.study_to_df(act_sto_config)
            df_filtered = fmt.optimization.StackedTransformerOptimization.ReluctanceModel.filter_loss_list_df(
                df, factor_min_dc_losses=act_filter_factor, factor_max_dc_losses=100)
            df_fem_reluctance = fmt.StackedTransformerOptimization.FemSimulation.fem_logs_to_df(
                df_filtered, fem_results_folder_path)
            # Assemble configuration path
            config_filepath = os.path.join(act_sto_config.stacked_transformer_optimization_directory,
                                           f"{act_sto_config.stacked_transformer_study_name}.pkl")
            config_on_disk = fmt.StackedTransformerOptimization.ReluctanceModel.load_config(config_filepath)

            # workaround for comma problem. Read a random csv file and set back the delimiter.
            pd.read_csv('~/Downloads/Pandas_trial.csv', header=0, index_col=0, delimiter=';')

            # sweep through all current waveforms
            i_l1_sorted = np.transpose(circuit_dto.calc_currents.i_l_1_sorted, (1, 2, 3, 0))
            angles_rad_sorted = np.transpose(circuit_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))

            re_simulate_numbers = df_fem_reluctance["number"].to_numpy()

            for re_simulate_number in re_simulate_numbers:
                print(f"{re_simulate_number=}")
                df_geometry_re_simulation_number = df_fem_reluctance[df_fem_reluctance["number"] == re_simulate_number]

                result_array = np.full_like(circuit_dto.calc_modulation.phi, np.nan)

                new_circuit_dto_directory = os.path.join(act_sto_config.stacked_transformer_optimization_directory,
                                                         "09_circuit_dtos_incl_transformer_losses")
                if not os.path.exists(new_circuit_dto_directory):
                    os.makedirs(new_circuit_dto_directory)

                if os.path.exists(os.path.join(new_circuit_dto_directory, f"{re_simulate_number}.pkl")):
                    print(f"Re-simulation of {circuit_dto.name} already exists. Skip.")
                else:
                    for vec_vvp in tqdm.tqdm(np.ndindex(circuit_dto.calc_modulation.phi.shape),
                                             total=len(circuit_dto.calc_modulation.phi.flatten())):
                        if debug:
                            print(f"{current_waveform=}")
                            print("----------------------")
                            print("Re-simulation of:")
                            print(f"   * Circuit study: {act_ginfo.circuit_study_name}")
                            print(f"   * Circuit trial: {circuit_id}")
                            print(f"   * Transformer study: {act_sto_config.transformer_study_name}")
                            print(f"   * Transformer re-simulation trial: {re_simulate_number}")


                        time = dct.functions_waveforms.full_angle_waveform_from_angles(
                            angles_rad_sorted[vec_vvp]) / 2 / np.pi / circuit_dto.input_config.fs
                        current = dct.functions_waveforms.full_current_waveform_from_currents(i_l1_sorted[vec_vvp])

                        current_waveform = np.array([time, current])

                        print(f"{current_waveform=}")
                        # workaround for comma problem. Read a random csv file and set back the delimiter.
                        pd.read_csv('~/Downloads/Pandas_trial.csv', header=0, index_col=0, delimiter=';')

                        volume, combined_losses, area_to_heat_sink = fmt.StackedTransformerOptimization.FemSimulation.full_simulation(
                            df_geometry_re_simulation_number, current_waveform, config_filepath, show_visual_outputs=False,
                            process_number=process_number)
                        result_array[vec_vvp] = combined_losses

                    results_dto = dct.StackedTransformerResults(
                        p_combined_losses=result_array,
                        volume=volume,
                        area_to_heat_sink=area_to_heat_sink,
                        circuit_trial_number=circuit_id,
                        stacked_transformer_trial_number=re_simulate_number
                    )

                    pickle_file = os.path.join(new_circuit_dto_directory, f"{int(re_simulate_number)}.pkl")
                    with open(pickle_file, 'wb') as output:
                        pickle.dump(results_dto, output, pickle.HIGHEST_PROTOCOL)

                    if debug:
                        # stop after one successful re-simulation run
                        break

    @staticmethod
    # Simulation handler. Later the simulation handler starts a process per list entry.
    def simulation_handler(act_ginfo: dct.GeneralInformation, target_number_trials: int,
                           filter_factor: float = 1.0, re_simulate: bool = False, debug: bool = False):
        """
        Control the multi simulation processes.

        :param act_ginfo : General information about the study
        :type  act_ginfo : dct.GeneralInformation:
        :param target_number_trials : Number of trials for the optimization
        :type  target_number_trials : int
        :param filter_factor : Filter factor to use filter the results (ASA: Later to merge with toml-data filter factor)
        :type  filter_factor : float
        :param re_simulate : Flag to control, if the point are to resimulate (ASA: Correct the parameter description)
        :type  re_simulate : bool
        :param debug : Debug mode flag
        :type  debug : bool
        """

        # Later this is to parallelize with multiple processes
        for act_sim_config in Transfsim.sim_config_list:
            # Debug switch
            if target_number_trials != 0:
                if debug:
                    # overwrite input number of trials with 100 for short simulation times
                    if target_number_trials > 100:
                        target_number_trials = 100

            Transfsim._simulation(act_sim_config[0], act_sim_config[1], act_ginfo, target_number_trials, filter_factor, re_simulate, debug)

            if debug:
                # stop after one circuit run
                break

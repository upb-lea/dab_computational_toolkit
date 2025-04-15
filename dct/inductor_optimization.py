"""Inductor optimization class."""
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
import dct

# configure root logger
logging.basicConfig(format='%(levelname)s,%(asctime)s:%(message)s', encoding='utf-8')
logging.getLogger().setLevel(logging.ERROR)


class InductorOptimization:
    """Optimization of the inductor."""

    # Simulation configuration list
    sim_config_list: list[list[int | fmt.InductorOptimizationDTO]] = []

    @staticmethod
    def init_configuration(toml_inductor: dct.TomlInductor, toml_prog_flow: dct.FlowControl, act_ginfo: dct.GeneralInformation) -> bool:
        """
        Initialize the configuration.

        :param toml_inductor: toml inductor configuration
        :type toml_inductor: dct.TomlInductor
        :param toml_prog_flow: toml program flow configuration
        :type toml_prog_flow: dct.FlowControl
        :param act_ginfo: General information about the study
        :type  act_ginfo: dct.GeneralInformation
        :return: True, if the configuration was successful initialized
        :rtype: bool
        """
        # Variable declaration
        # Return variable initialized to True (ASA: Usage is to add later, currently not used
        initialization_successful = True

        # Insulation parameter
        act_insulations = fmt.InductorInsulationDTO(primary_to_primary=toml_inductor.insulations.primary_to_primary,
                                                    core_bot=toml_inductor.insulations.core_bot,
                                                    core_top=toml_inductor.insulations.core_top,
                                                    core_right=toml_inductor.insulations.core_right,
                                                    core_left=toml_inductor.insulations.core_left)

        # Init the material data source
        act_material_data_sources = fmt.InductorMaterialDataSources(
            permeability_datasource=fmt.MaterialDataSource.Measurement,
            permeability_datatype=fmt.MeasurementDataType.ComplexPermeability,
            permeability_measurement_setup=fmt.MeasurementSetup.MagNet,
            permittivity_datasource=fmt.MaterialDataSource.ManufacturerDatasheet,
            permittivity_datatype=fmt.MeasurementDataType.ComplexPermittivity,
            permittivity_measurement_setup=fmt.MeasurementSetup.LEA_MTB_small_signal
        )

        # Create fix part of io_config
        io_config_gen = fmt.InductorOptimizationDTO(
            inductor_study_name=toml_prog_flow.configuration_data_files.inductor_configuration_file.replace(".toml", ""),
            core_name_list=toml_inductor.design_space.core_name_list,
            material_name_list=toml_inductor.design_space.material_name_list,
            core_inner_diameter_list=toml_inductor.design_space.core_inner_diameter_list,
            window_h_list=toml_inductor.design_space.window_h_list,
            window_w_list=toml_inductor.design_space.window_w_list,
            litz_wire_list=toml_inductor.design_space.litz_wire_list,
            insulations=act_insulations,
            target_inductance=0.0,
            temperature=toml_inductor.boundary_conditions.temperature,
            time_current_vec=[np.array([0]), np.array([0])],
            inductor_optimization_directory="",
            material_data_sources=act_material_data_sources)

        # Empty the list
        InductorOptimization.sim_config_list = []

        # Create the io_config_list for all trials
        for circuit_trial_number in act_ginfo.filtered_list_id:
            circuit_filepath = os.path.join(act_ginfo.circuit_study_path, act_ginfo.circuit_study_name, "filtered_results", f"{circuit_trial_number}.pkl")
            # Check filename
            if os.path.isfile(circuit_filepath):
                # Read results from circuit optimization
                circuit_dto = dct.HandleDabDto.load_from_file(circuit_filepath)
                # get the peak current waveform
                sorted_max_angles, i_l_1_max_current_waveform = dct.HandleDabDto.get_max_peak_waveform_inductor(
                    circuit_dto, False)
                time = sorted_max_angles / 2 / np.pi / circuit_dto.input_config.fs
                # Generate new io_config
                next_io_config = copy.deepcopy(io_config_gen)
                act_time_current_vec = np.array([time, i_l_1_max_current_waveform])
                # Add dynamic values to next_io_config
                next_io_config.circuit_trial_number = circuit_trial_number

                next_io_config.target_inductance = circuit_dto.input_config.Lc1
                next_io_config.time_current_vec = act_time_current_vec
                next_io_config.inductor_optimization_directory = os.path.join(
                    act_ginfo.inductor_study_path, str(circuit_trial_number),
                    toml_prog_flow.configuration_data_files.inductor_configuration_file.replace(".toml", ""))
                InductorOptimization.sim_config_list.append([circuit_trial_number, next_io_config])
            else:
                print(f"Wrong path or file {circuit_filepath} does not exists!")

        return initialization_successful

    # Simulation handler. Later the simulation handler starts a process per list entry.
    @staticmethod
    def _simulation(circuit_id: int, act_io_config: fmt.InductorOptimizationDTO, act_ginfo: dct.GeneralInformation,
                    target_number_trials: int, factor_min_dc_losses: float, factor_max_dc_losses: float, re_simulate: bool, debug: bool):
        """
        Perform the simulation.

        :param circuit_id : Name of the filtered optimal electrical circuit
        :type  circuit_id : int
        :param act_io_config : inductor configuration for the optimization
        :type  act_io_config : fmt.InductorOptimizationDTO
        :param act_ginfo : General information about the study
        :type  act_ginfo : dct.GeneralInformation:
        :param target_number_trials : Number of trials for the optimization
        :type  target_number_trials : int
        :param factor_min_dc_losses : Filter factor to use filter the results (ASA: Later to merge with toml-data filter factor)
        :type  factor_min_dc_losses : float
        :param re_simulate : Flag to control, if the point are to re-simulate (ASA: Correct the parameter description)
        :type  re_simulate : bool
        :param debug: Debug mode flag
        :type debug: bool
        """
        # Variable declaration
        # Process_number are unclear (Usage in femmt)
        process_number = 1

        # Load configuration
        circuit_dto = dct.HandleDabDto.load_from_file(os.path.join(act_ginfo.circuit_study_path,
                                                                   act_ginfo.circuit_study_name,
                                                                   "filtered_results", f"{circuit_id}.pkl"))
        # Check number of trials
        if target_number_trials > 0:
            fmt.optimization.InductorOptimization.ReluctanceModel.start_proceed_study(act_io_config, target_number_trials=target_number_trials)
        else:
            print(f"Target number of trials = {target_number_trials} which are less equal 0!. No simulation is performed")

        # Plot options
        # fmt.optimization.InductorOptimization.ReluctanceModel.show_study_results(io_config)
        # fmt.optimization.InductorOptimization.ReluctanceModel.df_plot_pareto_front(df, df_filtered, label_list=["all", "front"], interactive=False)

        # perform FEM simulations
        if factor_min_dc_losses != 0:
            df = fmt.optimization.InductorOptimization.ReluctanceModel.study_to_df(act_io_config)
            df_filtered = fmt.optimization.InductorOptimization.ReluctanceModel.filter_loss_list_df(df,
                                                                                                    factor_min_dc_losses=factor_min_dc_losses,
                                                                                                    factor_max_dc_losses=factor_max_dc_losses)
            if debug:
                # reduce dataset to the fist 5 entries
                df_filtered = df_filtered.iloc[:5]

            fmt.InductorOptimization.FemSimulation.fem_simulations_from_reluctance_df(df_filtered, act_io_config,
                                                                                      process_number=process_number)

        if re_simulate:
            fem_results_folder_path = os.path.join(act_io_config.inductor_optimization_directory,
                                                   "02_fem_simulation_results")
            df = fmt.optimization.InductorOptimization.ReluctanceModel.study_to_df(act_io_config)
            df_filtered = (fmt.optimization.InductorOptimization.ReluctanceModel.filter_loss_list_df
                           (df, factor_min_dc_losses=factor_min_dc_losses, factor_max_dc_losses=factor_max_dc_losses))
            df_fem_reluctance = fmt.InductorOptimization.FemSimulation.fem_logs_to_df(df_filtered, fem_results_folder_path)
            # fmt.InductorOptimization.FemSimulation.fem_vs_reluctance_pareto(df_fem_reluctance)

            config_filepath = os.path.join(act_io_config.inductor_optimization_directory, f"{act_io_config.inductor_study_name}.pkl")

            # workaround for comma problem. Read a random csv file and set back the delimiter.
            # pd.read_csv('Pandas_trial.csv', header=0, index_col=0, delimiter=';')

            # sweep through all current waveforms
            i_l1_sorted = np.transpose(circuit_dto.calc_currents.i_l_1_sorted, (1, 2, 3, 0))
            angles_rad_sorted = np.transpose(circuit_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))

            re_simulate_numbers = df_fem_reluctance["number"].to_numpy()

            for re_simulate_number in re_simulate_numbers:
                print(f"{re_simulate_number=}")
                df_geometry_re_simulation_number = df_fem_reluctance[
                    df_fem_reluctance["number"] == float(re_simulate_number)]

                print(df_geometry_re_simulation_number.head())

                result_array = np.full_like(circuit_dto.calc_modulation.phi, np.nan)

                new_circuit_dto_directory = os.path.join(act_io_config.inductor_optimization_directory,
                                                         "09_circuit_dtos_incl_inductor_losses")
                if not os.path.exists(new_circuit_dto_directory):
                    os.makedirs(new_circuit_dto_directory)

                if os.path.exists(os.path.join(new_circuit_dto_directory, f"{re_simulate_number}.pkl")):
                    print(f"Re-simulation of {circuit_dto.name} already exists. Skip.")
                else:
                    for vec_vvp in tqdm.tqdm(np.ndindex(circuit_dto.calc_modulation.phi.shape),
                                             total=len(circuit_dto.calc_modulation.phi.flatten())):
                        time, unique_indices = np.unique(
                            dct.functions_waveforms.full_angle_waveform_from_angles(
                                angles_rad_sorted[vec_vvp]) / 2 / np.pi / circuit_dto.input_config.fs,
                            return_index=True)
                        current = dct.functions_waveforms.full_current_waveform_from_currents(i_l1_sorted[vec_vvp])[
                            unique_indices]

                        current_waveform = np.array([time, current])
                        if debug:
                            print(f"{current_waveform=}")
                            print("----------------------")
                            print("Re-simulation of:")
                            print(f"   * Circuit study: {act_ginfo.circuit_study_name}")
                            print(f"   * Circuit trial: {circuit_id}")
                            print(f"   * Inductor study: {act_io_config.inductor_study_name}")
                            print(f"   * Inductor re-simulation trial: {re_simulate_number}")

                        volume, combined_losses, area_to_heat_sink = fmt.InductorOptimization.FemSimulation.full_simulation(
                            df_geometry_re_simulation_number, current_waveform=current_waveform,
                            inductor_config_filepath=config_filepath,
                            process_number=process_number)
                        result_array[vec_vvp] = combined_losses

                    inductor_losses = dct.InductorResults(
                        p_combined_losses=result_array,
                        volume=volume,
                        area_to_heat_sink=area_to_heat_sink,
                        circuit_trial_number=circuit_id,
                        inductor_trial_number=re_simulate_number,
                    )

                    pickle_file = os.path.join(new_circuit_dto_directory, f"{int(re_simulate_number)}.pkl")
                    with open(pickle_file, 'wb') as output:
                        pickle.dump(inductor_losses, output, pickle.HIGHEST_PROTOCOL)

                    if debug:
                        # stop after one successful re-simulation run
                        break

    # Simulation handler. Later the simulation handler starts a process per list entry.
    @staticmethod
    def simulation_handler(act_ginfo: dct.GeneralInformation, target_number_trials: int,
                           factor_min_dc_losses: float = 1.0, factor_dc_max_losses: float = 100,
                           re_simulate: bool = False, debug: bool = False):
        """
        Control the multi simulation processes.

        :param act_ginfo : General information about the study
        :type  act_ginfo : dct.GeneralInformation:
        :param target_number_trials : Number of trials for the optimization
        :type  target_number_trials : int
        :param factor_min_dc_losses : Filter factor to use filter the results (ASA: Later to merge with toml-data filter factor)
        :type  factor_min_dc_losses : float
        :param factor_dc_max_losses: Filter factor for the maximum losses, related to the minimum DC losses
        :type factor_dc_max_losses: float
        :param re_simulate : Flag to control, if the point are to re-simulate (ASA: Correct the parameter description)
        :type  re_simulate : bool
        :param debug : Debug mode flag
        :type  debug : bool
        """
        # Later this is to parallelize with multiple processes
        for act_sim_config in InductorOptimization.sim_config_list:
            # Debug switch
            if target_number_trials != 0:
                if debug:
                    # overwrite input number of trials with 100 for short simulation times
                    if target_number_trials > 100:
                        target_number_trials = 100

            InductorOptimization._simulation(act_sim_config[0], act_sim_config[1], act_ginfo, target_number_trials,
                                             factor_min_dc_losses, factor_dc_max_losses, re_simulate, debug)

            if debug:
                # stop after one circuit run
                break

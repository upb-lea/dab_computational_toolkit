"""Stacked transformer optimization class."""

# python libraries
import os
import copy
import logging
import os.path
import pickle
import time
import threading

# 3rd party libraries
import numpy as np
import tqdm

# own libraries
import dct.transformer_optimization_dtos
import femmt as fmt
from dct.server_ctl_dtos import ProgressData

logger = logging.getLogger(__name__)

class TransformerOptimization:
    """Optimization of the transformer."""

    # List with configurations to optimize and lock variable
    _optimization_config_list: list[dct.transformer_optimization_dtos.TransformerOptimizationDto]
    _t_lock_stat: threading.Lock

    def __init__(self) -> None:
        """Initialize the configuration list for the transformer optimizations."""
        self._optimization_config_list = []
        self._t_lock_stat: threading.Lock = threading.Lock()
        self._number_performed_calculations: int = 0

    def generate_optimization_list(self, toml_transformer: dct.TomlTransformer, study_data: dct.StudyData, filter_data: dct.FilterData) -> bool:
        """
        Initialize the configuration.

        :param toml_transformer: transformer toml file
        :type toml_transformer: dct.TomlTransformer
        :param study_data: Study data
        :type study_data: dct.StudyData
        :param filter_data: Information about the filtered circuit designs
        :type filter_data: dct.FilterData
        :return: True, if the configuration was successful initialized
        :rtype: bool
        """
        is_list_generation_successful = False

        act_insulation = fmt.StoInsulation(
            # insulation for top core window
            iso_window_top_core_top=toml_transformer.insulation.iso_window_top_core_top,
            iso_window_top_core_bot=toml_transformer.insulation.iso_window_top_core_bot,
            iso_window_top_core_left=toml_transformer.insulation.iso_window_top_core_left,
            iso_window_top_core_right=toml_transformer.insulation.iso_window_top_core_right,
            # insulation for bottom core window
            iso_window_bot_core_top=toml_transformer.insulation.iso_window_bot_core_top,
            iso_window_bot_core_bot=toml_transformer.insulation.iso_window_bot_core_bot,
            iso_window_bot_core_left=toml_transformer.insulation.iso_window_bot_core_left,
            iso_window_bot_core_right=toml_transformer.insulation.iso_window_bot_core_right,
            # winding-to-winding insulation
            iso_primary_to_primary=toml_transformer.insulation.iso_primary_to_primary,
            iso_secondary_to_secondary=toml_transformer.insulation.iso_secondary_to_secondary,
            iso_primary_to_secondary=toml_transformer.insulation.iso_primary_to_secondary
        )

        # Init the material data source
        material_data_sources = fmt.StackedTransformerMaterialDataSources(
            permeability_datasource=fmt.MaterialDataSource.Measurement,
            permeability_datatype=fmt.MeasurementDataType.ComplexPermeability,
            permeability_measurement_setup=fmt.MeasurementSetup.MagNet,
            permittivity_datasource=fmt.MaterialDataSource.ManufacturerDatasheet,
            permittivity_datatype=fmt.MeasurementDataType.ComplexPermittivity,
            permittivity_measurement_setup=fmt.MeasurementSetup.LEA_LK
        )

        # Create fix part of io_config
        sto_config = fmt.StoSingleInputConfig(
            stacked_transformer_study_name=study_data.study_name,
            # target parameters  initialized with default values
            l_s12_target=0,
            l_h_target=0,
            n_target=0,
            # operating point: current waveforms and temperature initialized with default values
            time_current_1_vec=np.ndarray([]),
            time_current_2_vec=np.ndarray([]),
            temperature=toml_transformer.boundary_conditions.temperature,  # ASA Later it becomes a dynamic value?
            # sweep parameters: geometry and materials
            n_p_top_min_max_list=toml_transformer.design_space.n_p_top_min_max_list,
            n_p_bot_min_max_list=toml_transformer.design_space.n_p_bot_min_max_list,
            material_list=toml_transformer.design_space.material_name_list,
            core_name_list=toml_transformer.design_space.core_name_list,
            core_inner_diameter_min_max_list=toml_transformer.design_space.core_inner_diameter_min_max_list,
            window_w_min_max_list=toml_transformer.design_space.window_w_min_max_list,
            window_h_bot_min_max_list=toml_transformer.design_space.window_h_bot_min_max_list,
            primary_litz_wire_list=toml_transformer.design_space.primary_litz_wire_list,
            secondary_litz_wire_list=toml_transformer.design_space.secondary_litz_wire_list,
            # maximum limitation for transformer total height and core volume
            max_transformer_total_height=toml_transformer.boundary_conditions.max_transformer_total_height,
            max_core_volume=toml_transformer.boundary_conditions.max_core_volume,
            # fix parameters: insulations
            insulations=act_insulation,
            # misc
            stacked_transformer_optimization_directory="",

            fft_filter_value_factor=toml_transformer.settings.fft_filter_value_factor,
            mesh_accuracy=toml_transformer.settings.mesh_accuracy,

            # data sources
            material_data_sources=material_data_sources
        )

        # Initialize the staticical data
        stat_data_init: ProgressData = ProgressData(start_time=0.0, run_time=0, nb_of_filtered_points=0, status=0)

        # Create the sto_config_list for all trials
        for circuit_trial_number in filter_data.filtered_list_id:
            circuit_filepath = os.path.join(filter_data.filtered_list_pathname, f"{circuit_trial_number}.pkl")

            # Check filename
            if os.path.isfile(circuit_filepath):
                # Read results from circuit optimization
                circuit_dto = dct.HandleDabDto.load_from_file(circuit_filepath)
                # get the peak current waveform
                sorted_max_angles, i_l_s_max_current_waveform, i_hf_2_max_current_waveform = dct.HandleDabDto.get_max_peak_waveform_transformer(
                    circuit_dto, False)
                time = sorted_max_angles / 2 / np.pi / circuit_dto.input_config.fs
                transformer_target_params = dct.HandleDabDto.export_transformer_target_parameters_dto(
                    dab_dto=circuit_dto)

                # Generate new sto_config
                next_io_config = copy.deepcopy(sto_config)
                # Add dynamic values to next_io_config
                # target parameters
                next_io_config.l_s12_target = float(transformer_target_params.l_s12_target)
                next_io_config.l_h_target = float(transformer_target_params.l_h_target)
                next_io_config.n_target = float(transformer_target_params.n_target)
                # operating point: current waveforms and temperature initialized with default values
                next_io_config.time_current_1_vec = transformer_target_params.time_current_1_vec
                next_io_config.time_current_2_vec = transformer_target_params.time_current_2_vec
                # misc
                next_io_config.stacked_transformer_optimization_directory\
                    = os.path.join(study_data.optimization_directory, str(circuit_trial_number), sto_config.stacked_transformer_study_name)
                transformer_dto = dct.transformer_optimization_dtos.TransformerOptimizationDto(
                    circuit_id=circuit_trial_number,
                    progress_data=copy.deepcopy(stat_data_init),
                    transformer_optimization_dto=next_io_config)

                self._optimization_config_list.append(transformer_dto)
            else:
                logger.info(f"Wrong path or file {circuit_filepath} does not exists!")

        if self._optimization_config_list:
            is_list_generation_successful = True

        return is_list_generation_successful

    def get_progress_data(self, filtered_list_id: int) -> ProgressData:
        """Provide the progress data of the optimization.

        :param filtered_list_id: List index of the filtered operation point from circuit
        :type  filtered_list_id: int

        :return: Progress data: Processing start time, actual processing time, number of filtered transformer Pareto front points and status.
        :rtype: ProgressData
        """
        # Variable deklaration and default initialisation
        ret_progress_data: ProgressData = ProgressData(
            start_time=0.0, run_time=0, nb_of_filtered_points=0,
            status=0)

        # Check for valid filtered_list_id
        if len(self._optimization_config_list) > filtered_list_id:
            # Lock statistical performance data access   (ASA: Possible Bug)
            # with self._t_lock_stat: -> ASA: Later to repair
            # Update statistical data if optimisation is running
            if self._optimization_config_list[filtered_list_id].progress_data.status == 1:
                self._optimization_config_list[filtered_list_id].progress_data.run_time = (
                    time.perf_counter() - self._optimization_config_list[filtered_list_id].progress_data.start_time)
                # Check for valid entry
                if self._optimization_config_list[filtered_list_id].progress_data.run_time < 0:
                    self._optimization_config_list[filtered_list_id].progress_data.run_time = 0.0
                    self._optimization_config_list[filtered_list_id].progress_data.start_time = time.perf_counter()
            else:
                ret_progress_data = copy.deepcopy(self._optimization_config_list[filtered_list_id].progress_data)

        return copy.deepcopy(ret_progress_data)

    def get_number_of_performed_calculations(self) -> int:
        """Provide the number of performed calculations.

        :return: int: Number of performed calculations
        :rtype: int
        """
        return self._number_performed_calculations

    def _optimize(self, circuit_id: int, act_sto_config: fmt.StoSingleInputConfig, filter_data: dct.FilterData,
                  act_target_number_trials: int, factor_dc_min_losses: float, factor_dc_max_losses: float, act_re_simulate: bool, debug: bool) -> int:
        """
        Perform the optimization.

        :param circuit_id: List of circuit trial numbers to perform transformer optimization
        :type circuit_id: list
        :param act_sto_config: Process number (in case of parallel computing)
        :type act_sto_config: int
        :param act_target_number_trials: Number of trials for the reluctance model optimization
        :type act_target_number_trials: int
        :param factor_dc_min_losses: Pareto filter, tolerance band = Multiplication of minimum losses
        :type factor_dc_min_losses: float
        :param factor_dc_max_losses: Pareto filter, tolerance band = Multiplication of maximum losses
        :type factor_dc_max_losses: float
        :param act_re_simulate: True to re-simulate all waveforms
        :type act_re_simulate: bool
        :param debug: True to debug, defaults to False
        :type debug: bool
        """
        # Variable declaration
        # Process_number used in femmt
        process_number = 1
        # Number of filtered operating points
        nb_of_filtered_points = 0

        # Load configuration
        circuit_dto = dct.HandleDabDto.load_from_file(os.path.join(filter_data.filtered_list_pathname, f"{circuit_id}.pkl"))
        # Check number of trials
        if act_target_number_trials > 0:
            fmt.optimization.StackedTransformerOptimization.ReluctanceModel.start_proceed_study(
                act_sto_config, target_number_trials=act_target_number_trials)
        else:
            logger.info(f"Target number of trials = {act_target_number_trials} which are less equal 0!. No simulation is performed")
            return 0

        # perform FEM simulations
        if factor_dc_min_losses != 0:
            df = fmt.optimization.StackedTransformerOptimization.ReluctanceModel.study_to_df(act_sto_config)
            df_filtered = fmt.optimization.StackedTransformerOptimization.ReluctanceModel.filter_loss_list_df(
                df, factor_min_dc_losses=factor_dc_min_losses, factor_max_dc_losses=factor_dc_max_losses)
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
                df, factor_min_dc_losses=factor_dc_min_losses, factor_max_dc_losses=100)
            df_fem_reluctance = fmt.StackedTransformerOptimization.FemSimulation.fem_logs_to_df(
                df_filtered, fem_results_folder_path)
            # Assemble configuration path
            config_filepath = os.path.join(act_sto_config.stacked_transformer_optimization_directory,
                                           f"{act_sto_config.stacked_transformer_study_name}.pkl")
            config_on_disk = fmt.StackedTransformerOptimization.ReluctanceModel.load_config(config_filepath)

            # workaround for comma problem. Read a random csv file and set back the delimiter.
            # pd.read_csv('~/Downloads/Pandas_trial.csv', header=0, index_col=0, delimiter=';')

            # sweep through all current waveforms
            i_l1_sorted = np.transpose(circuit_dto.calc_currents.i_l_1_sorted, (1, 2, 3, 0))
            angles_rad_sorted = np.transpose(circuit_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))

            re_simulate_numbers = df_fem_reluctance["number"].to_numpy()

            # Overtake the filtered operation points
            nb_of_filtered_points = len(re_simulate_numbers)

            for re_simulate_number in re_simulate_numbers:
                logger.info(f"{re_simulate_number=}")
                df_geometry_re_simulation_number = df_fem_reluctance[df_fem_reluctance["number"] == re_simulate_number]

                result_array = np.full_like(circuit_dto.calc_modulation.phi, np.nan)

                new_circuit_dto_directory = os.path.join(act_sto_config.stacked_transformer_optimization_directory,
                                                         "09_circuit_dtos_incl_transformer_losses")
                if not os.path.exists(new_circuit_dto_directory):
                    os.makedirs(new_circuit_dto_directory)

                if os.path.exists(os.path.join(new_circuit_dto_directory, f"{re_simulate_number}.pkl")):
                    logger.info(f"Re-simulation of {circuit_dto.name} already exists. Skip.")
                else:
                    for vec_vvp in tqdm.tqdm(np.ndindex(circuit_dto.calc_modulation.phi.shape),
                                             total=len(circuit_dto.calc_modulation.phi.flatten())):

                        time = dct.functions_waveforms.full_angle_waveform_from_angles(
                            angles_rad_sorted[vec_vvp]) / 2 / np.pi / circuit_dto.input_config.fs
                        current = dct.functions_waveforms.full_current_waveform_from_currents(i_l1_sorted[vec_vvp])

                        current_waveform = np.array([time, current])

                        if debug:
                            logger.info(f"{current_waveform=}")
                            logger.info("----------------------")
                            logger.info("Re-simulation of:")
                            logger.info(f"   * Circuit study: {filter_data.circuit_study_name}")
                            logger.info(f"   * Circuit trial: {circuit_id}")
                            logger.info(f"   * Transformer study: {act_sto_config.transformer_study_name}")
                            logger.info(f"   * Transformer re-simulation trial: {re_simulate_number}")

                        logger.info(f"{current_waveform=}")
                        # workaround for comma problem. Read a random csv file and set back the delimiter.
                        # pd.read_csv('~/Downloads/Pandas_trial.csv', header=0, index_col=0, delimiter=';')

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

                # Increment performed calculation counter
                self._number_performed_calculations = self._number_performed_calculations + 1

        # returns the number of filtered results
        return nb_of_filtered_points

    # Simulation handler. Later the simulation handler starts a process per list entry.
    def optimization_handler(self, filter_data: dct.FilterData, target_number_trials: int,
                             factor_dc_min_losses: float = 1.0, factor_dc_max_losses: float = 100,
                             enable_operating_range_simulation: bool = False, debug: bool = False) -> None:
        """
        Control the multi simulation processes.

        :param filter_data : Information about the filtered circuit designs
        :type  filter_data : dct.FilterData
        :param target_number_trials: Number of trials for the optimization
        :type  target_number_trials: int
        :param factor_dc_min_losses: Filter factor for the offset, related to the minimum DC losses
        :type  factor_dc_min_losses: float
        :param factor_dc_max_losses: Filter factor for the maximum losses, related to the minimum DC losses
        :type factor_dc_max_losses: float
        :param enable_operating_range_simulation: True to perform the simulations for all operating points
        :type  enable_operating_range_simulation: bool
        :param debug: Debug mode flag
        :type  debug: bool
        """
        # Later this is to parallelize with multiple processes
        for act_optimization_configuration in self._optimization_config_list:
            # Debug switch
            if target_number_trials != 0:
                if debug:
                    # overwrite input number of trials with 100 for short simulation times
                    if target_number_trials > 100:
                        target_number_trials = 100

            # Update statistical data
            # with self._t_lock_stat:
            act_optimization_configuration.progress_data.start_time = time.perf_counter()
            act_optimization_configuration.progress_data.status = 1

            nb_fil_pt = self._optimize(act_optimization_configuration.circuit_id,
                                       act_optimization_configuration.transformer_optimization_dto,
                                       filter_data, target_number_trials, factor_dc_min_losses,
                                       factor_dc_max_losses, enable_operating_range_simulation, debug)

            # Update statistical data
            #  with self._t_lock_stat:
            act_optimization_configuration.progress_data.run_time = time.perf_counter() - act_optimization_configuration.progress_data.start_time
            act_optimization_configuration.progress_data.nb_of_filtered_points = nb_fil_pt
            act_optimization_configuration.progress_data.status = 2

            if debug:
                # stop after one circuit run
                break

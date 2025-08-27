"""Stacked transformer optimization class."""

# python libraries
import os
import copy
import logging
import os.path
import pickle
import threading
from multiprocessing import Pool, cpu_count, current_process

# 3rd party libraries
import numpy as np
import tqdm

# own libraries
import dct.transformer_optimization_dtos
import femmt as fmt
from dct.boundary_check import CheckCondition as c_flag
from dct.server_ctl_dtos import ProgressData
from dct.server_ctl_dtos import ProgressStatus
from dct.server_ctl_dtos import RunTimeMeasurement as RunTime

logger = logging.getLogger(__name__)

class TransformerOptimization:
    """Optimization of the transformer."""

    # List with configurations to optimize and lock variable
    _optimization_config_list: list[dct.transformer_optimization_dtos.TransformerOptimizationDto]
    _t_lock_stat: threading.Lock
    _progress_run_time: RunTime

    def __init__(self) -> None:
        """Initialize the configuration list for the transformer optimizations."""
        self._optimization_config_list = []
        self._t_lock_stat: threading.Lock = threading.Lock()
        self._number_performed_calculations: int = 0
        self._progress_run_time: RunTime = RunTime()

    @staticmethod
    def verify_optimization_parameter(toml_transformer: dct.TomlTransformer) -> tuple[bool, str]:
        """Verify the input parameter ranges.

        :param toml_transformer: toml inductor configuration
        :type toml_transformer: dct.TomlInductor
        :return: True, if the configuration was consistent
        :rtype: bool
        """
        # Variable declaration
        inconsistency_report: str = ""
        is_consistent: bool = True
        toml_check_min_max_values_list: list[tuple[list[float], str]]
        toml_check_value_list: list[tuple[float, str]]

        # Design space parameter check
        group_name = "design_space"
        # Check core_name_list
        if len(toml_transformer.design_space.core_name_list) != 0:
            # Get available keywords
            keyword_dictionary: dict = fmt.core_database()
            # Perform dictionary check
            for keyword_entry in toml_transformer.design_space.core_name_list:
                is_check_passed, issue_report = dct.BoundaryCheck.check_dictionary(
                    keyword_dictionary, keyword_entry, f"{group_name}: core_name_list")
                # Check if boundary check fails
                if not is_check_passed:
                    inconsistency_report = inconsistency_report + issue_report
                    is_consistent = False
        else:
            toml_check_min_max_values_list = (
                [(toml_transformer.design_space.core_inner_diameter_min_max_list, f"{group_name}: core_inner_diameter_min_max_list"),
                 (toml_transformer.design_space.window_h_bot_min_max_list, f"{group_name}: window_h_bot_min_max_list"),
                 (toml_transformer.design_space.window_w_min_max_list, f"{group_name}: window_w_min_max_list")])

            # Perform the boundary check
            is_check_passed, issue_report = dct.BoundaryCheck.check_float_min_max_values_list(
                0, 5, toml_check_min_max_values_list, c_flag.check_exclusive, c_flag.check_exclusive)
            if not is_check_passed:
                inconsistency_report = inconsistency_report + issue_report
                is_consistent = False

        # Convert min_max-list from integer to float values
        float_n_p_top_min_max_list = dct.BoundaryCheck.convert_int_list_to_float_list(toml_transformer.design_space.n_p_top_min_max_list)
        float_n_p_bot_min_max_list = dct.BoundaryCheck.convert_int_list_to_float_list(toml_transformer.design_space.n_p_bot_min_max_list)
        # Setup check value list
        toml_check_min_max_values_list = (
            [(float_n_p_top_min_max_list, f"{group_name}: n_p_top_min_max_list"),
             (float_n_p_bot_min_max_list, f"{group_name}: n_p_bot_min_max_list")])

        # Perform the boundary check
        is_check_passed, issue_report = dct.BoundaryCheck.check_float_min_max_values_list(
            0, 10000, toml_check_min_max_values_list, c_flag.check_exclusive, c_flag.check_inclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Check litz_wire_list
        # Get available keywords
        keyword_dictionary = fmt.litz_database()
        # Perform dictionary check for primary litz
        for keyword_entry in toml_transformer.design_space.primary_litz_wire_list:
            is_check_passed, issue_report = dct.BoundaryCheck.check_dictionary(keyword_dictionary, keyword_entry, f"{group_name}: litz_wire_name_list")
            # Check if boundary check fails
            if not is_check_passed:
                inconsistency_report = inconsistency_report + issue_report
                is_consistent = False
        # Perform dictionary check for primary litz
        for keyword_entry in toml_transformer.design_space.secondary_litz_wire_list:
            is_check_passed, issue_report = dct.BoundaryCheck.check_dictionary(keyword_dictionary, keyword_entry, f"{group_name}: litz_wire_name_list")
            # Check if boundary check fails
            if not is_check_passed:
                inconsistency_report = inconsistency_report + issue_report
                is_consistent = False

        # Insulation parameter check
        group_name = "insulation"
        # Create a list of parameter to check
        toml_check_value_list = (
            [(toml_transformer.insulation.iso_primary_to_primary, f"{group_name}: iso_primary_to_primary"),
             (toml_transformer.insulation.iso_secondary_to_secondary, f"{group_name}: iso_secondary_to_secondary"),
             (toml_transformer.insulation.iso_primary_to_secondary, f"{group_name}: iso_primary_to_secondary"),
             (toml_transformer.insulation.iso_window_top_core_top, f"{group_name}: iso_window_top_core_top"),
             (toml_transformer.insulation.iso_window_top_core_bot, f"{group_name}: iso_window_top_core_bot"),
             (toml_transformer.insulation.iso_window_top_core_left, f"{group_name}: iso_window_top_core_left"),
             (toml_transformer.insulation.iso_window_top_core_right, f"{group_name}: iso_window_top_core_right"),
             (toml_transformer.insulation.iso_window_bot_core_top, f"{group_name}: iso_window_bot_core_top"),
             (toml_transformer.insulation.iso_window_bot_core_bot, f"{group_name}: iso_window_bot_core_bot"),
             (toml_transformer.insulation.iso_window_bot_core_left, f"{group_name}: iso_window_bot_core_left"),
             (toml_transformer.insulation.iso_window_bot_core_right, f"{group_name}: iso_window_bot_core_right")])

        # Perform insulation value check
        # Perform the boundary check
        is_check_passed, issue_report = dct.BoundaryCheck.check_float_value_list(
            0, 0.1, toml_check_value_list, c_flag.check_exclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Perform boundary condition check
        group_name = "boundary_condition"
        # Perform the boundary check
        is_check_passed, issue_report = dct.BoundaryCheck.check_float_value(
            -40, 175, toml_transformer.boundary_conditions.temperature, f"{group_name}: temperature", c_flag.check_inclusive, c_flag.check_inclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        toml_check_value_list = (
            [(toml_transformer.boundary_conditions.max_transformer_total_height, f"{group_name}: max_transformer_total_height"),
             (toml_transformer.boundary_conditions.max_core_volume, f"{group_name}: max_core_volume")])
        # Perform the boundary check
        is_check_passed, issue_report = dct.BoundaryCheck.check_float_value_list(
            0, 5, toml_check_value_list, c_flag.check_exclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Perform setting check
        group_name = "setting"
        # Perform the boundary check
        is_check_passed, issue_report = dct.BoundaryCheck.check_float_value(
            0, 1, toml_transformer.settings.fft_filter_value_factor, f"{group_name}: fft_filter_value_factor", c_flag.check_inclusive, c_flag.check_inclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False
        is_check_passed, issue_report = dct.BoundaryCheck.check_float_value(
            0, 1, toml_transformer.settings.mesh_accuracy, f"{group_name}: mesh_accuracy", c_flag.check_exclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Perform filter_distance value check
        group_name = "filter_distance"
        # Perform the boundary check
        is_check_passed, issue_report = dct.BoundaryCheck.check_float_min_max_values(
            0, 100, toml_transformer.filter_distance.factor_dc_losses_min_max_list,
            f"{group_name}: factor_dc_losses_min_max_list", c_flag.check_inclusive, c_flag.check_inclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        return is_consistent, inconsistency_report

    def initialize_transformer_optimization_list(self, toml_transformer: dct.TomlTransformer, study_data: dct.StudyData, filter_data: dct.FilterData) -> bool:
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

        # Verify optimization parameter
        is_consistent, issue_report = dct.TransformerOptimization.verify_optimization_parameter(toml_transformer)
        if not is_consistent:
            raise ValueError(
                "Transformer optimization parameter are inconsistent!\n",
                issue_report)

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

        # Initialize the material data source
        material_data_sources = fmt.MaterialDataSources(
            permeability_datasource=toml_transformer.material_data_sources.permeability_datasource,
            permittivity_datasource=toml_transformer.material_data_sources.permittivity_datasource,
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
            temperature=toml_transformer.boundary_conditions.temperature,
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

        # Initialize the statistical data
        stat_data_init: ProgressData = ProgressData(run_time=0, number_of_filtered_points=0, progress_status=ProgressStatus.Idle)

        # Create the sto_config_list for all filtered circuit trials
        for circuit_trial_file in filter_data.filtered_list_files:
            circuit_filepath = os.path.join(filter_data.filtered_list_pathname, f"{circuit_trial_file}.pkl")

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
                # target parameters
                next_io_config.l_s12_target = float(transformer_target_params.l_s12_target)
                next_io_config.l_h_target = float(transformer_target_params.l_h_target)
                next_io_config.n_target = float(transformer_target_params.n_target)
                # operating point: current waveforms and temperature initialized with default values
                next_io_config.time_current_1_vec = transformer_target_params.time_current_1_vec
                next_io_config.time_current_2_vec = transformer_target_params.time_current_2_vec
                # misc
                next_io_config.stacked_transformer_optimization_directory\
                    = os.path.join(study_data.optimization_directory, str(circuit_trial_file), sto_config.stacked_transformer_study_name)
                transformer_dto = dct.transformer_optimization_dtos.TransformerOptimizationDto(
                    circuit_filtered_point_filename=circuit_trial_file,
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
        # Variable declaration and default initialization
        ret_progress_data: ProgressData = ProgressData(
            run_time=0, number_of_filtered_points=0,
            progress_status=ProgressStatus.Idle)

        # Check for valid filtered_list_id
        if len(self._optimization_config_list) > filtered_list_id:
            # Lock statistical performance data access (ASA: Possible Bug)
            with self._t_lock_stat:
                # Check if list is in progress
                if self._optimization_config_list[filtered_list_id].progress_data.progress_status == ProgressStatus.InProgress:
                    # Update statistical data
                    self._optimization_config_list[filtered_list_id].progress_data.run_time = self._progress_run_time.get_runtime()

                # Create a copy of actual data
                ret_progress_data = copy.deepcopy(self._optimization_config_list[filtered_list_id].progress_data)

        return ret_progress_data

    def get_number_of_performed_calculations(self) -> int:
        """Provide the number of performed calculations.

        :return: int: Number of performed calculations
        :rtype: int
        """
        return self._number_performed_calculations

    @staticmethod
    def _optimize_reluctance_model(circuit_filtered_point_file: str, act_sto_config: fmt.StoSingleInputConfig, filter_data: dct.FilterData,
                                   act_target_number_trials: int, factor_dc_losses_min_max_list: list[float], debug: dct.Debug) -> int:
        """
        Perform the optimization.

        :param circuit_filtered_point_file: Filename of the filtered optimal electrical circuit
        :type circuit_filtered_point_file: str
        :param act_sto_config: Process number (in case of parallel computing)
        :type act_sto_config: int
        :param act_target_number_trials: Number of trials for the reluctance model optimization
        :type act_target_number_trials: int
        :param factor_dc_losses_min_max_list: Pareto filter, tolerance band = Multiplication of minimum/maximum losses
        :type factor_dc_losses_min_max_list: float
        :param debug: Debug DTO
        :type debug: dct.Debug
        """
        # Number of filtered operating points
        number_of_filtered_points = 0

        # Load configuration
        circuit_dto = dct.HandleDabDto.load_from_file(os.path.join(filter_data.filtered_list_pathname, f"{circuit_filtered_point_file}.pkl"))
        # Check number of trials
        if act_target_number_trials > 0:
            fmt.optimization.StackedTransformerOptimization.ReluctanceModel.start_proceed_study(
                act_sto_config, target_number_trials=act_target_number_trials)
        else:
            logger.info(f"Target number of trials = {act_target_number_trials} which are less equal 0!. No simulation is performed")
            return 0

        # Filter reluctance model results
        df = fmt.optimization.StackedTransformerOptimization.ReluctanceModel.study_to_df(act_sto_config)
        df_filtered = fmt.optimization.StackedTransformerOptimization.ReluctanceModel.filter_loss_list_df(
            df, factor_min_dc_losses=factor_dc_losses_min_max_list[0], factor_max_dc_losses=factor_dc_losses_min_max_list[1])
        if debug.general.is_debug:
            # reduce dataset to the given number from the debug configuration
            df_filtered = df_filtered.iloc[:debug.transformer.number_reluctance_working_point_max]

        # Assemble configuration path
        config_filepath = os.path.join(act_sto_config.stacked_transformer_optimization_directory,
                                       f"{act_sto_config.stacked_transformer_study_name}.pkl")

        # sweep through all current waveforms
        i_l1_sorted = np.transpose(circuit_dto.calc_currents.i_l_1_sorted, (1, 2, 3, 0))
        angles_rad_sorted = np.transpose(circuit_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))

        all_operation_point_geometry_numbers_list = df_filtered["number"].to_numpy()

        # Overtake the filtered operation points
        number_of_filtered_points = len(all_operation_point_geometry_numbers_list)

        logger.info(f"Full-operating point simulation list: {all_operation_point_geometry_numbers_list}")

        for single_geometry_number in tqdm.tqdm(all_operation_point_geometry_numbers_list):
            df_geometry_re_simulation_number = df_filtered[df_filtered["number"] == single_geometry_number]

            result_array = np.full_like(circuit_dto.calc_modulation.phi, np.nan)

            new_circuit_dto_directory = os.path.join(act_sto_config.stacked_transformer_optimization_directory,
                                                     "08_circuit_dtos_incl_reluctance_transformer_losses")
            if not os.path.exists(new_circuit_dto_directory):
                os.makedirs(new_circuit_dto_directory)

            if os.path.exists(os.path.join(new_circuit_dto_directory, f"{single_geometry_number}.pkl")):
                logger.info(f"Re-simulation of {circuit_dto.name} already exists. Skip.")
            else:
                for vec_vvp in np.ndindex(circuit_dto.calc_modulation.phi.shape):

                    time = dct.functions_waveforms.full_angle_waveform_from_angles(
                        angles_rad_sorted[vec_vvp]) / 2 / np.pi / circuit_dto.input_config.fs
                    current = dct.functions_waveforms.full_current_waveform_from_currents(i_l1_sorted[vec_vvp])

                    current_waveform = np.array([time, current])

                    logger.debug(f"{current_waveform=}")
                    logger.debug("----------------------")
                    logger.debug("Re-simulation of:")
                    logger.debug(f"   * Circuit study: {filter_data.circuit_study_name}")
                    logger.debug(f"   * Circuit trial: {circuit_filtered_point_file}")
                    logger.debug(f"   * Transformer study: {act_sto_config.stacked_transformer_study_name}")
                    logger.debug(f"   * Transformer re-simulation trial: {single_geometry_number}")

                    volume, combined_losses, area_to_heat_sink = fmt.StackedTransformerOptimization.ReluctanceModel.full_simulation(
                        df_geometry_re_simulation_number, current_waveform, config_filepath)
                    result_array[vec_vvp] = combined_losses

                results_dto = dct.StackedTransformerResults(
                    p_combined_losses=result_array,
                    volume=volume,
                    area_to_heat_sink=area_to_heat_sink,
                    circuit_trial_file=circuit_filtered_point_file,
                    stacked_transformer_trial_number=single_geometry_number
                )

                pickle_file = os.path.join(new_circuit_dto_directory, f"{int(single_geometry_number)}.pkl")
                with open(pickle_file, 'wb') as output:
                    pickle.dump(results_dto, output, pickle.HIGHEST_PROTOCOL)

        # returns the number of filtered results
        return number_of_filtered_points

    # Simulation handler. Later the simulation handler starts a process per list entry.
    def optimization_handler_reluctance_model(self, filter_data: dct.FilterData, target_number_trials: int,
                                              factor_dc_losses_min_max_list: list[float] | None, debug: dct.Debug) -> None:
        """
        Control the multi simulation processes.

        :param filter_data : Information about the filtered circuit designs
        :type  filter_data : dct.FilterData
        :param target_number_trials: Number of trials for the optimization
        :type  target_number_trials: int
        :param factor_dc_losses_min_max_list: Filter factor for the offset, related to the minimum/maximum DC losses
        :type  factor_dc_losses_min_max_list: float
        :param debug: Debug DTO
        :type  debug: dct.Debug
        """
        if factor_dc_losses_min_max_list is None:
            factor_dc_losses_min_max_list = [0.01, 100]

        number_cpus = cpu_count()

        with Pool(processes=number_cpus) as pool:
            parameters = []

            for count, act_optimization_configuration in enumerate(self._optimization_config_list):
                if debug.general.is_debug:
                    # in debug mode, stop when number of configuration parameters has reached the same as parallel cores are used
                    if count == number_cpus:
                        break
                # Update statistical data
                # with self._t_lock_stat:
                #     self._progress_run_time.reset_start_trigger()
                #     act_optimization_configuration.progress_data.run_time = self._progress_run_time.get_runtime()
                #     act_optimization_configuration.progress_data.progress_status = ProgressStatus.InProgress

                parameters.append((
                    act_optimization_configuration.circuit_filtered_point_filename,
                    act_optimization_configuration.transformer_optimization_dto,
                    filter_data, target_number_trials, factor_dc_losses_min_max_list,
                    debug))

                # # Update statistical data
                # with self._t_lock_stat:
                #     self._progress_run_time.stop_trigger()
                #     act_optimization_configuration.progress_data.run_time = self._progress_run_time.get_runtime()
                #     act_optimization_configuration.progress_data.number_of_filtered_points = number_of_filtered_point
                #     act_optimization_configuration.progress_data.progress_status = ProgressStatus.Done
                #     # Increment performed calculation counter
                #     self._number_performed_calculations = self._number_performed_calculations + 1

            pool.starmap(func=TransformerOptimization._optimize_reluctance_model, iterable=parameters)

    def fem_simulation_handler(self, filter_data: dct.FilterData, factor_dc_losses_min_max_list: list[float] | None, debug: dct.Debug) -> None:
        """
        Control the multi simulation processes.

        :param filter_data: Information about the filtered designs
        :type  filter_data: dct.FilterData
        :param factor_dc_losses_min_max_list: Filter factor for min and max losses to use filter the results
        :type  factor_dc_losses_min_max_list: float
        :param debug: Debug DTO
        :type debug: dct.Debug
        """
        if factor_dc_losses_min_max_list is None:
            factor_dc_losses_min_max_list = [1.0, 100]

        number_cpus = cpu_count()

        with Pool(processes=number_cpus) as pool:
            parameters = []
            for count, act_optimization_configuration in enumerate(self._optimization_config_list):
                if debug.general.is_debug:
                    # in debug mode, stop when number of configuration parameters has reached the same as parallel cores are used
                    if count == number_cpus:
                        break

                parameters.append((act_optimization_configuration.circuit_filtered_point_filename,
                                   act_optimization_configuration.transformer_optimization_dto,
                                   filter_data,
                                   factor_dc_losses_min_max_list,
                                   debug))

            pool.starmap(func=TransformerOptimization._fem_simulation, iterable=parameters)

    @staticmethod
    def _fem_simulation(circuit_filtered_point_file: str, act_sto_config: fmt.StoSingleInputConfig, filter_data: dct.FilterData,
                        factor_dc_losses_min_max_list: list[float], debug: dct.Debug) -> None:
        """
        Perform the optimization.

        :param circuit_filtered_point_file: Filename of the filtered optimal electrical circuit
        :type  circuit_filtered_point_file: str
        :param act_sto_config: stacked transformer configuration for the optimization
        :type  act_sto_config: fmt.StackedTransformerOptimizationDTO
        :param filter_data: Contains information about filtered circuit designs
        :type  filter_data: dct.FilterData
        :param factor_dc_losses_min_max_list: Filter factor to use filter the results min and max values
        :type  factor_dc_losses_min_max_list: list[float]
        :param debug: Debug DTO
        :type debug: dct.Debug
        """
        # Number of filtered operating points
        number_of_filtered_points = 0

        process_number = current_process().name

        # Load configuration
        circuit_dto = dct.HandleDabDto.load_from_file(os.path.join(filter_data.filtered_list_pathname, f"{circuit_filtered_point_file}.pkl"))

        # Filter study. Use same filter as in the reluctance model
        df = fmt.optimization.StackedTransformerOptimization.ReluctanceModel.study_to_df(act_sto_config)
        df_filtered = fmt.optimization.StackedTransformerOptimization.ReluctanceModel.filter_loss_list_df(
            df, factor_min_dc_losses=factor_dc_losses_min_max_list[0], factor_max_dc_losses=factor_dc_losses_min_max_list[1])
        if debug.general.is_debug:
            # reduce dataset to the given number from the debug configuration
            df_filtered = df_filtered.iloc[:debug.transformer.number_fem_working_point_max]

        # Assemble configuration path
        config_filepath = os.path.join(act_sto_config.stacked_transformer_optimization_directory,
                                       f"{act_sto_config.stacked_transformer_study_name}.pkl")

        # sweep through all current waveforms
        i_l1_sorted = np.transpose(circuit_dto.calc_currents.i_l_1_sorted, (1, 2, 3, 0))
        angles_rad_sorted = np.transpose(circuit_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))

        re_simulate_numbers = df_filtered["number"].to_numpy()

        # Overtake the filtered operation points
        number_of_filtered_points = len(re_simulate_numbers)

        for re_simulate_number in re_simulate_numbers:
            logger.info(f"{re_simulate_number=}")
            df_geometry_re_simulation_number = df_filtered[df_filtered["number"] == re_simulate_number]

            result_array = np.full_like(circuit_dto.calc_modulation.phi, np.nan)

            new_circuit_dto_directory = os.path.join(act_sto_config.stacked_transformer_optimization_directory,
                                                     "09_circuit_dtos_incl_transformer_losses")
            if not os.path.exists(new_circuit_dto_directory):
                os.makedirs(new_circuit_dto_directory)

            if os.path.exists(os.path.join(new_circuit_dto_directory, f"{re_simulate_number}.pkl")):
                logger.info(f"Re-simulation of {circuit_dto.name} already exists. Skip.")
            else:
                # The femmt simulation (full_simulation()) can raise different errors, most of them are geometry errors
                # e.g. winding is not fitting in the winding window
                try:
                    for vec_vvp in tqdm.tqdm(np.ndindex(circuit_dto.calc_modulation.phi.shape),
                                             total=len(circuit_dto.calc_modulation.phi.flatten())):

                        time = dct.functions_waveforms.full_angle_waveform_from_angles(
                            angles_rad_sorted[vec_vvp]) / 2 / np.pi / circuit_dto.input_config.fs
                        current = dct.functions_waveforms.full_current_waveform_from_currents(i_l1_sorted[vec_vvp])

                        current_waveform = np.array([time, current])

                        logger.debug(f"{current_waveform=}")
                        logger.debug("----------------------")
                        logger.debug("Re-simulation of:")
                        logger.debug(f"   * Circuit study: {filter_data.circuit_study_name}")
                        logger.debug(f"   * Circuit trial: {circuit_filtered_point_file}")
                        logger.debug(f"   * Transformer study: {act_sto_config.stacked_transformer_study_name}")
                        logger.debug(f"   * Transformer re-simulation trial: {re_simulate_number}")

                        volume, combined_losses, area_to_heat_sink = fmt.StackedTransformerOptimization.FemSimulation.full_simulation(
                            df_geometry_re_simulation_number, current_waveform, config_filepath, show_visual_outputs=False,
                            process_number=process_number)

                        result_array[vec_vvp] = combined_losses

                    results_dto = dct.StackedTransformerResults(
                        p_combined_losses=result_array,
                        volume=volume,
                        area_to_heat_sink=area_to_heat_sink,
                        circuit_trial_file=circuit_filtered_point_file,
                        stacked_transformer_trial_number=re_simulate_number
                    )

                    pickle_file = os.path.join(new_circuit_dto_directory, f"{int(re_simulate_number)}.pkl")
                    with open(pickle_file, 'wb') as output:
                        pickle.dump(results_dto, output, pickle.HIGHEST_PROTOCOL)
                except:
                    logger.info(f"Re-simulation of transformer geometry {re_simulate_number} not possible due to non-possible geometry.")

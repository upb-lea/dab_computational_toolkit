"""Inductor optimization class."""
# python libraries
import os
import pickle
import logging
import copy
import threading
from multiprocessing import Pool, cpu_count, current_process

# 3rd party libraries
import numpy as np
import tqdm

# own libraries
import femmt as fmt
import dct
from dct.boundary_check import CheckCondition as c_flag
from dct.components.inductor_optimization_dtos import InductorOptimizationDto
from dct.components.component_dtos import InductorResults
from dct.server_ctl_dtos import ProgressData
from dct.server_ctl_dtos import ProgressStatus
from dct.server_ctl_dtos import RunTimeMeasurement as RunTime
from dct.datasets_dtos import StudyData
from dct.datasets_dtos import FilterData
import dct.topology.dab.dab_functions_waveforms as dabwav
import dct.topology.dab.dab_datasets as dab_dset
from dct.constant_path import CIRCUIT_INDUCTOR_RELUCTANCE_LOSSES_FOLDER, CIRCUIT_INDUCTOR_LOSSES_FOLDER
from dct.components.component_dtos import InductorRequirements
from dct.toml_checker import TomlInductor
# configure root logger
logger = logging.getLogger(__name__)

class InductorOptimization:
    """Optimization of the inductor."""

    # Declaration of member types
    _optimization_config_list: list[InductorOptimizationDto]
    _i_lock_stat: threading.Lock
    _progress_run_time: RunTime

    def __init__(self) -> None:
        """Initialize the configuration list for the inductor optimizations."""
        self._i_lock_stat: threading.Lock = threading.Lock()
        self._optimization_config_list = []
        self._number_performed_calculations: int = 0
        self._progress_run_time: RunTime = RunTime()

    @staticmethod
    def verify_optimization_parameter(toml_inductor: dct.TomlInductor) -> tuple[bool, str]:
        """Verify the input parameter ranges.

        :param toml_inductor: toml inductor configuration
        :type toml_inductor: dct.TomlInductor
        :return: True, if the configuration was consistent
        :rtype: bool
        """
        # Variable declaration
        inconsistency_report: str = ""
        is_consistent: bool = True
        keyword_dictionary: dict
        toml_check_min_max_values_list: list[tuple[list[float], str]]
        toml_check_value_list: list[tuple[float, str]]

        # Design space parameter check
        # Check core_name_list
        if len(toml_inductor.design_space.core_name_list) != 0:
            # Get available keywords
            keyword_dictionary = fmt.core_database()
            # Perform dictionary check
            for keyword_entry in toml_inductor.design_space.core_name_list:
                is_check_passed, issue_report = dct.BoundaryCheck.check_dictionary(keyword_dictionary, keyword_entry, "core_name_list")
                # Check if boundary check fails
                if not is_check_passed:
                    inconsistency_report = inconsistency_report + issue_report
                    is_consistent = False
        else:

            toml_check_min_max_values_list = (
                [(toml_inductor.design_space.core_inner_diameter_min_max_list, "core_inner_diameter_min_max_list"),
                 (toml_inductor.design_space.window_h_min_max_list, "window_h_min_max_list"),
                 (toml_inductor.design_space.window_w_min_max_list, "window_w_min_max_list")])

            # Perform the boundary check
            is_check_passed, issue_report = dct.BoundaryCheck.check_float_min_max_values_list(
                0, 5, toml_check_min_max_values_list, c_flag.check_exclusive, c_flag.check_exclusive)
            if not is_check_passed:
                inconsistency_report = inconsistency_report + issue_report
                is_consistent = False

        # Check litz_wire_list
        # Get available keywords
        keyword_dictionary = fmt.litz_database()
        # Perform dictionary check
        for keyword_entry in toml_inductor.design_space.litz_wire_name_list:
            is_check_passed, issue_report = dct.BoundaryCheck.check_dictionary(keyword_dictionary, keyword_entry, "litz_wire_name_list")
            # Check if boundary check fails
            if not is_check_passed:
                inconsistency_report = inconsistency_report + issue_report
                is_consistent = False

        # Insulation parameter check
        toml_check_value_list = (
            [(toml_inductor.insulations.primary_to_primary, "primary_to_primary"),
             (toml_inductor.insulations.core_bot, "core_bot"), (toml_inductor.insulations.core_top, "core_top"),
             (toml_inductor.insulations.core_right, "core_right"), (toml_inductor.insulations.core_left, "core_left")])

        # Perform insulation value check
        # Perform the boundary check
        is_check_passed, issue_report = dct.BoundaryCheck.check_float_value_list(
            0, 0.1, toml_check_value_list, c_flag.check_exclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Perform temperature value check
        # Perform the boundary check
        is_check_passed, issue_report = dct.BoundaryCheck.check_float_value(
            -40, 175, toml_inductor.boundary_conditions.temperature, "temperature", c_flag.check_inclusive, c_flag.check_inclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Perform filter_distance value check
        group_name = "filter_distance"
        # Perform the boundary check
        is_check_passed, issue_report = dct.BoundaryCheck.check_float_min_max_values(
            0, 100, toml_inductor.filter_distance.factor_dc_losses_min_max_list,
            f"{group_name}: factor_dc_losses_min_max_list", c_flag.check_exclusive, c_flag.check_inclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        return is_consistent, inconsistency_report

    def initialize_inductor_optimization_list(self, toml_inductor: TomlInductor, inductor_study_data: StudyData,
                                              inductor_requirements_list: list[InductorRequirements]) -> None:
        """
        Initialize the inductor optimization.

        :param toml_inductor: inductor data
        :type toml_inductor: TomlInductor
        :param inductor_study_data: inductor study data
        :type inductor_study_data: StudyData
        :param inductor_requirements_list: list with inductor requirements
        :type inductor_requirements_list: list[InductorRequirements]
        """
        # Create the io_config_list for all trials
        for inductor_requirements in inductor_requirements_list:
            circuit_id = inductor_requirements.circuit_id
            trial_directory = os.path.join(inductor_study_data.optimization_directory, circuit_id, inductor_study_data.study_name)

            # catch mypy type issue
            if not isinstance(inductor_requirements, InductorRequirements):
                raise TypeError("circuit DTO file is incomplete.")

            # Insulation parameter
            act_insulations = fmt.InductorInsulationDTO(primary_to_primary=toml_inductor.insulations.primary_to_primary,
                                                        core_bot=toml_inductor.insulations.core_bot,
                                                        core_top=toml_inductor.insulations.core_top,
                                                        core_right=toml_inductor.insulations.core_right,
                                                        core_left=toml_inductor.insulations.core_left)

            # Initialize the material data source
            act_material_data_sources = fmt.MaterialDataSources(
                permeability_datasource=toml_inductor.material_data_sources.permeability_datasource,
                permittivity_datasource=toml_inductor.material_data_sources.permittivity_datasource,
            )

            # Create fix part of io_config
            inductor_requirements_dto = fmt.InductorOptimizationDTO(
                inductor_study_name=inductor_study_data.study_name,
                core_name_list=toml_inductor.design_space.core_name_list,
                material_name_list=toml_inductor.design_space.material_name_list,
                core_inner_diameter_min_max_list=toml_inductor.design_space.core_inner_diameter_min_max_list,
                window_h_min_max_list=toml_inductor.design_space.window_h_min_max_list,
                window_w_min_max_list=toml_inductor.design_space.window_w_min_max_list,
                litz_wire_name_list=toml_inductor.design_space.litz_wire_name_list,
                insulations=act_insulations,
                target_inductance=inductor_requirements.target_inductance,
                temperature=toml_inductor.boundary_conditions.temperature,
                time_current_vec=[inductor_requirements.time_vec, inductor_requirements.current_vec],
                inductor_optimization_directory=os.path.join(inductor_study_data.optimization_directory, circuit_id, inductor_study_data.study_name),
                material_data_sources=act_material_data_sources)

            # Initialize the statistical data
            stat_data_init: ProgressData = ProgressData(run_time=0, number_of_filtered_points=0,
                                                        progress_status=ProgressStatus.Idle)

            inductor_optimization_dto = InductorOptimizationDto(
                circuit_filtered_point_filename=circuit_id,
                progress_data=copy.deepcopy(stat_data_init),
                inductor_optimization_dto=inductor_requirements_dto)

            self._optimization_config_list.append(inductor_optimization_dto)

    def get_progress_data(self, filtered_list_id: int) -> ProgressData:
        """Provide the progress data of the optimization.

        :param filtered_list_id: List index of the filtered operation point from circuit
        :type  filtered_list_id: int

        :return: Progress data: Processing start time, actual processing time, number of filtered inductor Pareto front points and status.
        :rtype: ProgressData
        """
        # Variable declaration and default initialization
        ret_progress_data: ProgressData = ProgressData(run_time=0, number_of_filtered_points=0,
                                                       progress_status=ProgressStatus.Idle)

        # Check for valid filtered_list_id
        if len(self._optimization_config_list) > filtered_list_id:
            # Lock statistical performance data access
            with self._i_lock_stat:
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
        with self._i_lock_stat:
            act_number_performed_calculations = self._number_performed_calculations

        return act_number_performed_calculations

    # Simulation handler. Later the simulation handler starts a process per list entry.
    @staticmethod
    def _optimize_reluctance_model(circuit_filtered_point_file: str, act_io_config: fmt.InductorOptimizationDTO, filter_data: FilterData,
                                   target_number_trials: int, factor_dc_losses_min_max_list: list[float], debug: dct.Debug) -> int:
        """
        Perform the optimization.

        :param circuit_filtered_point_file: Filename of the filtered optimal electrical circuit
        :type  circuit_filtered_point_file: str
        :param act_io_config: inductor configuration for the optimization
        :type  act_io_config: fmt.InductorOptimizationDTO
        :param filter_data: Contains information about filtered circuit designs
        :type  filter_data: FilterData
        :param target_number_trials: Number of trials for the optimization
        :type  target_number_trials: int
        :param factor_dc_losses_min_max_list: Filter factor to use filter the results min and max values
        :type  factor_dc_losses_min_max_list: list[float]
        :param debug: Debug mode flag
        :type debug: bool
        """
        number_of_filtered_points = 0

        # Load configuration
        circuit_dto = dab_dset.HandleDabDto.load_from_file(os.path.join(filter_data.filtered_list_pathname, f"{circuit_filtered_point_file}.pkl"))
        # Check number of trials
        if target_number_trials > 0:
            fmt.optimization.InductorOptimization.ReluctanceModel.start_proceed_study(act_io_config, target_number_trials=target_number_trials)
        else:
            logger.info(f"Target number of trials = {target_number_trials} which are less equal 0!. No simulation is performed")
            return 0

        # filter the reluctance model data Pareto front
        df = fmt.optimization.InductorOptimization.ReluctanceModel.study_to_df(act_io_config)
        df_filtered = fmt.optimization.InductorOptimization.ReluctanceModel.filter_loss_list_df(
            df, factor_min_dc_losses=factor_dc_losses_min_max_list[0], factor_max_dc_losses=factor_dc_losses_min_max_list[1])
        if debug.general.is_debug:
            # reduce dataset to the given number from the debug configuration
            df_filtered = df_filtered.iloc[:debug.inductor.number_reluctance_working_point_max]

        config_filepath = os.path.join(act_io_config.inductor_optimization_directory, f"{act_io_config.inductor_study_name}.pkl")

        # sweep through all current waveforms
        i_l1_sorted = np.transpose(circuit_dto.calc_currents.i_l_1_sorted, (1, 2, 3, 0))
        angles_rad_sorted = np.transpose(circuit_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))

        all_operation_point_geometry_numbers_list = df_filtered["number"].to_numpy()

        # Overtake the filtered operation points
        number_of_filtered_points = len(all_operation_point_geometry_numbers_list)

        logger.info(f"Full-operating point simulation list: {all_operation_point_geometry_numbers_list}")

        # simulate all operating points
        for single_geometry_number in tqdm.tqdm(all_operation_point_geometry_numbers_list):
            df_geometry_re_simulation_number = df_filtered[df_filtered["number"] == float(single_geometry_number)]

            logger.debug(f"single_geometry_number: \n"
                         f"    {df_geometry_re_simulation_number.head()}")

            combined_loss_array = np.full_like(circuit_dto.calc_modulation.phi, np.nan)

            new_circuit_dto_directory = os.path.join(act_io_config.inductor_optimization_directory,
                                                     CIRCUIT_INDUCTOR_RELUCTANCE_LOSSES_FOLDER)
            if not os.path.exists(new_circuit_dto_directory):
                os.makedirs(new_circuit_dto_directory)

            if os.path.exists(os.path.join(new_circuit_dto_directory, f"{single_geometry_number}.pkl")):
                logger.info(f"Re-simulation of {circuit_dto.circuit_id} already exists. Skip.")
            else:
                # The femmt simulation (full_simulation()) can raise different errors, most of them are geometry errors
                # e.g. winding is not fitting in the winding window
                try:
                    for vec_vvp in np.ndindex(circuit_dto.calc_modulation.phi.shape):
                        time, unique_indices = np.unique(dabwav.full_angle_waveform_from_angles(
                            angles_rad_sorted[vec_vvp]) / 2 / np.pi / circuit_dto.input_config.fs, return_index=True)
                        current = dabwav.full_current_waveform_from_currents(i_l1_sorted[vec_vvp])[unique_indices]

                        current_waveform = np.array([time, current])
                        logger.debug(f"{current_waveform=}")
                        logger.debug("All operating point simulation of:")
                        logger.debug(f"   * Circuit study: {filter_data.circuit_study_name}")
                        logger.debug(f"   * Circuit trial: {circuit_filtered_point_file}")
                        logger.debug(f"   * Inductor study: {act_io_config.inductor_study_name}")
                        logger.debug(f"   * Inductor re-simulation trial: {single_geometry_number}")

                        volume, combined_losses, area_to_heat_sink = fmt.InductorOptimization.ReluctanceModel.full_simulation(
                            df_geometry_re_simulation_number, current_waveform=current_waveform,
                            inductor_config_filepath=config_filepath)
                        combined_loss_array[vec_vvp] = combined_losses

                    inductor_losses = InductorResults(
                        loss_array=combined_loss_array,
                        volume=volume,
                        area_to_heat_sink=area_to_heat_sink,
                        circuit_id=circuit_filtered_point_file,
                        inductor_id=single_geometry_number
                    )

                    pickle_file = os.path.join(new_circuit_dto_directory, f"{int(single_geometry_number)}.pkl")
                    with open(pickle_file, 'wb') as output:
                        pickle.dump(inductor_losses, output, pickle.HIGHEST_PROTOCOL)
                except:
                    logger.info(f"Re-simulation of inductor geometry {single_geometry_number} not possible due to non-possible geometry.")

        # returns the number of filtered results
        return number_of_filtered_points

    def optimization_handler_reluctance_model(self, filter_data: FilterData, target_number_trials: int,
                                              factor_dc_losses_min_max_list: list[float] | None, debug: dct.Debug) -> None:
        """
        Control the multi simulation processes.

        :param filter_data: Information about the filtered designs
        :type  filter_data: FilterData
        :param target_number_trials: Number of trials for the optimization
        :type  target_number_trials: int
        :param factor_dc_losses_min_max_list: Filter factor for min and max losses to use filter the results
        :type  factor_dc_losses_min_max_list: float
        :param debug: True to use debug mode which stops earlier
        :type debug: bool
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

                # Update statistical data
                # with self._i_lock_stat:
                #     # Start the progress time measurement
                #     self._progress_run_time.reset_start_trigger()
                #     act_optimization_configuration.progress_data.run_time = self._progress_run_time.get_runtime()
                #     act_optimization_configuration.progress_data.progress_status = ProgressStatus.InProgress

                parameters.append((
                    act_optimization_configuration.circuit_filtered_point_filename,
                    act_optimization_configuration.inductor_optimization_dto,
                    filter_data,
                    target_number_trials,
                    factor_dc_losses_min_max_list,
                    debug
                ))

                # Update statistical data
                # with self._i_lock_stat:
                #     self._progress_run_time.stop_trigger()
                #     act_optimization_configuration.progress_data.run_time = self._progress_run_time.get_runtime()
                #     act_optimization_configuration.progress_data.progress_status = ProgressStatus.Done
                #     act_optimization_configuration.progress_data.number_of_filtered_points = number_of_filtered_points
                #     # Increment performed calculation counter
                #     self._number_performed_calculations = self._number_performed_calculations + 1

            pool.starmap(func=InductorOptimization._optimize_reluctance_model, iterable=parameters)

    def fem_simulation_handler(self, filter_data: FilterData, factor_dc_losses_min_max_list: list[float] | None, debug: dct.Debug) -> None:
        """
        Control the multi simulation processes.

        :param filter_data: Information about the filtered designs
        :type  filter_data: FilterData
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
                                   act_optimization_configuration.inductor_optimization_dto,
                                   filter_data,
                                   factor_dc_losses_min_max_list,
                                   debug))

            pool.starmap(func=InductorOptimization._fem_simulation, iterable=parameters)

    # Simulation handler. Later the simulation handler starts a process per list entry.
    @staticmethod
    def _fem_simulation(circuit_id: str, act_io_config: fmt.InductorOptimizationDTO, filter_data: FilterData,
                        factor_dc_losses_min_max_list: list[float], debug: dct.Debug) -> None:
        """
        Perform the optimization.

        :param circuit_id: Filename of the filtered optimal electrical circuit
        :type  circuit_id: str
        :param act_io_config: inductor configuration for the optimization
        :type  act_io_config: fmt.InductorOptimizationDTO
        :param filter_data: Contains information about filtered circuit designs
        :type  filter_data: FilterData
        :param factor_dc_losses_min_max_list: Filter factor to use filter the results min and max values
        :type  factor_dc_losses_min_max_list: list[float]
        :param debug: Debug DTO
        :type debug: dct.Debug
        """
        number_of_filtered_points = 0

        process_number = current_process().name

        # Load configuration
        circuit_dto = dab_dset.HandleDabDto.load_from_file(os.path.join(filter_data.filtered_list_pathname, f"{circuit_id}.pkl"))

        df = fmt.optimization.InductorOptimization.ReluctanceModel.study_to_df(act_io_config)
        df_filtered = fmt.optimization.InductorOptimization.ReluctanceModel.filter_loss_list_df(
            df, factor_min_dc_losses=factor_dc_losses_min_max_list[0], factor_max_dc_losses=factor_dc_losses_min_max_list[1])
        if debug.general.is_debug:
            # reduce dataset to the fist given number from the debug configuration file
            df_filtered = df_filtered.iloc[:debug.inductor.number_fem_working_point_max]

        config_filepath = os.path.join(act_io_config.inductor_optimization_directory, f"{act_io_config.inductor_study_name}.pkl")

        # sweep through all current waveforms
        i_l1_sorted = np.transpose(circuit_dto.calc_currents.i_l_1_sorted, (1, 2, 3, 0))
        angles_rad_sorted = np.transpose(circuit_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))

        all_operation_point_geometry_numbers_list = df_filtered["number"].to_numpy()

        # Overtake the filtered operation points
        number_of_filtered_points = len(all_operation_point_geometry_numbers_list)

        logger.info(f"Full-operating point simulation list: {all_operation_point_geometry_numbers_list}")

        # simulate all operating points
        for inductor_id in tqdm.tqdm(all_operation_point_geometry_numbers_list):
            try:
                # generate a new df with only a single entry (with the geometry data)
                df_geometry_re_simulation_number = df_filtered[df_filtered["number"] == float(inductor_id)]

                logger.debug(f"single_geometry_number: \n"
                             f"    {df_geometry_re_simulation_number.head()}")

                combined_loss_array = np.full_like(circuit_dto.calc_modulation.phi, np.nan)

                new_circuit_dto_directory = os.path.join(act_io_config.inductor_optimization_directory,
                                                         CIRCUIT_INDUCTOR_LOSSES_FOLDER)
                if not os.path.exists(new_circuit_dto_directory):
                    os.makedirs(new_circuit_dto_directory)

                if os.path.exists(os.path.join(new_circuit_dto_directory, f"{inductor_id}.pkl")):
                    logger.info(f"Re-simulation of {circuit_dto.circuit_id} already exists. Skip.")
                else:
                    for vec_vvp in tqdm.tqdm(np.ndindex(circuit_dto.calc_modulation.phi.shape),
                                             total=len(circuit_dto.calc_modulation.phi.flatten())):
                        time, unique_indices = np.unique(dabwav.full_angle_waveform_from_angles(
                            angles_rad_sorted[vec_vvp]) / 2 / np.pi / circuit_dto.input_config.fs, return_index=True)
                        current = dabwav.full_current_waveform_from_currents(i_l1_sorted[vec_vvp])[unique_indices]

                        current_waveform = np.array([time, current])
                        logger.debug(f"{current_waveform=}")
                        logger.debug("All operating point simulation of:")
                        logger.debug(f"   * Circuit study: {filter_data.circuit_study_name}")
                        logger.debug(f"   * Circuit ID: {circuit_id}")
                        logger.debug(f"   * Inductor study: {act_io_config.inductor_study_name}")
                        logger.debug(f"   * Inductor ID: {inductor_id}")

                        volume, combined_losses, area_to_heat_sink = fmt.InductorOptimization.FemSimulation.full_simulation(
                            df_geometry_re_simulation_number, current_waveform=current_waveform,
                            inductor_config_filepath=config_filepath, process_number=process_number, print_derivations=False)
                        combined_loss_array[vec_vvp] = combined_losses

                    inductor_losses = InductorResults(
                        loss_array=combined_loss_array,
                        volume=volume,
                        area_to_heat_sink=area_to_heat_sink,
                        circuit_id=circuit_id,
                        inductor_id=inductor_id
                    )

                    pickle_file = os.path.join(new_circuit_dto_directory, f"{int(inductor_id)}.pkl")
                    with open(pickle_file, 'wb') as output:
                        pickle.dump(inductor_losses, output, pickle.HIGHEST_PROTOCOL)
            except:
                logger.warning(f"for number {inductor_id} an operation point exceeds the boundary!")

"""Capacitor selection."""

# python libraries
import logging
import os
from multiprocessing import Pool, cpu_count
import copy
import pickle

# 3rd party libraries
import numpy as np
import pandas as pd
import tqdm

# own libraries
import pecst
from dct.components.capacitor_optimization_dtos import CapacitorOptimizationDto
from dct.toml_checker import TomlCapacitorSelection, Debug
from dct.datasets_dtos import StudyData
from dct.datasets_dtos import FilterData
from dct.server_ctl_dtos import ProgressData, ProgressStatus
from dct.components.component_requirements import CapacitorRequirements
from dct.components.capacitor_optimization_dtos import CapacitorResults
from dct.constant_path import CIRCUIT_CAPACITOR_LOSS_FOLDER

logger = logging.getLogger(__name__)

class CapacitorSelection:
    """Select suitable capacitors."""

    c_df: pd.DataFrame
    _optimization_config_list: list[CapacitorOptimizationDto]

    def __init__(self):
        self._optimization_config_list = []

    @staticmethod
    def verify_optimization_parameter(toml_capacitor: TomlCapacitorSelection) -> tuple[bool, str]:
        """
        Verify the parameters from toml file for the capacitor optimization.

        Dummy method so far.

        :param toml_capacitor: capacitor toml file to check
        :type toml_capacitor: TomlCapacitorSelection
        :return: is_consistent, issue_report
        :rtype: tuple[bool, str]
        """
        if toml_capacitor:
            pass
        return True, ""

    def initialize_capacitor_selection(self, toml_capacitor_list: list[TomlCapacitorSelection], capacitor_study_data: StudyData,
                                       circuit_filter_data: FilterData,
                                       capacitor_requirements_list: list[CapacitorRequirements]) -> None:
        """
        Initialize the capacitor selection.

        :param toml_capacitor_list: capacitor data in a list
        :type toml_capacitor_list: list[TomlCapacitorSelection]
        :param capacitor_study_data: capacitor study data
        :type capacitor_study_data: StudyData
        :param circuit_filter_data: filtered circuit data
        :type circuit_filter_data: FilterData
        :param capacitor_requirements_list: list with capacitor requirements
        :type capacitor_requirements_list: list[CapacitorRequirements]
        """
        pecst.download_esr_csv_files()

        # Create the io_config_list for all trials
        for capacitor_requirements in capacitor_requirements_list:

            circuit_id = capacitor_requirements.circuit_id
            trial_directory = os.path.join(capacitor_study_data.optimization_directory, circuit_id, capacitor_study_data.study_name)

            # catch mypy type issue
            if not isinstance(capacitor_requirements, CapacitorRequirements):
                raise TypeError("circuit DTO file is incomplete.")

            # generate capacitor requirements from circuit simulation data
            capacitor_requirements_dto = pecst.CapacitorRequirements(
                maximum_peak_to_peak_voltage_ripple=toml_capacitor_list[capacitor_requirements.capacitor_number_in_circuit].maximum_peak_to_peak_voltage_ripple,
                current_waveform_for_op_max_current=np.array([capacitor_requirements.time_vec,
                                                              capacitor_requirements.current_vec]),
                v_dc_for_op_max_voltage=capacitor_requirements.v_dc_max,
                temperature_ambient=toml_capacitor_list[capacitor_requirements.capacitor_number_in_circuit].temperature_ambient,
                voltage_safety_margin_percentage=toml_capacitor_list[capacitor_requirements.capacitor_number_in_circuit].voltage_safety_margin_percentage,
                capacitor_type_list=[pecst.CapacitorType.FilmCapacitor],
                maximum_number_series_capacitors=toml_capacitor_list[capacitor_requirements.capacitor_number_in_circuit].maximum_number_series_capacitors,
                capacitor_tolerance_percent=pecst.CapacitanceTolerance.TenPercent,
                lifetime_h=toml_capacitor_list[capacitor_requirements.capacitor_number_in_circuit].lifetime_h,
                results_directory=trial_directory)

            # Initialize the statistical data
            stat_data_init: ProgressData = ProgressData(run_time=0, number_of_filtered_points=0,
                                                        progress_status=ProgressStatus.Idle)

            capacitor_optimization_dto = CapacitorOptimizationDto(
                circuit_id=circuit_id,
                progress_data=copy.deepcopy(stat_data_init),
                capacitor_optimization_dto=capacitor_requirements_dto,
                capacitor_number_in_circuit=capacitor_requirements.capacitor_number_in_circuit)

            self._optimization_config_list.append(capacitor_optimization_dto)

    @staticmethod
    def _start_optimization(circuit_id: str, act_cst_config: pecst.CapacitorRequirements, filter_data: FilterData,
                            capacitor_requirements: CapacitorRequirements, debug: Debug) -> int:
        # capacitor requirements
        _, c_db_df_list = pecst.select_capacitors(act_cst_config)

        c_db_df = pd.concat(c_db_df_list)

        if not os.path.exists(act_cst_config.results_directory):
            os.makedirs(act_cst_config.results_directory)
        c_db_df.to_csv(f"{act_cst_config.results_directory}/results.csv")

        df_filtered = pecst.filter_df(c_db_df)
        if debug.general.is_debug:
            # reduce dataset to the given number from the debug configuration
            df_filtered = df_filtered.iloc[:debug.capacitor_1.number_working_point_max]
        # save Pareto front designs (reduced in case of active debugging)
        df_filtered.to_csv(f"{act_cst_config.results_directory}/results_filtered.csv")

        all_operation_point_ordering_codes_list = df_filtered["ordering code"].to_numpy()
        all_operation_point_volume_list = df_filtered["volume_total"].to_numpy()
        all_operation_point_area_list = df_filtered["area_total"].to_numpy()
        all_operation_point_n_series_list = df_filtered["in_series_needed"].to_numpy()
        all_operation_point_n_parallel_list = df_filtered["in_parallel_needed"].to_numpy()

        # Overtake the filtered operation points
        number_of_filtered_points = len(all_operation_point_ordering_codes_list)

        logger.info(f"Full-operating point simulation list: {all_operation_point_ordering_codes_list}")

        # simulate all operating points
        for count, ordering_code in enumerate(tqdm.tqdm(all_operation_point_ordering_codes_list)):

            volume_total = all_operation_point_volume_list[count]
            area_total = all_operation_point_area_list[count]
            df_geometry_re_simulation_number = df_filtered[df_filtered["ordering code"] == ordering_code]
            n_series = all_operation_point_n_series_list[count]
            n_parallel = all_operation_point_n_parallel_list[count]

            logger.debug(f"ordering_code: \n"
                         f"    {df_geometry_re_simulation_number.head()}")

            loss_total_array = np.full_like(capacitor_requirements.current_array[:, :, :, 0], np.nan)

            new_circuit_dto_directory = os.path.join(act_cst_config.results_directory, CIRCUIT_CAPACITOR_LOSS_FOLDER)
            if not os.path.exists(new_circuit_dto_directory):
                os.makedirs(new_circuit_dto_directory)

            if os.path.exists(os.path.join(new_circuit_dto_directory, f"{ordering_code}.pkl")):
                logger.info(f"Re-simulation of {ordering_code} already exists. Skip.")
            else:
                for vec_vvp in np.ndindex(capacitor_requirements.current_array[:, :, :, 0].shape):
                    time = capacitor_requirements.time_array[vec_vvp]
                    current = capacitor_requirements.current_array[vec_vvp]

                    current_waveform = np.array([time, current])
                    logger.debug(f"{current_waveform=}")
                    logger.debug("All operating point simulation of:")
                    logger.debug(f"   * Circuit study: {filter_data.circuit_study_name}")
                    logger.debug(f"   * Circuit trial: {circuit_id}")
                    logger.debug(f"   * Inductor re-simulation trial: {ordering_code}")

                    [frequency_list, current_amplitude_list, _] = pecst.fft(current_waveform, plot='no', mode='time', title='fft input current')

                    loss_per_capacitor = pecst.power_loss_film_capacitor(ordering_code, frequency_list, current_amplitude_list,
                                                                         number_parallel_capacitors=n_parallel)

                    loss_total_array[vec_vvp] = loss_per_capacitor * n_series * n_parallel

                capacitor_losses = CapacitorResults(
                    loss_total_array=loss_total_array,
                    volume_total=volume_total,
                    area_total=area_total,
                    circuit_id=circuit_id,
                    capacitor_id=df_geometry_re_simulation_number['ordering code'].values[0],
                    n_series=n_series,
                    n_parallel=n_parallel,
                    capacitor_number_in_circuit=capacitor_requirements.capacitor_number_in_circuit
                )

                pickle_file = os.path.join(new_circuit_dto_directory, f"{ordering_code}.pkl")
                with open(pickle_file, 'wb') as output:
                    pickle.dump(capacitor_losses, output, pickle.HIGHEST_PROTOCOL)

        # returns the number of filtered results
        return number_of_filtered_points

    def optimization_handler(self, filter_data: FilterData, capacitor_requirements_list: list[CapacitorRequirements],
                             debug: Debug) -> None:
        """
        Control the multi simulation processes.

        :param filter_data: Information about the filtered designs
        :type  filter_data: dct.FilterData
        :param debug: True to use debug mode which stops earlier
        :type debug: bool
        :param capacitor_requirements_list: list with capacitor requirements
        :type capacitor_requirements_list: list[CapacitorRequirements]
        """
        number_cpus = cpu_count()

        with Pool(processes=number_cpus) as pool:
            parameters = []
            for count, act_optimization_configuration in enumerate(self._optimization_config_list):
                if debug.general.is_debug:
                    # in debug mode, stop when number of configuration parameters has reached the same as parallel cores are used
                    if count == number_cpus:
                        break

                capacitor_requirements = capacitor_requirements_list[count]

                parameters.append((
                    act_optimization_configuration.circuit_id,
                    act_optimization_configuration.capacitor_optimization_dto,
                    filter_data,
                    capacitor_requirements,
                    debug
                ))

            pool.starmap(func=CapacitorSelection._start_optimization, iterable=parameters)

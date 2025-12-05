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
from dct.capacitor_optimization_dtos import CapacitorOptimizationDto
from dct.toml_checker import TomlCapacitorSelection, Debug
# ASA Need to be moved to general class
from dct.topology.dab.dab_datasets import HandleDabDto
from dct.datasets_dtos import StudyData
from dct.datasets_dtos import FilterData
from dct.server_ctl_dtos import ProgressData, ProgressStatus
from dct.topology.dab.dab_datasets_dtos import ComponentRequirements
from dct.topology.dab.dab_functions_waveforms import full_current_waveform_from_currents, full_angle_waveform_from_angles
from dct.capacitor_optimization_dtos import CapacitorResults

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

    def initialize_capacitor_selection(self, toml_capacitor: TomlCapacitorSelection, capacitor_study_data: StudyData, circuit_filter_data: FilterData) -> None:
        """
        Initialize the capacitor selection.

        :param toml_capacitor: capacitor data
        :type toml_capacitor: TomlCapacitorSelection
        :param capacitor_study_data: capacitor study data
        :type capacitor_study_data: StudyData
        :param circuit_filter_data: filtered circuit data
        :type circuit_filter_data: FilterData
        """
        pecst.download_esr_csv_files()

        # Create the io_config_list for all trials
        for circuit_trial_file in circuit_filter_data.filtered_list_files:
            circuit_filepath = os.path.join(circuit_filter_data.filtered_list_pathname, f"{circuit_trial_file}.pkl")
            # Check filename
            if os.path.isfile(circuit_filepath):
                # Read results from circuit optimization
                circuit_dto = HandleDabDto.load_from_file(circuit_filepath)
                trial_directory = os.path.join(capacitor_study_data.optimization_directory, circuit_trial_file, capacitor_study_data.study_name)

                # catch mypy type issue
                if not isinstance(circuit_dto.component_requirements, ComponentRequirements):
                    raise TypeError("circuit DTO file is incomplete.")

                # generate capacitor requirements from circuit simulation data
                capacitor_requirements_dto = pecst.CapacitorRequirements(
                    maximum_peak_to_peak_voltage_ripple=toml_capacitor.maximum_peak_to_peak_voltage_ripple,
                    current_waveform_for_op_max_current=np.array([circuit_dto.component_requirements.capacitor_requirements[0].time,
                                                                  circuit_dto.component_requirements.capacitor_requirements[0].i_max_rms_current_waveform]),
                    v_dc_for_op_max_voltage=circuit_dto.component_requirements.capacitor_requirements[0].v_max,
                    temperature_ambient=toml_capacitor.temperature_ambient,
                    voltage_safety_margin_percentage=toml_capacitor.voltage_safety_margin_percentage,
                    capacitor_type_list=[pecst.CapacitorType.FilmCapacitor],
                    maximum_number_series_capacitors=toml_capacitor.maximum_number_series_capacitors,
                    capacitor_tolerance_percent=pecst.CapacitanceTolerance.TenPercent,
                    lifetime_h=toml_capacitor.lifetime_h,
                    results_directory=trial_directory)

                # Initialize the statistical data
                stat_data_init: ProgressData = ProgressData(run_time=0, number_of_filtered_points=0,
                                                            progress_status=ProgressStatus.Idle)

                capacitor_dto = CapacitorOptimizationDto(
                    circuit_filtered_point_filename=circuit_trial_file,
                    progress_data=copy.deepcopy(stat_data_init),
                    capacitor_optimization_dto=capacitor_requirements_dto)

                self._optimization_config_list.append(capacitor_dto)
            else:
                logger.info(f"Wrong path or file {circuit_filepath} does not exists!")

    @staticmethod
    def _start_optimization(circuit_filtered_point_file: str, act_cst_config: pecst.CapacitorRequirements, filter_data: FilterData,
                            debug: Debug) -> int:
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

        # Load configuration
        circuit_dto = HandleDabDto.load_from_file(os.path.join(filter_data.filtered_list_pathname, f"{circuit_filtered_point_file}.pkl"))

        # sweep through all current waveforms
        i_l1_sorted = np.transpose(circuit_dto.calc_currents.i_l_1_sorted, (1, 2, 3, 0))
        angles_rad_sorted = np.transpose(circuit_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))

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

            loss_total_array = np.full_like(circuit_dto.calc_modulation.phi, np.nan)

            new_circuit_dto_directory = os.path.join(act_cst_config.results_directory, "01_circuit_dtos_incl_capacitor_loss")
            if not os.path.exists(new_circuit_dto_directory):
                os.makedirs(new_circuit_dto_directory)

            if os.path.exists(os.path.join(new_circuit_dto_directory, f"{ordering_code}.pkl")):
                logger.info(f"Re-simulation of {circuit_dto.name} already exists. Skip.")
            else:
                for vec_vvp in np.ndindex(circuit_dto.calc_modulation.phi.shape):
                    time, unique_indices = np.unique(full_angle_waveform_from_angles(
                                                     angles_rad_sorted[vec_vvp]) / 2 / np.pi / circuit_dto.input_config.fs, return_index=True)
                    current = full_current_waveform_from_currents(i_l1_sorted[vec_vvp])[unique_indices]

                    current_waveform = np.array([time, current])
                    logger.debug(f"{current_waveform=}")
                    logger.debug("All operating point simulation of:")
                    logger.debug(f"   * Circuit study: {filter_data.circuit_study_name}")
                    logger.debug(f"   * Circuit trial: {circuit_filtered_point_file}")
                    logger.debug(f"   * Inductor re-simulation trial: {ordering_code}")

                    [frequency_list, current_amplitude_list, _] = pecst.fft(current_waveform, plot='no', mode='time', title='fft input current')

                    loss_per_capacitor = pecst.power_loss_film_capacitor(ordering_code, frequency_list, current_amplitude_list,
                                                                         number_parallel_capacitors=n_parallel)

                    loss_total_array[vec_vvp] = loss_per_capacitor * n_series * n_parallel

                capacitor_losses = CapacitorResults(
                    loss_total_array=loss_total_array,
                    volume_total=volume_total,
                    area_total=area_total,
                    circuit_trial_file=circuit_filtered_point_file,
                    capacitor_order_number=df_geometry_re_simulation_number['ordering code'].values[0],
                    n_series=n_series,
                    n_parallel=n_parallel
                )

                pickle_file = os.path.join(new_circuit_dto_directory, f"{ordering_code}.pkl")
                with open(pickle_file, 'wb') as output:
                    pickle.dump(capacitor_losses, output, pickle.HIGHEST_PROTOCOL)

        # returns the number of filtered results
        return number_of_filtered_points

    def optimization_handler(self, filter_data: FilterData, debug: Debug) -> None:
        """
        Control the multi simulation processes.

        :param filter_data: Information about the filtered designs
        :type  filter_data: dct.FilterData
        :param debug: True to use debug mode which stops earlier
        :type debug: bool
        """
        number_cpus = cpu_count()

        with Pool(processes=number_cpus) as pool:
            parameters = []
            for count, act_optimization_configuration in enumerate(self._optimization_config_list):
                if debug.general.is_debug:
                    # in debug mode, stop when number of configuration parameters has reached the same as parallel cores are used
                    if count == number_cpus:
                        break

                parameters.append((
                    act_optimization_configuration.circuit_filtered_point_filename,
                    act_optimization_configuration.capacitor_optimization_dto,
                    filter_data,
                    debug
                ))

            pool.starmap(func=CapacitorSelection._start_optimization, iterable=parameters)

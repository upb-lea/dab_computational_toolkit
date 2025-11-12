"""Capacitor selection."""

# python libraries
import logging
import os
from multiprocessing import Pool, cpu_count
from dct.server_ctl_dtos import ProgressData
from dct.server_ctl_dtos import ProgressStatus
import copy
import pickle

# 3rd party libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import tqdm

# own libraries
import pecst
from dct.capacitor_optimization_dtos import CapacitorOptimizationDto
from dct.toml_checker import TomlCapacitorSelection
from dct.datasets import HandleDabDto
from dct.datasets_dtos import StudyData, FilterData
from dct.toml_checker import Debug

logger = logging.getLogger(__name__)

class CapacitorSelection:
    c_df: pd.DataFrame
    _optimization_config_list: list[CapacitorOptimizationDto]

    def __init__(self):
        self._optimization_config_list = []


    def initialize_capacitor_selection(self, toml_capacitor: TomlCapacitorSelection, study_data: StudyData, circuit_filter_data: FilterData):
        pecst.download_esr_csv_files()

        # Create the io_config_list for all trials
        for circuit_trial_file in circuit_filter_data.filtered_list_files:
            circuit_filepath = os.path.join(circuit_filter_data.filtered_list_pathname, f"{circuit_trial_file}.pkl")
            # Check filename
            if os.path.isfile(circuit_filepath):
                # Read results from circuit optimization
                circuit_dto = HandleDabDto.load_from_file(circuit_filepath)

                optimization_directory = os.path.join(study_data.optimization_directory, circuit_trial_file, study_data.study_name)

                # figure out worst case working point for the capacitor per circuit design
                sorted_max_rms_angles, i_c1_max_rms_current_waveform = HandleDabDto.get_max_rms_waveform_capacitor(circuit_dto, plot=False)
                time = sorted_max_rms_angles / (2 * np.pi * circuit_dto.input_config.fs)
                v_max = np.max(circuit_dto.input_config.mesh_v1)

                # generate capacitor requirements from circuit simulation data
                capacitor_requirements_dto = pecst.CapacitorRequirements(
                    maximum_peak_to_peak_voltage_ripple=toml_capacitor.maximum_peak_to_peak_voltage_ripple,
                    current_waveform_for_op_max_current=np.array([time, i_c1_max_rms_current_waveform]),
                    v_dc_for_op_max_voltage=v_max,
                    temperature_ambient=toml_capacitor.temperature_ambient,
                    voltage_safety_margin_percentage=toml_capacitor.voltage_safety_margin_percentage,
                    capacitor_type_list=[pecst.CapacitorType.FilmCapacitor],
                    maximum_number_series_capacitors=toml_capacitor.maximum_number_series_capacitors,
                    capacitor_tolerance_percent=pecst.CapacitanceTolerance.TenPercent,
                    lifetime_h=toml_capacitor.lifetime_h,
                    results_directory=optimization_directory
                )

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
                            debug: Debug):
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

        # config_filepath = os.path.join(act_cst_config.inductor_optimization_directory, f"{act_cst_config.inductor_study_name}.pkl")

        # Load configuration
        circuit_dto = HandleDabDto.load_from_file(os.path.join(filter_data.filtered_list_pathname, f"{circuit_filtered_point_file}.pkl"))

        # sweep through all current waveforms
        i_l1_sorted = np.transpose(circuit_dto.calc_currents.i_l_1_sorted, (1, 2, 3, 0))
        angles_rad_sorted = np.transpose(circuit_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))

        all_operation_point_ordering_codes_list = df_filtered["ordering code"].to_numpy()

        # Overtake the filtered operation points
        number_of_filtered_points = len(all_operation_point_ordering_codes_list)

        logger.info(f"Full-operating point simulation list: {all_operation_point_ordering_codes_list}")

        # simulate all operating points
        for ordering_code in tqdm.tqdm(all_operation_point_ordering_codes_list):
            df_geometry_re_simulation_number = df_filtered[df_filtered["ordering code"] == ordering_code]

            logger.debug(f"ordering_code: \n"
                         f"    {df_geometry_re_simulation_number.head()}")

        #     combined_loss_array = np.full_like(circuit_dto.calc_modulation.phi, np.nan)
        #
        #     new_circuit_dto_directory = os.path.join(act_cst_config.inductor_optimization_directory, "08_circuit_dtos_incl_reluctance_inductor_losses")
        #     if not os.path.exists(new_circuit_dto_directory):
        #         os.makedirs(new_circuit_dto_directory)
        #
        #     if os.path.exists(os.path.join(new_circuit_dto_directory, f"{ordering_code}.pkl")):
        #         logger.info(f"Re-simulation of {circuit_dto.name} already exists. Skip.")
        #     else:
        #         # The femmt simulation (full_simulation()) can raise different errors, most of them are geometry errors
        #         # e.g. winding is not fitting in the winding window
        #         try:
        #             for vec_vvp in np.ndindex(circuit_dto.calc_modulation.phi.shape):
        #                 time, unique_indices = np.unique(dct.functions_waveforms.full_angle_waveform_from_angles(
        #                     angles_rad_sorted[vec_vvp]) / 2 / np.pi / circuit_dto.input_config.fs, return_index=True)
        #                 current = dct.functions_waveforms.full_current_waveform_from_currents(i_l1_sorted[vec_vvp])[unique_indices]
        #
        #                 current_waveform = np.array([time, current])
        #                 logger.debug(f"{current_waveform=}")
        #                 logger.debug("All operating point simulation of:")
        #                 logger.debug(f"   * Circuit study: {filter_data.circuit_study_name}")
        #                 logger.debug(f"   * Circuit trial: {circuit_filtered_point_file}")
        #                 logger.debug(f"   * Inductor study: {act_cst_config.inductor_study_name}")
        #                 logger.debug(f"   * Inductor re-simulation trial: {ordering_code}")
        #
        #                 volume, combined_losses, area_to_heat_sink = fmt.InductorOptimization.ReluctanceModel.full_simulation(
        #                     df_geometry_re_simulation_number, current_waveform=current_waveform,
        #                     inductor_config_filepath=config_filepath)
        #                 combined_loss_array[vec_vvp] = combined_losses
        #
        #             inductor_losses = dct.InductorResults(
        #                 p_combined_losses=combined_loss_array,
        #                 volume=volume,
        #                 area_to_heat_sink=area_to_heat_sink,
        #                 circuit_trial_file=circuit_filtered_point_file,
        #                 inductor_trial_number=ordering_code,
        #             )
        #
        #             pickle_file = os.path.join(new_circuit_dto_directory, f"{int(ordering_code)}.pkl")
        #             with open(pickle_file, 'wb') as output:
        #                 pickle.dump(inductor_losses, output, pickle.HIGHEST_PROTOCOL)
        #         except:
        #             logger.info(f"Re-simulation of inductor geometry {ordering_code} not possible due to non-possible geometry.")
        #
        # # returns the number of filtered results
        # return number_of_filtered_points

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

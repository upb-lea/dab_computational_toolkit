"""Summary pareto optimizations."""
# python libraries
import os
import pickle
import logging
import threading
import copy

# 3rd party libraries
import pandas as pd
import numpy as np

# own libraries
from dct import ProgressStatus
from dct.components.heat_sink_optimization import ThermalCalcSupport
from dct.components.capacitor_optimization_dtos import CapacitorResults
from dct.components.heat_sink_dtos import HeatSinkBoundaryConditions, ComponentCooling
from dct.toml_checker import TomlHeatSink
from dct.datasets_dtos import StudyData, FilterData

import hct
from dct.server_ctl_dtos import ProgressData
from dct.server_ctl_dtos import RunTimeMeasurement as RunTime
from dct.constant_path import (CIRCUIT_INDUCTOR_RELUCTANCE_LOSSES_FOLDER, CIRCUIT_TRANSFORMER_RELUCTANCE_LOSSES_FOLDER,
                               CIRCUIT_INDUCTOR_FEM_LOSSES_FOLDER, CIRCUIT_TRANSFORMER_FEM_LOSSES_FOLDER,
                               CIRCUIT_CAPACITOR_LOSS_FOLDER)

logger = logging.getLogger(__name__)

class SummaryProcessing:
    """Perform the summary calculation based on optimization results."""

    _s_lock_stat: threading.Lock
    _progress_run_time: RunTime
    _progress_data: ProgressData
    # Areas and transistor cooling parameter
    copper_coin_area_1: float
    transistor_b1_cooling: ComponentCooling
    copper_coin_area_2: float
    transistor_b2_cooling: ComponentCooling

    # Thermal resistance
    r_th_per_unit_area_ind_heat_sink: float
    r_th_per_unit_area_xfmr_heat_sink: float

    # Heat sink boundary condition parameter
    heat_sink_boundary_conditions: HeatSinkBoundaryConditions
    # Thermal calculation support class
    thr_sup: ThermalCalcSupport

    def __init__(self):
        """Initialize the configuration list for the circuit optimizations."""
        self._s_lock_stat = threading.Lock()
        # Initialize the statistical data (For more configuration it needs to become instance instead of static
        self._progress_run_time = RunTime()
        self._progress_data = ProgressData(run_time=0, number_of_filtered_points=0,
                                           progress_status=ProgressStatus.Idle)

        # Heat sink boundary condition parameter
        self.heat_sink_boundary_conditions = HeatSinkBoundaryConditions(0, 0)
        # Thermal calculation support class
        self.thr_sup = ThermalCalcSupport()





    def get_progress_data(self) -> ProgressData:
        """Provide the progress data of the optimization.

        :return: Progress data: Processing start time, actual processing time, number of filtered operation points and status.
        :rtype: ProgressData
        """
        # Lock statistical performance data access
        with self._s_lock_stat:
            # Check if list is in progress
            if self._progress_data.progress_status == ProgressStatus.InProgress:
                # Update statistical data if optimization is running
                self._progress_data.run_time = self._progress_run_time.get_runtime()

        return copy.deepcopy(self._progress_data)

    @staticmethod
    def _generate_component_id_list_from_pkl_files(act_dir_name: str) -> tuple[bool, list[str]]:
        """Generate a list of the IDs from pickle filenames (inductor, transformer, capacitor).

        :param act_dir_name : Name of the directory containing the files
        :type  act_dir_name : str

        :return: tuple of bool and result list. True, if the directory exists and contains minimum one file
        :rtype: tuple
        """
        # Variable declaration
        component_id_list: list[str] = []
        is_component_id_list_generated = False

        # Check if target folder exists
        if os.path.exists(act_dir_name):
            # Create list of filepaths
            file_list = os.listdir(act_dir_name)
            # Filter basename without extension
            for file_name in file_list:
                # Create file path
                file_path = os.path.join(act_dir_name, file_name)
                # Check if it is a file
                if os.path.isfile(file_path):
                    component_id = os.path.splitext(os.path.basename(file_name))[0]
                    # Check file type
                    file_extension = os.path.splitext(os.path.basename(file_name))[1]
                    if file_extension == '.pkl':
                        component_id_list.append(component_id)
                    else:
                        logger.info(f"File {component_id}{file_extension} has no extension '.pkl'!")
                else:
                    logger.info(f"File'{file_path}' does not exists!")
        else:
            logger.info(f"Path {act_dir_name} does not exists!")

        if component_id_list:
            is_component_id_list_generated = True

        return is_component_id_list_generated, component_id_list

    def generate_result_database(self, inductor_study_data: StudyData, transformer_study_data: StudyData,
                                 summary_data: StudyData, act_inductor_study_names: list[str],
                                 act_stacked_transformer_study_names: list[str], filter_data: FilterData,
                                 capacitor_1_study_data: StudyData, capacitor_2_study_data: StudyData,
                                 act_capacitor_1_study_names: list[str], act_capacitor_2_study_names: list[str], is_pre_summary: bool,
                                 r_th_per_unit_area_ind_heat_sink: float, r_th_per_unit_area_xfmr_heat_sink: float) -> pd.DataFrame:
        """Generate a database df by summaries the calculation results.

        :param inductor_study_data: inductor study data
        :type inductor_study_data: StudyData
        :param transformer_study_data: transformer study data
        :type transformer_study_data: StudyData
        :param summary_data: Information about the summary name and path
        :type summary_data: StudyData
        :param capacitor_1_study_data: List of names with capacitor studies which are to process
        :type capacitor_1_study_data: list[str]
        :param capacitor_2_study_data: List of names with capacitor studies which are to process
        :type capacitor_2_study_data: list[str]
        :param act_capacitor_1_study_names: List of names with capacitor studies which are to process
        :type  act_capacitor_1_study_names: list[str]
        :param act_capacitor_2_study_names: List of names with capacitor studies which are to process
        :type  act_capacitor_2_study_names: list[str]
        :param act_inductor_study_names: List of names with inductor studies which are to process
        :type  act_inductor_study_names: list[str]
        :param act_stacked_transformer_study_names: List of names with transformer studies which are to process
        :type  act_stacked_transformer_study_names: list[str]
        :param filter_data: filtered result lists
        :type filter_data: FilterData
        :param is_pre_summary: True for pre-summary, False for summary
        :type is_pre_summary: bool
        :return: DataFrame with result information of the pareto front
        :rtype:  pd.DataFrame
        """
        # Variable declaration

        if is_pre_summary:
            # pre summary using reluctance model results (inductive components)
            inductor_result_directory = CIRCUIT_INDUCTOR_RELUCTANCE_LOSSES_FOLDER
            transformer_result_directory = CIRCUIT_TRANSFORMER_RELUCTANCE_LOSSES_FOLDER
        else:
            # final summary using FEM results (inductive components)
            inductor_result_directory = CIRCUIT_INDUCTOR_FEM_LOSSES_FOLDER
            transformer_result_directory = CIRCUIT_TRANSFORMER_FEM_LOSSES_FOLDER

        # Start the progress time measurement
        with self._s_lock_stat:
            self._progress_run_time.reset_start_trigger()
            self._progress_data.run_time = self._progress_run_time.get_runtime()
            self._progress_data.progress_status = ProgressStatus.InProgress

        # Result DataFrame
        df = pd.DataFrame()

        # iterate circuit numbers
        for circuit_id in filter_data.filtered_list_files:

            df_inductors = pd.DataFrame()
            df_transformers = pd.DataFrame()
            df_capacitors_1 = pd.DataFrame()
            df_capacitors_2 = pd.DataFrame()

            # Assemble pkl-filename
            circuit_id_filepath = os.path.join(filter_data.filtered_list_pathname, f"{circuit_id}.pkl")

            # Get circuit results
            with open(circuit_id_filepath, 'rb') as pickle_file_data:
                circuit_dto = pickle.load(pickle_file_data)

            # Calculate the thermal values
            if not circuit_dto.calc_losses:  # mypy avoid follow-up issues
                raise ValueError("Incomplete loss calculation.")


            logger.debug(f"{act_inductor_study_names=}")

            circuit_data = {
                "circuit_id": circuit_id,
                "circuit_t_j_max": circuit_dto.circuit_thermal.t_j_max,
                "circuit_r_th_ib_jhs": circuit_dto.circuit_thermal.r_th_jhs,
                "circuit_area": circuit_dto.circuit_thermal.area,
                "circuit_loss_array": circuit_dto.circuit_thermal.loss_array,
                "circuit_temperature_heat_sink_max_array": circuit_dto.circuit_thermal.temperature_heat_sink_max_array
            }
            df_circuit_local = pd.DataFrame([circuit_data])

            # iterate inductor study
            for inductor_study_name in act_inductor_study_names:
                inductor_study_results_filepath = os.path.join(inductor_study_data.optimization_directory, circuit_id,
                                                               inductor_study_name,
                                                               inductor_result_directory)

                # Generate magnetic list
                is_inductor_list_generated, inductor_id_list = (
                    SummaryProcessing._generate_component_id_list_from_pkl_files(inductor_study_results_filepath))

                if not is_inductor_list_generated:
                    logger.info(f"Path {inductor_study_results_filepath} does not exists or does not contains any pkl-files!")
                    # Next circuit
                    continue

                # iterate inductor numbers
                logger.debug(f"{inductor_id_list=}")
                for inductor_id in inductor_id_list:
                    inductor_id_filepath = os.path.join(inductor_study_results_filepath, f"{inductor_id}.pkl")

                    # Get inductor results
                    with open(inductor_id_filepath, 'rb') as pickle_file_data:
                        inductor_dto = pickle.load(pickle_file_data)

                    if inductor_dto.circuit_id != circuit_id:
                        raise ValueError(f"{inductor_dto.circuit_id=} != {circuit_id}")
                    if int(inductor_dto.inductor_id) != int(inductor_id):
                        raise ValueError(f"{inductor_dto.inductor_id=} != {inductor_id}")

                    inductance_loss_matrix = inductor_dto.loss_array

                    logger.debug(f"{act_stacked_transformer_study_names=}")

                    inductor_data = {
                        # inductor
                        "inductor_study_name": inductor_study_name,
                        "inductor_id": inductor_id,
                        "inductor_volume": inductor_dto.volume,
                        "inductor_loss_array": inductance_loss_matrix,
                        "inductor_t_max": 0,
                        "inductor_area": inductor_dto.area_to_heat_sink,
                    }
                    df_single_inductor = pd.DataFrame([inductor_data])

                    df_inductors = pd.concat([df_inductors, df_single_inductor], axis=0)

            logger.info(f"{df_inductors=}")

            # iterate transformer study
            for stacked_transformer_study_name in act_stacked_transformer_study_names:
                stacked_transformer_study_results_filepath = os.path.join(transformer_study_data.optimization_directory,
                                                                          circuit_id,
                                                                          stacked_transformer_study_name,
                                                                          transformer_result_directory)

                # Check, if stacked transformer number list cannot be generated
                is_transformer_list_generated, transformer_id_list = (
                    SummaryProcessing._generate_component_id_list_from_pkl_files(stacked_transformer_study_results_filepath))

                if not is_transformer_list_generated:
                    logger.info(f"Path {stacked_transformer_study_results_filepath} does not exists or does not contains any pkl-files!")
                    # Next circuit
                    continue

                logger.debug(f"{transformer_id_list=}")

                # iterate transformer numbers
                for transformer_id in transformer_id_list:
                    transformer_id_filepath = os.path.join(stacked_transformer_study_results_filepath, f"{transformer_id}.pkl")

                    # get transformer results
                    with open(transformer_id_filepath, 'rb') as pickle_file_data:
                        transformer_dto = pickle.load(pickle_file_data)

                    if transformer_dto.circuit_id != circuit_id:
                        raise ValueError(f"{transformer_dto.circuit_id=} != {circuit_id}")
                    if int(transformer_dto.transformer_id) != int(transformer_id):
                        raise ValueError(f"{transformer_dto.transformer_id=} != {transformer_id}")

                    logger.debug(f"{act_capacitor_1_study_names=}")

                    transformer_data = {
                        # transformer
                        "transformer_study_name": stacked_transformer_study_name,
                        "transformer_id": transformer_id,
                        "transformer_volume": transformer_dto.volume,
                        "transformer_loss_array": transformer_dto.loss_array,
                        "transformer_t_max": 0,
                        "transformer_area": transformer_dto.area_to_heat_sink,
                    }

                    df_single_transformer = pd.DataFrame([transformer_data])
                    df_transformers = pd.concat([df_transformers, df_single_transformer], axis=0)

                logger.info(f"{df_transformers=}")

            for capacitor_1_study_name in act_capacitor_1_study_names:
                # Assemble directory name for capacitor 1 results
                capacitor_1_filepath_results = os.path.join(capacitor_1_study_data.optimization_directory,
                                                            circuit_id,
                                                            capacitor_1_study_name,
                                                            CIRCUIT_CAPACITOR_LOSS_FOLDER)

                # Check, if stacked transformer number list cannot be generated
                is_capacitor_1_list_generated, capacitor_1_id_list = (
                    SummaryProcessing._generate_component_id_list_from_pkl_files(capacitor_1_filepath_results))
                if not is_capacitor_1_list_generated:
                    logger.info(f"Path {capacitor_1_filepath_results} does not exists or does not contains any pkl-files!")
                    # Next circuit
                    continue
                logger.debug(f"{capacitor_1_id_list=}")

                # iterate capacitor 1 numbers
                for capacitor_1_id in capacitor_1_id_list:
                    capacitor_1_filepath_number = os.path.join(capacitor_1_filepath_results, f"{capacitor_1_id}.pkl")

                    # get capacitor 1 results
                    with open(capacitor_1_filepath_number, 'rb') as pickle_file_data:
                        capacitor_1_dto: CapacitorResults = pickle.load(pickle_file_data)

                    if capacitor_1_dto.circuit_id != circuit_id:
                        raise ValueError(f"{capacitor_1_dto.circuit_id=} != {circuit_id}")
                    if capacitor_1_dto.capacitor_id != capacitor_1_id:
                        raise ValueError(f"{capacitor_1_dto.capacitor_id=} != {capacitor_1_id}")

                    capacitor_1_data = {
                        # capacitor 1
                        "capacitor_1_study_name": capacitor_1_study_name,
                        "capacitor_1_id": capacitor_1_id,
                        "capacitor_1_volume": capacitor_1_dto.volume_total,
                        "capacitor_1_loss_array": capacitor_1_dto.loss_total_array,
                        "capacitor_1_area": capacitor_1_dto.area_total,
                        "capacitor_1_n_parallel": capacitor_1_dto.n_parallel,
                        "capacitor_1_n_series": capacitor_1_dto.n_series,
                    }
                    df_single_capacitor_1 = pd.DataFrame([capacitor_1_data])
                    df_capacitors_1 = pd.concat([df_capacitors_1, df_single_capacitor_1], axis=0)

                logger.info(f"{df_capacitors_1=}")

            logger.debug(f"{act_capacitor_2_study_names=}")
            for capacitor_2_study_name in act_capacitor_2_study_names:
                # Assemble directory name for capacitor 1 results
                capacitor_2_filepath_results = os.path.join(capacitor_2_study_data.optimization_directory,
                                                            circuit_id,
                                                            capacitor_2_study_name,
                                                            CIRCUIT_CAPACITOR_LOSS_FOLDER)

                # Check, if stacked transformer number list cannot be generated
                is_capacitor_2_list_generated, capacitor_2_id_list = (
                    SummaryProcessing._generate_component_id_list_from_pkl_files(capacitor_2_filepath_results))
                if not is_capacitor_2_list_generated:
                    logger.info(f"Path {capacitor_2_filepath_results} does not exists or does not contains any pkl-files!")
                    # Next circuit
                    continue
                logger.debug(f"{capacitor_2_id_list=}")

                # iterate capacitor 2 numbers
                for capacitor_2_id in capacitor_2_id_list:
                    capacitor_2_filepath_number = os.path.join(capacitor_2_filepath_results,
                                                               f"{capacitor_2_id}.pkl")

                    # get capacitor 1 results
                    with open(capacitor_2_filepath_number, 'rb') as pickle_file_data:
                        capacitor_2_dto = pickle.load(pickle_file_data)

                    if capacitor_2_dto.circuit_id != circuit_id:
                        raise ValueError(f"{capacitor_2_dto.circuit_id=} != {circuit_id}")
                    if capacitor_2_dto.capacitor_id != capacitor_2_id:
                        raise ValueError(f"{capacitor_2_dto.capacitor_id=} != {capacitor_2_id}")

                    capacitor_2_data = {
                        # capacitor 2
                        "capacitor_2_study_name": capacitor_2_study_name,
                        "capacitor_2_id": capacitor_2_id,
                        "capacitor_2_volume": capacitor_2_dto.volume_total,
                        "capacitor_2_loss_array": capacitor_2_dto.loss_total_array,
                        "capacitor_2_area": capacitor_2_dto.area_total,
                        "capacitor_2_n_parallel": capacitor_2_dto.n_parallel,
                        "capacitor_2_n_series": capacitor_2_dto.n_series,
                    }
                    df_single_capacitor_2 = pd.DataFrame([capacitor_2_data])
                    df_capacitors_2 = pd.concat([df_capacitors_2, df_single_capacitor_2], axis=0)

            logger.info(f"{df_capacitors_2=}")

            # merge different df by creating a common key
            df_circuit_local['key'] = 0
            df_inductors['key'] = 0
            df_transformers['key'] = 0
            df_capacitors_1['key'] = 0
            df_capacitors_2['key'] = 0

            # perform cross join
            df_intermediate_1 = df_circuit_local.merge(df_inductors, on='key', how='outer')
            df_intermediate_2 = df_intermediate_1.merge(df_transformers, on='key', how='outer')
            df_intermediate_3 = df_intermediate_2.merge(df_capacitors_1, on='key', how='outer')
            # TODO: Needs fix, as data for capacitor 2 is currently missing
            # fix df_intermediate_3 to df_intermediate_4 in the following
            # df_intermediate_4 = df_intermediate_3.merge(df_capacitor_2, on='key', how='outer')

            del df_intermediate_3['key']

            logger.info(f"{df_intermediate_3=}")

            df = pd.concat([df, df_intermediate_3], axis=0)

        # Calculate the total area as sum of circuit,  inductor and transformer area df-command is like vector sum v1[:]=v2[:]+v3[:])
        # heat sink area, capacitors do not need heat sink area
        df["total_area"] = df["circuit_area"].apply(lambda x: np.sum(x)) + df["inductor_area"] + df["transformer_area"]

        # TODO: Fix needed as capacitor 2 is not considered currently
        df["total_mean_loss"] = (df["circuit_loss_array"].apply(lambda x: np.sum([np.mean(y) for y in x], axis=0)) + \
                                 df["inductor_loss_array"].apply(np.mean) + \
                                 df["transformer_loss_array"].apply(np.mean) + \
                                 df["capacitor_1_loss_array"].apply(np.mean)  # + np.mean(df["capacitor_2_loss_array"])
                                 )
        # TODO: Fix needed, as capacitor 2 is not considered currently
        df["volume_wo_heat_sink"] = df["transformer_volume"] + df["inductor_volume"] + df["capacitor_1_volume"]  # + df["capacitor_2_volume"]

        # TODO: Fix needed capacitor 2+  df["capacitor_2_loss_array"]
        df["total_loss_array"] = (df["inductor_loss_array"] + df["circuit_loss_array"] + \
                                  df["transformer_loss_array"] + df["capacitor_1_loss_array"])

        # Calculate the thermal resistance according r_th = 1/lambda * l / A
        # For inductor: r_th_per_unit_area_ind_heat_sink = 1/lambda * l
        df["r_th_ind_heat_sink"] = r_th_per_unit_area_ind_heat_sink / df["inductor_area"]
        df["temperature_inductor_heat_sink_max_array"] = 125 - df["r_th_ind_heat_sink"] * df["inductor_loss_array"]
        # For transformer: r_th_per_unit_area_xfmr_heat_sink = 1/lambda * l.
        df["r_th_xfmr_heat_sink"] = r_th_per_unit_area_xfmr_heat_sink / df["transformer_area"]
        df["temperature_xfmr_heat_sink_max_array"] = 125 - df["r_th_xfmr_heat_sink"] * df["transformer_loss_array"]

        # maximum heat sink temperatures (minimum of all the maximum temperatures of single components)
        df["t_min_array"] = df["circuit_temperature_heat_sink_max_array"].apply(lambda x: np.minimum(*x))
        df["t_min_array"] = df.apply(lambda x: np.minimum(x["t_min_array"], x["temperature_inductor_heat_sink_max_array"]), axis=1)
        df["t_min_array"] = df.apply(lambda x: np.minimum(x["t_min_array"], x["temperature_xfmr_heat_sink_max_array"]), axis=1)
        df["t_min_array"] = df.apply(lambda x: np.minimum(x["t_min_array"], self.heat_sink_boundary_conditions.t_hs_max), axis=1)

        # maximum delta temperature over the heat sink
        df["delta_t_max_heat_sink_array"] = df["t_min_array"] - self.heat_sink_boundary_conditions.t_ambient

        df["r_th_heat_sink_target_array"] = df["delta_t_max_heat_sink_array"] / df["total_loss_array"]

        df["r_th_heat_sink"] = df["r_th_heat_sink_target_array"].apply(lambda x: x.min())

        # Save results to file (ASA : later to store only on demand)
        df.to_csv(f"{summary_data.optimization_directory}/df_wo_hs.csv")

        # Start the progress time measurement
        with self._s_lock_stat:
            self._progress_run_time.stop_trigger()
            self._progress_data.run_time = self._progress_run_time.get_runtime()

        # return the database
        return df

    def select_heat_sink_configuration(self, heat_sink_study_data: StudyData, summary_data: StudyData, act_df_for_hs: pd.DataFrame) -> None:
        """Select the heat sink configuration from calculated heat sink pareto front.

        :param heat_sink_study_data: Information about the heat sink study name and study path
        :type  heat_sink_study_data: StudyData
        :param summary_data: Information about the summary name and path
        :type summary_data: StudyData
        :param act_df_for_hs: DataFrame with result information of the pareto front for heat sink selection
        :type  act_df_for_hs: pd.DataFrame
        """
        # Variable declaration

        # Continue the progress time measurement
        with self._s_lock_stat:
            self._progress_run_time.continue_trigger()
            self._progress_data.run_time = self._progress_run_time.get_runtime()
            self._progress_data.progress_status = ProgressStatus.InProgress

        # load heat sink
        hs_config_filepath = os.path.join(heat_sink_study_data.optimization_directory,
                                          f"{heat_sink_study_data.study_name}.pkl")
        hs_config = hct.Optimization.load_config(hs_config_filepath)
        # Debug ASA Missing true simulations for remaining function

        df_hs = hct.Optimization.study_to_df(hs_config)

        # generate full summary as panda database operation
        logger.info(df_hs.loc[df_hs["values_1"] < 1]["values_0"].nsmallest(n=1).values[0])
        act_df_for_hs["heat_sink_volume"] = act_df_for_hs["r_th_heat_sink"].apply(
            lambda r_th_max: df_hs.loc[df_hs["values_1"] < r_th_max]["values_0"].nsmallest(n=1).values[0] \
            if np.any(df_hs.loc[df_hs["values_1"] < r_th_max]["values_0"].nsmallest(n=1).values) else None)

        act_df_for_hs["total_volume"] = act_df_for_hs["volume_wo_heat_sink"] + act_df_for_hs["heat_sink_volume"]

        # save full summary
        act_df_for_hs.to_csv(f"{summary_data.optimization_directory}/df_w_hs.csv")

        # Update statistical data for summary processing finalized
        # Update statistical data
        with self._s_lock_stat:
            self._progress_run_time.stop_trigger()
            self._progress_data.run_time = self._progress_run_time.get_runtime()
            self._progress_data.progress_status = ProgressStatus.Done

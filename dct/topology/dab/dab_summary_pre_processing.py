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
import dct
from dct import ProgressStatus
from dct.components.heat_sink_optimization import ThermalCalcSupport
from dct.components.capacitor_optimization_dtos import CapacitorResults
import hct
import dct.topology.dab.dab_datasets as dab_dset
from dct.server_ctl_dtos import ProgressData
from dct.server_ctl_dtos import RunTimeMeasurement as RunTime
from dct.constant_path import (CIRCUIT_INDUCTOR_RELUCTANCE_LOSSES_FOLDER, CIRCUIT_TRANSFORMER_RELUCTANCE_LOSSES_FOLDER,
                               CIRCUIT_INDUCTOR_FEM_LOSSES_FOLDER, CIRCUIT_TRANSFORMER_FEM_LOSSES_FOLDER,
                               CIRCUIT_CAPACITOR_LOSS_FOLDER)

logger = logging.getLogger(__name__)

class DabSummaryPreProcessing:
    """Perform the summary calculation based on optimization results."""

    _s_lock_stat: threading.Lock
    _progress_run_time: RunTime
    _progress_data: ProgressData
    # Areas and transistor cooling parameter
    copper_coin_area_1: float
    transistor_b1_cooling: dct.TransistorCooling
    copper_coin_area_2: float
    transistor_b2_cooling: dct.TransistorCooling

    # Thermal resistance
    r_th_per_unit_area_ind_heat_sink: float
    r_th_per_unit_area_xfmr_heat_sink: float

    # Heat sink boundary condition parameter
    heat_sink_boundary_conditions: dct.HeatSinkBoundaryConditions
    # Thermal calculation support class
    thr_sup: ThermalCalcSupport

    def __init__(self):
        """Initialize the configuration list for the circuit optimizations."""
        self._s_lock_stat = threading.Lock()
        # Initialize the statistical data (For more configuration it needs to become instance instead of static
        self._progress_run_time = RunTime()
        self._progress_data = ProgressData(run_time=0, number_of_filtered_points=0,
                                           progress_status=ProgressStatus.Idle)

        # Areas and transistor cooling parameter
        self.copper_coin_area_1 = 0
        self.transistor_b1_cooling = dct.TransistorCooling(0, 0)
        self.copper_coin_area_2 = 0
        self.transistor_b2_cooling = dct.TransistorCooling(0, 0)

        # Thermal resistance
        self.r_th_per_unit_area_ind_heat_sink = 0
        self.r_th_per_unit_area_xfmr_heat_sink = 0

        # Heat sink boundary condition parameter
        self.heat_sink_boundary_conditions = dct.HeatSinkBoundaryConditions(0, 0)
        # Thermal calculation support class
        self.thr_sup = ThermalCalcSupport()

    def init_thermal_configuration(self, act_heat_sink_data: dct.TomlHeatSink) -> bool:
        """Initialize the thermal parameter of the connection points for the transistors, inductor and transformer.

        :param act_heat_sink_data: toml file with configuration data
        :type act_heat_sink_data: dct.TomlHeatSink

        :return: True, if the thermal parameter of the connection points was successful initialized
        :rtype: bool
        """
        # Variable declaration
        # Return variable initialized to True
        successful_init = True
        transformer_cooling: dct.MagneticElementCooling
        # Thermal parameter for bridge transistor 1: List [tim_thickness, tim_conductivity]
        self.transistor_b1_cooling = dct.TransistorCooling(
            tim_thickness=act_heat_sink_data.thermal_resistance_data.transistor_b1_cooling[0],
            tim_conductivity=act_heat_sink_data.thermal_resistance_data.transistor_b1_cooling[1])

        # Thermal parameter for bridge transistor 2: List [tim_thickness, tim_conductivity]
        self.transistor_b2_cooling = dct.TransistorCooling(
            tim_thickness=act_heat_sink_data.thermal_resistance_data.transistor_b2_cooling[0],
            tim_conductivity=act_heat_sink_data.thermal_resistance_data.transistor_b2_cooling[1])

        # Thermal parameter for inductor: r_th per area: List [tim_thickness, tim_conductivity]
        inductor_tim_thickness = act_heat_sink_data.thermal_resistance_data.inductor_cooling[0]
        inductor_tim_conductivity = act_heat_sink_data.thermal_resistance_data.inductor_cooling[1]

        # Check on zero
        if inductor_tim_conductivity > 0:
            # Calculate the thermal resistance per unit area as term from the formula r_th = 1/lambda * l / A
            # r_th_per_unit_area_ind_heat_sink = 1/lambda * l. Later r_th = r_th_per_unit_area_ind_heat_sink / A
            self.r_th_per_unit_area_ind_heat_sink = inductor_tim_thickness / inductor_tim_conductivity
        else:
            logger.info(f"inductor cooling tim conductivity value must be greater zero, but is {inductor_tim_conductivity}!")
            successful_init = False

        # Thermal parameter for inductor: r_th per area: List [tim_thickness, tim_conductivity]
        transformer_tim_thickness = act_heat_sink_data.thermal_resistance_data.transformer_cooling[0]
        transformer_tim_conductivity = act_heat_sink_data.thermal_resistance_data.transformer_cooling[1]

        transformer_cooling = dct.MagneticElementCooling(
            tim_thickness=transformer_tim_thickness,
            tim_conductivity=transformer_tim_conductivity
        )
        # Check on zero ( ASA: Maybe in general all configuration files are to check for validity in advanced. In this case the check can be removed.)
        if transformer_tim_conductivity > 0:
            # Calculate the thermal resistance per unit area as term from the formula r_th = 1/lambda * l / A
            # r_th_per_unit_area_xfmr_heat_sink = 1/lambda * l. Later r_th = r_th_per_unit_area_xfmr_heat_sink / A
            self.r_th_per_unit_area_xfmr_heat_sink = transformer_tim_thickness / transformer_tim_conductivity
        else:
            logger.info(f"transformer cooling tim conductivity value must be greater zero, but is {transformer_tim_conductivity}!")
            successful_init = False

        self.heat_sink_boundary_conditions = dct.HeatSinkBoundaryConditions(t_ambient=act_heat_sink_data.boundary_conditions.t_ambient,
                                                                            t_hs_max=act_heat_sink_data.boundary_conditions.t_hs_max)
        # Return if initialization was successful performed (True)
        return successful_init

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
    def _generate_number_list_from_pkl_files(act_dir_name: str) -> tuple[bool, list[str]]:
        """Generate a list of the numbers from pickle filenames.

        :param act_dir_name : Name of the directory containing the files
        :type  act_dir_name : str

        :return: tuple of bool and result list. True, if the directory exists and contains minimum one file
        :rtype: tuple
        """
        # Variable declaration
        result_numbers: list[str] = []
        is_list_generated = False

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
                    device_number = os.path.splitext(os.path.basename(file_name))[0]
                    # Check file type
                    extension = os.path.splitext(os.path.basename(file_name))[1]
                    if extension == '.pkl':
                        result_numbers.append(device_number)
                    else:
                        logger.info(f"File {device_number}{extension} has no extension '.pkl'!")
                else:
                    logger.info(f"File'{file_path}' does not exists!")
        else:
            logger.info(f"Path {act_dir_name} does not exists!")

        if result_numbers:
            is_list_generated = True

        return is_list_generated, result_numbers

    def generate_result_database(self, inductor_study_data: dct.StudyData, transformer_study_data: dct.StudyData,
                                 summary_data: dct.StudyData, act_inductor_study_names: list[str],
                                 act_stacked_transformer_study_names: list[str], filter_data: dct.FilterData,
                                 capacitor_1_study_data: dct.StudyData, capacitor_2_study_data: dct.StudyData,
                                 act_capacitor_1_study_names: list[str], act_capacitor_2_study_names: list[str], is_pre_summary: bool) -> pd.DataFrame:
        """Generate a database df by summaries the calculation results.

        :param inductor_study_data: inductor study data
        :type inductor_study_data: dct.StudyData
        :param transformer_study_data: transformer study data
        :type transformer_study_data: dct.StudyData
        :param summary_data: Information about the summary name and path
        :type summary_data: dct.StudyData
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
        :type filter_data: dct.FilterData
        :param is_pre_summary: True for pre-summary, False for summary
        :type is_pre_summary: bool
        :return: DataFrame with result information of the pareto front
        :rtype:  pd.DataFrame
        """
        # Variable declaration

        if is_pre_summary:
            inductor_result_directory = CIRCUIT_INDUCTOR_RELUCTANCE_LOSSES_FOLDER
            transformer_result_directory = CIRCUIT_TRANSFORMER_RELUCTANCE_LOSSES_FOLDER
        else:
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

            df_circuit = pd.DataFrame()
            df_inductor = pd.DataFrame()
            df_transformer = pd.DataFrame()
            df_capacitor_1 = pd.DataFrame()
            df_capacitor_2 = pd.DataFrame()

            # Assemble pkl-filename
            circuit_id_filepath = os.path.join(filter_data.filtered_list_pathname, f"{circuit_id}.pkl")

            # Get circuit results
            circuit_dto = dab_dset.HandleDabDto.load_from_file(circuit_id_filepath)

            # Calculate the thermal values
            if not circuit_dto.calc_losses:  # mypy avoid follow-up issues
                raise ValueError("Incomplete loss calculation.")

            # Begin: ASA: No influence by inductor or transformer ################################
            # get transistor results
            total_transistor_cond_loss_matrix \
                = 4 * (circuit_dto.calc_losses.p_m1_conduction + circuit_dto.calc_losses.p_m2_conduction)

            b1_transistor_cond_loss_matrix = circuit_dto.calc_losses.p_m1_conduction
            b2_transistor_cond_loss_matrix = circuit_dto.calc_losses.p_m2_conduction
            # End: ASA: No influence by inductor or transformer ################################
            # Begin: ASA: No influence by inductor or transformer ################################
            # get all the losses in a matrix
            r_th_copper_coin_1, copper_coin_area_1 = self.thr_sup.calculate_r_th_copper_coin(
                circuit_dto.input_config.transistor_dto_1.cooling_area)
            r_th_copper_coin_2, copper_coin_area_2 = self.thr_sup.calculate_r_th_copper_coin(
                circuit_dto.input_config.transistor_dto_2.cooling_area)

            circuit_r_th_tim_1 = self.thr_sup.calculate_r_th_tim(copper_coin_area_1, self.transistor_b1_cooling)
            circuit_r_th_tim_2 = self.thr_sup.calculate_r_th_tim(copper_coin_area_2, self.transistor_b2_cooling)

            circuit_r_th_1_jhs = circuit_dto.input_config.transistor_dto_1.r_th_jc + r_th_copper_coin_1 + circuit_r_th_tim_1
            circuit_r_th_2_jhs = circuit_dto.input_config.transistor_dto_2.r_th_jc + r_th_copper_coin_2 + circuit_r_th_tim_2

            circuit_heat_sink_max_1_matrix = (
                circuit_dto.input_config.transistor_dto_1.t_j_max_op - circuit_r_th_1_jhs * b1_transistor_cond_loss_matrix)
            circuit_heat_sink_max_2_matrix = (
                circuit_dto.input_config.transistor_dto_2.t_j_max_op - circuit_r_th_2_jhs * b2_transistor_cond_loss_matrix)
            # End: ASA: No influence by inductor or transformer ################################

            logger.debug(f"{act_inductor_study_names=}")

            circuit_data = {
                # circuit
                "circuit_id": circuit_id,
                "circuit_t_j_max_1": circuit_dto.input_config.transistor_dto_1.t_j_max_op,
                "circuit_t_j_max_2": circuit_dto.input_config.transistor_dto_2.t_j_max_op,
                "circuit_r_th_ib_jhs_1": circuit_r_th_1_jhs,
                "circuit_r_th_ib_jhs_2": circuit_r_th_2_jhs,
                "circuit_area": 4 * (copper_coin_area_1 + copper_coin_area_2),
                "circuit_loss_array": total_transistor_cond_loss_matrix,
                "circuit_loss_mean": np.mean(total_transistor_cond_loss_matrix),
                "circuit_heat_sink_max_1_array": circuit_heat_sink_max_1_matrix,
                "circuit_heat_sink_max_2_array": circuit_heat_sink_max_2_matrix,
            }
            df_circuit_local = pd.DataFrame([circuit_data])

            # iterate inductor study
            for inductor_study_name in act_inductor_study_names:
                inductor_filepath_results = os.path.join(inductor_study_data.optimization_directory, circuit_id,
                                                         inductor_study_name,
                                                         inductor_result_directory)

                # Generate magnetic list
                is_inductor_list_generated, inductor_id_list = (
                    DabSummaryPreProcessing._generate_number_list_from_pkl_files(inductor_filepath_results))

                if not is_inductor_list_generated:
                    logger.info(f"Path {inductor_filepath_results} does not exists or does not contains any pkl-files!")
                    # Next circuit
                    continue

                # iterate inductor numbers
                logger.debug(f"{inductor_id_list=}")
                for inductor_id in inductor_id_list:
                    inductor_filepath_number = os.path.join(inductor_filepath_results, f"{inductor_id}.pkl")

                    # Get inductor results
                    with open(inductor_filepath_number, 'rb') as pickle_file_data:
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
                        "inductor_loss_mean": np.mean(inductance_loss_matrix),
                        "inductor_t_max": 0,
                        "inductor_area": inductor_dto.area_to_heat_sink,
                    }
                    local_df_inductor = pd.DataFrame([inductor_data])

                    df_inductor = pd.concat([df_inductor, local_df_inductor], axis=0)

            logger.info(f"{df_inductor=}")

            # iterate transformer study
            for stacked_transformer_study_name in act_stacked_transformer_study_names:
                stacked_transformer_filepath_results = os.path.join(transformer_study_data.optimization_directory,
                                                                    circuit_id,
                                                                    stacked_transformer_study_name,
                                                                    transformer_result_directory)

                # Check, if stacked transformer number list cannot be generated
                is_transformer_list_generated, transformer_id_list = (
                    DabSummaryPreProcessing._generate_number_list_from_pkl_files(stacked_transformer_filepath_results))

                if not is_transformer_list_generated:
                    logger.info(f"Path {stacked_transformer_filepath_results} does not exists or does not contains any pkl-files!")
                    # Next circuit
                    continue

                logger.debug(f"{transformer_id_list=}")

                # iterate transformer numbers
                for transformer_id in transformer_id_list:
                    transformer_id_filepath = os.path.join(stacked_transformer_filepath_results, f"{transformer_id}.pkl")

                    # get transformer results
                    with open(transformer_id_filepath, 'rb') as pickle_file_data:
                        transformer_dto = pickle.load(pickle_file_data)

                    if transformer_dto.circuit_id != circuit_id:
                        raise ValueError(f"{transformer_dto.circuit_id=} != {circuit_id}")
                    if int(transformer_dto.transformer_id) != int(transformer_id):
                        raise ValueError(f"{transformer_dto.transformer_id=} != {transformer_id}")

                    transformer_loss_matrix = transformer_dto.loss_array

                    logger.debug(f"{act_capacitor_1_study_names=}")

                    transformer_data = {
                        # transformer
                        "transformer_study_name": stacked_transformer_study_name,
                        "transformer_id": transformer_id,
                        "transformer_volume": transformer_dto.volume,
                        "transformer_loss_array": transformer_dto.loss_array,
                        "transformer_loss_mean": np.mean(transformer_dto.loss_array),
                        "transformer_t_max": 0,
                        "transformer_area": transformer_dto.area_to_heat_sink,
                    }

                    local_df_transformer = pd.DataFrame([transformer_data])
                    df_transformer = pd.concat([df_transformer, local_df_transformer], axis=0)

                logger.info(f"{df_transformer=}")

            for capacitor_1_study_name in act_capacitor_1_study_names:
                # Assemble directory name for capacitor 1 results
                capacitor_1_filepath_results = os.path.join(capacitor_1_study_data.optimization_directory,
                                                            circuit_id,
                                                            capacitor_1_study_name,
                                                            CIRCUIT_CAPACITOR_LOSS_FOLDER)

                # Check, if stacked transformer number list cannot be generated
                is_capacitor_1_list_generated, capacitor_1_id_list = (
                    DabSummaryPreProcessing._generate_number_list_from_pkl_files(capacitor_1_filepath_results))
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
                        "capacitor_1_loss_mean": np.mean(capacitor_1_dto.loss_total_array),
                        "capacitor_1_area": capacitor_1_dto.area_total,
                        "capacitor_1_n_parallel": capacitor_1_dto.n_parallel,
                        "capacitor_1_n_series": capacitor_1_dto.n_series,
                    }
                    local_df_capacitor_1 = pd.DataFrame([capacitor_1_data])
                    df_capacitor_1 = pd.concat([df_capacitor_1, local_df_capacitor_1], axis=0)

                logger.info(f"{df_capacitor_1=}")

            logger.debug(f"{act_capacitor_2_study_names=}")
            for capacitor_2_study_name in act_capacitor_2_study_names:
                # Assemble directory name for capacitor 1 results
                capacitor_2_filepath_results = os.path.join(capacitor_2_study_data.optimization_directory,
                                                            circuit_id,
                                                            capacitor_2_study_name,
                                                            CIRCUIT_CAPACITOR_LOSS_FOLDER)

                # Check, if stacked transformer number list cannot be generated
                is_capacitor_2_list_generated, capacitor_2_id_list = (
                    DabSummaryPreProcessing._generate_number_list_from_pkl_files(capacitor_2_filepath_results))
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
                        "capacitor_2_loss_mean": np.mean(capacitor_2_dto.loss_total_array),
                        "capacitor_2_area": capacitor_2_dto.area_total,
                        "capacitor_2_n_parallel": capacitor_2_dto.n_parallel,
                        "capacitor_2_n_series": capacitor_2_dto.n_series,
                    }
                    local_df_capacitor_2 = pd.DataFrame([capacitor_2_data])
                    df_capacitor_2 = pd.concat([df_capacitor_2, local_df_capacitor_2], axis=0)

            logger.info(f"{df_capacitor_2=}")

            # merge df's by creating a common key
            df_circuit_local['key'] = 0
            df_inductor['key'] = 0
            df_transformer['key'] = 0
            df_capacitor_1['key'] = 0
            df_capacitor_2['key'] = 0

            # perform cross join
            df_intermediate_1 = df_circuit_local.merge(df_inductor, on='key', how='outer')
            df_intermediate_2 = df_intermediate_1.merge(df_transformer, on='key', how='outer')
            df_intermediate_3 = df_intermediate_2.merge(df_capacitor_1, on='key', how='outer')
            # TODO: Needs fix, as data for capacitor 2 is currently missing
            # fix df_intermediate_3 to df_intermediate_4 in the following
            # df_intermediate_4 = df_intermediate_3.merge(df_capacitor_2, on='key', how='outer')

            del df_intermediate_3['key']

            logger.info(f"{df_intermediate_3=}")

            df = pd.concat([df, df_intermediate_3], axis=0)

        # Calculate the total area as sum of circuit,  inductor and transformer area df-command is like vector sum v1[:]=v2[:]+v3[:])
        # heat sink area, capacitors do not need heat sink area
        df["total_area"] = df["circuit_area"] + df["inductor_area"] + df["transformer_area"]

        # TODO: Fix needed as capacitor 2 is not considered currently
        df["total_mean_loss"] = (df["circuit_loss_array"].apply(np.mean) + df["inductor_loss_array"].apply(np.mean) + \
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
        df["r_th_ind_heat_sink"] = self.r_th_per_unit_area_ind_heat_sink / df["inductor_area"]
        df["temperature_inductor_heat_sink_max_array"] = 125 - df["r_th_ind_heat_sink"] * df["inductor_loss_array"]
        # For transformer: r_th_per_unit_area_xfmr_heat_sink = 1/lambda * l.
        df["r_th_xfmr_heat_sink"] = self.r_th_per_unit_area_xfmr_heat_sink / df["transformer_area"]
        df["temperature_xfmr_heat_sink_max_array"] = 125 - df["r_th_xfmr_heat_sink"] * df["transformer_loss_array"]

        # maximum heat sink temperatures (minimum of all the maximum temperatures of single components)
        df["t_min_array"] = df.apply(lambda x: np.minimum(x["circuit_heat_sink_max_1_array"], x["circuit_heat_sink_max_2_array"]), axis=1)
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

    def select_heat_sink_configuration(self, heat_sink_study_data: dct.StudyData, summary_data: dct.StudyData, act_df_for_hs: pd.DataFrame) -> None:
        """Select the heat sink configuration from calculated heat sink pareto front.

        :param heat_sink_study_data: Information about the heat sink study name and study path
        :type  heat_sink_study_data: dct.StudyData
        :param summary_data: Information about the summary name and path
        :type summary_data: dct.StudyData
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

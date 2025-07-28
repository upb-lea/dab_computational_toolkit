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
from dct.heat_sink_optimization import ThermalCalcSupport
import hct
from dct.server_ctl_dtos import ProgressData
from dct.server_ctl_dtos import RunTimeMeasurement as RunTime


logger = logging.getLogger(__name__)

class DctSummaryProcessing:
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

    def init_thermal_configuration(self, act_thermal_data: dct.TomlHeatSinkSummaryData) -> bool:
        """Initialize the thermal parameter of the connection points for the transistors, inductor and transformer.

        :param act_thermal_data : toml file with configuration data
        :type  act_thermal_data : dct.TomlHeatSinkSummaryData

        :return: True, if the thermal parameter of the connection points was successful initialized
        :rtype: bool
        """
        # Variable declaration
        # Return variable initialized to True
        successful_init = True
        transformer_cooling: dct.InductiveElementCooling
        # Thermal parameter for bridge transistor 1: List [tim_thickness, tim_conductivity]
        self.transistor_b1_cooling = dct.TransistorCooling(
            tim_thickness=act_thermal_data.transistor_b1_cooling[0],
            tim_conductivity=act_thermal_data.transistor_b1_cooling[1])

        # Thermal parameter for bridge transistor 2: List [tim_thickness, tim_conductivity]
        self.transistor_b2_cooling = dct.TransistorCooling(
            tim_thickness=act_thermal_data.transistor_b2_cooling[0],
            tim_conductivity=act_thermal_data.transistor_b2_cooling[1])

        # Thermal parameter for inductor: r_th per area: List [tim_thickness, tim_conductivity]
        inductor_tim_thickness = act_thermal_data.inductor_cooling[0]
        inductor_tim_conductivity = act_thermal_data.inductor_cooling[1]

        # Check on zero
        if inductor_tim_conductivity > 0:
            # Calculate the thermal resistance per unit area as term from the formula r_th = 1/lambda * l / A
            # r_th_per_unit_area_ind_heat_sink = 1/lambda * l. Later r_th = r_th_per_unit_area_ind_heat_sink / A
            self.r_th_per_unit_area_ind_heat_sink = inductor_tim_thickness / inductor_tim_conductivity
        else:
            logger.info(f"inductor cooling tim conductivity value must be greater zero, but is {inductor_tim_conductivity}!")
            successful_init = False

        # Thermal parameter for inductor: r_th per area: List [tim_thickness, tim_conductivity]
        # ASA: Rename database class from InductiveElementCooling to MagneticElementCooling
        transformer_tim_thickness = act_thermal_data.transformer_cooling[0]
        transformer_tim_conductivity = act_thermal_data.transformer_cooling[1]

        transformer_cooling = dct.InductiveElementCooling(
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

        # Heat sink parameter:  List [t_ambient, t_hs_max]
        self.heat_sink_boundary_conditions = dct.HeatSinkBoundaryConditions(t_ambient=act_thermal_data.heat_sink[0],
                                                                            t_hs_max=act_thermal_data.heat_sink[1])
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
    def _generate_magnetic_number_list(act_dir_name: str) -> tuple[bool, list[str]]:
        """Generate a list of the numbers from filenames.

        :param act_dir_name : Name of the directory containing the files
        :type  act_dir_name : str

        :return: tuple of bool and result list. True, if the directory exists and contains minimum one file
        :rtype: tuple
        """
        # Variable declaration
        magnetic_result_numbers: list[str] = []
        is_magnetic_list_generated = False

        # Check if target folder 09_circuit_dtos_incl_inductor_losses is created
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
                        magnetic_result_numbers.append(device_number)
                    else:
                        logger.info(f"File {device_number}{extension} has no extension '.pkl'!")
                else:
                    logger.info(f"File'{file_path}' does not exists!")
        else:
            logger.info(f"Path {act_dir_name} does not exists!")

        if magnetic_result_numbers:
            is_magnetic_list_generated = True

        return is_magnetic_list_generated, magnetic_result_numbers

    def generate_result_database(self, inductor_study_data: dct.StudyData, transformer_study_data: dct.StudyData,
                                 summary_data: dct.StudyData, act_inductor_study_names: list[str],
                                 act_stacked_transformer_study_names: list[str], filter_data: dct.FilterData) -> pd.DataFrame:
        """Generate a database df by summaries the calculation results.

        :param inductor_study_data: inductor study data
        :type inductor_study_data: dct.StudyData
        :param transformer_study_data: transformer study data
        :type transformer_study_data: dct.StudyData
        :param summary_data: Information about the summary name and path
        :type summary_data: dct.StudyData
        :param act_inductor_study_names: List of names with inductor studies which are to process
        :type  act_inductor_study_names: list[str]
        :param act_stacked_transformer_study_names: List of names with transformer studies which are to process
        :type  act_stacked_transformer_study_names: list[str]
        :param filter_data: filtered result lists
        :type filter_data: dct.FilterData
        :return: DataFrame with result information of the pareto front
        :rtype:  pd.DataFrame
        """
        # Variable declaration

        # Start the progress time measurement
        with self._s_lock_stat:
            self._progress_run_time.reset_start_trigger()
            self._progress_data.run_time = self._progress_run_time.get_runtime()
            self._progress_data.progress_status = ProgressStatus.InProgress

        # Result DataFrame
        df = pd.DataFrame()

        # iterate circuit numbers
        for circuit_trial_file in filter_data.filtered_list_files:
            # Assemble pkl-filename
            circuit_filepath_number = os.path.join(filter_data.filtered_list_pathname, f"{circuit_trial_file}.pkl")

            # Get circuit results
            circuit_dto = dct.HandleDabDto.load_from_file(circuit_filepath_number)

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

            logger.info(f"{circuit_trial_file=}")

            # iterate inductor study
            for inductor_study_name in act_inductor_study_names:

                # Assemble directory name for inductor results:.../09_circuit_dtos_incl_inductor_losses
                inductor_filepath_results = os.path.join(inductor_study_data.optimization_directory, circuit_trial_file,
                                                         inductor_study_name,
                                                         "09_circuit_dtos_incl_inductor_losses")

                # Generate magnetic list
                is_inductor_list_generated, inductor_full_operating_range_list = (
                    DctSummaryProcessing._generate_magnetic_number_list(inductor_filepath_results))
                if not is_inductor_list_generated:
                    logger.info(f"Path {inductor_filepath_results} does not exists or does not contains any pkl-files!")
                    # Next circuit
                    continue

                # iterate inductor numbers
                for inductor_number in inductor_full_operating_range_list:
                    inductor_filepath_number = os.path.join(inductor_filepath_results, f"{inductor_number}.pkl")

                    # Get inductor results
                    with open(inductor_filepath_number, 'rb') as pickle_file_data:
                        inductor_dto = pickle.load(pickle_file_data)

                    if inductor_dto.circuit_trial_file != circuit_trial_file:
                        raise ValueError(f"{inductor_dto.circuit_trial_file=} != {circuit_trial_file}")
                    if int(inductor_dto.inductor_trial_number) != int(inductor_number):
                        raise ValueError(f"{inductor_dto.inductor_trial_number=} != {inductor_number}")

                    inductance_loss_matrix = inductor_dto.p_combined_losses

                    # iterate transformer study
                    for stacked_transformer_study_name in act_stacked_transformer_study_names:

                        # Assemble directory name for transformer  results:.../09_circuit_dtos_incl_transformer_losses
                        stacked_transformer_filepath_results = os.path.join(transformer_study_data.optimization_directory,
                                                                            circuit_trial_file,
                                                                            stacked_transformer_study_name,
                                                                            "09_circuit_dtos_incl_transformer_losses")

                        # Check, if stacked transformer number list cannot be generated
                        is_transformer_list_generated, stacked_transformer_full_operating_range_list = (
                            DctSummaryProcessing._generate_magnetic_number_list(stacked_transformer_filepath_results))
                        if not is_transformer_list_generated:
                            logger.info(f"Path {stacked_transformer_filepath_results} does not exists or does not contains any pkl-files!")
                            # Next circuit
                            continue

                        # iterate transformer numbers
                        for stacked_transformer_number in stacked_transformer_full_operating_range_list:
                            stacked_transformer_filepath_number = os.path.join(stacked_transformer_filepath_results, f"{stacked_transformer_number}.pkl")

                            # get transformer results
                            with open(stacked_transformer_filepath_number, 'rb') as pickle_file_data:
                                transformer_dto = pickle.load(pickle_file_data)

                            if transformer_dto.circuit_trial_file != circuit_trial_file:
                                raise ValueError(f"{transformer_dto.circuit_trial_file=} != {circuit_trial_file}")
                            if int(transformer_dto.stacked_transformer_trial_number) != int(stacked_transformer_number):
                                raise ValueError(f"{transformer_dto.stacked_transformer_trial_number=} != {stacked_transformer_number}")

                            transformer_loss_matrix = transformer_dto.p_combined_losses

                            total_loss_matrix = (inductor_dto.p_combined_losses + total_transistor_cond_loss_matrix + \
                                                 transformer_dto.p_combined_losses)

                            # maximum loss indices
                            max_loss_all_index = np.unravel_index(total_loss_matrix.argmax(), np.shape(total_loss_matrix))
                            # Calculate losses of circuit1 and 2
                            max_loss_circuit_1_index = np.unravel_index(b1_transistor_cond_loss_matrix.argmax(),
                                                                        np.shape(b1_transistor_cond_loss_matrix))
                            max_loss_circuit_2_index = np.unravel_index(b2_transistor_cond_loss_matrix.argmax(),
                                                                        np.shape(b2_transistor_cond_loss_matrix))

                            max_loss_inductor_index = np.unravel_index(inductance_loss_matrix.argmax(), np.shape(inductance_loss_matrix))
                            max_loss_transformer_index = np.unravel_index(transformer_loss_matrix.argmax(), np.shape(transformer_loss_matrix))
                            # Calculate the thermal resistance according r_th = 1/lambda * l / A
                            # For inductor: r_th_per_unit_area_ind_heat_sink = 1/lambda * l
                            r_th_ind_heat_sink = self.r_th_per_unit_area_ind_heat_sink / inductor_dto.area_to_heat_sink
                            temperature_inductor_heat_sink_max_matrix = 125 - r_th_ind_heat_sink * inductance_loss_matrix
                            # For transformer: r_th_per_unit_area_xfmr_heat_sink = 1/lambda * l.
                            r_th_xfmr_heat_sink = self.r_th_per_unit_area_xfmr_heat_sink / transformer_dto.area_to_heat_sink
                            temperature_xfmr_heat_sink_max_matrix = 125 - r_th_xfmr_heat_sink * transformer_loss_matrix

                            # maximum heat sink temperatures (minimum of all the maximum temperatures of single components)
                            t_min_matrix = np.minimum(circuit_heat_sink_max_1_matrix, circuit_heat_sink_max_2_matrix)
                            t_min_matrix = np.minimum(t_min_matrix, temperature_inductor_heat_sink_max_matrix)
                            t_min_matrix = np.minimum(t_min_matrix, temperature_xfmr_heat_sink_max_matrix)
                            t_min_matrix = np.minimum(t_min_matrix, self.heat_sink_boundary_conditions.t_hs_max)

                            # maximum delta temperature over the heat sink
                            delta_t_max_heat_sink_matrix = t_min_matrix - self.heat_sink_boundary_conditions.t_ambient

                            r_th_heat_sink_target_matrix = delta_t_max_heat_sink_matrix / total_loss_matrix

                            r_th_target = r_th_heat_sink_target_matrix.min()

                            data = {
                                # circuit
                                "circuit_trial_file": circuit_trial_file,
                                "circuit_mean_loss": np.mean(total_transistor_cond_loss_matrix),
                                "circuit_max_all_loss": total_transistor_cond_loss_matrix[max_loss_all_index],
                                "circuit_max_circuit_ib_loss": total_transistor_cond_loss_matrix[max_loss_circuit_1_index],
                                "circuit_max_circuit_ob_loss": total_transistor_cond_loss_matrix[max_loss_circuit_2_index],
                                "circuit_max_inductor_loss": total_transistor_cond_loss_matrix[max_loss_inductor_index],
                                "circuit_max_transformer_loss": total_transistor_cond_loss_matrix[max_loss_transformer_index],
                                "circuit_t_j_max_1": circuit_dto.input_config.transistor_dto_1.t_j_max_op,
                                "circuit_t_j_max_2": circuit_dto.input_config.transistor_dto_2.t_j_max_op,
                                "circuit_r_th_ib_jhs_1": circuit_r_th_1_jhs,
                                "circuit_r_th_ib_jhs_2": circuit_r_th_2_jhs,
                                "circuit_heat_sink_temperature_max_1": circuit_heat_sink_max_1_matrix[max_loss_circuit_1_index],
                                "circuit_heat_sink_temperature_max_2": circuit_heat_sink_max_2_matrix[max_loss_circuit_2_index],
                                "circuit_area": 4 * (copper_coin_area_1 + copper_coin_area_2),
                                # inductor
                                "inductor_study_name": inductor_study_name,
                                "inductor_number": inductor_number,
                                "inductor_volume": inductor_dto.volume,
                                "inductor_mean_loss": np.mean(inductance_loss_matrix),
                                "inductor_max_all_loss": inductance_loss_matrix[max_loss_all_index],
                                "inductor_max_circuit_ib_loss": inductance_loss_matrix[max_loss_circuit_1_index],
                                "inductor_max_circuit_ob_loss": inductance_loss_matrix[max_loss_circuit_2_index],
                                "inductor_max_inductor_loss": inductance_loss_matrix[max_loss_inductor_index],
                                "inductor_max_transformer_loss": inductance_loss_matrix[max_loss_transformer_index],
                                "inductor_t_max": 0,
                                "inductor_heat_sink_temperature_max": temperature_inductor_heat_sink_max_matrix[max_loss_inductor_index],
                                "inductor_area": inductor_dto.area_to_heat_sink,
                                # transformer
                                "transformer_study_name": stacked_transformer_study_name,
                                "transformer_number": stacked_transformer_number,
                                "transformer_volume": transformer_dto.volume,
                                "transformer_mean_loss": np.mean(transformer_dto.p_combined_losses),
                                "transformer_max_all_loss": transformer_loss_matrix[max_loss_all_index],
                                "transformer_max_circuit_ib_loss": transformer_loss_matrix[max_loss_circuit_1_index],
                                "transformer_max_circuit_ob_loss": transformer_loss_matrix[max_loss_circuit_2_index],
                                "transformer_max_inductor_loss": transformer_loss_matrix[max_loss_inductor_index],
                                "transformer_max_transformer_loss": transformer_loss_matrix[max_loss_transformer_index],
                                "transformer_t_max": 0,
                                "transformer_heat_sink_temperature_max": temperature_xfmr_heat_sink_max_matrix[max_loss_transformer_index],
                                "transformer_area": transformer_dto.area_to_heat_sink,

                                # summary
                                "total_losses": total_loss_matrix[max_loss_all_index],

                                # heat sink
                                "r_th_heat_sink": r_th_target
                            }
                            local_df = pd.DataFrame([data])

                            df = pd.concat([df, local_df], axis=0)

        # Calculate the total area as sum of circuit,  inductor and transformer area df-command is like vector sum v1[:]=v2[:]+v3[:])
        df["total_area"] = df["circuit_area"] + df["inductor_area"] + df["transformer_area"]
        df["total_mean_loss"] = df["circuit_mean_loss"] + df["inductor_mean_loss"] + df["transformer_mean_loss"]
        df["volume_wo_heat_sink"] = df["transformer_volume"] + df["inductor_volume"]
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

        act_df_for_hs["total_volume"] = act_df_for_hs["transformer_volume"] + act_df_for_hs["inductor_volume"] + act_df_for_hs["heat_sink_volume"]

        # save full summary
        act_df_for_hs.to_csv(f"{summary_data.optimization_directory}/df_w_hs.csv")

        # Update statistical data for summary processing finalized
        # Update statistical data
        with self._s_lock_stat:
            self._progress_run_time.stop_trigger()
            self._progress_data.run_time = self._progress_run_time.get_runtime()
            self._progress_data.progress_status = ProgressStatus.Done

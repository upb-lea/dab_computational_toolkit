"""Pareto optimization classes and functions."""
# Python libraries
import os
import logging
import json
import pickle
import datetime
import threading
import copy

# 3rd party libraries
import optuna
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import deepdiff
import dct.sampling as sampling

import dct.datasets_dtos
# own libraries
import dct.datasets_dtos as d_dtos
import dct.circuit_optimization_dtos as circuit_dtos
import dct.datasets as d_sets
import transistordatabase as tdb
from dct.boundary_check import CheckCondition as c_flag
from dct import toml_checker as tc
from dct.server_ctl_dtos import ProgressData
from dct.server_ctl_dtos import ProgressStatus
from dct.server_ctl_dtos import RunTimeMeasurement as RunTime
from dct.circuit_enums import SamplingEnum

logger = logging.getLogger(__name__)

class CircuitOptimization:
    """Optimize the DAB converter regarding maximum ZVS coverage and minimum conduction losses."""

    # Declaration of member types
    _c_lock_stat: threading.Lock
    _progress_data: ProgressData
    _progress_run_time: RunTime
    _dab_config: circuit_dtos.CircuitParetoDabDesign | None
    _study_in_memory: optuna.Study | None
    _study_in_storage: optuna.Study | None
    _fixed_parameters: d_dtos.FixedParameters | None

    def __init__(self):
        """Initialize the configuration list for the circuit optimizations."""
        # Variable allocation
        self._c_lock_stat = threading.Lock()
        # Initialize the statistical data (For more configuration it needs to become instance instead of static
        self._progress_data = ProgressData(run_time=0, number_of_filtered_points=0,
                                           progress_status=ProgressStatus.Idle)
        self._progress_run_time = RunTime()
        self._dab_config = None
        self._is_study_available = False

        self._study_in_memory = None
        self._study_in_storage = None
        self._fixed_parameters = None

    @staticmethod
    def load_filepaths(project_directory: str) -> circuit_dtos.ParetoFilePaths:
        """
        Load file path of the subdirectories of the project.

        :param project_directory: project directory file path
        :type project_directory: str
        :return: File path in a DTO
        :rtype: p_dtos.ParetoFilePaths
        """
        # ASA: TODO: Merge ginfo and set_up_folder_structure
        filepath_config = f"{project_directory}/filepath_config.json"
        if os.path.exists(filepath_config):
            with open(filepath_config, 'r', encoding='utf8') as json_file:
                loaded_file = json.load(json_file)
        else:
            raise ValueError("Project does not exist.")

        file_path_dto = circuit_dtos.ParetoFilePaths(
            circuit=loaded_file["circuit"],
            transformer=loaded_file["transformer"],
            inductor=loaded_file["inductor"],
            heat_sink=loaded_file["heat_sink"]
        )
        return file_path_dto

    def save_config(self) -> None:
        """Save the actual configuration file as pickle file on the disk."""
        # Check if a configuration is loaded
        if self._dab_config is None:
            logger.warning("Circuit configuration is empty!\n    Configuration is not saved!")
            return

        filepaths = CircuitOptimization.load_filepaths(self._dab_config.project_directory)

        os.makedirs(self._dab_config.project_directory, exist_ok=True)
        with open(f"{filepaths.circuit}/{self._dab_config.circuit_study_name}/{self._dab_config.circuit_study_name}.pkl", 'wb') as output:
            pickle.dump(self._dab_config, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_stored_config(circuit_project_directory: str, circuit_study_name: str) -> circuit_dtos.CircuitParetoDabDesign:
        """
        Load pickle configuration file from disk.

        :param circuit_project_directory: project directory
        :type circuit_project_directory: str
        :param circuit_study_name: name of the circuit study
        :type circuit_study_name: str
        :return: Configuration file as p_dtos.DabDesign
        :rtype: p_dtos.CircuitParetoDabDesign
        """
        filepaths = CircuitOptimization.load_filepaths(circuit_project_directory)
        config_pickle_filepath = os.path.join(filepaths.circuit, circuit_study_name, f"{circuit_study_name}.pkl")

        with open(config_pickle_filepath, 'rb') as pickle_file_data:
            loaded_pareto_dto = pickle.load(pickle_file_data)
            if not isinstance(loaded_pareto_dto, circuit_dtos.CircuitParetoDabDesign):
                raise TypeError(f"Loaded pickle file {loaded_pareto_dto} not of type CircuitParetoDabDesign.")

        return loaded_pareto_dto

    @staticmethod
    def verify_optimization_parameter(toml_circuit: tc.TomlCircuitParetoDabDesign) -> tuple[bool, str]:
        """Verify the input parameter ranges.

        :param toml_circuit: toml inductor configuration
        :type toml_circuit: dct.TomlInductor
        :return: True, if the configuration was consistent
        :rtype: bool
        """
        # Variable declaration
        inconsistency_report: str = ""
        is_inconsistent: bool = False
        toml_check_keyword_list: list[tuple[list[str], str]]
        toml_check_min_max_values_list: list[tuple[list[float], str]]
        toml_check_value_list: list[tuple[float, str]]
        toml_check_min_max_value_multi_list: list[tuple[list[float], str, list[float], str]]

        # Design space parameter check
        # Create dictionary from transistor database list
        db = tdb.DatabaseManager()
        db.set_operation_mode_json()
        db.update_from_fileexchange(True)

        # Get available keywords
        keyword_list: list[str] = db.get_transistor_names_list()
        keyword_dictionary: dict = {}
        # Create dictionary
        for keyword_entry in keyword_list:
            keyword_dictionary[keyword_entry] = 0

        # Check transistors
        toml_check_keyword_list = (
            [(toml_circuit.design_space.transistor_1_name_list, "transistor_1_name_list"),
             (toml_circuit.design_space.transistor_2_name_list, "transistor_2_name_list")])

        # Perform the boundary check
        for check_keyword in toml_check_keyword_list:
            if len(check_keyword[0]) == 0:
                inconsistency_report = f"   Circuit parameter: List {check_keyword[1]} is empty!\n"
                is_inconsistent = True
            else:
                # Perform dictionary check
                for keyword_entry in check_keyword[0]:
                    is_check_failed, issue_report = dct.BoundaryCheck.check_dictionary(keyword_dictionary, keyword_entry, check_keyword[1])
                    # Check if boundary check fails
                    if is_check_failed:
                        inconsistency_report = inconsistency_report + issue_report
                        is_inconsistent = True

        # Check switching frequency range
        float_f_s_min_max_list = dct.BoundaryCheck.convert_min_max_values_to_float(toml_circuit.design_space.f_s_min_max_list)
        is_check_failed, issue_report = dct.BoundaryCheck.check_float_min_max_values(
            1000, 1e7, float_f_s_min_max_list, "f_s_min_max_list", c_flag.check_exclusive, c_flag.check_exclusive)
        if is_check_failed:
            inconsistency_report = inconsistency_report + issue_report
            is_inconsistent = True

        # Check l_s_min_max_list, l_1_min_max_list and l_2__min_max_list
        toml_check_min_max_values_list = (
            [(toml_circuit.design_space.l_s_min_max_list, "l_s_min_max_list"),
             (toml_circuit.design_space.l_1_min_max_list, "l_1_min_max_list"),
             (toml_circuit.design_space.l_2__min_max_list, "l_2__min_max_list")])

        # Perform the boundary check
        is_check_failed, issue_report = dct.BoundaryCheck.check_float_min_max_values_list(
            0, 1, toml_check_min_max_values_list, c_flag.check_exclusive, c_flag.check_exclusive)
        if is_check_failed:
            inconsistency_report = inconsistency_report + issue_report
            is_inconsistent = True

        is_check_failed, issue_report = dct.BoundaryCheck.check_float_min_max_values(
            0, 100, toml_circuit.design_space.n_min_max_list, "n_min_max_list", c_flag.check_exclusive, c_flag.check_exclusive)
        if is_check_failed:
            inconsistency_report = inconsistency_report + issue_report
            is_inconsistent = True

        # Check l_s_min_max_list, l_1_min_max_list and l_2__min_max_list
        toml_check_min_max_values_list = (
            [(toml_circuit.design_space.l_s_min_max_list, "l_s_min_max_list"),
             (toml_circuit.design_space.l_1_min_max_list, "l_1_min_max_list"),
             (toml_circuit.design_space.l_2__min_max_list, "l_2__min_max_list")])

        # Perform the boundary check
        is_check_failed, issue_report = dct.BoundaryCheck.check_float_min_max_values_list(
            0, 1, toml_check_min_max_values_list, c_flag.check_inclusive, c_flag.check_exclusive)
        if is_check_failed:
            inconsistency_report = inconsistency_report + issue_report
            is_inconsistent = True

        # Check c_par_1 and c_par_2
        toml_check_value_list = (
            [(toml_circuit.design_space.c_par_1, "c_par_1"),
             (toml_circuit.design_space.c_par_2, "c_par_2")])

        # Perform the boundary check
        # Check c_par_1 and c_par_2
        is_check_failed, issue_report = dct.BoundaryCheck.check_float_value_list(
            0, 1, toml_check_value_list, c_flag.check_exclusive, c_flag.check_exclusive)
        if is_check_failed:
            inconsistency_report = inconsistency_report + issue_report
            is_inconsistent = True

        # Output range parameter and sampling parameter check
        # Init is_user_point_list_consistent-flag
        is_user_point_list_consistent = False
        # Evaluate list length
        len_check1 = len(toml_circuit.sampling.v1_additional_user_point_list) == len(toml_circuit.sampling.v2_additional_user_point_list)
        len_check2 = len(toml_circuit.sampling.p_additional_user_point_list) == len(toml_circuit.sampling.additional_user_weighting_point_list)
        len_check3 = len(toml_circuit.sampling.v1_additional_user_point_list) == len(toml_circuit.sampling.p_additional_user_point_list)
        # Check if the additional user point lists are consistent
        if len_check1 == len_check2 and len_check2 == len_check3:
            is_user_point_list_consistent = True

        # Check v1_min_max_list and v2_min_max_list
        toml_check_min_max_value_multi_list = (
            [(toml_circuit.output_range.v1_min_max_list, "v1_min_max_list",
              toml_circuit.sampling.v1_additional_user_point_list, "v1_additional_user_point_list"),
             (toml_circuit.output_range.v2_min_max_list, "v2_min_max_list",
              toml_circuit.sampling.v2_additional_user_point_list, "v2_additional_user_point_list")])

        # Perform the boundary check
        for check_parameter in toml_check_min_max_value_multi_list:
            is_check_failed, issue_report = dct.BoundaryCheck.check_float_min_max_values(
                0, 1500, check_parameter[0], check_parameter[1], c_flag.check_exclusive, c_flag.check_exclusive)
            if is_check_failed:
                inconsistency_report = inconsistency_report + issue_report
                is_inconsistent = True
            elif is_user_point_list_consistent:
                for voltage_value in check_parameter[2]:
                    is_check_failed, issue_report = dct.BoundaryCheck.check_float_value(
                        check_parameter[0][0], check_parameter[0][1], voltage_value, check_parameter[3], c_flag.check_inclusive, c_flag.check_inclusive)
                    if is_check_failed:
                        inconsistency_report = inconsistency_report + issue_report
                        is_inconsistent = True

        # Perform the boundary check  of p_min_max_list
        is_check_failed, issue_report = dct.BoundaryCheck.check_float_min_max_values(
            -100000, 100000, toml_circuit.output_range.p_min_max_list, "p_min_max_list", c_flag.check_exclusive, c_flag.check_exclusive)
        if is_check_failed:
            inconsistency_report = inconsistency_report + issue_report
            is_inconsistent = True
        elif is_user_point_list_consistent:
            for power_value in toml_circuit.sampling.p_additional_user_point_list:
                is_check_failed, issue_report = dct.BoundaryCheck.check_float_value(
                    toml_circuit.output_range.p_min_max_list[0], toml_circuit.output_range.p_min_max_list[1], power_value,
                    "p_additional_user_point_list", c_flag.check_inclusive, c_flag.check_inclusive)
                if is_check_failed:
                    inconsistency_report = inconsistency_report + issue_report
                    is_inconsistent = True

        # Remaining Sampling parameter check
        # Check v1_additional_user_point_list
        # Initialize variable
        weighting_sum: float = 0.0
        # Perform the boundary check  of p_min_max_list
        for weight_value in toml_circuit.sampling.additional_user_weighting_point_list:
            is_check_failed, issue_report = dct.BoundaryCheck.check_float_value(
                0, 1, weight_value, "additional_user_weighting_point_list", c_flag.check_inclusive, c_flag.check_inclusive)
            if is_check_failed:
                inconsistency_report = inconsistency_report + issue_report
                is_inconsistent = True

            weighting_sum = weighting_sum + weight_value

        # Check the sum
        weighting_sum = 0.0
        # Perform the boundary check  of p_min_max_list
        if weighting_sum > 1 or weighting_sum < 0:
            is_inconsistent = True
            inconsistency_report = inconsistency_report + f"    The additional weighting point list sum of {weighting_sum} is out of range!"

        # Perform filter_distance value check
        group_name = "filter_distance"
        # Perform the boundary check for number_filtered_designs
        is_check_failed, issue_report = dct.BoundaryCheck.check_float_value(
            0, 100, float(toml_circuit.filter_distance.number_filtered_designs),
            f"{group_name}: number_filtered_designs", c_flag.check_exclusive, c_flag.check_ignore)
        if is_check_failed:
            inconsistency_report = inconsistency_report + issue_report
            is_inconsistent = True

        # Perform the boundary check for number_filtered_designs
        is_check_failed, issue_report = dct.BoundaryCheck.check_float_value(
            0.01, 100, toml_circuit.filter_distance.difference_percentage, f"{group_name}: difference_percentage", c_flag.check_exclusive, c_flag.check_ignore)
        if is_check_failed:
            inconsistency_report = inconsistency_report + issue_report
            is_inconsistent = True

        return is_inconsistent, inconsistency_report

    def get_config(self) -> circuit_dtos.CircuitParetoDabDesign | None:
        """
        Return the actual loaded configuration file.

        :return: Configuration file as p_dtos.DabDesign
        :rtype: p_dtos.CircuitParetoDabDesign
        """
        if self._dab_config is None:
            logger.warning("Configuration is not loaded!")

        return copy.deepcopy(self._dab_config)

    def get_progress_data(self) -> ProgressData:
        """Provide the progress data of the optimization.

        :return: Progress data: Processing start time, actual processing time, number of filtered operation points and status.
        :rtype: ProgressData
        """
        # Lock statistical performance data access
        with self._c_lock_stat:
            # Check if list is in progress
            if self._progress_data.progress_status == ProgressStatus.InProgress:
                # Update statistical data if optimization is running
                self._progress_data.run_time = self._progress_run_time.get_runtime()

        return copy.deepcopy(self._progress_data)

    def get_actual_pareto_html(self) -> str:
        """
        Read the current Pareto front from running optimization process.

        :return: Pareto front html page
        :rtype: str
        """
        # Variable declaration
        pareto_html: str = ""

        if self._study_in_memory is not None:
            fig = optuna.visualization.plot_pareto_front(self._study_in_memory)
            pareto_html = fig.to_html(full_html=False)

        return pareto_html

    @staticmethod
    def get_pareto_html(study_name: str, path_name: str) -> str:
        """
        Read the current Pareto front from running optimization process.

        :param study_name: Name of the optuna study
        :type  study_name: str
        :param path_name: Path where optuna study is located
        :type  path_name: str

        :return: Pareto front html page
        :rtype: str
        """
        # Variable declaration
        pareto_html = ""

        # Load study from file
        try:
            study = optuna.load_study(study_name=study_name, storage="sqlite:///" + path_name)
            fig = optuna.visualization.plot_pareto_front(study)
            pareto_html = fig.to_html(full_html=False)
        except KeyError:
            logger.warning(f"Study with name '{study_name}' are not found.")

        return pareto_html

    @staticmethod
    def _objective(trial: optuna.Trial, dab_config: circuit_dtos.CircuitParetoDabDesign, fixed_parameters: d_dtos.FixedParameters) -> tuple:
        """
        Objective function to optimize.

        :param dab_config: DAB optimization configuration file
        :type dab_config: p_dtos.CircuitParetoDabDesign
        :param trial: optuna trial
        :type trial: optuna.Trial
        :param fixed_parameters: fixed parameters (loaded transistor DTOs)
        :type fixed_parameters: d_dtos.FixedParameters

        :return:
        """
        f_s_suggest = trial.suggest_int('f_s_suggest', dab_config.design_space.f_s_min_max_list[0], dab_config.design_space.f_s_min_max_list[1])
        l_s_suggest = trial.suggest_float('l_s_suggest', dab_config.design_space.l_s_min_max_list[0], dab_config.design_space.l_s_min_max_list[1])
        l_1_suggest = trial.suggest_float('l_1_suggest', dab_config.design_space.l_1_min_max_list[0], dab_config.design_space.l_1_min_max_list[1])
        l_2__suggest = trial.suggest_float('l_2__suggest', dab_config.design_space.l_2__min_max_list[0], dab_config.design_space.l_2__min_max_list[1])
        n_suggest = trial.suggest_float('n_suggest', dab_config.design_space.n_min_max_list[0], dab_config.design_space.n_min_max_list[1])
        transistor_1_name_suggest = trial.suggest_categorical('transistor_1_name_suggest', dab_config.design_space.transistor_1_name_list)
        transistor_2_name_suggest = trial.suggest_categorical('transistor_2_name_suggest', dab_config.design_space.transistor_2_name_list)

        for _, transistor_dto in enumerate(fixed_parameters.transistor_1_dto_list):
            if transistor_dto.name == transistor_1_name_suggest:
                transistor_1_dto: d_dtos.TransistorDTO = transistor_dto

        for _, transistor_dto in enumerate(fixed_parameters.transistor_2_dto_list):
            if transistor_dto.name == transistor_2_name_suggest:
                transistor_2_dto: d_dtos.TransistorDTO = transistor_dto

        dab_calc = d_sets.HandleDabDto.init_config(
            name=dab_config.circuit_study_name,
            mesh_v1=fixed_parameters.mesh_v1,
            mesh_v2=fixed_parameters.mesh_v2,
            mesh_p=fixed_parameters.mesh_p,
            sampling=dab_config.sampling,
            n=n_suggest,
            ls=l_s_suggest,
            fs=f_s_suggest,
            lc1=l_1_suggest,
            lc2=l_2__suggest / n_suggest ** 2,
            c_par_1=dab_config.design_space.c_par_1,
            c_par_2=dab_config.design_space.c_par_2,
            transistor_dto_1=transistor_1_dto,
            transistor_dto_2=transistor_2_dto
        )

        if (np.any(np.isnan(dab_calc.calc_modulation.phi)) or np.any(np.isnan(dab_calc.calc_modulation.tau1)) \
                or np.any(np.isnan(dab_calc.calc_modulation.tau2))):
            return float('nan'), float('nan')

        # Calculate the cost function.
        i_cost_matrix = dab_calc.calc_currents.i_hf_1_rms ** 2 + dab_calc.calc_currents.i_hf_2_rms ** 2
        # consider weighting
        i_cost_matrix_weighted = i_cost_matrix * fixed_parameters.mesh_weights

        # Mean for not-NaN values, as there will be too many NaN results.
        i_cost = np.mean(i_cost_matrix_weighted[~np.isnan(i_cost_matrix_weighted)])

        return dab_calc.calc_modulation.mask_zvs_coverage * 100, i_cost

    @staticmethod
    def calculate_fixed_parameters(act_dab_config: circuit_dtos.CircuitParetoDabDesign) -> d_dtos.FixedParameters:
        """
        Calculate time-consuming parameters which are same for every single simulation.

        :param act_dab_config: DAB circuit configuration
        :type act_dab_config: circuit_dtos.CircuitParetoDabDesign
        :return: Fix parameters (transistor DTOs)
        :rtype: d_dtos.FixedParameters
        """
        transistor_1_dto_list = []
        transistor_2_dto_list = []

        for transistor in act_dab_config.design_space.transistor_1_name_list:
            transistor_1_dto_list.append(d_sets.HandleTransistorDto.tdb_to_transistor_dto(transistor))

        for transistor in act_dab_config.design_space.transistor_2_name_list:
            transistor_2_dto_list.append(d_sets.HandleTransistorDto.tdb_to_transistor_dto(transistor))

        # choose sampling method
        if act_dab_config.sampling.sampling_method == SamplingEnum.meshgrid:
            steps_per_dimension = int(np.ceil(np.power(act_dab_config.sampling.sampling_points, 1 / 3)))
            logger.info(f"Number of sampling points has been updated from {act_dab_config.sampling.sampling_points} to {steps_per_dimension ** 3}.")
            logger.info("Note: meshgrid sampling does not take user-given operating points into account")
            v1_operating_points, v2_operating_points, p_operating_points = np.meshgrid(
                np.linspace(act_dab_config.output_range.v1_min_max_list[0], act_dab_config.output_range.v1_min_max_list[1], steps_per_dimension),
                np.linspace(act_dab_config.output_range.v2_min_max_list[0], act_dab_config.output_range.v2_min_max_list[1], steps_per_dimension),
                np.linspace(act_dab_config.output_range.p_min_max_list[0], act_dab_config.output_range.p_min_max_list[1], steps_per_dimension),
                sparse=False)
        elif act_dab_config.sampling.sampling_method == SamplingEnum.latin_hypercube:
            v1_operating_points, v2_operating_points, p_operating_points = sampling.latin_hypercube(
                act_dab_config.output_range.v1_min_max_list[0], act_dab_config.output_range.v1_min_max_list[1],
                act_dab_config.output_range.v2_min_max_list[0], act_dab_config.output_range.v2_min_max_list[1],
                act_dab_config.output_range.p_min_max_list[0], act_dab_config.output_range.p_min_max_list[1],
                total_number_points=act_dab_config.sampling.sampling_points,
                dim_1_user_given_points_list=act_dab_config.sampling.v1_additional_user_point_list,
                dim_2_user_given_points_list=act_dab_config.sampling.v2_additional_user_point_list,
                dim_3_user_given_points_list=act_dab_config.sampling.p_additional_user_point_list,
                sampling_random_seed=act_dab_config.sampling.sampling_random_seed)
        else:
            raise ValueError(f"sampling_method '{act_dab_config.sampling.sampling_method}' not available.")

        logger.debug(f"{v1_operating_points=}")

        # calculate weighting

        if act_dab_config.sampling.sampling_method == SamplingEnum.meshgrid:
            weight_sum = 0
            given_user_points = 0
        else:
            weight_sum = np.sum(act_dab_config.sampling.additional_user_weighting_point_list)
            logger.debug(f"{weight_sum=}")
            given_user_points = len(act_dab_config.sampling.v1_additional_user_point_list)
        logger.debug(f"{given_user_points=}")
        logger.debug(f"{v1_operating_points.size=}")

        if weight_sum > 1 or weight_sum < 0:
            raise ValueError("Sum of weighting point list must be within 0 and 1.")
        else:
            leftover_auto_weight = (1 - weight_sum) / (v1_operating_points.size - given_user_points)
            logger.info(f"Auto-weight given for all other {v1_operating_points.size - given_user_points} operating points: {leftover_auto_weight}")
            # default case, same weights for all points
            weights = np.full_like(v1_operating_points, leftover_auto_weight)
            # for user point weightings, both lists must be filled.
            if act_dab_config.sampling.additional_user_weighting_point_list and act_dab_config.sampling.sampling_method != SamplingEnum.meshgrid:
                logger.debug("Given user weighting point list detected, fill up with user-given weights.")
                weights[-len(act_dab_config.sampling.additional_user_weighting_point_list):] = act_dab_config.sampling.additional_user_weighting_point_list
            logger.debug(f"{weights=}")
            logger.debug(f"Double check: Sum of weights = {np.sum(weights)}")

        return d_dtos.FixedParameters(
            transistor_1_dto_list=transistor_1_dto_list,
            transistor_2_dto_list=transistor_2_dto_list,
            mesh_v1=np.atleast_3d(v1_operating_points),
            mesh_v2=np.atleast_3d(v2_operating_points),
            mesh_p=np.atleast_3d(p_operating_points),
            mesh_weights=np.atleast_3d(weights)
        )

    def run_optimization_sqlite(self, act_number_trials: int) -> None:
        """Proceed a study which is stored as sqlite database.

        :type act_number_trials: int
        :param act_number_trials: Number of optimization trials
        """
        if self._dab_config is None:
            logger.warning("Circuit configuration is not initialized!")
            return
        elif self._fixed_parameters is None:
            logger.warning("Parameter calculation is missing!")
            return
        elif self._study_in_memory is None:
            logger.warning("Study is not initialized!")
            return

        # Function to execute
        func = lambda trial: CircuitOptimization._objective(trial, self._dab_config, self._fixed_parameters)

        try:
            self._study_in_memory.optimize(func, n_trials=act_number_trials, n_jobs=1, show_progress_bar=True)
        except KeyboardInterrupt:
            pass

    def run_optimization_mysql(self, act_storage_url: str, act_number_trials: int) -> None:
        """Proceed a study which is stored as sqlite database.

        :param act_storage_url: url-Name of the database path
        :type act_storage_url: str
        :param act_number_trials: Number of trials adding to the existing study
        :type  act_number_trials: int
        """
        if self._dab_config is None:
            logger.warning("Circuit configuration is not initialized!")
            return
        elif self._fixed_parameters is None:
            logger.warning("Parameter calculation is missing!")
            return

        # Function to execute
        func = lambda trial: CircuitOptimization._objective(trial, self._dab_config, self._fixed_parameters)

        # Each process create his own study instance with the same database and study name
        act_study = optuna.load_study(storage=act_storage_url, study_name=self._dab_config.circuit_study_name)
        # Run optimization
        try:
            act_study.optimize(func, n_trials=act_number_trials, n_jobs=1, show_progress_bar=True)
        except KeyboardInterrupt:
            pass
        finally:
            # study_in_storage.add_trials(act_study_name.trials[-number_trials:])
            logger.info(f"Finished {act_number_trials} trials.")
            logger.info(f"current time: {datetime.datetime.now()}")
            # Save method from RAM-Disk to where ever (Currently opened by missing RAM-DISK)

    def start_proceed_study(self, dab_config: circuit_dtos.CircuitParetoDabDesign, number_trials: int,
                            database_type: str = 'sqlite',
                            sampler: optuna.samplers.BaseSampler = optuna.samplers.NSGAIIISampler()) -> None:
        """Proceed a study which is stored as sqlite database.

        :param dab_config: DAB optimization configuration file
        :type dab_config: p_dtos.CircuitParetoDabDesign
        :param number_trials: Number of trials adding to the existing study
        :type number_trials: int
        :param database_type: storage database, e.g. 'sqlite' or 'mysql'
        :type  database_type: str
        :param sampler: optuna.samplers.NSGAIISampler() or optuna.samplers.NSGAIIISampler(). Note about the brackets () !! Default: NSGAIII
        :type sampler: optuna.sampler-object
        """
        # Overtake configuration (Later this is to do by 'generate_optimization_list' or correspondent method
        self._dab_config = copy.deepcopy(dab_config)

        filepaths = CircuitOptimization.load_filepaths(self._dab_config.project_directory)

        circuit_study_working_directory = os.path.join(filepaths.circuit, self._dab_config.circuit_study_name)
        circuit_study_sqlite_database = os.path.join(circuit_study_working_directory, f"{self._dab_config.circuit_study_name}.sqlite3")

        if os.path.exists(circuit_study_sqlite_database):
            logger.info("Existing circuit study found. Proceeding.")
        else:
            os.makedirs(f"{filepaths.circuit}/{self._dab_config.circuit_study_name}", exist_ok=True)

        # set logging verbosity: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.logger.set_verbosity.html#optuna.logging.set_verbosity
        # .INFO: all messages (default)
        # .WARNING: fails and warnings
        # .ERROR: only errors
        optuna.logging.set_verbosity(optuna.logging.ERROR)

        # check for differences with the old configuration file
        config_on_disk_filepath = f"{filepaths.circuit}/{self._dab_config.circuit_study_name}/{self._dab_config.circuit_study_name}.pkl"
        if os.path.exists(config_on_disk_filepath):
            config_on_disk = CircuitOptimization.load_stored_config(self._dab_config.project_directory, self._dab_config.circuit_study_name)
            difference = deepdiff.DeepDiff(config_on_disk, self._dab_config, ignore_order=True, significant_digits=10)
            if difference:
                raise ValueError("Configuration file has changed from previous simulation.\n"
                                 f"Program is terminated.\n"
                                 f"Difference: {difference}")

        directions = ['maximize', 'minimize']

        # Calculate the fixed parameters
        self._fixed_parameters = CircuitOptimization.calculate_fixed_parameters(self._dab_config)

        # Update statistical data
        with self._c_lock_stat:
            self._progress_data.run_time = self._progress_run_time.get_runtime()
            self._progress_run_time.reset_start_trigger()
            self._progress_data.progress_status = ProgressStatus.InProgress

        # introduce study in storage, e.g. sqlite or mysql
        if database_type == 'sqlite':
            # Note: for sqlite operation, there needs to be three slashes '///' even before the path '/home/...'
            # Means, in total there are four slashes including the path itself '////home/.../database.sqlite3'
            storage = f"sqlite:///{circuit_study_sqlite_database}"

            # Create study object in drive
            self._study_in_storage = optuna.create_study(study_name=dab_config.circuit_study_name,
                                                         storage=storage,
                                                         directions=directions,
                                                         load_if_exists=True, sampler=sampler)

            # Create study object in memory
            self._study_in_memory = optuna.create_study(study_name=dab_config.circuit_study_name, directions=directions, sampler=sampler)
            # If trials exists, add them to study_in_memory
            self._study_in_memory.add_trials(self._study_in_storage.trials)
            # Inform about sampler type
            logger.info(f"Sampler is {self._study_in_storage.sampler.__class__.__name__}")
            # actual number of trials
            overtaken_no_trials = len(self._study_in_memory.trials)
            # Start optimization
            self.run_optimization_sqlite(number_trials)
            # Store memory to storage
            self._study_in_storage.add_trials(self._study_in_memory.trials[-number_trials:])
            logger.info(f"Add {number_trials} new calculated trials to existing {overtaken_no_trials} trials ="
                        f"{len(self._study_in_memory.trials)} trials.")
            logger.info(f"current time: {datetime.datetime.now()}")
            self.save_config()

            self.save_study_results_pareto(show_results=False)

        elif database_type == 'mysql':

            # connection to MySQL-database
            storage_url = "mysql+pymysql://oaml_optuna:optuna@localhost/optuna_db"

            # Create storage-object for Optuna on drive (Later RAMDISK)
            storage_mysql = optuna.storages.RDBStorage(storage_url)
            # storage = "mysql://oaml_optuna:optuna@localhost/optuna_db"

            # Create study object in drive
            self._study_in_storage = optuna.create_study(study_name=dab_config.circuit_study_name,
                                                         storage=storage_mysql,
                                                         directions=directions,
                                                         load_if_exists=True, sampler=sampler)

            # Inform about sampler type
            logger.info(f"Sampler is {self._study_in_storage.sampler.__class__.__name__}")
            # Start optimization
            self.run_optimization_mysql(storage_url, number_trials)

            # Stop time measurement
            self._progress_run_time.stop_trigger()

        # Set flag _is_study_available to indicate, that the study is available for filtering
        self._is_study_available = True

        # Parallelization Test with mysql
        # Number of processes
        #   num_processes = 1
        # Process list
        #    processes = []
        # Loop to start the processes
        #   for proc in range(num_processes):
        #       logger.info(f"Process {proc} started")
        #       p = multiprocessing.Process(target=Optimization.run_optimization,
        #                                   args=(storage_url, dab_config.circuit_study_name,
        #                                         number_trials, dab_config,fixed_parameters
        #                                        )
        #                                  )
        #       p.start()
        #       processes.append(p)

        # Wait for joining
        #   for proc in processes:
        # wait until each process is joined
        #       p.join()
        #       logger.info(f"Process {proc} joins")

    def save_study_results_pareto(self, show_results: bool = False) -> None:
        """Show the results of a study.

        A local .html file is generated under config.working_directory to store the interactive plotly plots on disk.

        :param show_results: True to directly open the browser to show the study results.
        :type show_results: bool
        """
        if self._study_in_storage is None:
            logger.warning("The study is not initialized!")
            return
        elif self._dab_config is None:
            logger.warning("Circuit configuration is not initialized!")
            return

        filepaths = CircuitOptimization.load_filepaths(self._dab_config.project_directory)

        fig = optuna.visualization.plot_pareto_front(self._study_in_storage, target_names=["ZVS coverage / %", r"i_\mathrm{cost}"])
        fig.update_layout(title=f"{self._dab_config.circuit_study_name} <br><sup>{self._dab_config.project_directory}</sup>")
        fig.write_html(
            f"{filepaths.circuit}/{self._dab_config.circuit_study_name}/{self._dab_config.circuit_study_name}"
            f"_{datetime.datetime.now().isoformat(timespec='minutes')}.html")
        if show_results:
            fig.show()

    @staticmethod
    def load_dab_dto_from_study(dab_config: circuit_dtos.CircuitParetoDabDesign, trial_number: int | None = None) -> dct.CircuitDabDTO:
        """
        Load a DAB-DTO from an optuna study.

        :param dab_config: DAB optimization configuration file
        :type dab_config: p_dtos.CircuitParetoDabDesign
        :param trial_number: trial number to load to the DTO
        :type trial_number: int
        :return:
        """
        if trial_number is None:
            raise NotImplementedError("needs to be implemented")

        filepaths = CircuitOptimization.load_filepaths(dab_config.project_directory)
        database_url = CircuitOptimization.create_sqlite_database_url(dab_config)

        loaded_study = optuna.create_study(study_name=dab_config.circuit_study_name,
                                           storage=database_url, load_if_exists=True)
        logger.info(f"The study '{dab_config.circuit_study_name}' contains {len(loaded_study.trials)} trials.")
        trials_dict = loaded_study.trials[trial_number].params

        fix_parameters = CircuitOptimization.calculate_fixed_parameters(dab_config)

        dab_dto = d_sets.HandleDabDto.init_config(
            name=str(trial_number),
            mesh_v1=fix_parameters.mesh_v1,
            mesh_v2=fix_parameters.mesh_v2,
            mesh_p=fix_parameters.mesh_p,
            sampling=dab_config.sampling,
            n=trials_dict["n_suggest"],
            ls=trials_dict["l_s_suggest"],
            fs=trials_dict["f_s_suggest"],
            lc1=trials_dict["l_1_suggest"],
            lc2=trials_dict["l_2__suggest"] / trials_dict["n_suggest"] ** 2,
            c_par_1=dab_config.design_space.c_par_1,
            c_par_2=dab_config.design_space.c_par_2,
            transistor_dto_1=trials_dict["transistor_1_name_suggest"],
            transistor_dto_2=trials_dict["transistor_2_name_suggest"]
        )

        return dab_dto

    def df_to_dab_dto_list(self, df: pd.DataFrame) -> list[d_dtos.CircuitDabDTO]:
        """
        Load a DAB-DTO from an optuna study.

        :param df: Pandas DataFrame to convert to the DAB-DTO list
        :type df: pd.DataFrame
        :return: List of DTO
        :rtype:  list[d_dtos.CircuitDabDTO]
        """
        dab_dto_list: list[d_dtos.CircuitDabDTO] = []

        # Check if configuration is not available or fixed parameters are not available
        if self._dab_config is None:
            logger.warning("Circuit configuration is not initialized!")
            return dab_dto_list
        elif self._fixed_parameters is None:
            logger.warning("Missing initialized fixed parameters!")
            return dab_dto_list

        logger.info(f"The study '{self._dab_config.circuit_study_name}' contains {len(df)} trials.")

        for index, _ in df.iterrows():
            transistor_dto_1 = d_sets.HandleTransistorDto.tdb_to_transistor_dto(df["params_transistor_1_name_suggest"][index])
            transistor_dto_2 = d_sets.HandleTransistorDto.tdb_to_transistor_dto(df["params_transistor_2_name_suggest"][index])

            dab_dto = d_sets.HandleDabDto.init_config(
                name=str(df["number"][index].item()),
                mesh_v1=self._fixed_parameters.mesh_v1,
                mesh_v2=self._fixed_parameters.mesh_v2,
                mesh_p=self._fixed_parameters.mesh_p,
                sampling=self._dab_config.sampling,
                n=df["params_n_suggest"][index].item(),
                ls=df["params_l_s_suggest"][index].item(),
                fs=df["params_f_s_suggest"][index].item(),
                lc1=df["params_l_1_suggest"][index].item(),
                lc2=df["params_l_2__suggest"][index].item() / df["params_n_suggest"][index].item() ** 2,
                c_par_1=self._dab_config.design_space.c_par_1,
                c_par_2=self._dab_config.design_space.c_par_2,
                transistor_dto_1=transistor_dto_1,
                transistor_dto_2=transistor_dto_2
            )
            dab_dto_list.append(dab_dto)

        return dab_dto_list

    @staticmethod
    def study_to_df(dab_config: circuit_dtos.CircuitParetoDabDesign) -> pd.DataFrame:
        """Create a DataFrame from a study.

        :param dab_config: DAB optimization configuration file
        :type dab_config: p_dtos.CircuitParetoDabDesign
        """
        filepaths = CircuitOptimization.load_filepaths(dab_config.project_directory)
        database_url = CircuitOptimization.create_sqlite_database_url(dab_config)
        loaded_study = optuna.create_study(study_name=dab_config.circuit_study_name, storage=database_url, load_if_exists=True)
        df = loaded_study.trials_dataframe()
        df.to_csv(f'{filepaths.circuit}/{dab_config.circuit_study_name}/{dab_config.circuit_study_name}.csv')
        return df

    @staticmethod
    def create_sqlite_database_url(dab_config: circuit_dtos.CircuitParetoDabDesign) -> str:
        """
        Create the DAB circuit optimization sqlite URL.

        :param dab_config: DAB optimization configuration file
        :type dab_config: p_dtos.CircuitParetoDabDesign
        :return: SQLite URL
        :rtype: str
        """
        filepaths = CircuitOptimization.load_filepaths(dab_config.project_directory)
        sqlite_storage_url = f"sqlite:///{filepaths.circuit}/{dab_config.circuit_study_name}/{dab_config.circuit_study_name}.sqlite3"
        return sqlite_storage_url

    @staticmethod
    def df_plot_pareto_front(df: pd.DataFrame, figure_size: tuple) -> None:
        """Plot an interactive Pareto diagram (losses vs. volume) to select the transformers to re-simulate.

        :param df: DataFrame, generated from an optuna study (exported by optuna)
        :type df: pd.DataFrame
        :param figure_size: figure size as x,y-tuple in mm, e.g. (160, 80)
        :type figure_size: tuple
        """
        logger.info(df.head())

        names = df["number"].to_numpy()
        # plt.figure()
        fig, ax = plt.subplots(figsize=[x / 25.4 for x in figure_size] if figure_size is not None else None, dpi=80)
        sc = plt.scatter(df["values_0"], df["values_1"], s=10)

        annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        def update_annot(ind):
            pos = sc.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            text = f"{[names[n] for n in ind['ind']]}"
            annot.set_text(text)
            annot.get_bbox_patch().set_alpha(0.4)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)

        plt.xlabel(r'ZVS coverage in \%')
        plt.ylabel(r'$i_\mathrm{HF,1}^2 + i_\mathrm{HF,2}^2$ in A')
        plt.grid()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def load_csv_to_df(csv_filepath: str) -> pd.DataFrame:
        """
        Load a csv file (previously stored from a Pandas DataFrame) back to a Pandas DataFrame.

        :param csv_filepath: File path of .csv file
        :type csv_filepath: str
        :return: loaded results from the given .csv file
        :rtype: pandas.DataFrame
        """
        df = pd.read_csv(csv_filepath, header=0, index_col=0)
        # reading a pandas DataFrame seems to change a global variable in the c subsystem
        # after reading csv values, there are issues running onelab/gmsh, as gmsh writes ',' instead '.' to its own files
        # reading the file again with setting back the delimiter to ';', is a workaround for the mentioned problem.
        pd.read_csv(csv_filepath, header=0, index_col=0, delimiter=';')
        return df

    @staticmethod
    def is_pareto_efficient(costs: np.ndarray, return_mask: bool = True) -> np.ndarray:
        """
        Find the pareto-efficient points.

        :param costs: An (n_points, n_costs) array
        :type costs: np.array
        :param return_mask: True to return a mask
        :type return_mask: bool
        :return: An array of indices of pareto-efficient points.
            If return_mask is True, this will be an (n_points, ) boolean array
            Otherwise it will be a (n_efficient_points, ) integer array of indices.
        :rtype: np.array
        """
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index < len(costs):
            non_dominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
            non_dominated_point_mask[next_point_index] = True  # type: ignore
            is_efficient = is_efficient[non_dominated_point_mask]  # Remove dominated points
            costs = costs[non_dominated_point_mask]
            next_point_index = int(np.sum(non_dominated_point_mask[:next_point_index])) + 1  # type: ignore
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype=bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient

    @staticmethod
    def pareto_front_from_df(df: pd.DataFrame, x: str = "values_0", y: str = "values_1") -> pd.DataFrame:
        """
        Calculate the Pareto front from a Pandas DataFrame. Return a Pandas DataFrame.

        :param df: Pandas DataFrame
        :type df: pd.DataFrame
        :param x: Name of x-parameter from df to show in Pareto plane
        :type x: str
        :param y: Name of y-parameter from df to show in Pareto plane
        :type y: str
        :return: Pandas DataFrame with pareto efficient points
        :rtype: pd.DataFrame
        """
        x_vec = df[x][~np.isnan(df[x])]
        y_vec = df[y][~np.isnan(df[x])]
        numpy_zip = np.column_stack((x_vec, y_vec))
        pareto_tuple_mask_vec = CircuitOptimization.is_pareto_efficient(numpy_zip)
        pareto_df = df[~np.isnan(df[x])][pareto_tuple_mask_vec]
        return pareto_df

    @staticmethod
    def filter_df(df: pd.DataFrame, x: str = "values_0", y: str = "values_1", factor_min_dc_losses: float = 1.2,
                  factor_max_dc_losses: float = 10) -> pd.DataFrame:
        """
        Remove designs with too high losses compared to the minimum losses.

        :param df: pandas DataFrame with study results
        :type df: pd.DataFrame
        :param x: x-value name for Pareto plot filtering
        :type x: str
        :param y: y-value name for Pareto plot filtering
        :type y: str
        :param factor_min_dc_losses: filter factor for the minimum dc losses
        :type factor_min_dc_losses: float
        :param factor_max_dc_losses: dc_max_loss = factor_max_dc_losses * min_available_dc_losses_in_pareto_front
        :type factor_max_dc_losses: float
        :returns: pandas DataFrame with Pareto front near points
        :rtype: pd.DataFrame
        """
        # figure out pareto front
        # pareto_volume_list, pareto_core_hyst_list, pareto_dto_list = self.pareto_front(volume_list, core_hyst_loss_list, valid_design_list)

        pareto_df: pd.DataFrame = CircuitOptimization.pareto_front_from_df(df, x, y)

        vector_to_sort = np.array([pareto_df[x], pareto_df[y]])

        # sorting 2d array by 1st row
        # https://stackoverflow.com/questions/49374253/sort-a-numpy-2d-array-by-1st-row-maintaining-columns
        sorted_vector = vector_to_sort[:, vector_to_sort[0].argsort()]
        x_pareto_vec = sorted_vector[0]
        y_pareto_vec = sorted_vector[1]

        total_losses_list = df[y][~np.isnan(df[y])].to_numpy()

        min_total_dc_losses = total_losses_list[np.argmin(total_losses_list)]
        loss_offset = factor_min_dc_losses * min_total_dc_losses

        ref_loss_max = np.interp(df[x], x_pareto_vec, y_pareto_vec) + loss_offset
        # clip losses to a maximum of the minimum losses
        ref_loss_max = np.clip(ref_loss_max, a_min=-1, a_max=factor_max_dc_losses * min_total_dc_losses)

        pareto_df_offset = df[df[y] < ref_loss_max]

        return pareto_df_offset

    def filter_study_results(self) -> None:
        """Filter the study result and use GeckoCIRCUITS for detailed calculation."""
        # Check if configuration is not available and if study is not available
        if self._dab_config is None:
            logger.warning("Circuit configuration is not loaded!")
            return
        elif not self._is_study_available:
            logger.warning("Study is not calculated. First run 'start_proceed_study'!")
            return
        elif self._study_in_storage is None:
            logger.warning("Study is not calculated. First run 'start_proceed_study'!")
            return

        filepaths = CircuitOptimization.load_filepaths(self._dab_config.project_directory)

        df = self._study_in_storage.trials_dataframe()
        df.to_csv(f'{filepaths.circuit}/{self._dab_config.circuit_study_name}/{self._dab_config.circuit_study_name}.csv')

        df = df[df["values_0"] == 100]

        smallest_dto_list: list[d_dtos.CircuitDabDTO] = []
        df_smallest_all = df.nsmallest(n=1, columns=["values_1"])
        df_smallest = df.nsmallest(n=1, columns=["values_1"])

        smallest_dto_list.append(self.df_to_dab_dto_list(df_smallest)[0])

        # Continue time measurement
        with self._c_lock_stat:
            self._progress_run_time.continue_trigger()

        for count in np.arange(0, self._dab_config.filter.number_filtered_designs - 1):
            logger.info("------------------")
            logger.info(f"{count=}")
            n_suggest = df_smallest['params_n_suggest'].item()
            f_s_suggest = df_smallest['params_f_s_suggest'].item()
            l_s_suggest = df_smallest['params_l_s_suggest'].item()
            l_1_suggest = df_smallest['params_l_1_suggest'].item()
            l_2__suggest = df_smallest['params_l_2__suggest'].item()
            transistor_1_name_suggest = df_smallest['params_transistor_1_name_suggest'].item()
            transistor_2_name_suggest = df_smallest['params_transistor_2_name_suggest'].item()

            # make sure to use parameters with minimum x % difference.
            difference = self._dab_config.filter.difference_percentage / 100

            df = df.loc[
                ~((df["params_n_suggest"].ge(n_suggest * (1 - difference)) & df["params_n_suggest"].le(n_suggest * (1 + difference))) & \
                  (df["params_f_s_suggest"].ge(f_s_suggest * (1 - difference)) & df["params_f_s_suggest"].le(f_s_suggest * (1 + difference))) & \
                  (df["params_l_s_suggest"].ge(l_s_suggest * (1 - difference)) & df["params_l_s_suggest"].le(l_s_suggest * (1 + difference))) & \
                  (df["params_l_1_suggest"].ge(l_1_suggest * (1 - difference)) & df["params_l_1_suggest"].le(l_1_suggest * (1 + difference))) & \
                  (df["params_l_2__suggest"].ge(l_2__suggest * (1 - difference)) & df["params_l_2__suggest"].le(l_2__suggest * (1 + difference))) & \
                  df["params_transistor_1_name_suggest"].isin([transistor_1_name_suggest]) & \
                  df["params_transistor_2_name_suggest"].isin([transistor_2_name_suggest])
                  )]

            df_smallest = df.nsmallest(n=1, columns=["values_1"])
            df_smallest_all = pd.concat([df_smallest_all, df_smallest], axis=0)

        smallest_dto_list = self.df_to_dab_dto_list(df_smallest_all)

        # join if necessary
        folders = CircuitOptimization.load_filepaths(self._dab_config.project_directory)

        dto_directory = os.path.join(folders.circuit, self._dab_config.circuit_study_name, "filtered_results")
        os.makedirs(dto_directory, exist_ok=True)
        for dto in smallest_dto_list:
            # dto = dct.HandleDabDto.add_gecko_simulation_results(dto, get_waveforms=True)
            dct.HandleDabDto.save(dto, dto.name, directory=dto_directory, timestamp=False)

        # Stop runtime measurement and update statistical data
        with self._c_lock_stat:
            self._progress_run_time.stop_trigger()
            self._progress_data.run_time = self._progress_run_time.get_runtime()
            self._progress_data.progress_status = ProgressStatus.Done

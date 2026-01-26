"""Pareto optimization classes and functions."""
# Python libraries
import os
import logging
import pickle
import datetime
import threading
import copy
from typing import Any

# 3rd party libraries
import optuna
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import deepdiff
import dct.sampling as sampling
from sklearn.cluster import KMeans
from scipy.signal import savgol_filter
from typing import cast, SupportsFloat

from dct import TomlHeatSink
# own libraries
from dct.topology.sbc import sbc_datasets_dtos as s_dtos
from dct.topology.sbc import sbc_circuit_topology_dtos as circuit_dtos
from dct.topology.sbc import sbc_datasets as d_sets
import transistordatabase as tdb
from dct.boundary_check import CheckCondition as c_flag
from dct.boundary_check import BoundaryCheck
from dct.topology.sbc import sbc_toml_checker as sbc_tc
from dct.datasets_dtos import StudyData, FilterData
from dct.server_ctl_dtos import ProgressData
from dct.server_ctl_dtos import ProgressStatus
from dct.server_ctl_dtos import RunTimeMeasurement as RunTime
from dct.circuit_enums import SamplingEnum
from dct.topology.circuit_optimization_base import CircuitOptimizationBase
from dct.datasets_dtos import PlotData
import dct.generalplotsettings as gps
from dct.components.component_dtos import ComponentRequirements
from dct.components.component_dtos import CapacitorRequirements, InductorRequirements, TransformerRequirements

logger = logging.getLogger(__name__)

class SbcCircuitOptimization(CircuitOptimizationBase[sbc_tc.TomlSbcGeneral, sbc_tc.TomlSbcCircuitParetoDesign]):
    """Optimize the SBC converter regarding maximum ZVS coverage and minimum conduction losses."""

    # Definition of constant values
    # Topological resource constants
    _number_of_required_capacitors: int = 0
    _number_of_required_inductors: int = 1
    _number_of_required_transformers: int = 0

    # Declaration of member types
    _c_lock_stat: threading.Lock
    _progress_data: ProgressData
    _progress_run_time: RunTime
    _sbc_config: circuit_dtos.CircuitParetoSbcDesign | None
    _study_in_memory: optuna.Study | None
    _study_in_storage: optuna.Study | None
    _fixed_parameters: s_dtos.FixedParameters | None

    def __init__(self) -> None:
        """Initialize the configuration list for the circuit optimizations."""
        # Call the constructor of the base class
        super().__init__()
        # Variable allocation
        self._c_lock_stat = threading.Lock()
        # Initialize the statistical data (For more configuration it needs to become instance instead of static
        self._progress_data = ProgressData(run_time=0, number_of_filtered_points=0,
                                           progress_status=ProgressStatus.Idle)
        self._progress_run_time = RunTime()
        self._sbc_config = None
        self._is_study_available = False

        self._study_in_memory = None
        self._study_in_storage = None
        self._fixed_parameters = None

        # General optimization parameter
        self._toml_general: sbc_tc.TomlSbcGeneral | None = None
        # Circuit optimization parameter
        self._toml_circuit: sbc_tc.TomlSbcCircuitParetoDesign | None = None

    def save_config(self) -> None:
        """Save the actual configuration file as pickle file on the disk."""
        # Check if a configuration is loaded
        if self._sbc_config is None:
            logger.warning("Circuit configuration is empty!\n    Configuration is not saved!")
            return

        os.makedirs(self.circuit_study_data.optimization_directory, exist_ok=True)
        with open(f"{self.circuit_study_data.optimization_directory}/{self._sbc_config.circuit_study_name}.pkl", 'wb') as output:
            pickle.dump(self._sbc_config, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_stored_config(act_circuit_study_data: StudyData) -> circuit_dtos.CircuitParetoSbcDesign:
        """
        Load pickle configuration file from disk.

        :param act_circuit_study_data: Information about the circuit study name and study path
        :type  act_circuit_study_data: StudyData
        :return: Configuration file as circuit_dtos.SbcDesign
        :rtype: circuit_dtos.CircuitParetoSbcDesign
        """
        config_pickle_filepath = os.path.join(act_circuit_study_data.optimization_directory,
                                              f"{act_circuit_study_data.study_name}.pkl")

        with open(config_pickle_filepath, 'rb') as pickle_file_data:
            loaded_pareto_dto = pickle.load(pickle_file_data)
            if not isinstance(loaded_pareto_dto, circuit_dtos.CircuitParetoSbcDesign):
                raise TypeError(f"Loaded pickle file {loaded_pareto_dto} not of type CircuitParetoSbcDesign.")

        return loaded_pareto_dto

    def load_and_verify_general_parameters(self, toml_dict: dict[str, Any]) -> tuple[bool, str]:
        """Verify the input parameter ranges.

        :param toml_dict: toml general configuration
        :type toml_dict: dict[str, Any]
        :return: True, if the configuration was consistent and empty string or False and report of the issues
        :rtype: tuple[bool, str]
        """
        # Variable declaration
        inconsistency_report: str = ""
        is_consistent: bool = True

        # Convert and check toml file content
        toml_general: sbc_tc.TomlSbcGeneral = sbc_tc.TomlSbcGeneral(**toml_dict)

        # Output range parameter and sampling parameter check
        group_name = "parameter_range or sampling"
        # Init is_user_point_list_consistent-flag
        is_user_point_list_consistent = False
        # Evaluate list length
        len_additional_user_v1 = len(toml_general.sampling.v1_additional_user_point_list)
        len_additional_user_v2 = len(toml_general.sampling.v2_additional_user_point_list)
        len_additional_user_i2 = len(toml_general.sampling.i2_additional_user_point_list)
        len_additional_user_w = len(toml_general.sampling.additional_user_weighting_point_list)
        len_check1 = len_additional_user_v1 == len_additional_user_v2 and len_additional_user_i2 == len_additional_user_w
        len_check2 = len_additional_user_i2 == len_additional_user_w
        # Check if the additional user point lists are consistent
        if len_check1 and len_check2:
            is_user_point_list_consistent = True
        else:
            act_report = f"    The number of list entries in v1_additional_user_point_list ({len_additional_user_v1}), "
            act_report = act_report + f"v2_additional_user_point_list ({len_additional_user_v2}),\n"
            act_report = act_report + f"    i2_additional_user_point_list ({len_additional_user_i2}) and "
            act_report = act_report + f"additional_user_weighting_point_list ({len_additional_user_w} "
            act_report = act_report + "needs to be the same!\n)"
            inconsistency_report = (inconsistency_report + act_report)
            is_consistent = False

        # Perform the boundary check of v1_min_max_list
        is_check_passed, issue_report = BoundaryCheck.check_float_min_max_values(
            0, 1000, toml_general.parameter_range.v1_min_max_list, "parameter_range: v1_min_max_list", c_flag.check_inclusive,
            c_flag.check_inclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False
        elif is_user_point_list_consistent:
            for power_value in toml_general.sampling.v1_additional_user_point_list:
                is_check_passed, issue_report = BoundaryCheck.check_float_value(
                    toml_general.parameter_range.v1_min_max_list[0], toml_general.parameter_range.v1_min_max_list[1],
                    power_value,
                    "sampling: v1_additional_user_point_list", c_flag.check_inclusive, c_flag.check_inclusive)
                if not is_check_passed:
                    inconsistency_report = inconsistency_report + issue_report
                    is_consistent = False

        # Perform the boundary check  of v2_min_max_list
        is_check_passed, issue_report = BoundaryCheck.check_float_min_max_values(
            0, 1000, toml_general.parameter_range.v2_min_max_list, "parameter_range: v2_min_max_list", c_flag.check_inclusive,
            c_flag.check_inclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False
        else:
            # Check lower boundary with respect to v1_min_max_list if v2_min_max_list is consistent
            if toml_general.parameter_range.v1_min_max_list[0] <= toml_general.parameter_range.v2_min_max_list[0]:
                issue_report = (f"Inconsistency in v2_min_max_list: Minimum of v1="
                                f"{toml_general.parameter_range.v1_min_max_list[0]} is less equal minimum "
                                f"of v2={toml_general.parameter_range.v1_min_max_list[0]}.")
                inconsistency_report = inconsistency_report + issue_report
                is_check_passed = False
            else:
                is_check_passed = True
            # Check upper boundary with respect to v1_min_max_list
            if toml_general.parameter_range.v1_min_max_list[1] <= toml_general.parameter_range.v2_min_max_list[1]:
                issue_report = (f"Inconsistency in v2_min_max_list: Maximum of v1="
                                f"{toml_general.parameter_range.v1_min_max_list[1]} is less equal maximum "
                                f"of v2={toml_general.parameter_range.v1_min_max_list[1]}.")
                inconsistency_report = inconsistency_report + issue_report
                is_check_passed = False
            if not is_check_passed:
                is_consistent = False
            elif is_user_point_list_consistent:
                for index, voltage_value in enumerate(toml_general.sampling.v2_additional_user_point_list):
                    is_check_passed, issue_report = BoundaryCheck.check_float_value(
                        toml_general.parameter_range.v2_min_max_list[0],
                        toml_general.parameter_range.v2_min_max_list[1], voltage_value,
                        "sampling: v2_additional_user_point_list", c_flag.check_inclusive, c_flag.check_inclusive)
                    if not is_check_passed:
                        inconsistency_report = inconsistency_report + issue_report
                        is_consistent = False
                    if voltage_value >= toml_general.sampling.v1_additional_user_point_list[index]:
                        issue_report = (f"Inconsistency in additional point list at index {index}."
                                        f"v1={toml_general.sampling.v2_additional_user_point_list[index]} "
                                        f"is less equal v2={voltage_value}.")
                        inconsistency_report = inconsistency_report + issue_report
                        is_consistent = False

        # Perform the boundary check  of i2_min_max_list
        is_check_passed, issue_report = BoundaryCheck.check_float_min_max_values(
            -300, 300, toml_general.parameter_range.i2_min_max_list, "parameter_range: i2_min_max_list", c_flag.check_inclusive,
            c_flag.check_inclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False
        elif is_user_point_list_consistent:
            for power_value in toml_general.sampling.i2_additional_user_point_list:
                is_check_passed, issue_report = BoundaryCheck.check_float_value(
                    toml_general.parameter_range.i2_min_max_list[0], toml_general.parameter_range.i2_min_max_list[1],
                    power_value,
                    "sampling: i2_additional_user_point_list", c_flag.check_inclusive, c_flag.check_inclusive)
                if not is_check_passed:
                    inconsistency_report = inconsistency_report + issue_report
                    is_consistent = False

        # Remaining Sampling parameter check
        group_name = "sampling"
        # Check additional_user_weighting_point_list
        # Initialize variable
        weighting_sum: float = 0.0
        # Perform the boundary check  of additional_user_weighting_point_list
        for weight_value in toml_general.sampling.additional_user_weighting_point_list:
            is_check_passed, issue_report = BoundaryCheck.check_float_value(
                0, 1, weight_value, "additional_user_weighting_point_list", c_flag.check_inclusive,
                c_flag.check_inclusive)
            if not is_check_passed:
                inconsistency_report = inconsistency_report + issue_report
                is_consistent = False

            weighting_sum = weighting_sum + weight_value

        # Check the sum
        if weighting_sum > 1:
            is_consistent = False
            act_report = "    The sum of parameter entries of parameter additional_user_weighting_point_list "
            act_report = act_report + f"{weighting_sum} has to be less equal 1!\n"
            inconsistency_report = inconsistency_report + act_report

        # Perform the boundary check for sampling points
        is_check_passed, issue_report = BoundaryCheck.check_float_value(
            0, 1, float(toml_general.sampling.sampling_points),
            f"{group_name}: sampling_points", c_flag.check_exclusive, c_flag.check_ignore)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Check sampling random seed
        # Perform the boundary check for number_filtered_designs
        is_check_passed, issue_report = BoundaryCheck.check_float_value(
            0, 1, float(toml_general.sampling.sampling_random_seed),
            f"{group_name}: sampling_random_seed", c_flag.check_inclusive, c_flag.check_ignore)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False
            # delete old parameter
            self._toml_general = None
        else:
            # Overtake the parameter
            self._toml_general = toml_general

        return is_consistent, inconsistency_report

    def load_and_verify_circuit_parameters(self, toml_dict: dict[str, Any], is_tdb_to_update: bool = False) -> tuple[bool, str]:
        """Load and verify the circuit input parameter.

        :param toml_dict: dictionary with circuit configuration
        :type  toml_dict: dict[str, Any]
        :param is_tdb_to_update: True to update the transistor database
        :type is_tdb_to_update: bool
        :return: True, if the configuration was consistent and empty string or False and report of the issues
        :rtype: tuple[bool, str]
        """
        # Variable declaration
        inconsistency_report: str = ""
        is_consistent: bool = True
        toml_check_keyword_list: list[tuple[list[str], str]]
        toml_check_value_list: list[tuple[float, str]]

        toml_circuit: sbc_tc.TomlSbcCircuitParetoDesign = sbc_tc.TomlSbcCircuitParetoDesign(**toml_dict)

        # Design space parameter check
        # Create dictionary from transistor database list
        db = tdb.DatabaseManager()
        db.set_operation_mode_json()
        if not is_tdb_to_update:
            # Debug ASA comment out
            # db.update_from_file_exchange(True)
            pass

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
                is_consistent = False
            else:
                # Perform dictionary check
                for keyword_entry in check_keyword[0]:
                    is_check_passed, issue_report = BoundaryCheck.check_dictionary(keyword_dictionary, keyword_entry, check_keyword[1])
                    # Check if boundary check fails
                    if not is_check_passed:
                        inconsistency_report = inconsistency_report + issue_report
                        is_consistent = False

        # Check switching frequency range
        float_f_s_min_max_list = BoundaryCheck.convert_int_list_to_float_list(toml_circuit.design_space.f_s_min_max_list)
        is_check_passed, issue_report = BoundaryCheck.check_float_min_max_values(
            1000, 1e7, float_f_s_min_max_list, "f_s_min_max_list", c_flag.check_exclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Check l_s_min_max_list
        is_check_passed, issue_report = BoundaryCheck.check_float_min_max_values(
            0, 1, toml_circuit.design_space.l_s_min_max_list,
            "parameter_range: l_s_min_max_list", c_flag.check_exclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Check l_s_min_max_list
        is_check_passed, issue_report = BoundaryCheck.check_float_min_max_values(
            0, 1, toml_circuit.design_space.l_s_min_max_list,
            "l_s_min_max_list", c_flag.check_inclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Check c_par_1 and c_par_2
        toml_check_value_list = (
            [(toml_circuit.design_space.c_par_1, "c_par_1"),
             (toml_circuit.design_space.c_par_2, "c_par_2")])

        # Perform the boundary check
        # Check c_par_1 and c_par_2
        is_check_passed, issue_report = BoundaryCheck.check_float_value_list(
            0, 1e-3, toml_check_value_list, c_flag.check_exclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Perform filter_distance value check
        group_name = "filter_distance"
        # Perform the boundary check for number_filtered_designs
        is_check_passed, issue_report = BoundaryCheck.check_float_value(
            0, 100, float(toml_circuit.filter_distance.number_filtered_designs),
            f"{group_name}: number_filtered_designs", c_flag.check_exclusive, c_flag.check_ignore)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Perform the boundary check for number_filtered_designs
        is_check_passed, issue_report = BoundaryCheck.check_float_value(
            0.01, 100, toml_circuit.filter_distance.difference_percentage,
            f"{group_name}: difference_percentage", c_flag.check_exclusive, c_flag.check_inclusive)

        # Perform thermal resistance data check
        group_name = "thermal_data"
        # Create the list
        toml_check_value_list1: list[tuple[float, str]] = []
        toml_check_value_list2: list[tuple[float, str]] = []

        # Perform list length check for transistor_b1_cooling
        if len(toml_circuit.thermal_data.transistor_b1_cooling) != 2:
            inconsistency_report = inconsistency_report + "    Number of values in parameter 'transistor_b1_cooling' is not equal 2!\n"
            is_consistent = False
        else:
            toml_check_value_list1.append(
                (toml_circuit.thermal_data.transistor_b1_cooling[0], f"{group_name}: transistor_b1_cooling[0]-tim_thickness"))
            toml_check_value_list2.append(
                (toml_circuit.thermal_data.transistor_b1_cooling[1], f"{group_name}: transistor_b1_cooling[1]-tim_conductivity"))

        # Perform list length check for transistor_b1_cooling
        if len(toml_circuit.thermal_data.transistor_b1_cooling) != 2:
            inconsistency_report = inconsistency_report + "    Number of values in parameter 'transistor_b2_cooling' is not equal 2!\n"
            is_consistent = False
        else:
            toml_check_value_list1.append(
                (toml_circuit.thermal_data.transistor_b1_cooling[0], f"{group_name}: transistor_b2_cooling[0]-tim_thickness"))
            toml_check_value_list2.append(
                (toml_circuit.thermal_data.transistor_b1_cooling[1], f"{group_name}: transistor_b2_cooling[1]-tim_conductivity"))

        # Perform the boundary check for tim-thickness
        is_check_passed, issue_report = dct.BoundaryCheck.check_float_value_list(
            0, 0.01, toml_check_value_list1, c_flag.check_exclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        # Perform the boundary check for tim-conductivity
        is_check_passed, issue_report = dct.BoundaryCheck.check_float_value_list(
            1, 100, toml_check_value_list2, c_flag.check_exclusive, c_flag.check_exclusive)
        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False

        if not is_check_passed:
            inconsistency_report = inconsistency_report + issue_report
            is_consistent = False
            # delete old parameter
            self._toml_circuit = None
        else:
            # Overtake the parameter
            self._toml_circuit = toml_circuit

        return is_consistent, inconsistency_report

    @staticmethod
    def _is_optimization_skippable(study_data: StudyData, filter_data: FilterData) -> tuple[bool, str]:
        # Variable declaration
        # skippable flag
        is_skippable: bool = False
        # Report string
        issue_report: str = ""

        # Check, if data are available (skip case)
        # Check if filtered results folder exists
        if os.path.exists(filter_data.filtered_list_pathname):
            # Add filtered result list
            for filtered_circuit_result in os.listdir(filter_data.filtered_list_pathname):
                if os.path.isfile(os.path.join(filter_data.filtered_list_pathname, filtered_circuit_result)):
                    filter_data.filtered_list_files.append(os.path.splitext(filtered_circuit_result)[0])

            if not filter_data.filtered_list_files:
                issue_report = f"Filtered results folder {filter_data.filtered_list_pathname} is empty."
            else:
                is_skippable = True
        else:
            issue_report = f"Filtered circuit results folder {filter_data.filtered_list_pathname} does not exist."

        # Return evaluation result
        return is_skippable, issue_report

    def initialize_circuit_optimization(self) -> bool:
        """
        Initialize the circuit_dto for circuit optimization.

        :return: True, if the configuration was successful initialized
        :rtype: bool
        """
        # Check if toml_circuit, toml_general and study data are still initialized
        if self._toml_general is None or self._toml_circuit is None or self.circuit_study_data is None:
            # Serious programming error: In verification check these variables should be initialized
            # This has to be guaranteed by the workflow
            raise ValueError("Serious programming error 'sbc multiple allocation failure'. Please write an issue!")

        # Initialize the circuit_dtos
        design_space = circuit_dtos.CircuitParetoDesignSpace(
            f_s_min_max_list=self._toml_circuit.design_space.f_s_min_max_list,
            l_s_min_max_list=self._toml_circuit.design_space.l_s_min_max_list,
            transistor_1_name_list=self._toml_circuit.design_space.transistor_1_name_list,
            transistor_2_name_list=self._toml_circuit.design_space.transistor_2_name_list,
            c_par_1=self._toml_circuit.design_space.c_par_1,
            c_par_2=self._toml_circuit.design_space.c_par_2

        )

        parameter_range = circuit_dtos.CircuitParameterRange(
            v1_min_max_list=self._toml_general.parameter_range.v1_min_max_list,
            duty_cycle_min_max_list=SbcCircuitOptimization.calculate_voltage_to_duty_cycle_min_max_list(
                self._toml_general.parameter_range),
            i2_min_max_list=self._toml_general.parameter_range.i2_min_max_list)

        filter = circuit_dtos.CircuitFilter(
            number_filtered_designs=self._toml_circuit.filter_distance.number_filtered_designs,
            difference_percentage=self._toml_circuit.filter_distance.difference_percentage
        )

        # None can not be handled by toml correct, so this is a workaround. By default, "random" in toml equals "None"
        local_sampling_random_seed: int | None = None
        # In case of a concrete seed was given, overwrite None with the given one
        if isinstance(self._toml_general.sampling.sampling_random_seed, int):
            local_sampling_random_seed = int(self._toml_general.sampling.sampling_random_seed)

        sampling = circuit_dtos.CircuitSampling(
            sampling_method=self._toml_general.sampling.sampling_method,
            sampling_points=self._toml_general.sampling.sampling_points,
            sampling_random_seed=local_sampling_random_seed,
            v1_additional_user_point_list=self._toml_general.sampling.v1_additional_user_point_list,
            duty_cycle_additional_user_point_list=self._toml_general.sampling.v2_additional_user_point_list,
            i2_additional_user_point_list=self._toml_general.sampling.i2_additional_user_point_list,
            additional_user_weighting_point_list=self._toml_general.sampling.additional_user_weighting_point_list
        )

        self._sbc_config = circuit_dtos.CircuitParetoSbcDesign(
            circuit_study_name=self.circuit_study_data.study_name,
            project_directory=self.project_directory,
            design_space=design_space,
            parameter_range=parameter_range,
            sampling=sampling,
            filter=filter)

        return True

    @staticmethod
    def calculate_voltage_to_duty_cycle_min_max_list(act_parameter_range: sbc_tc.TomlSbcParameterRange) -> list[float]:
        """Calculate the duty cycle list based on v1 and v2 lists.

        :param act_parameter_range: Parameter range of voltages and current
        :type  act_parameter_range: sbc_tc.TomlSbcParameterRange
        :return: Duty cycle minimum-maximum list
        :rtype:  list[float]
        """
        # Variable declaration
        duty_cycle_min_max_list: list[float] = []
        # Minimum value calculated from maximum v1-voltage and minimum v2-voltage
        duty_cycle_min_max_list.append(act_parameter_range.v2_min_max_list[0] / act_parameter_range.v1_min_max_list[1])
        # Maximum value calculated from minimum v1-voltage and maximum v2-voltage
        duty_cycle_min_max_list.append(act_parameter_range.v2_min_max_list[1] / act_parameter_range.v1_min_max_list[0])

        duty_cycle_min_max_list.sort()
        if duty_cycle_min_max_list[1] >= 0.999:
            duty_cycle_min_max_list[1] = 0.999

        return duty_cycle_min_max_list

    def get_config(self) -> circuit_dtos.CircuitParetoSbcDesign | None:
        """
        Return the actual loaded configuration file.

        :return: Configuration file as circuit_dtos.SbcDesign
        :rtype: circuit_dtos.CircuitParetoSbcDesign
        """
        if self._sbc_config is None:
            logger.warning("Configuration is not loaded!")

        return copy.deepcopy(self._sbc_config)

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
    def _objective(trial: optuna.Trial, sbc_config: circuit_dtos.CircuitParetoSbcDesign, fixed_parameters: s_dtos.FixedParameters) -> tuple:
        """
        Objective function to optimize.

        :param sbc_config: Synchronous buck converter optimization configuration file
        :type sbc_config: circuit_dtos.CircuitParetoSbcDesign
        :param trial: optuna trial
        :type trial: optuna.Trial
        :param fixed_parameters: fixed parameters (loaded transistor DTOs)
        :type fixed_parameters: s_dtos.FixedParameters

        :return:
        """
        # Variable declaration
        transistor_1_dto: s_dtos.TransistorDTO | None = None

        # Get new suggestion from optimizer
        f_s_suggest = trial.suggest_int('f_s_suggest', sbc_config.design_space.f_s_min_max_list[0], sbc_config.design_space.f_s_min_max_list[1])
        l_s_suggest = trial.suggest_float('l_s_suggest', sbc_config.design_space.l_s_min_max_list[0], sbc_config.design_space.l_s_min_max_list[1])
        transistor_1_name_suggest = trial.suggest_categorical('transistor_1_name_suggest', sbc_config.design_space.transistor_1_name_list)

        # Copy transistor data based on suggested transistor name
        for transistor_dto in fixed_parameters.transistor_1_dto_list:
            if transistor_dto.name == transistor_1_name_suggest:
                transistor_1_dto = transistor_dto

        if transistor_1_dto is None:
            return float('nan'), float('nan')

        # Overtake suggested values
        sbc_calc = d_sets.HandleSbcDto.init_config(
            name=sbc_config.circuit_study_name,
            mesh_v=fixed_parameters.mesh_v,
            mesh_duty_cycle=fixed_parameters.mesh_duty_cycle,
            mesh_i=fixed_parameters.mesh_i,
            sampling=sbc_config.sampling,
            ls=l_s_suggest,
            fs=f_s_suggest,
            transistor_dto_1=transistor_1_dto,
            transistor_dto_2=transistor_1_dto
        )

        # 0 = ripple, 1=volume , 2= one function
        debug_selection = 1
        # False = switching and conduction loss, True = only switching loss
        debug_consider_only_switch_loss = False
        # Debug selection
        i_ripple_or_volume_cost_value: np.ndarray = np.array(0)
        # Debug i_ripple
        if debug_selection == 0:
            # Set the ripple current square value (ripple power) as sum of the weights
            i_ripple_or_volume_cost_value = fixed_parameters.mesh_weights.ravel() @ sbc_calc.calc_currents.i_ripple.ravel()

        # Debug volume proxy L
        if debug_selection == 1:
            # Set the inductor volume as sum of the weights
            i_ripple_or_volume_cost_value = fixed_parameters.mesh_weights.ravel() @ sbc_calc.calc_volume_inductor_proxy.ravel()

        # Calculate transistor losses
        # Debug Only switching losses ASA: Later to correct
        if debug_consider_only_switch_loss:
            i_loss_cost_value = fixed_parameters.mesh_weights.ravel() @ (sbc_calc.calc_losses.p_hs_switch + sbc_calc.calc_losses.p_ls_switch)
        else:
            i_loss_cost_value = fixed_parameters.mesh_weights.ravel() @ sbc_calc.calc_losses.p_sbc_total

        # Debug basic pareto behavior according fixed functions
        if debug_selection == 2:
            i_ripple_or_volume_cost_value = np.array(20+5e-5/l_s_suggest)
            # i_ripple_or_volume_cost_value = (l_s_suggest * fixed_parameters.mesh_weights.ravel())
            # i_loss_cost_value = 30+f_s_suggest/100000
            i_loss_cost_value = fixed_parameters.mesh_weights.ravel() @ sbc_calc.calc_losses.p_sbc_total

        return i_ripple_or_volume_cost_value, i_loss_cost_value

    @staticmethod
    def calculate_fixed_parameters(act_sbc_config: circuit_dtos.CircuitParetoSbcDesign) -> s_dtos.FixedParameters:
        """
        Calculate time-consuming parameters which are same for every single simulation.

        :param act_sbc_config: Sbc circuit configuration
        :type act_sbc_config: circuit_dtos.CircuitParetoSbcDesign
        :return: Fix parameters (transistor DTOs)
        :rtype: s_dtos.FixedParameters
        """
        transistor_1_dto_list = []
        transistor_2_dto_list = []

        for transistor in act_sbc_config.design_space.transistor_1_name_list:
            transistor_1_dto_list.append(d_sets.HandleTransistorDto.tdb_to_transistor_dto(transistor))

        for transistor in act_sbc_config.design_space.transistor_2_name_list:
            transistor_2_dto_list.append(d_sets.HandleTransistorDto.tdb_to_transistor_dto(transistor))

        # choose sampling method
        if act_sbc_config.sampling.sampling_method == SamplingEnum.meshgrid:
            steps_per_dimension = int(np.ceil(np.power(act_sbc_config.sampling.sampling_points, 1 / 3)))
            logger.info(f"Number of sampling points has been updated from {act_sbc_config.sampling.sampling_points} to {steps_per_dimension ** 3}.")
            logger.info("Note: meshgrid sampling does not take user-given operating points into account")
            v_operating_points, duty_cylce_operating_points, i_operating_points = np.meshgrid(
                np.linspace(act_sbc_config.parameter_range.v1_min_max_list[0], act_sbc_config.parameter_range.v1_min_max_list[1],
                            steps_per_dimension),
                np.linspace(act_sbc_config.parameter_range.duty_cycle_min_max_list[0],
                            act_sbc_config.parameter_range.duty_cycle_min_max_list[1], steps_per_dimension),
                np.linspace(act_sbc_config.parameter_range.i2_min_max_list[0], act_sbc_config.parameter_range.i2_min_max_list[1],
                            steps_per_dimension),
                sparse=False)
        elif act_sbc_config.sampling.sampling_method == SamplingEnum.latin_hypercube:
            v_operating_points, duty_cylce_operating_points, i_operating_points = sampling.latin_hypercube(
                act_sbc_config.parameter_range.v1_min_max_list[0], act_sbc_config.parameter_range.v1_min_max_list[1],
                act_sbc_config.parameter_range.duty_cycle_min_max_list[0], act_sbc_config.parameter_range.duty_cycle_min_max_list[1],
                act_sbc_config.parameter_range.i2_min_max_list[0], act_sbc_config.parameter_range.i2_min_max_list[1],
                total_number_points=act_sbc_config.sampling.sampling_points,
                dim_1_user_given_points_list=act_sbc_config.sampling.v1_additional_user_point_list,
                dim_2_user_given_points_list=act_sbc_config.sampling.duty_cycle_additional_user_point_list,
                dim_3_user_given_points_list=act_sbc_config.sampling.i2_additional_user_point_list,
                sampling_random_seed=act_sbc_config.sampling.sampling_random_seed)
        else:
            raise ValueError(f"sampling_method '{act_sbc_config.sampling.sampling_method}' not available.")

        logger.debug(f"{v_operating_points=}")

        # calculate weighting

        if act_sbc_config.sampling.sampling_method == SamplingEnum.meshgrid:
            weight_sum = 0
            given_user_points = 0
        else:
            weight_sum = np.sum(act_sbc_config.sampling.additional_user_weighting_point_list)
            logger.debug(f"{weight_sum=}")
            given_user_points = len(act_sbc_config.sampling.v1_additional_user_point_list)
        logger.debug(f"{given_user_points=}")
        logger.debug(f"{v_operating_points.size=}")

        if weight_sum > 1 or weight_sum < 0:
            raise ValueError("Sum of weighting point list must be within 0 and 1.")
        else:
            leftover_auto_weight = (1 - weight_sum) / (v_operating_points.size - given_user_points)
            logger.info(f"Auto-weight given for all other {v_operating_points.size - given_user_points} operating points: {leftover_auto_weight}")
            # default case, same weights for all points
            weights = np.full_like(v_operating_points, leftover_auto_weight)
            # for user point weightings, both lists must be filled.
            if act_sbc_config.sampling.additional_user_weighting_point_list and act_sbc_config.sampling.sampling_method != SamplingEnum.meshgrid:
                logger.debug("Given user weighting point list detected, fill up with user-given weights.")
                weights[-len(act_sbc_config.sampling.additional_user_weighting_point_list):] = act_sbc_config.sampling.additional_user_weighting_point_list
            logger.debug(f"{weights=}")
            logger.debug(f"Double check: Sum of weights = {np.sum(weights)}")

        return s_dtos.FixedParameters(
            transistor_1_dto_list=transistor_1_dto_list,
            transistor_2_dto_list=transistor_2_dto_list,
            mesh_v=np.atleast_3d(v_operating_points),
            mesh_duty_cycle=np.atleast_3d(duty_cylce_operating_points),
            mesh_i=np.atleast_3d(i_operating_points),
            mesh_weights=np.atleast_3d(weights)
        )

    def run_optimization_sqlite(self, act_number_trials: int) -> None:
        """Proceed a study which is stored as sqlite database.

        :type act_number_trials: int
        :param act_number_trials: Number of optimization trials
        """
        if self._sbc_config is None:
            logger.warning("Circuit configuration is not initialized!")
            return
        elif self._fixed_parameters is None:
            logger.warning("Parameter calculation is missing!")
            return
        elif self._study_in_memory is None:
            logger.warning("Study is not initialized!")
            return

        # Function to execute
        func = lambda trial: SbcCircuitOptimization._objective(trial, self._sbc_config, self._fixed_parameters)

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
        if self._sbc_config is None:
            logger.warning("Circuit configuration is not initialized!")
            return
        elif self._fixed_parameters is None:
            logger.warning("Parameter calculation is missing!")
            return

        # Function to execute
        func = lambda trial: SbcCircuitOptimization._objective(trial, self._sbc_config, self._fixed_parameters)

        # Each process create his own study instance with the same database and study name
        act_study = optuna.load_study(storage=act_storage_url, study_name=self._sbc_config.circuit_study_name)
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

    def start_proceed_study(self, number_trials: int, database_type: str = 'sqlite',
                            sampler: optuna.samplers.BaseSampler = optuna.samplers.NSGAIIISampler()) -> None:
        """Proceed a study which is stored as sqlite database.

        :param number_trials: Number of trials adding to the existing study
        :type number_trials: int
        :param database_type: storage database, e.g. 'sqlite' or 'mysql'
        :type  database_type: str
        :param sampler: optuna.samplers.NSGAIISampler() or optuna.samplers.NSGAIIISampler(). Note about the brackets () !! Default: NSGAIII
        :type sampler: optuna.sampler-object
        """
        if self._sbc_config is None:
            logger.warning("Method 'initialize_circuit_optimization' is not called!\n"
                           "    No list is generated so that no optimization can be performed!")
            return

        circuit_study_sqlite_database = os.path.join(self.circuit_study_data.optimization_directory,
                                                     f"{self._sbc_config.circuit_study_name}.sqlite3")

        if os.path.exists(circuit_study_sqlite_database):
            logger.info("Existing circuit study found. Proceeding.")
        else:
            os.makedirs(f"{self.circuit_study_data.optimization_directory}", exist_ok=True)

        # set logging verbosity: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.logger.set_verbosity.html#optuna.logging.set_verbosity
        # .INFO: all messages (default)
        # .WARNING: fails and warnings
        # .ERROR: only errors
        optuna.logging.set_verbosity(optuna.logging.ERROR)

        # check for differences with the old configuration file
        config_on_disk_filepath = f"{self.circuit_study_data.optimization_directory}/{self._sbc_config.circuit_study_name}.pkl"
        if os.path.exists(config_on_disk_filepath):
            config_on_disk = SbcCircuitOptimization.load_stored_config(self.circuit_study_data)
            difference = deepdiff.DeepDiff(config_on_disk, self._sbc_config, ignore_order=True, significant_digits=10)
            if difference:
                raise ValueError("Configuration file has changed from previous simulation.\n"
                                 f"Program is terminated.\n"
                                 f"Difference: {difference}")

        # directions = ['maximize', 'minimize']
        directions = ["minimize", "minimize"]

        # Calculate the fixed parameters
        self._fixed_parameters = SbcCircuitOptimization.calculate_fixed_parameters(self._sbc_config)

        # Debug
        print(f"i=f{self._fixed_parameters.mesh_i}")
        print(f"V_in=f{self._fixed_parameters.mesh_v}")
        print(f"duty=f{self._fixed_parameters.mesh_duty_cycle}")
        print(f"w=f{self._fixed_parameters.mesh_weights}")

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
            self._study_in_storage = optuna.create_study(study_name=self._sbc_config.circuit_study_name,
                                                         storage=storage,
                                                         directions=directions,
                                                         load_if_exists=True, sampler=sampler)

            # Create study object in memory
            self._study_in_memory = optuna.create_study(study_name=self._sbc_config.circuit_study_name, directions=directions, sampler=sampler)
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
            self._study_in_storage = optuna.create_study(study_name=self._sbc_config.circuit_study_name,
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
        #                                   args=(storage_url, sbc_config.circuit_study_name,
        #                                         number_trials, sbc_config,fixed_parameters
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
        elif self._sbc_config is None:
            logger.warning("Circuit configuration is not initialized!")
            return

        fig = optuna.visualization.plot_pareto_front(self._study_in_storage, target_names=["ZVS coverage / %", r"i_\mathrm{cost}"])
        fig.update_layout(title=f"{self._sbc_config.circuit_study_name} <br><sup>{self._sbc_config.project_directory}</sup>")
        fig.write_html(
            f"{self.circuit_study_data.optimization_directory}"
            f"_{datetime.datetime.now().isoformat(timespec='minutes')}.html")
        if show_results:
            fig.show()

    @staticmethod
    def load_sbc_dto_from_study(act_study_data: StudyData, sbc_config: circuit_dtos.CircuitParetoSbcDesign,
                                trial_number: int | None = None) -> s_dtos.SbcCircuitDTO:
        """
        Load a SBC-DTO from an optuna study.

        :param act_study_data: Information about the circuit study name and study path
        :type  act_study_data: StudyData
        :param sbc_config: SBC optimization configuration file
        :type  sbc_config: circuit_dtos.CircuitParetoSbcDesign
        :param trial_number: trial number to load to the DTO
        :type trial_number: int
        :return: DTO
        :rtype:  s_dtos.SbcCircuitDTO
        """
        if trial_number is None:
            raise NotImplementedError("needs to be implemented")

        database_url = SbcCircuitOptimization.create_sqlite_database_url(act_study_data)

        loaded_study = optuna.create_study(study_name=act_study_data.study_name,
                                           storage=database_url, load_if_exists=True)
        logger.info(f"The study '{act_study_data.study_name}' contains {len(loaded_study.trials)} trials.")
        trials_dict = loaded_study.trials[trial_number].params

        fix_parameters = SbcCircuitOptimization.calculate_fixed_parameters(sbc_config)

        sbc_dto = d_sets.HandleSbcDto.init_config(
            name=str(trial_number),
            mesh_v=fix_parameters.mesh_v,
            mesh_duty_cycle=fix_parameters.mesh_duty_cycle,
            mesh_i=fix_parameters.mesh_i,
            sampling=sbc_config.sampling,
            ls=trials_dict["l_s_suggest"],
            fs=trials_dict["f_s_suggest"],
            transistor_dto_1=trials_dict["transistor_1_name_suggest"],
            transistor_dto_2=trials_dict["transistor_2_name_suggest"]
        )

        return sbc_dto

    def df_to_sbc_dto_list(self, df: pd.DataFrame) -> list[s_dtos.SbcCircuitDTO]:
        """
        Load a SBC-DTO from an optuna study.

        :param df: Pandas DataFrame to convert to the SBC-DTO list
        :type df: pd.DataFrame
        :return: List of DTO
        :rtype:  list[s_dtos.SbcCircuitDTO]
        """
        sbc_dto_list: list[s_dtos.SbcCircuitDTO] = []

        # Check if configuration is not available or fixed parameters are not available
        if self._sbc_config is None:
            logger.warning("Circuit configuration is not initialized!")
            return sbc_dto_list
        elif self._fixed_parameters is None:
            logger.warning("Missing initialized fixed parameters!")
            return sbc_dto_list

        logger.info(f"The study '{self._sbc_config.circuit_study_name}' contains {len(df)} trials.")

        index: int
        for idx, _ in df.iterrows():

            index = cast(int, idx)

            transistor_dto_1 = d_sets.HandleTransistorDto.tdb_to_transistor_dto(str(df.at[index, "params_transistor_1_name_suggest"]))
            transistor_dto_2 = d_sets.HandleTransistorDto.tdb_to_transistor_dto(str(df.at[index, "params_transistor_1_name_suggest"]))

            sbc_dto = d_sets.HandleSbcDto.init_config(
                name=str(df.at[index, "number"]),
                mesh_v=self._fixed_parameters.mesh_v,
                mesh_duty_cycle=self._fixed_parameters.mesh_duty_cycle,
                mesh_i=self._fixed_parameters.mesh_i,
                sampling=self._sbc_config.sampling,
                ls=float(cast(SupportsFloat, df.at[index, "params_l_s_suggest"])),
                fs=float(cast(SupportsFloat, df.at[index, "params_f_s_suggest"])),
                transistor_dto_1=transistor_dto_1,
                transistor_dto_2=transistor_dto_2
            )
            sbc_dto_list.append(sbc_dto)

        return sbc_dto_list

    @staticmethod
    def study_to_df(act_study_data: StudyData) -> pd.DataFrame:
        """Create a DataFrame from a study.

        :param act_study_data: Information about the circuit study name and study path
        :type  act_study_data: StudyData
        :return: study result data transferred to Pandas dataframe
        :rtype: pd.DataFrame
        """
        database_url = SbcCircuitOptimization.create_sqlite_database_url(act_study_data)
        loaded_study = optuna.create_study(study_name=act_study_data.study_name, storage=database_url, load_if_exists=True)
        df = loaded_study.trials_dataframe()
        df.to_csv(f'{act_study_data.optimization_directory}/{act_study_data.study_name}.csv')
        return df

    @staticmethod
    def create_sqlite_database_url(act_study_data: StudyData) -> str:
        """
        Create the SBC circuit optimization sqlite URL.

        :param act_study_data: Information about the circuit study name and study path
        :type  act_study_data: StudyData
        :return: SQLite URL
        :rtype: str
        """
        sqlite_storage_url = f"sqlite:///{act_study_data.optimization_directory}/{act_study_data.study_name}.sqlite3"
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
        next_point_index = 0

        while next_point_index < len(costs):
            # points, which are dominated
            dominated = np.all(costs[next_point_index] <= costs, axis=1)

            # Remove dominating points
            dominated[next_point_index] = False
            is_efficient = is_efficient[~dominated]
            costs = costs[~dominated]

            next_point_index += 1

        is_efficient_mask = np.zeros(n_points, dtype=bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask

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
        pareto_tuple_mask_vec = SbcCircuitOptimization.is_pareto_efficient(numpy_zip)
        pareto_tuple_mask_vec = pareto_tuple_mask_vec.astype(bool)
        pareto_df = df[pareto_tuple_mask_vec]

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

        pareto_df: pd.DataFrame = SbcCircuitOptimization.pareto_front_from_df(df, x, y)

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

        pareto_df_offset: pd.DataFrame = df[df[y] < ref_loss_max]

        return pareto_df_offset

    @staticmethod
    def hybrid_pareto_sampling(pareto_matrix: np.ndarray, n_points: int = 8) -> np.ndarray:
        """
        Filter points from pareto front by hybrid-strategy: Extremes + Knees + Density.

        This method filters points from a Pareto front using a hybrid strategy that combines three complementary
        selection criteria: extremes, knees, and density.
        Extremes: Points that represent the best values in at least one objective, ensuring that the boundary solutions
        with maximal trade-offs are preserved.
        Knees: Points located at significant curvature changes along the Pareto front, highlighting well-balanced compromises
        where small sacrifices in one objective yield large gains in others.
        Density: Points selected based on the distribution density of solutions to reduce redundancy by representing clusters
        with fewer, but diverse, representatives.
        By integrating these strategies, the function provides a compact, representative subset of the Pareto front
        that balances multiple selection approaches.

        :param pareto_matrix: pareto front points with shape (n,2)
        :type pareto_matrix: np.ndarray
        :param n_points: Number of filtered points
        :type n_points: int
        :returns: filtered points
        :rtype: np.ndarray
        """
        # Constant values
        # Filter window and filter order
        smooth_window: int = 11
        filter_order: int = 3
        # Prevent division by 0
        prev_diff_zero = 1e-12

        # Number of selected points with point density method(round up)
        n_points_cdf: int = (n_points+1) // 2

        # Copy, sort and normalize
        pareto_matrix_copy = pareto_matrix.copy().astype(float)
        # Sort to target 1
        pareto_matrix_copy = pareto_matrix_copy[np.argsort(pareto_matrix_copy[:, 0])]
        # Reduce similar x values (Strong monotone falling) (function returns [value, index]
        _, unique_x_indices = np.unique(pareto_matrix_copy[:, 0], return_index=True)
        pareto_matrix_copy = pareto_matrix_copy[unique_x_indices]
        # Normalize the curve to [0,1]
        pareto_norm = (pareto_matrix_copy - pareto_matrix_copy.min(axis=0)) / (pareto_matrix_copy.max(axis=0) - pareto_matrix_copy.min(axis=0))

        # #### Debug  ###############################################################
        df_filtered_par = pd.DataFrame({"x": pareto_norm[:, 0], "y": pareto_norm[:, 1]})
        # Store the data
        # df_filtered_par.to_csv("path to define + /Pareto_data.csv")
        # ###########################################################################

        # Curvature analysis due to the knee
        # Smooth the curve with Savitzky–Golay-Filter
        smooth_target_2 = savgol_filter(pareto_norm[:, 1], smooth_window, filter_order)
        # #### Debug  ###############################################################
        df_filtered_file = pd.DataFrame({"x": pareto_norm[:, 0], "y": smooth_target_2})
        # Store the data
        # df_filtered_file.to_csv("path to define + Pareto_filter_data.csv")
        # ###########################################################################

        # First and second derivative
        d_target_2 = np.gradient(smooth_target_2, pareto_norm[:, 0])
        d2_target_2 = np.gradient(d_target_2, pareto_norm[:, 0])
        # curvature calculation (Formula at 13.3.8 see url)
        # https://math.libretexts.org/Bookshelves/Calculus/Calculus_(OpenStax)/13%3A_Vector-Valued_Functions/13.03%3A_Arc_Length_and_Curvature
        curvature = np.abs(d2_target_2) / (1 + d_target_2 ** 2) ** 1.5

        # Length of the curve
        d_target_1 = np.diff(pareto_norm[:, 0])
        d_target_2 = np.diff(pareto_norm[:, 1])
        d_curve = np.sqrt(d_target_1 ** 2 + d_target_2 ** 2)
        curve = np.concatenate(([0], np.cumsum(d_curve)))
        curve = curve / curve[-1]  # Standardize to [0,1]

        # Weighting: Curvature * local density
        density = np.ones_like(curve)
        for i in range(1, len(curve) - 1):
            density[i] = 0.5 * (curve[i + 1] - curve[i - 1])
        inv_density = 1 / (density + prev_diff_zero)
        weight = curvature * inv_density

        # Selection of points according point density (inclusive extreme values)
        result_point_vector = np.linspace(0, 1, n_points_cdf)
        cdf = np.cumsum(weight)
        cdf = cdf / cdf[-1]
        # Create point density index vector
        cdf_filtered_id: list[int] = []
        # For loop over result_point_vector
        for t in result_point_vector:
            idx = int(np.argmin(np.abs(cdf - t)))
            cdf_filtered_id.append(idx)

        # Remove same points
        cdf_filtered_id = np.unique(cdf_filtered_id).tolist()
        n_points_cdf = len(cdf_filtered_id)

        # Set number of cluster points
        n_points_cluster = n_points - n_points_cdf

        # Find clusters in regions of high point density
        km = KMeans(n_clusters=n_points_cluster, n_init=15)
        km.fit(pareto_norm)
        cluster_centers = km.cluster_centers_

        cluster_filtered_id: list = []
        # For loop over result_point_vector
        for cluster_center in cluster_centers:
            # Square of  distance of assigned points to the cluster
            dists = np.sum((pareto_norm - cluster_center) ** 2, axis=1)
            # Point index with minimal distance
            idx = int(np.argmin(dists))
            cluster_filtered_id.append(idx)

        # Hybrid strategy: Combine points density selection and cluster points
        chosen_idx = np.concatenate([cdf_filtered_id, cluster_filtered_id])
        # Remove duplicates
        chosen_idx = np.unique(chosen_idx)
        # Return the result in original units
        pareto_matrix_result: np.ndarray = pareto_matrix_copy[chosen_idx]

        return pareto_matrix_result

    def filter_study_results(self) -> tuple[bool, str]:
        """Filter the study result."""
        # Variable initialization
        is_filter_available: bool = False
        issue_report: str = ""

        # Check if configuration is not available and if study is not available
        if self._sbc_config is None:
            issue_report = "Circuit configuration is not loaded!"
            return is_filter_available, issue_report
        if not self._is_study_available:
            issue_report = "Study is not calculated. First run 'start_proceed_study'!"
            return is_filter_available, issue_report
        if self._study_in_storage is None:
            issue_report = "Study is not calculated. First run 'start_proceed_study'!"
            return is_filter_available, issue_report

        is_filter_available = True

        # Evaluate previous result
        if not is_filter_available:
            logger.warning(issue_report)
            return is_filter_available, issue_report

        df = self._study_in_storage.trials_dataframe()
        df.to_csv(f'{self.circuit_study_data.optimization_directory}/{self._sbc_config.circuit_study_name}.csv')

        df_pareto_front = SbcCircuitOptimization.pareto_front_from_df(df)

        # Filter points from pareto front with hybrid algorithm
        filtered_points: np.ndarray = SbcCircuitOptimization.hybrid_pareto_sampling(
            df_pareto_front[["values_0", "values_1"]].to_numpy(), self._sbc_config.filter.number_filtered_designs)

        # Assign the identified points to the dataframe index
        filtered_indices: list[int] = []
        for point in filtered_points:
            # boolean mask for rows of identical content
            mask = (df["values_0"].values == point[0]) & (df["values_1"].values == point[1])
            # Find index in data frame
            idx = int(np.where(mask)[0][0])
            filtered_indices.append(idx)

        # Create selection data frame
        df_selected = df.loc[filtered_indices]

        # Continue time measurement
        with self._c_lock_stat:
            self._progress_run_time.continue_trigger()

        # Store the result to filtered point folder
        selected_dto_list = self.df_to_sbc_dto_list(df_selected)

        dto_directory = self.filter_data.filtered_list_pathname
        os.makedirs(dto_directory, exist_ok=True)
        for dto in selected_dto_list:
            # Calculate component requirement
            dto = d_sets.HandleSbcDto.generate_components_target_requirements(dto, self.circuit_study_data.study_name)
            d_sets.HandleSbcDto.save(dto, dto.circuit_id, directory=dto_directory, timestamp=False)

        # Update the filtered result list
        self.filter_data.filtered_list_files = []
        for filtered_circuit_result in os.listdir(dto_directory):
            # Check if it is a file
            if os.path.isfile(os.path.join(dto_directory, filtered_circuit_result)):
                self.filter_data.filtered_list_files.append(os.path.splitext(filtered_circuit_result)[0])

        # Evaluate the result: Has the list minimum one entry
        if len(self.filter_data.filtered_list_files) == 0:
            is_filter_available = False
            issue_report = "No design could be selected!"

        # Stop runtime measurement and update statistical data
        with self._c_lock_stat:
            self._progress_run_time.stop_trigger()
            self._progress_data.run_time = self._progress_run_time.get_runtime()
            self._progress_data.progress_status = ProgressStatus.Done

        return is_filter_available, issue_report

    def get_capacitor_requirements(self) -> list[CapacitorRequirements]:
        """Get the capacitor requirements.

        :return: Capacitor Requirements
        :rtype: CapacitorRequirements
        """
        # Variable declaration
        capacitor_requirements_list: list[CapacitorRequirements]

        # Dummy access with self to satisfy ruff
        if self._sbc_config is None:
            capacitor_requirements_list = []
        else:
            capacitor_requirements_list = []

        # Capacitor requirements still not generated
        return capacitor_requirements_list

    def get_inductor_requirements(self) -> list[InductorRequirements]:
        """Get the inductor requirements.

        :return: Inductor Requirements
        :rtype: InductorRequirements
        """
        inductor_requirements_list: list[InductorRequirements] = []
        for circuit_id_file in self.filter_data.filtered_list_files:
            circuit_id_filepath = os.path.join(self.filter_data.filtered_list_pathname, f"{circuit_id_file}.pkl")
            circuit_dto = d_sets.HandleSbcDto.load_from_file(circuit_id_filepath)
            if not isinstance(circuit_dto.component_requirements, ComponentRequirements):
                # due to mypy checker
                raise TypeError("Loaded component requirements have wrong type.")

            if not isinstance(circuit_dto.component_requirements.inductor_requirements[0], InductorRequirements):
                # due to mypy checker
                raise TypeError("Loaded capacitor requirements have wrong type.")

            # Add inductor requirements to the list
            inductor_requirements_list = inductor_requirements_list + circuit_dto.component_requirements.inductor_requirements

        return inductor_requirements_list

    def get_transformer_requirements(self) -> list[TransformerRequirements]:
        """Get the transformer requirements.

        :return: Transformer Requirements
        :rtype: TransformerRequirements
        """
        # Variable declaration
        transformer_requirements_list: list[TransformerRequirements]

        # Dummy access with self to satisfy ruff
        if self._sbc_config is None:
            transformer_requirements_list = []
        else:
            transformer_requirements_list = []

        return transformer_requirements_list

    @staticmethod
    def get_circuit_plot_data(act_study_data: StudyData) -> PlotData:
        """Provide the circuit data to plot.

        :param act_study_data: Information about the circuit study name and study path
        :type  act_study_data: StudyData
        :return: Plot data and legend
        :rtype: PlotData
        """
        # Load circuit study data
        df_circuit = SbcCircuitOptimization.study_to_df(act_study_data)

        # Convert from list[Series[Any]] to list[list[float]]
        circuit_x_values_list: list[list[float]] = [df_circuit["values_0"].to_list()]
        circuit_y_values_list: list[list[float]] = [df_circuit["values_1"].to_list()]

        # Extract circuit plot data from data frame
        circuit_plot_data: PlotData = (
            PlotData(x_values_list=circuit_x_values_list, y_values_list=circuit_y_values_list,
                     color_list=[gps.colors()["black"]], alpha=0.5,
                     x_label=r"$\mathcal{L}_\mathrm{v}$ / \%", y_label=r"$\mathcal{L}_\mathrm{i}$ / A²",
                     label_list=[None], fig_name_path=act_study_data.study_name))

        return circuit_plot_data

    @staticmethod
    def get_number_of_required_capacitors() -> int:
        """Get the number of  required capacitors.

        :return: Number of capacitors required by the actual topology
        :rtype: int
        """
        return SbcCircuitOptimization._number_of_required_capacitors

    @staticmethod
    def get_number_of_required_inductors() -> int:
        """Get the number of  required inductors.

        :return: Number of inductors required by the actual topology
        :rtype: int
        """
        return SbcCircuitOptimization._number_of_required_inductors

    @staticmethod
    def get_number_of_required_transformers() -> int:
        """Get the number of  required transformers.

        :return: Number of transformers required by the actual topology
        :rtype: int
        """
        return SbcCircuitOptimization._number_of_required_transformers

    @staticmethod
    def generate_general_toml(file_path: str) -> None:
        """
        Generate the default DabCircuitConf.toml file.

        :param file_path: filename including absolute path
        :type file_path: str
        """
        toml_data = '''
        [default_data] # After update this configuration file according your project delete this line to validate it
        [parameter_range]
            v1_min_max_list=[47, 48]
            v2_min_max_list=[20, 24]
            i2_min_max_list=[15.9, 16]
        
        [sampling]
            sampling_method="latin_hypercube"
            sampling_points=8
            sampling_random_seed=10
            v1_additional_user_point_list=[]
            v2_additional_user_point_list=[]
            i2_additional_user_point_list=[]
            additional_user_weighting_point_list=[]
        
        [misc]
            min_efficiency_percent=80
            control_board_volume=10e-6
        '''
        with open(file_path, 'w') as output:
            output.write(toml_data)

    @staticmethod
    def generate_circuit_toml(file_path: str) -> None:
        """Generate the default DabCircuitConf.toml file.

        :param file_path: filename including absolute path
        :type file_path: str
        """
        toml_data = '''
        [default_data] # After update this configuration file according your project delete this line to validate it
        [design_space]
            f_s_min_max_list=[50e3, 300e3]
            l_s_min_max_list=[20e-6, 900e-6]
            l_1_min_max_list=[10e-6, 10e-3]
            l_2__min_max_list=[10e-6, 1e-3]
            n_min_max_list=[3, 7]
            transistor_1_name_list=['CREE_C3M0065100J', 'CREE_C3M0120100J']
            transistor_2_name_list=['CREE_C3M0060065J', 'CREE_C3M0120065J']
            c_par_1=16e-12
            c_par_2=16e-12
    
        [output_range]
            v1_min_max_list=[690, 710]
            v2_min_max_list=[175, 295]
            p_min_max_list=[-2000, 2200]
    
        [sampling]
            sampling_method="latin_hypercube"
            sampling_points=4
            sampling_random_seed=10
            v1_additional_user_point_list=[700]
            v2_additional_user_point_list=[180]
            p_additional_user_point_list=[2000]
    
        [filter_distance]
            number_filtered_designs = 1
            difference_percentage = 5

        [thermal_data]
            # [tim_thickness, tim_conductivity]
            transistor_b1_cooling = [1e-3,12.0]
            transistor_b2_cooling = [1e-3,12.0]            
           
        '''
        with open(file_path, 'w') as output:
            output.write(toml_data)

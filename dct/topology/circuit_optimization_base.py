"""Main control program to optimize the DAB converter."""

# python libraries
from abc import ABC, abstractmethod
from typing import Generic, TypeVar, Any
import os

# 3rd party libraries
from pydantic import BaseModel
import optuna

# Own libraries
from dct.datasets_dtos import StudyData, FilterData
from dct.server_ctl_dtos import ProgressData
from dct.topology.component_requirements_from_circuit import CapacitorRequirements

# Type of general optimization parameter
T_G_D = TypeVar("T_G_D", bound="TomlGData")
# Type of circuit optimization parameter
T_C_D = TypeVar("T_C_D", bound="TomlCData")


class TomlGData(BaseModel, ABC):
    """Contains the general toml data."""

    pass

class TomlCData(BaseModel, ABC):
    """Contains the circuit toml data."""

    pass


class CircuitOptimizationBase(Generic[T_G_D, T_C_D], ABC):
    """Represent the base class for electrical converter optimization depending on the topology."""

    def __init__(self) -> None:
        """Initialize the member variables."""
        self.project_directory: str = ""
        self.circuit_study_data: StudyData = StudyData()
        self.filter_data: FilterData = FilterData()

    def init_study_information(self, study_name: str, project_directory: str, sub_directory: str) -> None:
        """Initialize the study information.

        :param study_name: Name of the study
        :type  study_name: str
        :param project_directory: Project directory name
        :type  project_directory: Project folder path
        :param sub_directory: Sub directory of the circuit optimization
        :type  sub_directory: Sub directory path
        """
        # Project directory
        self.project_directory = project_directory
        # Circuit path
        self.circuit_study_data = StudyData(
            study_name=study_name, optimization_directory=os.path.join(project_directory, sub_directory, study_name))

        # Filtered data path and list
        self.filter_data = FilterData(
            circuit_study_name=self.circuit_study_data.study_name,
            filtered_list_pathname=os.path.join(self.circuit_study_data.optimization_directory,
                                                "filtered_results")
        )

    def is_circuit_optimization_skippable(self) -> tuple[bool, str]:
        """Control procedure of skippable optimization check.

        :return: True, if the optimization is skippable and empty string or False and report of the issues
        :rtype: tuple[bool, str]
        """
        if self.circuit_study_data is None and self.filter_data is None:
            # The method to initialize study information needs to be called first.
            # This has to be guaranteed by the workflow
            return False, "Serious programming error 1b. Please write an issue!"

        # Check, if all data are available
        is_skipable, issue_report = self.__class__._is_optimization_skippable(self.circuit_study_data, self.filter_data)

        # return result
        return is_skipable, issue_report

    # Abstract methods
    @abstractmethod
    def load_and_verify_general_parameters(self, toml_dict: dict[str, Any]) -> tuple[bool, str]:
        """Verify the input parameter ranges.

        :param toml_dict: toml general configuration
        :type toml_dict: dict[str, Any]
        :return: True, if the configuration was consistent and empty string or False and report of the issues
        :rtype: tuple[bool, str]
        """
        pass

    @abstractmethod
    def load_and_verify_circuit_parameters(self, toml_dict: dict[str, Any], is_tdb_to_update: bool = False) -> tuple[bool, str]:
        """Load and verify the circuit input parameter.

        :param toml_dict: toml general configuration
        :type  toml_dict: dict[str, Any]
        :param is_tdb_to_update: indicated, if the transistor database is up to date
        :type  is_tdb_to_update: bool
        :return: True, if the configuration was consistent and empty string or False and report of the issues
        :rtype: tuple[bool, str]
        """
        pass

    @staticmethod
    @abstractmethod
    def _is_optimization_skippable(study_data: StudyData, filter_data: FilterData) -> tuple[bool, str]:
        """Verify the circuit optimization is skippable.

        :param study_data: Study data information
        :type  study_data: StudyData
        :param filter_data: Filtered data
        :type  filter_data: FilterData
        :return: True, if the configuration was consistent and empty string or False and report of the issues
        :rtype: tuple[bool, str]
        """
        pass

    @abstractmethod
    def initialize_circuit_optimization(self) -> bool:
        """Initialize the circuit optimization configuration.

        :return: True, if the configuration is initialized
        :rtype: bool
        """
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def filter_study_results(self) -> tuple[bool, str]:
        """Filter the study result and (later to implement->) use ngspice for detailed calculation.

        :return: True, if the study results are filtered and empty string or False and report of the issues
        :rtype: tuple[bool, str]
        """
        pass

    # Server/ Performance methods

    @abstractmethod
    def get_progress_data(self) -> ProgressData:
        """Provide the progress data of the optimization.

        :return: Progress data: Processing start time, actual processing time, number of filtered operation points and status.
        :rtype: ProgressData
        """
        pass

    @abstractmethod
    def get_actual_pareto_html(self) -> str:
        """
        Read the current Pareto front from running optimization process.

        :return: Pareto front html page
        :rtype: str
        """
        pass

    @staticmethod
    @abstractmethod
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
        pass

    @staticmethod
    @abstractmethod
    def get_capacitor_requirements(circuit_filepath: str) -> list[CapacitorRequirements]:
        """Get the capacitor requirements.

        :param circuit_filepath: circuit filepath
        :type circuit_filepath: str
        :return: Capacitor requirements
        :rtype: CapacitorRequirements
        """
        pass

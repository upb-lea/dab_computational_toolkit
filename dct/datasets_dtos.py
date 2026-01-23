"""DTOs for general datasets."""

# python libraries
import dataclasses
import os
import logging
import dct.toml_checker as TomlConf

# 3rd party libraries

# own libraries
from dct.circuit_enums import CalcModeEnum

# Initialize logger
logger = logging.getLogger(__name__)

class StudyData:
    """Data class containing all general information to perform a study."""

    def __init__(self, study_name: str, optimization_directory: str,
                 calculation_mode: CalcModeEnum = CalcModeEnum.new_mode, number_of_trials: int = 0):
        """
        Initialize the member variables.

        If no member variables are provided, its will be initialized as empty strings, which corresponds to invalid values.
        :param optimization_directory: Drive location folder to the optimization data
        :type  optimization_directory: str
        :param circuit_study_name: Name of the study
        :type  circuit_study_name: str
        """
        self.study_name = study_name
        self.optimization_directory = optimization_directory
        self.calculation_mode = calculation_mode
        self.number_of_trials = number_of_trials
        # Check if optimization_directory is not empty
        if optimization_directory != "":
            # Create optimization_directory if not exists
            os.makedirs(optimization_directory, exist_ok=True)
        
    @staticmethod
    def check_study_data(study_path: str, study_name: str) -> bool:
        """
        Verify if the study path and sqlite3-database file exists.

        Works for all types of studies (circuit, inductor, transformer, heat sink).
        :param study_path: drive location folder to the study
        :type  study_path: str
        :param study_name: Name of the study
        :type  study_name: str
        :return: True, if the optimization could be performed successful
        :rtype: bool
        """
        # Variable definition
        # return value initialization to false
        is_study_existing = False

        # check path
        if os.path.exists(study_path) or study_path == "":
            # Assemble file name
            study_name = study_name + ".sqlite3"
            target_file = os.path.join(study_path, study_name)
            # check filename
            if os.path.isfile(target_file):
                is_study_existing = True
            else:
                logger.info(f"File {target_file} does not exists!")
        else:
            logger.info(f"Path {study_path} does not exists!")

        # True = study exists
        return is_study_existing

@dataclasses.dataclass
class FilterData:
    """Information about the filtered circuit designs."""

    def __init__(self, filtered_list_pathname: str = "", circuit_study_name: str = ""):
        """
        Initialize the member variables.

        If no member variables are provided, its will be initialized as empty strings, which corresponds to invalid values.
        :param filtered_list_pathname: drive location folder to the filtered data
        :type  filtered_list_pathname: str
        :param circuit_study_name: Name of the study
        :type  circuit_study_name: str
        """
        # Initialize the member variables
        self.filtered_list_files: list[str] = []
        self.filtered_list_pathname: str = filtered_list_pathname
        self.circuit_study_name: str = circuit_study_name


@dataclasses.dataclass
class PlotData:
    """Contains data to plot with plot function."""

    x_values_list: list[list[float]]
    y_values_list: list[list[float]]
    color_list: list[str]
    alpha: float
    x_label: str
    y_label: str
    label_list: list[str | None]
    fig_name_path: str
    xlim: list | None = None
    ylim: list | None = None

@dataclasses.dataclass
class CapacitorConfiguration:
    """Study data and toml-file for capacitor configuration."""

    study_data: StudyData
    capacitor_toml_data: TomlConf.TomlCapacitorSelection | None = None

@dataclasses.dataclass
class InductorConfiguration:
    """Study data and toml-file for inductor configuration."""

    study_data: StudyData
    inductor_toml_data: TomlConf.TomlInductor | None = None

@dataclasses.dataclass
class TransformerConfiguration:
    """Study data and toml-file for transformer configuration."""

    study_data: StudyData
    transformer_toml_data: TomlConf.TomlTransformer | None = None

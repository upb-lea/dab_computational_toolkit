"""Main control program to optimize the DAB converter."""

# python libraries
import datetime
import copy
import os
import shutil
import sys
import threading
import tomllib
import zipfile
import fnmatch
import time
from multiprocessing import Queue
from typing import Any
import logging
import logging.config
from importlib.metadata import version

# 3rd party libraries
import json


# own libraries
import dct
from dct import toml_checker as tc
from dct.server_ctl_dtos import ProgressStatus
from dct import server_ctl_dtos as srv_ctl_dtos
from dct.datasets_dtos import StudyData
# Circuit, inductor, transformer and heat sink optimization class
from dct.topology.circuit_optimization_base import CircuitOptimizationBase
from dct import CapacitorSelection
from dct import InductorOptimization
from dct import TransformerOptimization
from dct import HeatSinkOptimization
from dct.plot_control import ParetoPlots
from dct import generate_logging_config
import dct.generate_toml as toml_gen
from dct.components.summary_processing import SummaryProcessing
from dct.server_ctl_dtos import ConfigurationDataEntryDto, SummaryDataEntryDto
from dct.server_ctl import DctServer as ServerCtl
from dct.server_ctl import ServerRequestData
from dct.server_ctl import RequestCmd
from dct.server_ctl import ParetoFrontSource
from dct.server_ctl_dtos import ProgressData
from dct.server_ctl_dtos import RunTimeMeasurement as RunTime
from dct.circuit_enums import CalcModeEnum, TopologyEnum
from dct.constant_path import (CIRCUIT_INDUCTOR_RELUCTANCE_LOSSES_FOLDER, CIRCUIT_INDUCTOR_FEM_LOSSES_FOLDER,
                               CIRCUIT_TRANSFORMER_RELUCTANCE_LOSSES_FOLDER, CIRCUIT_TRANSFORMER_FEM_LOSSES_FOLDER,
                               FILTERED_RESULTS_PATH, RELUCTANCE_COMPLETE_FILE, CIRCUIT_CAPACITOR_LOSS_FOLDER,
                               SIMULATION_COMPLETE_FILE, PROCESSING_COMPLETE_FILE)

logger = logging.getLogger(__name__)

class DctMainCtl:
    """Main class for control dab-optimization."""

    def __init__(self) -> None:
        """Initialize the member variable of the DctMainCtl-class."""
        # List data for server communication
        self._circuit_list: list[srv_ctl_dtos.ConfigurationDataEntryDto]
        self._inductor_main_list: list[srv_ctl_dtos.MagneticDataEntryDto]
        self._transformer_main_list: list[srv_ctl_dtos.MagneticDataEntryDto]
        self._inductor_list: list[srv_ctl_dtos.ConfigurationDataEntryDto]
        self._transformer_list: list[srv_ctl_dtos.ConfigurationDataEntryDto]
        self._heat_sink_list: list[srv_ctl_dtos.ConfigurationDataEntryDto]
        self._summary_list: list[srv_ctl_dtos.SummaryDataEntryDto]
        # Processing time data (Prepared for future implementation with  multiple configurations)
        self._total_time: RunTime
        self._inductor_progress_time: list[RunTime] = []
        self._transformer_progress_time: list[RunTime] = []

        # Optimization class instances
        # circuit_optimization is missing due to static class. Needs to be changed to instance class too.
        self._filtered_list_files: list[str] = []
        self._circuit_optimization: CircuitOptimizationBase | None = None
        self._capacitor_selection: CapacitorSelection | None = None
        self._capacitor_2_selection: CapacitorSelection | None = None
        self._inductor_optimization: InductorOptimization | None = None
        self._transformer_optimization: TransformerOptimization | None = None
        self._heat_sink_optimization: HeatSinkOptimization | None = None
        self._summary_processing: SummaryProcessing | None = None
        self._summary_processing: SummaryProcessing | None = None

        # Filtered point results in case of skip
        self._inductor_number_filtered_analytic_points_skip_list: list[int] = []
        self._inductor_number_filtered_simulation_points_skip_list: list[int] = []
        self._transformer_number_filtered_analytic_points_skip_list: list[int] = []
        self._transformer_number_filtered_simulation_points_skip_list: list[int] = []
        self._heat_sink_number_filtered_points_skip: list[int] = []

        # Key input variables
        self._key_input_lock: threading.Lock = threading.Lock()
        self._key_input_string: str = ""
        self._break_point_flag = False
        self._break_point_message: str = ""

    @staticmethod
    def log_software_versions(filepath: str) -> None:
        """
        Log the software versions of selected packages used to generate the results.

        :param filepath: file path and file name of the logging file
        :type filepath: str
        """
        with open(filepath, "w") as file:
            file.write(
                f'dct=={version("dct")}\n'
                f'optuna=={version("optuna")}\n' 
                f'femmt=={version("femmt")}\n'
                f'materialdatabase=={version("materialdatabase")}\n'
                f'transistordatabase=={version("transistordatabase")}\n'
                f'hct=={version("hct")}\n'
                f'mag-net-hub=={version("mag-net-hub")}\n'
                f'numpy=={version("numpy")}\n'
                f'pandas=={version("pandas")}\n'
                f'matplotlib=={version("matplotlib")}\n')

    @staticmethod
    def set_up_folder_structure(toml_prog_flow: tc.FlowControl) -> None:
        """
        Set up the folder structure for the subprojects.

        :param toml_prog_flow: Flow control toml file
        :type toml_prog_flow: tc.FlowControl
        """
        # Convert relative  paths to absolute paths
        project_directory = os.path.abspath(toml_prog_flow.general.project_directory)
        circuit_path = os.path.join(project_directory, toml_prog_flow.circuit.subdirectory)
        capacitor_1_path = os.path.join(project_directory, toml_prog_flow.capacitor_1.subdirectory)
        capacitor_2_path = os.path.join(project_directory, toml_prog_flow.capacitor_2.subdirectory)
        inductor_path = os.path.join(project_directory, toml_prog_flow.inductor.subdirectory)
        transformer_path = os.path.join(project_directory, toml_prog_flow.transformer.subdirectory)
        heat_sink_path = os.path.join(project_directory, toml_prog_flow.heat_sink.subdirectory)
        pre_summary_path = os.path.join(project_directory, toml_prog_flow.pre_summary.subdirectory)
        summary_path = os.path.join(project_directory, toml_prog_flow.summary.subdirectory)

        path_dict = {'circuit': circuit_path,
                     'capacitor_1': capacitor_1_path,
                     'capacitor_2': capacitor_2_path,
                     'inductor': inductor_path,
                     'transformer': transformer_path,
                     'heat_sink': heat_sink_path,
                     'pre_summary': pre_summary_path,
                     'summary': summary_path}

        for _, value in path_dict.items():
            os.makedirs(value, exist_ok=True)

        json_filepath = os.path.join(project_directory, "filepath_config.json")

        with open(json_filepath, 'w', encoding='utf8') as json_file:
            json.dump(path_dict, json_file, ensure_ascii=False, indent=4)

    @staticmethod
    def load_toml_file(toml_file: str) -> tuple[bool, dict[str, Any]]:
        """
        Load the toml configuration data to a dictionary.

        :param toml_file : File name of the toml-file
        :type  toml_file : str
        :return: True, if the data could be loaded successful and the loaded dictionary
        :rtype: bool, dict
        """
        # return value initialization to false and toml Data to empty
        is_toml_file_existing = False
        config: dict[str, Any] = {}

        # Separate filename and path
        toml_file_directory = os.path.dirname(toml_file)

        # check path
        if os.path.exists(toml_file_directory) or toml_file_directory == "":
            # check filename
            if os.path.isfile(toml_file):
                with open(toml_file, "rb") as f:
                    try:
                        config = tomllib.load(f)
                        is_toml_file_existing = True
                    except (tomllib.TOMLDecodeError) as e:
                        # File is not conform to toml-format
                        logger.warning(f"toml-file is not conform to toml-format:\n{e}")
            else:
                logger.warning(f"File {toml_file} does not exists!")
        else:
            logger.warning(f"Path {toml_file_directory} does not exists!")

        return is_toml_file_existing, config

    @staticmethod
    def load_generate_logging_config(logging_config_file: str) -> None:
        """
        Read the logging configuration file and configure the logger.

        Generate a default logging configuration file in case it does not exist.

        :param logging_config_file: File name of the logging configuration file
        :type logging_config_file: str
        """
        # Separate filename and path
        logging_conf_file_directory = os.path.dirname(logging_config_file)

        # check path
        if os.path.exists(logging_conf_file_directory) or logging_conf_file_directory == "":
            # check filename
            if os.path.isfile(logging_config_file):
                with open(logging_config_file, "rb") as f:
                    try:
                        logging.config.fileConfig(logging_config_file)
                    except:
                        logger.warning(f"Logging configuration file {logging_config_file} is inconsistent.")
                    else:
                        logger.info(f"Found existing logging configuration {logging_config_file}.")
            else:
                logger.info("Generate a new logging.conf file.")
                generate_logging_config(logging_conf_file_directory)
                # Reset to standard file name
                logging_config_file = os.path.join(logging_conf_file_directory, "logging.conf")
                if os.path.isfile(logging_config_file):
                    with open(logging_config_file, "rb") as f:
                        logging.config.fileConfig(logging_config_file)
                else:
                    raise ValueError("logging.conf can not be generated.")
        else:
            logger.warning(f"Path {logging_conf_file_directory} does not exists!")

    def generate_conf_file(self, path: str) -> bool:
        """
        Create and save the configuration file.

        Generate following default configuration files within the path:
        DabCircuitConf.toml, DabInductorConf.toml, DabTransformerConf.toml and DabHeatSinkConf.toml,

        :param path : Location of the configuration
        :type  path : str
        :return: true, if the files are stored successfully
        :rtype: bool

        """
        # Dummy code to satisfy ruff
        return_value: bool = self._break_point_flag and False

        return return_value

    @staticmethod
    def delete_study_content(folder_name: str, study_file_name: str = "") -> None:
        """
        Delete the study files and the femmt folders.

        If a new study is to generate the old obsolete files and folders needs to be deleted.

        :param folder_name : Location of the study files
        :type  folder_name : str
        :param study_file_name : Name of the study files (without extension)
        :type  study_file_name : str
        """
        # Variable declaration
        is_study_found: bool = False

        # Check if folder exists
        if os.path.exists(folder_name):
            # Delete all content of the folder
            for item in os.listdir(folder_name):
                # Create the full pathname
                full_path = os.path.join(folder_name, item)
                # Check if it is a folder
                if os.path.isdir(full_path):
                    # Delete the folder
                    shutil.rmtree(full_path)
                # Check if it is the target file name
                elif os.path.isfile(full_path) and os.path.splitext(item)[0] == study_file_name:
                    # Delete this file
                    os.remove(full_path)
                    # Set the flag that study is found and deleted
                    is_study_found = True
            # Check, if the study is not found
            if not is_study_found:
                logger.info(f"File of study {study_file_name} does not exists in {folder_name}!")
        else:
            logger.info(f"Path {folder_name} does not exists!")

    def user_input_break_point(self, break_point_key: str, info: str) -> None:
        """
        Continue, wait for user input or stop the program according breakpoint configuration.

        :param  break_point_key: Breakpoint configuration keyword
        :type   break_point_key: str
        :param info: Information text displayed at this breakpoint (if program waits or stops).
        :type  info: str
        """
        # Still not defined
        pass

    @staticmethod
    def get_number_of_pkl_files(filtered_file_path: str) -> int:
        """Count the number of files with extension 'pkl'.

        If the optimization is skipped the number of filtered points reflected by the number of pkl-files
        needs to be count for status information.

        :param filtered_file_path: drive location path to the 'pkl'-file folder
        :type  filtered_file_path: str
        :return: Number of pkl-files within the folder corresponds to number of filtered operation points
        :rtype: int
        """
        # Number of pkl-files in this folder
        number_of_files = 0

        # check path
        if os.path.exists(filtered_file_path):
            # Loop over the files
            for filename in os.listdir(filtered_file_path):
                if filename.endswith('.pkl') and os.path.isfile(os.path.join(filtered_file_path, filename)):
                    number_of_files = number_of_files + 1
        else:
            logger.info(f"Path {filtered_file_path} does not exists!")

        # Return the number of files with extension pkl
        return number_of_files

    def check_breakpoint(self, break_point_key: str, info: str) -> None:
        """
        Continue, wait for user input or stop the program according breakpoint configuration.

        :param  break_point_key: Breakpoint configuration keyword
        :type  break_point_key: str
        :param info: Information text displayed at this breakpoint (if program waits or stops).
        :type  info: str
        """
        # Check if breakpoint stops the program
        if break_point_key == "stop":
            print("Program stops cause by breakpoint at: '"+info+"'!")
            # stop program
            sys.exit()

        elif break_point_key == "pause":
            # Information
            print("Active breakpoint at: '"+info+"'!\n")
            print("'C'=continue, 'S'=stop the program. Please enter your choice")
            key_inp = "x"
            # Notify server about the breakpoint
            self._break_point_flag = True
            # Consume previous key input
            self._get_key_inp()
            # Add breakpoint message
            self._break_point_message = "Active breakpoint at: '"+info+"'!\n"
            # Wait for keyboard entry or server command
            while key_inp != "c" and key_inp != "C" and key_inp != "s" and key_inp != "S" and self._break_point_flag:
                key_inp = self._get_key_inp()
                # Decrease the load
                time.sleep(0.25)

            # Check result
            if key_inp == "s" or key_inp == "S":
                print("User stops the program!")
                # stop program
                sys.exit()
        else:
            # Remove breakpoint message
            self._break_point_message = ""

    @staticmethod
    def generate_zip_archive(toml_prog_flow: tc.FlowControl) -> None:
        """
        Generate a zip archive from the given simulation results to transfer to another computer.

        Remove unnecessary file structure before performing the zip operation, e.g. the 00_femmt_simulation results directory.

        :param toml_prog_flow: Flow control toml file
        :type toml_prog_flow: tc.FlowControl
        """
        folder_selection = [toml_prog_flow.general.project_directory]

        # Exclude folders that should not be included in the zip archive
        folder_exclusion = ['00_femmt_simulation']

        # Define the path to the zip archive
        zip_file_name = f'{toml_prog_flow.general.project_directory}_archived_{datetime.datetime.now().strftime("%Y-%m-%d_%H:%M")}.zip'

        # Check if path exists
        if os.path.exists(folder_selection[0]):

            # Get target folder name for the zip file
            zip_file_path = os.path.join(os.path.dirname(folder_selection[0]), zip_file_name)

            # Create the zip archive
            with zipfile.ZipFile(zip_file_path, 'w') as zip_archive:
                for folder in folder_selection:
                    for root, dirs, files in os.walk(folder):
                        for dir in dirs:
                            dir_path = os.path.join(root, dir)
                            relative_path = os.path.relpath(dir_path, folder)
                            if fnmatch.fnmatch(relative_path, '*/' + folder_exclusion[0] + '/*') or relative_path.endswith('/' + folder_exclusion[0]):
                                dirs.remove(dir)  # Exclude the subfolder
                        for file in files:
                            file_path = os.path.join(root, file)
                            relative_path = os.path.relpath(file_path, folder)
                            if relative_path.startswith(folder_exclusion[0] + '/') or relative_path.endswith('/' + folder_exclusion[0]):
                                continue  # Exclude the file if it is in an excluded folder
                            zip_archive.write(file_path, relative_path)
            # Notify user
            logger.info(f"Zip archive created:{zip_file_path}")

        else:
            # Warn user
            logger.warning(f"Path {folder_selection[0]} does not exists!")

    @staticmethod
    def _is_skippable(act_study_data: StudyData, complete_file_name: str, is_sqlite_check_enabled: bool = False,
                      circuit_filtered_index_list: list[str] | None = None) -> tuple[bool, str]:
        """Verify, if the optimization is skippable.

        The verification bases on the availability of a sqlite database file and
        the check of optimization complete by usage of a complete_file.

        :param act_study_data: Information about the study name and study path
        :type  act_study_data: StudyData
        :param complete_file_name: file name which contains the completion list (without path)
        :type  complete_file_name: str
        :param is_sqlite_check_enabled: Flag, which indicates, if the sqlite database check is enabled
        :type  is_sqlite_check_enabled: bool
        :param circuit_filtered_index_list: List with the name of filtered results
        :type  circuit_filtered_index_list: str
        :return: True, if the file could be created, False if the file could not create, e.g. no pkl-files is found.
        :rtype: bool
        """
        # Variable declaration and initialization
        is_processing_complete: bool
        issue_report: str

        # Check is_sqlite_check_enabled-flag
        if is_sqlite_check_enabled:
            # Check if index_list is not empty
            if circuit_filtered_index_list:
                # For loop to check, if sqlite database for all filtered values are available
                for id_entry in (circuit_filtered_index_list):
                    # Assemble pathname
                    results_datapath = os.path.join(act_study_data.optimization_directory,
                                                    str(id_entry), act_study_data.study_name)
                    # Check, if data are available (skip case)
                    if not StudyData.check_study_data(results_datapath, act_study_data.study_name):
                        raise ValueError(
                            f"Study {act_study_data.study_name} in path {results_datapath} "
                            "does not exist or file path is wrong. No sqlite3-database found!")
            else:
                # Check, if data are available (skip case)
                if not StudyData.check_study_data(act_study_data.optimization_directory, act_study_data.study_name):
                    raise ValueError(f"Study {act_study_data.study_name} in path {act_study_data.optimization_directory} "
                                     "does not exist or file path is wrong. No sqlite3-database found!")

        # Check if the processing is completed for all designs
        is_processing_complete, issue_report = DctMainCtl._is_processing_complete(act_study_data.optimization_directory,
                                                                                  complete_file_name)

        return is_processing_complete, issue_report

    @staticmethod
    def _set_processing_complete(base_directory: str, subdirectory: str, complete_file_name: str,
                                 circuit_filtered_index_list: list[str] | None = None) -> None:
        """Create the 'processing_complete.json' file to indicate the completion of the calculation.

        At the end of an optimization the 'processing_complete.json' will be created. This is for verification,
        that the optimization is complete. In case of skip this file will be use to check the completion of optimization.
        The verification bases on all created pkl-files while optimization. E.g. the path to these files are assembled
        by , 'base_directory'/filter_results_list'-entry/'subdirectory'. If the filter_results_list is empty only
        'base_directory' is taken as path to pkl-files. An exception occurs, if base_directory does not exist.

        :param base_directory: Directory for 'processing_complete.json' and start point for sub directory
        :type  base_directory: str
        :param subdirectory: (relative) subdirectory path to pkl-files
        :type  subdirectory: str
        :param circuit_filtered_index_list: List with the name of filtered results
        :type  circuit_filtered_index_list: str
        :param complete_file_name: List with the name of filtered results
        :type  complete_file_name: str
        :return: True, if the file could be created, False if the file could not create, e.g. no pkl-files is found.
        :rtype: bool
        """
        # Variable declaration and initialization
        path_list: list[str] = []
        pkl_file_list: list[str] = []
        processing_complete_file = os.path.join(base_directory, complete_file_name)

        # Check if file exists
        # Check path
        if not (os.path.exists(base_directory) or base_directory == ""):
            raise ValueError(f"Path {base_directory} does not exists!")

        # Create path list
        if circuit_filtered_index_list:
            for entry in circuit_filtered_index_list:
                path_list.append(os.path.join(base_directory, entry, subdirectory))
        else:
            # pkl-files searched in base_directory/subdirectory
            path_list.append(os.path.join(base_directory, subdirectory))

        # Create pkl-file completion list
        for path_entry in path_list:
            # Check path
            if not (os.path.exists(path_entry) or path_entry == ""):
                raise ValueError(f"pkl-file path {path_entry} does not exists!")
            # Create list
            for filename in os.listdir(path_entry):
                if filename.endswith(".pkl"):
                    pkl_file_list.append(os.path.join(path_entry, filename))
        # Store processing_complete_file
        with open(processing_complete_file, "w", encoding="utf-8") as file_handle:
            json.dump(pkl_file_list, file_handle, indent=2)

    @staticmethod
    def _is_processing_complete(base_directory: str, act_complete_file_name: str) -> tuple[bool, str]:
        """Verify the completion of the calculation by usage of the 'processing_complete.json' file.

        Check, if all files exists, which are listed in 'processing_complete.json' file
        If no 'processing_complete.json' file exists at base_directory or the list is empty
        the verification fails. An exception occurs, if base_directory does not exist.

        :param base_directory: Directory for 'processing_complete.json' file
        :type  base_directory: str
        :return: True, if the verification is performed successful and empty string or False and report of the issues
        :rtype: tuple[bool, str]
        """
        # Variable declaration and initialization
        issue_report: str = ""
        pkl_file_list: list[str] = []
        processing_complete_file = os.path.join(base_directory, act_complete_file_name)
        is_processing_complete = False

        # Check if file exists
        # Check path
        if os.path.exists(base_directory) or base_directory == "":
            # Check filename
            if os.path.isfile(processing_complete_file):
                # Load the pkl_file_list from file
                with open(processing_complete_file, "r", encoding="utf-8") as file_handle:
                    pkl_file_list = json.load(file_handle)
                    # Check if list is not empty
                    if not pkl_file_list:
                        issue_report = (f"List in file {act_complete_file_name} is empty!")
            else:
                issue_report = issue_report + (f"File {act_complete_file_name} in path {base_directory} does not exists!\n")
        else:
            raise ValueError(f"Path {base_directory} does not exists!")

        # Check if list is not empty
        if pkl_file_list:
            # Set return value to true
            is_processing_complete = True
            # Loop over all entries
            for entry in pkl_file_list:
                # Check if the file exists
                if not os.path.isfile(entry):
                    issue_report = issue_report + (f"File {entry} does not exists!\n")
                    is_processing_complete = False

        return is_processing_complete, issue_report

    @staticmethod
    def _delete_processing_complete(base_directory: str, act_complete_file_name: str) -> bool:
        """Delete the 'processing_complete.json' file.

        Remark: An exception occurs, if base_directory does not exist.

        :param base_directory: Directory for 'processing_complete.json' file
        :type  base_directory: str
        :return: True, if the file can be deleted or does not exist, otherwise False
        :rtype: bool
        """
        # Variable declaration and initialization
        processing_complete_file = os.path.join(base_directory, act_complete_file_name)
        is_file_removed: bool = False

        # Check path
        if os.path.exists(base_directory) or base_directory == "":
            # Check if file exists
            if os.path.isfile(processing_complete_file):
                # Delete the file
                try:
                    os.remove(processing_complete_file)
                    is_file_removed = True
                except PermissionError:
                    logger.warning(f"File {processing_complete_file} is write protected!")
                except Exception as e:
                    logger.warning(f"Deletion error of file {processing_complete_file}: {e}")
            else:
                is_file_removed = True
        else:
            raise ValueError(f"Path {base_directory} does not exists!")

        return is_file_removed

    @staticmethod
    def _update_calculation_mode(circuit_calculation_mode: CalcModeEnum, dependent_study_data: StudyData) -> None:
        """Update the calculation mode of the dependent study according circuit_calculation_mode.

        if circuit_calculation_mode is new_mode the calculation mode of the dependent study data is set to new
        if circuit_calculation_mode is continues_mode the calculation mode of the dependent study data
        is set to continues_mode, if it's mode was not new_mode.

        :param calculation_mode_value: calculation mode from circuit study data
        :type  calculation_mode_value: CalcModeEnum
        :param dependent_study_data: Dependent study data
        :type  dependent_study_data: StudyData
        """
        # Check depending on study mode
        # new_mode of depending on study is kept independent of circuit study calculation mode
        if dependent_study_data.calculation_mode != CalcModeEnum.new_mode:
            # Circuit calculation mode new_mode forces new_mode for dependent study
            if circuit_calculation_mode == CalcModeEnum.new_mode:
                dependent_study_data.calculation_mode = CalcModeEnum.new_mode
            # Circuit calculation mode continue_mode request minimum continue_mode for dependent study
            elif circuit_calculation_mode == CalcModeEnum.continue_mode:
                dependent_study_data.calculation_mode = CalcModeEnum.continue_mode

    @staticmethod
    def _get_calculation_mode(calculation_mode_value: str) -> CalcModeEnum:
        """Provide the correspondent calculation mode based on the calculation mode value string.

        :param calculation_mode_value: Calculation mode value, which corresponds to a string
        :type  calculation_mode_value: str
        :return: Calculation mode
        :rtype: CalcModeEnum
        """
        # Variable declaration and initialization
        result_enum: CalcModeEnum

        if calculation_mode_value == CalcModeEnum.new_mode.value:
            result_enum = CalcModeEnum.new_mode
        elif calculation_mode_value == CalcModeEnum.continue_mode.value:
            result_enum = CalcModeEnum.continue_mode
        elif calculation_mode_value == CalcModeEnum.skip_mode.value:
            result_enum = CalcModeEnum.skip_mode
        else:
            raise ValueError(f"Enum value {calculation_mode_value} is invalid!\n"
                             f"only {CalcModeEnum.new_mode.value}, {CalcModeEnum.continue_mode.value}, "
                             f"and {CalcModeEnum.skip_mode.value} are valid!\n")

        return result_enum

    @staticmethod
    def get_initialization_queue_data(act_toml_prog_flow: tc.FlowControl) \
        -> tuple[list[ConfigurationDataEntryDto], list[srv_ctl_dtos.MagneticDataEntryDto], list[ConfigurationDataEntryDto],
                 list[srv_ctl_dtos.MagneticDataEntryDto], list[ConfigurationDataEntryDto],
                 list[ConfigurationDataEntryDto], list[SummaryDataEntryDto]]:
        """Initialize the lists of configuration data.

        :param act_toml_prog_flow: Flow control toml file (reference)
        :type act_toml_prog_flow: tc.FlowControl
        :return: List of configuration data for data transfer: circuit, transformer, inductor, heat sink and summary.
                 Each configuration gets one entry of the list.
        :rtype: list[ConfigurationDataEntryDto], list[srv_ctl_dtos.MagneticDataEntryDto], list[ConfigurationDataEntryDto],
                list[srv_ctl_dtos.MagneticDataEntryDto], list[ConfigurationDataEntryDto],
                list[ConfigurationDataEntryDto], list[ConfigurationDataEntryDto], list[SummaryDataEntryDto]
        """
        # Variable declaration and initialization

        # Initialize the statistical data
        progress_data_init: ProgressData = ProgressData(run_time=0, number_of_filtered_points=0,
                                                        progress_status=ProgressStatus.Idle)

        # Create data for the queue objects
        # Circuit
        circuit_list: list[srv_ctl_dtos.ConfigurationDataEntryDto] = []
        # Workaround until circuit_configuration_file is a list
        configuration_file_list = [act_toml_prog_flow.configuration_data_files.circuit_configuration_file]
        # Add entries per configuration
        for entry in configuration_file_list:
            circuit_main_entry = srv_ctl_dtos.ConfigurationDataEntryDto(
                configuration_name=entry,
                number_of_trials=act_toml_prog_flow.circuit.number_of_trials,
                progress_data=copy.deepcopy(progress_data_init))
            #  If optimization is skipped, set the status to 'skip'
            if act_toml_prog_flow.circuit.calculation_mode == CalcModeEnum.skip_mode.value:
                circuit_main_entry.progress_data.progress_status = ProgressStatus.Skipped
            # Overtake the list entries
            circuit_list.append(circuit_main_entry)

        # Inductor
        inductor_main_list: list[srv_ctl_dtos.MagneticDataEntryDto] = []
        # Inductor list (entry per configuration)
        inductor_list: list[srv_ctl_dtos.ConfigurationDataEntryDto] = []
        # Workaround until inductor_configuration_file is no list
        configuration_file_list = [act_toml_prog_flow.configuration_data_files.inductor_configuration_file]
        # Add entries per configuration
        for entry in configuration_file_list:
            # Update dependent magnetic lists (homepage 1)
            inductor_main_entry = srv_ctl_dtos.MagneticDataEntryDto(
                magnetic_configuration_name=entry,
                number_performed_calculations=0,
                number_calculations=0,
                progress_data=copy.deepcopy(progress_data_init))
            # Update dependent magnetic lists (homepage 2)
            inductor_data = srv_ctl_dtos.ConfigurationDataEntryDto(
                configuration_name=entry,
                number_of_trials=act_toml_prog_flow.inductor.number_of_trials,
                progress_data=copy.deepcopy(progress_data_init))
            #  If optimization is skipped, set the status to 'skip'
            if act_toml_prog_flow.inductor.calculation_mode == CalcModeEnum.skip_mode.value:
                inductor_main_entry.progress_data.progress_status = ProgressStatus.Skipped
                inductor_data.progress_data.progress_status = ProgressStatus.Skipped
            # Add elements to list
            inductor_main_list.append(inductor_main_entry)
            inductor_list.append(inductor_data)

        # Transformer
        transformer_main_list: list[srv_ctl_dtos.MagneticDataEntryDto] = []
        # Transformer data (List per configuration)
        transformer_list: list[srv_ctl_dtos.ConfigurationDataEntryDto] = []
        # Workaround until transformer_configuration_file is no list
        configuration_file_list = [act_toml_prog_flow.configuration_data_files.transformer_configuration_file]
        # Add entries per configuration
        for entry in configuration_file_list:
            # Update dependent magnetic lists (homepage 1)
            transformer_main_entry = srv_ctl_dtos.MagneticDataEntryDto(
                magnetic_configuration_name=entry,
                number_performed_calculations=0,
                number_calculations=0,
                progress_data=copy.deepcopy(progress_data_init))
            # Update dependent magnetic lists (homepage 2)
            transformer_data = srv_ctl_dtos.ConfigurationDataEntryDto(
                configuration_name=entry,
                number_of_trials=act_toml_prog_flow.transformer.number_of_trials,
                progress_data=copy.deepcopy(progress_data_init))
            #  If optimization is skipped, set the status to 'skip'
            if act_toml_prog_flow.transformer.calculation_mode == CalcModeEnum.skip_mode.value:
                transformer_main_entry.progress_data.progress_status = ProgressStatus.Skipped
                transformer_data.progress_data.progress_status = ProgressStatus.Skipped
            # Add elements to list
            transformer_main_list.append(transformer_main_entry)
            transformer_list.append(transformer_data)

        # Heat_sink data (List per configuration)
        heat_sink_list: list[srv_ctl_dtos.ConfigurationDataEntryDto] = []
        # Workaround until transformer_configuration_file is no list
        configuration_file_list = [act_toml_prog_flow.configuration_data_files.heat_sink_configuration_file]
        # Add entries per configuration
        for entry in configuration_file_list:
            heat_sink_data = srv_ctl_dtos.ConfigurationDataEntryDto(
                configuration_name=entry,
                number_of_trials=act_toml_prog_flow.heat_sink.number_of_trials,
                progress_data=copy.deepcopy(progress_data_init))
            #  If optimization is skipped, set the status to 'skip'
            if act_toml_prog_flow.heat_sink.calculation_mode == CalcModeEnum.skip_mode.value:
                heat_sink_data.progress_data.progress_status = ProgressStatus.Skipped
            heat_sink_list.append(heat_sink_data)

        # Summary data (List per circuit configuration)
        summary_list: list[srv_ctl_dtos.SummaryDataEntryDto] = []
        # Workaround until transformer_configuration_file is no list
        configuration_file_list = [act_toml_prog_flow.configuration_data_files.circuit_configuration_file]
        # Add entries per configuration
        for entry in configuration_file_list:
            summary_data = srv_ctl_dtos.SummaryDataEntryDto(
                configuration_name=entry,
                number_of_combinations=0,
                progress_data=copy.deepcopy(progress_data_init))
            summary_list.append(summary_data)

        # Return the queue data object
        return circuit_list, inductor_main_list, inductor_list, transformer_main_list, transformer_list, heat_sink_list, summary_list

    def _get_page_main_data(self) -> srv_ctl_dtos.QueueMainData:
        """Provide the actual statistical information of the optimization.

        :return: Data for the queue consisting of actual information about the circuit, heat sink and summary and processing times.
        :rtype: srv_ctl_dtos.QueueMainData
        """
        # Circuit
        # Check if circuit optimization is initialized,if not initialized progress data are valid
        if self._circuit_optimization is not None:
            # Check if circuit optimization is not skipped
            if self._circuit_list[0].progress_data.progress_status != ProgressStatus.Skipped:
                self._circuit_list[0].progress_data = self._circuit_optimization.get_progress_data()
        # Workaround for number of filtered points
        self._circuit_list[0].progress_data.number_of_filtered_points = len(self._filtered_list_files)
        # Inductor
        # Check if inductor is initialized,if not initialized progress data are valid
        if self._inductor_optimization is not None:
            self._inductor_main_list[0].number_performed_calculations = self._inductor_optimization.get_number_of_performed_calculations()
            self._inductor_main_list[0].progress_data.run_time = self._inductor_progress_time[0].get_runtime()
        # Transformer
        # Check if transformer is initialized,if not initialized progress data are valid
        if self._transformer_optimization is not None:
            self._transformer_main_list[0].number_performed_calculations = self._transformer_optimization.get_number_of_performed_calculations()
            self._transformer_main_list[0].progress_data.run_time = self._transformer_progress_time[0].get_runtime()

        # Heat sink
        # Check if heat sink is initialized,if not initialized progress data are valid
        if self._heat_sink_optimization is not None:
            self._heat_sink_list[0].progress_data = self._heat_sink_optimization.get_progress_data()

        # Summary data
        # Check if summary processing is initialized,if not initialized progress data are valid
        if self._summary_processing is not None:
            self._summary_list[0].progress_data = self._summary_processing.get_progress_data()

        # Assemble queue data
        queue_main_data_response: srv_ctl_dtos.QueueMainData \
            = srv_ctl_dtos.QueueMainData(circuit_list=self._circuit_list,
                                         inductor_main_list=self._inductor_main_list,
                                         transformer_main_list=self._transformer_main_list,
                                         heat_sink_list=self._heat_sink_list,
                                         summary_list=self._summary_list,
                                         total_process_time=self._total_time.get_runtime(),
                                         break_point_notification=self._break_point_message
                                         )

        return queue_main_data_response

    def _get_page_detail_data(self, filtered_pt_id: int) -> srv_ctl_dtos.QueueDetailData:
        """Provide the actual statistical information of the optimization.

        :param  filtered_pt_id: Index of the filtered circuit operation point
        :type   filtered_pt_id: int

        :return: Data for the queue consisting of actual information about the circuit, transformer, inductor, heat sink and summary
                 and processing times of a specific filtered circuit operation point.
        :rtype: srv_ctl_dtos.QueueMainData
        """
        # Circuit
        filtered_points_name_list: list[tuple[str, int] | Any] = []
        # Workaround: Convert the filtered file list
        entry_id = 0
        for entry in self._filtered_list_files:
            list_item = [entry, entry_id]
            entry_id = entry_id + 1
            filtered_points_name_list.append(list_item)

        # Check if circuit optimization is initialized,if not initialized progress data are valid
        if self._circuit_optimization is not None:
            # Check if circuit optimization is not skipped
            if self._circuit_list[0].progress_data.progress_status != ProgressStatus.Skipped:
                self._circuit_list[0].progress_data = self._circuit_optimization.get_progress_data()
        # Add remaining data
        circuit_data_entry: srv_ctl_dtos.CircuitConfigurationDataDto = srv_ctl_dtos.CircuitConfigurationDataDto(
            configuration_name=self._circuit_list[0].configuration_name,
            number_of_trials=self._circuit_list[0].number_of_trials,
            filtered_points_name_list=filtered_points_name_list,
            progress_data=self._circuit_list[0].progress_data
        )

        # Inductor
        # Check if inductor is initialized,if not initialized progress data are valid
        if self._inductor_optimization is not None:
            self._inductor_list[0].progress_data = self._inductor_optimization.get_progress_data(filtered_pt_id)
        elif self._inductor_list[0].progress_data.progress_status == ProgressStatus.Skipped:
            self._inductor_list[0].progress_data.number_of_filtered_points = self._inductor_number_filtered_analytic_points_skip_list[filtered_pt_id]
        # Transformer
        # Check if transformer is initialized,if not initialized progress data are valid
        if self._transformer_optimization is not None:
            self._transformer_list[0].progress_data = self._transformer_optimization.get_progress_data(filtered_pt_id)
        elif self._inductor_list[0].progress_data.progress_status == ProgressStatus.Skipped:
            self._transformer_list[0].progress_data.number_of_filtered_points = self._transformer_number_filtered_analytic_points_skip_list[filtered_pt_id]
        # Heat sink
        # Check if heat sink is initialized,if not initialized progress data are valid
        if self._heat_sink_optimization is not None:
            self._heat_sink_list[0].progress_data = self._heat_sink_optimization.get_progress_data()
        # Summary data
        # Check if summary processing is initialized,if not initialized progress data are valid
        if self._summary_processing is not None:
            self._summary_list[0].progress_data = self._summary_processing.get_progress_data()

        # Assemble queue data
        queue_detail_data_response: srv_ctl_dtos.QueueDetailData = (
            srv_ctl_dtos.QueueDetailData(circuit_data=circuit_data_entry,
                                         inductor_list=self._inductor_list,
                                         transformer_list=self._transformer_list,
                                         heat_sink_list=self._heat_sink_list,
                                         summary_data=self._summary_list[0],
                                         conf_process_time=self._total_time.get_runtime(),
                                         break_point_notification=self._break_point_message))

        return queue_detail_data_response

    def _request_pareto_front(self, pareto_source: ParetoFrontSource, c_configuration_index: int,
                              item_configuration_index: int, c_filtered_point_index: int) -> srv_ctl_dtos.QueueParetoFrontData:
        """Provide the actual statistical information of the optimization.

        :param  c_configuration_index: Index of the circuit configuration
        :type   c_configuration_index: int
        :param  pareto_source: Type of the Pareto front source
        :type   pareto_source: ParetoFrontSource (enumeration)
        :param  c_filtered_point_index: Index of the filtered circuit operation point
        :type   c_filtered_point_index: int

        :return: Html page of the requested pareto front
        :rtype:  str
        """
        # Variable declaration
        response_data = srv_ctl_dtos.QueueParetoFrontData(pareto_front_optuna="",
                                                          validity=False,
                                                          evaluation_info="Unexpected error!")

        # Circuit (Later to create a instance variable)
        filtered_points_name_list: list[tuple[str, int] | Any] = []
        # Workaround: Convert the filtered file list
        entry_id = 0
        for entry in self._filtered_list_files:
            list_item = [entry, entry_id]
            entry_id = entry_id + 1
            filtered_points_name_list.append(list_item)

        # Flag for available pareto file
        is_pareto_file_available: bool = False

        # Verify the input data
        if not c_configuration_index < len(self._circuit_list) or c_configuration_index < 0:
            response_data.evaluation_info = "Index of circuit configuration is invalid!"
            return response_data
        # Check if circuit  optimization is initialized, if not initialized progress data are valid
        if self._circuit_optimization is None:
            response_data.evaluation_info = "Warning: Circuit optimization is not initialized!"
            return response_data
        # Process the input data to find the Pareto front data
        if pareto_source == ParetoFrontSource.pareto_circuit:
            # Pareto front of circuit
            if self._circuit_list[c_configuration_index].progress_data.progress_status == ProgressStatus.InProgress:
                # Get Pareto front from memory
                response_data.evaluation_info = (
                    f"Circuit configuration file: {self._circuit_list[c_configuration_index].configuration_name}")
                response_data.pareto_front_optuna = self._circuit_optimization.get_actual_pareto_html()
            elif not self._circuit_list[c_configuration_index].progress_data.progress_status == ProgressStatus.Idle:
                # Get Pareto front from file
                if StudyData.check_study_data(CircuitOptimizationBase.circuit_study_data.optimization_directory,
                                              CircuitOptimizationBase.circuit_study_data.study_name):
                    response_data.evaluation_info = f"Circuit configuration: {self._circuit_list[c_configuration_index].configuration_name}"
                    response_data.pareto_front_optuna = self._circuit_optimization.get_pareto_html(
                        CircuitOptimizationBase.circuit_study_data.study_name,
                        os.path.join(CircuitOptimizationBase.circuit_study_data.optimization_directory,
                                     CircuitOptimizationBase.circuit_study_data.study_name+".sqlite3"))
            else:
                response_data.evaluation_info = "Pareto front calculation still not started!"
        # Pareto front of inductor
        elif pareto_source == ParetoFrontSource.pareto_inductor:
            # Verify the input data
            if not item_configuration_index < len(self._inductor_list) or item_configuration_index < 0:
                response_data.evaluation_info = "Index of circuit configuration is invalid!"
                return response_data
            else:
                # Check if inductor optimization is initialized, if not initialized progress data are valid
                if self._inductor_optimization is not None:
                    # Get progress data of selected filtered point
                    self._inductor_list[0].progress_data = self._inductor_optimization.get_progress_data(c_filtered_point_index)
                    if self._inductor_list[0].progress_data == 1:
                        # Get Pareto front from memory (Still not available. Femmt-update needed)
                        # response_data.evaluation_info = f"Inductor configuration: {self._inductor_list[i_configuration_index].conf_name}"
                        response_data.evaluation_info = "Pareto front calculation is started..."
                        # response_data.pareto_front_optuna = self.inductor_optimization.get_actual_pareto_html()
                    elif not self._inductor_list[0].progress_data.progress_status == ProgressStatus.Idle:
                        is_pareto_file_available = True
                if is_pareto_file_available or self._inductor_list[0].progress_data.progress_status == ProgressStatus.Skipped:
                    # Get Pareto front from file
                    # Assemble path name
                    sqlite_file_path = os.path.join(self._inductor_study_data.optimization_directory,
                                                    filtered_points_name_list[c_filtered_point_index][0],
                                                    self._inductor_study_data.study_name)
                    if StudyData.check_study_data(sqlite_file_path, self._inductor_study_data.study_name):
                        response_data.evaluation_info = (f"Inductor configuration file: {self._inductor_list[item_configuration_index].configuration_name}"
                                                         f" of filtered point: {filtered_points_name_list[c_filtered_point_index][0]}")
                        response_data.pareto_front_optuna = self._circuit_optimization.get_pareto_html(
                            self._inductor_study_data.study_name, os.path.join(sqlite_file_path, self._inductor_study_data.study_name + ".sqlite3"))
                else:
                    # Pareto front is not available
                    response_data.evaluation_info = "Pareto front calculation still not started!"
        # Pareto front of transformer
        elif pareto_source == ParetoFrontSource.pareto_transformer:
            # Verify the input data
            if not item_configuration_index < len(self._transformer_list) or item_configuration_index < 0:
                response_data.evaluation_info = "Index of circuit configuration is invalid!"
                return response_data
            else:
                # Check if transformer optimization is initialized,if not initialized progress data are valid
                if self._transformer_optimization is not None:
                    # Get progress data of selected filtered point
                    self._transformer_list[0].progress_data = self._transformer_optimization.get_progress_data(c_filtered_point_index)
                    if self._transformer_list[0].progress_data == 1:
                        # Get Pareto front from memory (Still not available. Femmt-update needed)
                        response_data.evaluation_info = "Pareto front calculation is started..."
                        # response_data.pareto_front_optuna = self.transformer_optimization.get_actual_pareto_html()
                    elif not self._transformer_list[0].progress_data.progress_status == ProgressStatus.Idle:
                        is_pareto_file_available = True
                if is_pareto_file_available or self._transformer_list[0].progress_data.progress_status == ProgressStatus.Skipped:
                    # Get Pareto front from file
                    # Assemble path name
                    sqlite_file_path = os.path.join(self._transformer_study_data.optimization_directory,
                                                    filtered_points_name_list[c_filtered_point_index][0],
                                                    self._transformer_study_data.study_name)
                    if StudyData.check_study_data(sqlite_file_path, self._transformer_study_data.study_name):
                        response_data.evaluation_info = (
                            f"Transformer configuration file: {self._transformer_list[item_configuration_index].configuration_name}"
                            f" of filtered point: {filtered_points_name_list[c_filtered_point_index][0]}"
                        )
                        response_data.pareto_front_optuna = self._circuit_optimization.get_pareto_html(
                            self._transformer_study_data.study_name, os.path.join(sqlite_file_path, self._transformer_study_data.study_name + ".sqlite3"))
                else:
                    # Pareto front is not available
                    response_data.evaluation_info = "Pareto front calculation still not started!"
        elif pareto_source == ParetoFrontSource.pareto_heat_sink:
            # Pareto front of heat sink
            # Verify the input data
            if not item_configuration_index < len(self._heat_sink_list) or item_configuration_index < 0:
                response_data.evaluation_info = "Index of circuit configuration is invalid!"
                return response_data
            else:
                # Check if heat sink optimization is initialized,if not initialized progress data are valid
                if self._heat_sink_optimization is not None:
                    # Get progress data of selected filtered point
                    self._heat_sink_list[item_configuration_index].progress_data = self._heat_sink_optimization.get_progress_data()
                    if self._transformer_list[item_configuration_index].progress_data == 1:
                        # Get Pareto front from memory (Still not available. Femmt-update needed)
                        response_data.evaluation_info = "Pareto front calculation is started..."
                        # response_data.pareto_front_optuna = self.transformer_optimization.get_actual_pareto_html()
                    elif not self._heat_sink_list[item_configuration_index].progress_data.progress_status == ProgressStatus.Idle:
                        is_pareto_file_available = True
                if is_pareto_file_available or self._heat_sink_list[item_configuration_index].progress_data.progress_status == ProgressStatus.Skipped:
                    # Get Pareto front from file
                    if StudyData.check_study_data(self._heat_sink_study_data.optimization_directory, self._heat_sink_study_data.study_name):
                        response_data.evaluation_info = f"Heat sink configuration file: {self._heat_sink_list[item_configuration_index].configuration_name}"
                        response_data.pareto_front_optuna = self._circuit_optimization.get_pareto_html(
                            self._heat_sink_study_data.study_name, os.path.join(self._heat_sink_study_data.optimization_directory,
                                                                                self._heat_sink_study_data.study_name + ".sqlite3"))
                else:
                    # Pareto front is not available
                    response_data.evaluation_info = "Pareto front calculation still not started!"
        elif pareto_source == ParetoFrontSource.pareto_summary:
            pass

        if not response_data.pareto_front_optuna == "":
            response_data.validity = True

        return response_data

    def _srv_response_queue(self, act_srv_request_queue: Queue, act_srv_response_queue: Queue) -> None:
        """Provide the response depending on the request.

        :param  act_srv_request_queue: Container (queue) for the request receives the requests of the server
        :type   filtered_pt_id: Queue
        :param  act_srv_response_queue: Container (queue) for the response to the server
        :type   act_srv_response_queue: Queue
        """
        while True:
            # Wait for the request
            request: ServerRequestData = act_srv_request_queue.get()
            # Evaluate request
            if request.request_cmd == RequestCmd.page_main:
                # Process request and send response
                srv_main_data = self._get_page_main_data()
                act_srv_response_queue.put(srv_main_data)
            elif request.request_cmd == RequestCmd.page_detail:
                # Process request and send response
                srv_detail_data = self._get_page_detail_data(request.c_filtered_point_index)
                act_srv_response_queue.put(srv_detail_data)
            elif request.request_cmd == RequestCmd.continue_opt:
                self._break_point_flag = False
                self._break_point_message = ""
                act_srv_response_queue.put(True)
            elif request.request_cmd == RequestCmd.pareto_front:
                pareto_front_data_queue = self._request_pareto_front(request.pareto_source, request.c_configuration_index,
                                                                     request.item_configuration_index, request.c_filtered_point_index)
                act_srv_response_queue.put(pareto_front_data_queue)

    def _get_key_inp(self) -> str:
        # Get the last key input with lock
        with self._key_input_lock:
            last_key_input_string = self._key_input_string
            self._key_input_string = ""
        return last_key_input_string

    def _key_input(self) -> None:
        """Provide the response depending on the request.

        :param  filtered_pt_id: Index of the filtered circuit operation point
        :type   filtered_pt_id: int

        :return: Queue with data consisting of the requested information and processing times of a specific filtered circuit operation point.
        :rtype: Queue
        """
        while True:
            # Wait for the request
            key_input_string: str = input()
            # Store the input
            with self._key_input_lock:
                self._key_input_string = key_input_string

    def run_optimization_from_toml_configurations(self, workspace_path: str) -> None:
        """Perform the main program.

        This function corresponds to 'main', which is called after the instance of the class are created.

        :param  workspace_path: Path to subfolder 'workspace' (if empty default path '../<path to this file>' is used)
        :type   workspace_path: str
        """
        # Server response thread handler
        _srv_response_handler = None
        # Queues for request and response
        srv_request_queue: Queue = Queue()
        srv_response_queue: Queue = Queue()
        # Flag to control the svr_response_thread
        srv_response_stop_flag = True

        # Initialize start time
        self._total_time = RunTime()

        # Start runtime measurement for the optimization
        self._total_time.reset_start_trigger()

        # Server class to control the workflow
        srv_ctl = ServerCtl()

        # Check if workspace path is not provided by argument
        if workspace_path == "":
            # Find process workspace
            workspace_path = os.path.dirname(os.path.abspath(__file__))
            # Join parent folder of workspace_path and workspace path to absolute path name
            workspace_path = os.path.join(os.path.dirname(workspace_path), "workspace")

        # Set directory to workspace path
        try:
            # Change to workspace
            os.chdir(workspace_path)
        except FileNotFoundError as exc:
            raise ValueError("Error: Workspace folder does not exists!") from exc
        except PermissionError as exc:
            raise ValueError("Error: No permission to change the folder!") from exc

        # Create absolute path
        workspace_path = os.path.abspath(workspace_path)

        # --------------------------
        # Logging
        # --------------------------
        # read logging for submodules
        logging_filename = os.path.join(workspace_path, "logging.conf")
        self.load_generate_logging_config(logging_filename)

        # --------------------------
        # Debug
        # --------------------------
        debug_toml_filepath = os.path.join(workspace_path, "debug_config.toml")
        is_debug_loaded, debug_dict = self.load_toml_file(debug_toml_filepath)

        if is_debug_loaded:
            print("debug.toml config found.")
            logger.info("debug.toml config found.")
            toml_debug = tc.Debug(**debug_dict)
        else:
            toml_debug = tc.Debug(general=tc.DebugGeneral(is_debug=False),
                                  capacitor_1=tc.DebugCapacitor(number_working_point_max=1),
                                  capacitor_2=tc.DebugCapacitor(number_working_point_max=1),
                                  inductor=tc.DebugInductor(number_reluctance_working_point_max=1,
                                                            number_fem_working_point_max=1),
                                  transformer=tc.DebugTransformer(number_reluctance_working_point_max=1,
                                                                  number_fem_working_point_max=1))

        # --------------------------
        # Flow control
        # --------------------------
        logger.debug("Read flow control file")
        # Load the configuration for program flow and check the validity
        file_path = os.path.join(workspace_path, "progFlow.toml")
        flow_control_loaded, dict_prog_flow = self.load_toml_file(file_path)

        if not flow_control_loaded:
            # Set default topology
            self._circuit_optimization = dct.topology.dab.DabCircuitOptimization()
            # Generate topology dependent default files
            toml_gen.generate_default_flow_control_toml(workspace_path)
            file_path = os.path.join(workspace_path, "DabGeneralConf.toml")
            self._circuit_optimization.generate_general_toml(file_path)
            file_path = os.path.join(workspace_path, "DabCircuitConf.toml")
            self._circuit_optimization.generate_circuit_toml(file_path)
            # Generate missing component default files
            toml_gen.generate_missing_toml_files(workspace_path)
            raise ValueError(f"Program flow toml file does not exist in path {workspace_path}.\n"
                             "A default program flow toml file and corresponding default configuration files are generated.\n"
                             "All generated default files needs to be updated before using.")

        # Verify toml data and transfer to class
        toml_prog_flow = tc.FlowControl(**dict_prog_flow)

        # Add absolute path to project data path
        toml_prog_flow.general.project_directory = os.path.join(workspace_path, toml_prog_flow.general.project_directory)

        self.set_up_folder_structure(toml_prog_flow)

        DctMainCtl.log_software_versions(os.path.join(os.path.abspath(toml_prog_flow.general.project_directory), "software_versions.txt"))

        # -----------------------------------------
        # Introduce study data and filter data DTOs
        # -----------------------------------------

        project_directory = os.path.abspath(toml_prog_flow.general.project_directory)

        self._capacitor_selection_data = StudyData(
            study_name=toml_prog_flow.configuration_data_files.capacitor_1_configuration_file.replace(".toml", ""),
            optimization_directory=os.path.join(project_directory, toml_prog_flow.capacitor_1.subdirectory,
                                                toml_prog_flow.configuration_data_files.circuit_configuration_file.replace(".toml", "")),
            calculation_mode=DctMainCtl._get_calculation_mode(toml_prog_flow.capacitor_1.calculation_mode)
        )

        self._capacitor_2_selection_data = StudyData(
            study_name=toml_prog_flow.configuration_data_files.capacitor_2_configuration_file.replace(".toml", ""),
            optimization_directory=os.path.join(project_directory, toml_prog_flow.capacitor_2.subdirectory,
                                                toml_prog_flow.configuration_data_files.circuit_configuration_file.replace(".toml", "")),
            calculation_mode=DctMainCtl._get_calculation_mode(toml_prog_flow.capacitor_2.calculation_mode)
        )

        self._inductor_study_data = StudyData(
            study_name=toml_prog_flow.configuration_data_files.inductor_configuration_file.replace(".toml", ""),
            optimization_directory=os.path.join(project_directory, toml_prog_flow.inductor.subdirectory,
                                                toml_prog_flow.configuration_data_files.circuit_configuration_file.replace(".toml", "")),
            calculation_mode=DctMainCtl._get_calculation_mode(toml_prog_flow.inductor.calculation_mode)
        )
        self._transformer_study_data = StudyData(
            study_name=toml_prog_flow.configuration_data_files.transformer_configuration_file.replace(".toml", ""),
            optimization_directory=os.path.join(project_directory, toml_prog_flow.transformer.subdirectory,
                                                toml_prog_flow.configuration_data_files.circuit_configuration_file.replace(".toml", "")),
            calculation_mode=DctMainCtl._get_calculation_mode(toml_prog_flow.transformer.calculation_mode)
        )
        self._heat_sink_study_data = StudyData(
            study_name=toml_prog_flow.configuration_data_files.heat_sink_configuration_file.replace(".toml", ""),
            optimization_directory=os.path.join(project_directory, toml_prog_flow.heat_sink.subdirectory,
                                                toml_prog_flow.configuration_data_files.heat_sink_configuration_file.replace(".toml", "")),
            calculation_mode=DctMainCtl._get_calculation_mode(toml_prog_flow.heat_sink.calculation_mode)
        )

        pre_summary_data = StudyData(study_name="pre_summary",
                                     optimization_directory=os.path.join(project_directory, toml_prog_flow.pre_summary.subdirectory),
                                     calculation_mode=DctMainCtl._get_calculation_mode(toml_prog_flow.pre_summary.calculation_mode))

        summary_data = StudyData(study_name="summary", optimization_directory=os.path.join(project_directory, toml_prog_flow.summary.subdirectory),
                                 calculation_mode=DctMainCtl._get_calculation_mode(toml_prog_flow.summary.calculation_mode))

        # Initialize the data for server monitoring (Only 1 circuit configuration is used, later to change)
        (self._circuit_list, self._inductor_main_list, self._inductor_list, self._transformer_main_list,
         self._transformer_list, self._heat_sink_list, self._summary_list) = self.get_initialization_queue_data(toml_prog_flow)

        # --------------------------
        # Select topology
        # --------------------------

        # Allocate and initialize circuit configuration
        if toml_prog_flow.general.topology == TopologyEnum.dab.value:
            self._circuit_optimization = dct.topology.dab.DabCircuitOptimization()
        elif toml_prog_flow.general.topology == TopologyEnum.sbc.value:
            self._circuit_optimization = dct.topology.sbc.SbcCircuitOptimization()
        else:
            raise ValueError("Serious programming error in topology selection. Please write an issue!")

        # --------------------------
        # General toml control
        # --------------------------
        logger.debug("Read general toml control")

        # Init general configuration
        is_general_toml_loaded, dict_general_toml = self.load_toml_file(toml_prog_flow.configuration_data_files.general_configuration_file)

        if not is_general_toml_loaded:
            file_path = os.path.join(workspace_path, toml_prog_flow.configuration_data_files.general_configuration_file)
            self._circuit_optimization.generate_general_toml(file_path)
            raise ValueError(f"General toml configuration file: {file_path} does not exist.\n"
                             f"A default file is generated and needs to be updated!")

        is_general_toml_consistent, general_issue_report = self._circuit_optimization.load_and_verify_general_parameters(dict_general_toml)

        if not is_general_toml_consistent:
            raise ValueError("General parameter in file ",
                             f"{toml_prog_flow.configuration_data_files.general_configuration_file} are inconsistent!\n", general_issue_report)

        # --------------------------
        # Circuit flow control
        # --------------------------
        logger.debug("Read circuit flow control")

        # Init study and path information
        self._circuit_optimization.init_study_information(
            toml_prog_flow.configuration_data_files.circuit_configuration_file.replace(".toml", ""),
            project_directory, toml_prog_flow.circuit.subdirectory,
            DctMainCtl._get_calculation_mode(toml_prog_flow.circuit.calculation_mode))

        # Init circuit configuration
        is_circuit_loaded, dict_circuit = DctMainCtl.load_toml_file(toml_prog_flow.configuration_data_files.circuit_configuration_file)

        if not is_circuit_loaded:
            file_path = os.path.join(workspace_path, toml_prog_flow.configuration_data_files.circuit_configuration_file)
            self._circuit_optimization.generate_general_toml(file_path)
            raise ValueError(f"Circuit configuration file: {file_path} does not exist.\n"
                             f"A default file is generated and needs to be updated!")

        is_consistent, issue_report = self._circuit_optimization.load_and_verify_circuit_parameters(dict_circuit)

        # Verify circuit parameters
        if not is_consistent:
            raise ValueError("Circuit optimization parameter in file ",
                             f"{toml_prog_flow.configuration_data_files.circuit_configuration_file} are inconsistent!\n", issue_report)

        # Check, if electrical optimization is to skip
        if self._circuit_optimization.circuit_study_data.calculation_mode == CalcModeEnum.skip_mode:
            # Check completion of process
            is_skippable, issue_report = DctMainCtl._is_skippable(self._circuit_optimization.circuit_study_data,
                                                                  PROCESSING_COMPLETE_FILE, True, [])
            # Evaluate the result of completion check
            if is_skippable:
                # Check topology dependent skip reason
                is_skippable, issue_report = self._circuit_optimization.is_circuit_optimization_skippable()

            # Evaluate the result of circuit check
            if not is_skippable:
                raise ValueError("Circuit optimization is not skippable:\n" + f"{issue_report}")
        else:
            circuit_calculation_mode = self._circuit_optimization.circuit_study_data.calculation_mode
            # If circuit calculation mode is not skipped, all further calculation modes are impacted
            DctMainCtl._update_calculation_mode(circuit_calculation_mode, self._capacitor_selection_data)
            DctMainCtl._update_calculation_mode(circuit_calculation_mode, self._capacitor_2_selection_data)
            DctMainCtl._update_calculation_mode(circuit_calculation_mode, self._inductor_study_data)
            DctMainCtl._update_calculation_mode(circuit_calculation_mode, self._transformer_study_data)

        # In case of CalcModeEnum.new_mode the old study is to delete
        if self._circuit_optimization.circuit_study_data.calculation_mode == CalcModeEnum.new_mode:
            # delete old circuit study data
            self.delete_study_content(self._circuit_optimization.circuit_study_data.optimization_directory,
                                      self._circuit_optimization.circuit_study_data.study_name)
            # Create the deleted filtered result folder
            os.makedirs(self._circuit_optimization.filter_data.filtered_list_pathname, exist_ok=True)

        # --------------------------
        # Capacitor 1 flow control
        # --------------------------
        logger.debug("Read capacitor 1 flow control")

        # Check if capacitor components are required
        if self._circuit_optimization.get_number_of_required_capacitors() > 0:
            # Init capacitor configuration
            is_capacitor_1_loaded, dict_capacitor_1 = self.load_toml_file(toml_prog_flow.configuration_data_files.capacitor_1_configuration_file)

            if not is_capacitor_1_loaded:
                file_path = os.path.join(workspace_path, toml_prog_flow.configuration_data_files.capacitor_1_configuration_file)
                toml_gen.generate_default_capacitor_toml(file_path)
                raise ValueError(f"General toml configuration file: {file_path} does not exist.\n"
                                 f"A default file is generated and needs to be updated!")

            # Verify toml data and transfer to class
            toml_capacitor_1 = tc.TomlCapacitorSelection(**dict_capacitor_1)

            if not is_general_toml_loaded:
                file_path = os.path.join(workspace_path,
                                         toml_prog_flow.configuration_data_files.general_configuration_file)
                self._circuit_optimization.generate_general_toml(file_path)
                raise ValueError(f"General toml configuration file: {file_path} does not exist.")

            is_general_toml_consistent, general_issue_report = self._circuit_optimization.load_and_verify_general_parameters(
                dict_general_toml)

            if not is_general_toml_consistent:
                raise ValueError("General parameter in file ",
                                 f"{toml_prog_flow.configuration_data_files.general_configuration_file} are inconsistent!\n",
                                 general_issue_report)

            if not is_capacitor_1_loaded:
                raise ValueError(f"Capacitor 1 configuration file: {toml_prog_flow.configuration_data_files.capacitor_1_configuration_file} does not exist.")

            # Verify capacitor parameters
            is_consistent, issue_report = dct.CapacitorSelection.verify_optimization_parameter(toml_capacitor_1)
            if not is_consistent:
                raise ValueError("Capacitor optimization parameter in file ",
                                 f"{toml_prog_flow.configuration_data_files.capacitor_1_configuration_file} are inconsistent!\n", issue_report)

            # Check, if capacitor1 selection is to skip
            if self._capacitor_selection_data.calculation_mode == CalcModeEnum.skip_mode:
                # Check if capacitor1 selection is skippable
                is_skippable, issue_report = DctMainCtl._is_skippable(self._capacitor_selection_data, PROCESSING_COMPLETE_FILE)
                # Evaluate the result of circuit check
                if not is_skippable:
                    logger.warning("Capacitor 1 selection is not skippable:\n" + f"{issue_report}")
                    self._capacitor_selection_data.calculation_mode = CalcModeEnum.continue_mode

            # In case of CalcModeEnum.new_mode the old study is to delete
            if self._capacitor_selection_data.calculation_mode == CalcModeEnum.new_mode:
                # delete old circuit study data
                self.delete_study_content(self._capacitor_selection_data.optimization_directory, self._capacitor_selection_data.study_name)
        else:
            # Skip capacitor optimization
            self._capacitor_selection_data.calculation_mode = CalcModeEnum.skip_mode

        # --------------------------
        # Capacitor 2 flow control
        # --------------------------
        logger.debug("Read capacitor 2 flow control")

        # Check if capacitor components are required
        if self._circuit_optimization.get_number_of_required_capacitors() > 0:
            # Init circuit configuration
            is_capacitor_2_loaded, dict_capacitor_2 = self.load_toml_file(toml_prog_flow.configuration_data_files.capacitor_2_configuration_file)

            if not is_capacitor_2_loaded:
                file_path = os.path.join(workspace_path, toml_prog_flow.configuration_data_files.capacitor_2_configuration_file)
                toml_gen.generate_default_capacitor_toml(file_path)
                raise ValueError(f"General toml configuration file: {file_path} does not exist.\n"
                                 f"A default file is generated and needs to be updated!")

            # Verify toml data and transfer to class
            toml_capacitor_2 = tc.TomlCapacitorSelection(**dict_capacitor_2)

            # Verify capacitor parameters
            is_consistent, issue_report = dct.CapacitorSelection.verify_optimization_parameter(toml_capacitor_2)
            if not is_consistent:
                raise ValueError("Capacitor optimization parameter in file ",
                                 f"{toml_prog_flow.configuration_data_files.capacitor_2_configuration_file} are inconsistent!\n", issue_report)

            # Check, if capacitor2 selection is to skip
            if self._capacitor_2_selection_data.calculation_mode == CalcModeEnum.skip_mode:
                # Check if capacitor1 selection is skippable
                is_skippable, issue_report = DctMainCtl._is_skippable(self._capacitor_2_selection_data, PROCESSING_COMPLETE_FILE)
                # Evaluate the result of circuit check
                if not is_skippable:
                    logger.warning("Capacitor 2 selection is not skippable:\n" + f"{issue_report}")
                    self._capacitor_2_selection_data.calculation_mode = CalcModeEnum.continue_mode

            # In case of CalcModeEnum.new_mode the old study is to delete
            if self._capacitor_2_selection_data.calculation_mode == CalcModeEnum.new_mode:
                # delete old circuit study data
                self.delete_study_content(self._capacitor_2_selection_data.optimization_directory, self._capacitor_2_selection_data.study_name)
        else:
            # Skip capacitor optimization
            self._capacitor_2_selection_data.calculation_mode = CalcModeEnum.skip_mode

        # --------------------------
        # Inductor flow control
        # --------------------------
        logger.debug("Read inductor flow control")

        # Check if inductor components are required
        if self._circuit_optimization.get_number_of_required_inductors() > 0:
            # Load the inductor-configuration parameter
            inductor_toml_filepath = toml_prog_flow.configuration_data_files.inductor_configuration_file
            is_inductor_loaded, inductor_dict = self.load_toml_file(toml_prog_flow.configuration_data_files.inductor_configuration_file)

            if not is_inductor_loaded:
                file_path = os.path.join(workspace_path, toml_prog_flow.configuration_data_files.inductor_configuration_file)
                toml_gen.generate_default_inductor_toml(file_path)
                raise ValueError(f"Inductor toml configuration file: {file_path} does not exist.\n"
                                 f"A default file is generated and needs to be updated!")

            # Verify toml data and transfer to class
            toml_inductor = dct.TomlInductor(**inductor_dict)

            # Verify optimization parameter
            is_consistent, issue_report = InductorOptimization.verify_optimization_parameter(toml_inductor)
            if not is_consistent:
                raise ValueError("Inductor optimization parameter in file ",
                                 f"{toml_prog_flow.configuration_data_files.inductor_configuration_file} are inconsistent!\n", issue_report)

            # Overtake the calculation mode
            inductor_sim_calculation_mode: CalcModeEnum = self._inductor_study_data.calculation_mode
            # Check, if inductor optimization is to skip
            if self._inductor_study_data.calculation_mode == CalcModeEnum.skip_mode:
                # Initialize _inductor_number_filtered_points_skip_list
                self._inductor_number_filtered_analytic_points_skip_list = []
                # Check if the optimization is skippable for analytic calculation
                is_skippable, issue_report = DctMainCtl._is_skippable(self._inductor_study_data, RELUCTANCE_COMPLETE_FILE, True,
                                                                      self._circuit_optimization.filter_data.filtered_list_files)
                # Evaluate if the optimization is skippable for analytic calculation
                if not is_skippable:
                    self._inductor_study_data.calculation_mode = CalcModeEnum.new_mode
                    inductor_sim_calculation_mode = CalcModeEnum.continue_mode
                    self._capacitor_selection_data.calculation_mode = CalcModeEnum.continue_mode
                    logger.warning("Inductor optimization (analytic and simulation part) are not skippable:\n"
                                   f"{issue_report}")
                else:
                    # Check if the optimization is skippable for simulation calculation
                    is_skippable, issue_report = DctMainCtl._is_skippable(self._inductor_study_data, SIMULATION_COMPLETE_FILE, True,
                                                                          self._circuit_optimization.filter_data.filtered_list_files)
                    # Evaluate if the optimization is skippable for simulation calculation
                    if not is_skippable:
                        logger.warning("Inductor optimization (simulation part) is not skippable:\n"
                                       f"{issue_report}")
                        inductor_sim_calculation_mode = CalcModeEnum.continue_mode

            # In case of CalcModeEnum.new_mode the old study is to delete
            if self._inductor_study_data.calculation_mode == CalcModeEnum.new_mode:
                # Delete old inductor study
                self.delete_study_content(self._inductor_study_data.optimization_directory)
        else:
            # Skip inductor optimization
            self._inductor_study_data.calculation_mode = CalcModeEnum.skip_mode

        # --------------------------
        # Transformer flow control
        # --------------------------
        logger.debug("Read transformer flow control")

        # Check if transformer components are required
        if self._circuit_optimization.get_number_of_required_transformers() > 0:
            # Load the transformer-configuration parameter
            transformer_toml_filepath = toml_prog_flow.configuration_data_files.transformer_configuration_file
            is_transformer_loaded, transformer_dict = self.load_toml_file(toml_prog_flow.configuration_data_files.transformer_configuration_file)
            toml_transformer = dct.TomlTransformer(**transformer_dict)

            if not is_transformer_loaded:
                file_path = os.path.join(workspace_path, toml_prog_flow.configuration_data_files.transformer_configuration_file)
                toml_gen.generate_default_transformer_toml(file_path)
                raise ValueError(f"Transformer toml configuration file: {file_path} does not exist.\n"
                                 f"A default file is generated and needs to be updated!")

            # Verify optimization parameter
            is_consistent, issue_report = TransformerOptimization.verify_optimization_parameter(toml_transformer)
            if not is_consistent:
                raise ValueError("Transformer optimization parameter in file ",
                                 f"{toml_prog_flow.configuration_data_files.transformer_configuration_file} are inconsistent!\n", issue_report)

            # Overtake the calculation mode
            transformer_sim_calculation_mode: CalcModeEnum = self._transformer_study_data.calculation_mode
            # Check, if transformer optimization is to skip
            if self._transformer_study_data.calculation_mode == CalcModeEnum.skip_mode:
                # Initialize _transformer_number_filtered_points_skip_list
                self._transformer_number_filtered_analytic_points_skip_list = []

                # Check if the optimization is skippable for simulation calculation
                is_skippable, issue_report = DctMainCtl._is_skippable(self._transformer_study_data, RELUCTANCE_COMPLETE_FILE, True,
                                                                      self._circuit_optimization.filter_data.filtered_list_files)
                # Evaluate if the optimization is skippable for analytic calculation
                if not is_skippable:
                    self._transformer_study_data.calculation_mode = CalcModeEnum.new_mode
                    transformer_sim_calculation_mode = CalcModeEnum.continue_mode
                    logger.warning("Transformer optimization (analytic and simulation part) are not skippable:\n"
                                   f"{issue_report}")
                else:
                    # Check if the optimization is skippable for simulation calculation
                    is_skippable, issue_report = DctMainCtl._is_skippable(self._transformer_study_data, SIMULATION_COMPLETE_FILE,
                                                                          True,
                                                                          self._circuit_optimization.filter_data.filtered_list_files)
                    # Evaluate if the optimization is skippable for simulation calculation
                    if not is_skippable:
                        transformer_sim_calculation_mode = CalcModeEnum.continue_mode
                        logger.warning("Transformer optimization (simulation part) is not skippable:\n"
                                       f"{issue_report}")

            # In case of CalcModeEnum.new_mode the old study is to delete
            if self._transformer_study_data.calculation_mode == CalcModeEnum.new_mode:
                # Delete old transformer study
                self.delete_study_content(self._transformer_study_data.optimization_directory)
        else:
            # Skip transformer optimization
            self._transformer_study_data.calculation_mode = CalcModeEnum.skip_mode
        # --------------------------
        # Heat sink flow control
        # --------------------------

        heat_sink_toml_filepath = toml_prog_flow.configuration_data_files.heat_sink_configuration_file
        is_heat_sink_loaded, heat_sink_dict = self.load_toml_file(heat_sink_toml_filepath)
        toml_heat_sink = dct.TomlHeatSink(**heat_sink_dict)
        if not is_heat_sink_loaded:
            file_path = os.path.join(workspace_path, toml_prog_flow.configuration_data_files.heat_sink_configuration_file)
            toml_gen.generate_default_heat_sink_toml(file_path)
            raise ValueError(f"Transformer toml configuration file: {file_path} does not exist.\n"
                             f"A default file is generated and needs to be updated!")
        # Verify optimization parameter
        is_consistent, issue_report = dct.HeatSinkOptimization.verify_optimization_parameter(toml_heat_sink)
        if not is_consistent:
            raise ValueError("Heat sink optimization parameter in file "
                             f"{toml_prog_flow.configuration_data_files.heat_sink_configuration_file} are inconsistent!\n", issue_report)

        # Check, if heat sink optimization is to skip
        if self._heat_sink_study_data.calculation_mode == CalcModeEnum.skip_mode:
            # Check if the optimization is skippable
            is_skippable, issue_report = DctMainCtl._is_skippable(self._heat_sink_study_data,
                                                                  PROCESSING_COMPLETE_FILE, True, [])

            # Evaluate if the optimization is skippable
            if not is_skippable:
                self._heat_sink_study_data.calculation_mode = CalcModeEnum.continue_mode
                logger.warning("heat sink optimization is not skippable:\n"
                               f"{issue_report}")

        # In case of CalcModeEnum.new_mode the old study is to delete
        if self._heat_sink_study_data.calculation_mode == CalcModeEnum.new_mode:
            # Delete old heat sink study
            self.delete_study_content(self._heat_sink_study_data.optimization_directory, self._heat_sink_study_data.study_name)

        # -- Start server  --------------------------------------------------------------------------------------------

        # Initialize the runtime timer
        self._circuit_progress_time = [RunTime()]
        self._inductor_progress_time = [RunTime()]
        self._transformer_progress_time = [RunTime()]
        self._heat_sink_progress_time = [RunTime()]
        self._summary_progress_time = [RunTime()]

        # Start the data exchange queue thread
        srv_response_stop_flag = False
        # _srv_response_handler = threading.Thread(target=self._srv_response_queue,
        #                                         args=(srv_request_queue, srv_response_queue), daemon=True)

        # _srv_response_handler.start()

        # Start the server
        # srv_ctl.start_dct_server(srv_request_queue, srv_response_queue, True)

        # Initialize key input handler
        # self._key_input_handler = threading.Thread(target=self._key_input,
        #                                            args=(srv_request_queue, srv_response_queue), daemon=True)

        # -- Start optimization  ----------------------------------------------------------------------------------------

        # Initialization thermal data
        if not self._circuit_optimization.init_thermal_circuit_configuration(toml_heat_sink):
            raise ValueError("Thermal data configuration not initialized!")
        print("dctmainctl")
        print(f"{self._circuit_optimization=}")
        print(f"{self._circuit_optimization.transistor_b1_cooling=}")

        # --------------------------
        # Circuit optimization
        # --------------------------
        logger.info("Start circuit optimization.")

        # Check, if electrical optimization is not to skip
        if not self._circuit_optimization.circuit_study_data.calculation_mode == CalcModeEnum.skip_mode:

            # Initialize circuit configuration
            self._circuit_optimization.initialize_circuit_optimization()

            # Delete processing complete indicator
            DctMainCtl._delete_processing_complete(self._circuit_optimization.circuit_study_data.optimization_directory,
                                                   PROCESSING_COMPLETE_FILE)
            # Perform circuit optimization
            self._circuit_optimization.start_proceed_study(number_trials=toml_prog_flow.circuit.number_of_trials)

        # Check breakpoint
        self.check_breakpoint(toml_prog_flow.breakpoints.circuit_pareto, "Electric Pareto front calculated")

        # Check, if electrical optimization is not to skip
        if not self._circuit_optimization.circuit_study_data.calculation_mode == CalcModeEnum.skip_mode:
            # Check if _circuit_optimization is not allocated, what corresponds to a serious programming error
            # Error is to prevent by the workflow
            if self._circuit_optimization is None:
                raise ValueError("Serious programming error '_circuit_optimization allocation failure'. Please write an issue!")

            # Calculate the filtered results
            is_filter_data_available, issue_report = self._circuit_optimization.filter_study_results()

            # Evaluate the filtered data result (Program stop has to be removed, if multiple circuit configuration are optimized)
            if not is_filter_data_available:
                raise ValueError("Filtered data error:"+issue_report)

            # Set processing complete indicator
            DctMainCtl._set_processing_complete(self._circuit_optimization.circuit_study_data.optimization_directory,
                                                FILTERED_RESULTS_PATH, PROCESSING_COMPLETE_FILE)

            # Workaround: Set filtered result id list here, later to handle in circuit_optimization
            self._filtered_list_files = self._circuit_optimization.filter_data.filtered_list_files

        # Set the number of calculations for the magnetic components
        self._inductor_main_list[0].number_calculations = len(self._filtered_list_files)
        self._transformer_main_list[0].number_calculations = len(self._filtered_list_files)

        # Check breakpoint
        self.check_breakpoint(toml_prog_flow.breakpoints.circuit_filtered, "Filtered value of electric Pareto front calculated")

        # --------------------------
        # Capacitor 1 selection
        # --------------------------
        logger.info("Start capacitor 1 selection")

        # Check, if capacitor optimization is not to skip
        if not self._capacitor_selection_data.calculation_mode == CalcModeEnum.skip_mode:
            # Allocate and initialize circuit configuration
            self._capacitor_selection = CapacitorSelection()

            capacitor_requirements_list = self._circuit_optimization.get_capacitor_requirements()

            self._capacitor_selection.initialize_capacitor_selection([toml_capacitor_1, toml_capacitor_2],
                                                                     capacitor_study_data=self._capacitor_selection_data,
                                                                     capacitor_requirements_list=capacitor_requirements_list)

            # Check, if old study is to delete, if available
            if self._capacitor_selection_data.calculation_mode == CalcModeEnum.new_mode:
                # delete old circuit study data
                self.delete_study_content(self._capacitor_selection_data.optimization_directory, self._capacitor_selection_data.study_name)

            # Delete processing complete indicator
            DctMainCtl._delete_processing_complete(self._capacitor_selection_data.optimization_directory, PROCESSING_COMPLETE_FILE)
            # Perform circuit optimization
            self._capacitor_selection.optimization_handler(filter_data=self._circuit_optimization.filter_data,
                                                           capacitor_requirements_list=capacitor_requirements_list,
                                                           debug=toml_debug)
            # Set processing complete indicator
            design_directory = os.path.join(self._capacitor_selection_data.study_name, CIRCUIT_CAPACITOR_LOSS_FOLDER)
            DctMainCtl._set_processing_complete(self._capacitor_selection_data.optimization_directory,
                                                design_directory, PROCESSING_COMPLETE_FILE,
                                                self._circuit_optimization.filter_data.filtered_list_files)

        # Check breakpoint
        self.check_breakpoint(toml_prog_flow.breakpoints.capacitor_1, "Capacitor 1 Pareto front calculated")

        # --------------------------
        # Inductor reluctance model optimization
        # --------------------------
        logger.info("Start inductor reluctance model optimization.")

        # Start the inductor processing time measurement
        self._inductor_progress_time[0].reset_start_trigger()

        # Allocate and initialize inductor configuration
        self._inductor_optimization = InductorOptimization()

        # Check, if inductor optimization is not to skip
        if not self._inductor_study_data.calculation_mode == CalcModeEnum.skip_mode:
            # Read requirement list and initialize inductor optimization
            inductor_requirements_list = self._circuit_optimization.get_inductor_requirements()
            self._inductor_optimization.initialize_inductor_optimization_list(toml_inductor, self._inductor_study_data, inductor_requirements_list)

            # Set the status to InProgress
            self._inductor_main_list[0].progress_data.progress_status = ProgressStatus.InProgress

            # Check, if inductor optimization is not to skip (cannot be skipped if circuit calculation mode is new)
            if not self._inductor_study_data.calculation_mode == CalcModeEnum.skip_mode:
                # Delete processing complete indicator
                DctMainCtl._delete_processing_complete(self._inductor_study_data.optimization_directory, RELUCTANCE_COMPLETE_FILE)
                # Perform inductor optimization
                self._inductor_optimization.optimization_handler_reluctance_model(
                    self._circuit_optimization.filter_data, toml_prog_flow.inductor.number_of_trials,
                    toml_inductor.filter_distance.factor_dc_losses_min_max_list, debug=toml_debug)
                # Set processing complete indicator
                design_directory = os.path.join(self._inductor_study_data.study_name, CIRCUIT_INDUCTOR_RELUCTANCE_LOSSES_FOLDER)
                DctMainCtl._set_processing_complete(self._inductor_study_data.optimization_directory,
                                                    design_directory, RELUCTANCE_COMPLETE_FILE,
                                                    self._circuit_optimization.filter_data.filtered_list_files)

            # Set the status to Done
            self._inductor_main_list[0].progress_data.progress_status = ProgressStatus.Done

            # Stop the inductor processing time measurement
            self._inductor_progress_time[0].stop_trigger()

        # Check breakpoint
        self.check_breakpoint(toml_prog_flow.breakpoints.inductor, "Inductor reluctance model Pareto front calculated")

        # --------------------------
        # Transformer reluctance model optimization
        # --------------------------
        logger.info("Start transformer reluctance model optimization.")

        # Start the transformer processing time measurement
        self._transformer_progress_time[0].reset_start_trigger()

        # Allocate and initialize transformer configuration
        self._transformer_optimization = TransformerOptimization()

        # Check, if transformer optimization is not to skip (cannot be skipped if circuit calculation mode is new)
        if not self._transformer_study_data.calculation_mode == CalcModeEnum.skip_mode:
            # Read requirement list and initialize inductor optimization
            transformer_requirements_list = self._circuit_optimization.get_transformer_requirements()
            self._transformer_optimization.initialize_transformer_optimization_list(toml_transformer, self._transformer_study_data,
                                                                                    transformer_requirements_list)

            # Set the status to InProgress
            self._transformer_main_list[0].progress_data.progress_status = ProgressStatus.InProgress

            if not self._transformer_study_data.calculation_mode == CalcModeEnum.skip_mode:
                # Delete processing complete indicator
                DctMainCtl._delete_processing_complete(self._transformer_study_data.optimization_directory, RELUCTANCE_COMPLETE_FILE)
                # Perform transformer optimization
                self._transformer_optimization.optimization_handler_reluctance_model(
                    self._circuit_optimization.filter_data, toml_prog_flow.transformer.number_of_trials,
                    toml_transformer.filter_distance.factor_dc_losses_min_max_list, debug=toml_debug)
                # Set processing complete indicator
                design_directory = os.path.join(self._transformer_study_data.study_name, CIRCUIT_TRANSFORMER_RELUCTANCE_LOSSES_FOLDER)
                DctMainCtl._set_processing_complete(self._transformer_study_data.optimization_directory,
                                                    design_directory, RELUCTANCE_COMPLETE_FILE,
                                                    self._circuit_optimization.filter_data.filtered_list_files)

            # Set the status to Done
            self._transformer_main_list[0].progress_data.progress_status = ProgressStatus.Done

            # Stop the transformer processing time measurement
            self._transformer_progress_time[0].stop_trigger()

        # Check breakpoint
        self.check_breakpoint(toml_prog_flow.breakpoints.transformer, "Transformer reluctance model Pareto front calculated")

        # --------------------------
        # Heat sink optimization
        # --------------------------
        logger.info("Start heat sink optimization.")

        # Check, if heat sink optimization is to skip
        if not self._heat_sink_study_data.calculation_mode == CalcModeEnum.skip_mode:
            # Allocate and initialize heat sink configuration
            self._heat_sink_optimization = HeatSinkOptimization()
            self._heat_sink_optimization.initialize_heat_sink_optimization(toml_heat_sink, toml_prog_flow)

            # Delete processing complete indicator
            DctMainCtl._delete_processing_complete(self._heat_sink_study_data.optimization_directory, PROCESSING_COMPLETE_FILE)
            # Perform heat sink optimization
            self._heat_sink_optimization.optimization_handler(toml_prog_flow.heat_sink.number_of_trials)
            # Set processing complete indicator
            DctMainCtl._set_processing_complete(self._heat_sink_study_data.optimization_directory,
                                                "", PROCESSING_COMPLETE_FILE)
        # Check breakpoint
        self.check_breakpoint(toml_prog_flow.breakpoints.heat_sink, "Heat sink Pareto front calculated")

        # --------------------------
        # Pre-summary calculation
        # --------------------------
        logger.info("Start pre-summary.")

        # Allocate summary data object
        self._summary_pre_processing = SummaryProcessing()

        if not self._summary_pre_processing.init_thermal_configuration(toml_heat_sink):
            raise ValueError("Thermal data configuration not initialized!")

        # Create list of inductor and transformer study (ASA: Currently not implemented in configuration files)
        inductor_study_names = [self._inductor_study_data.study_name]
        stacked_transformer_study_names = [self._transformer_study_data.study_name]
        capacitor_1_study_names = [self._capacitor_selection_data.study_name]
        capacitor_2_study_names = [self._capacitor_2_selection_data.study_name]
        # Start summary processing by generating the DataFrame from calculated simulation results

        s_df = self._summary_pre_processing.generate_result_database(
            self._inductor_study_data, self._transformer_study_data, pre_summary_data,
            inductor_study_names, stacked_transformer_study_names, self._circuit_optimization.filter_data,
            self._capacitor_selection_data, self._capacitor_2_selection_data,
            capacitor_1_study_names, capacitor_2_study_names, is_pre_summary=True,
            r_th_per_unit_area_ind_heat_sink=self._summary_pre_processing.r_th_per_unit_area_ind_heat_sink,
            r_th_per_unit_area_xfmr_heat_sink=self._summary_pre_processing.r_th_per_unit_area_xfmr_heat_sink,
            heat_sink_boundary_conditions=self._summary_pre_processing.heat_sink_boundary_conditions
        )
        #  Select the needed heat sink configuration
        self._summary_pre_processing.select_heat_sink_configuration(self._heat_sink_study_data, pre_summary_data, s_df)

        # Check breakpoint
        self.check_breakpoint(toml_prog_flow.breakpoints.pre_summary, "Pre-summary is calculated")
        self.generate_zip_archive(toml_prog_flow)

        ParetoPlots.plot_circuit_results(self._circuit_optimization, pre_summary_data.optimization_directory)
        ParetoPlots.plot_inductor_results(self._inductor_study_data, self._circuit_optimization.filter_data.filtered_list_files,
                                          pre_summary_data.optimization_directory)
        ParetoPlots.plot_transformer_results(self._transformer_study_data, self._circuit_optimization.filter_data.filtered_list_files,
                                             pre_summary_data.optimization_directory)
        ParetoPlots.plot_heat_sink_results(self._heat_sink_study_data, pre_summary_data.optimization_directory)
        ParetoPlots.plot_summary(pre_summary_data, self._circuit_optimization)

        # --------------------------
        # Inductor FEM simulation
        # --------------------------
        logger.info("Start inductor FEM simulations.")

        # Check, if inductor FEM simulation is not to skip (cannot be skipped if circuit calculation mode is new)
        if not inductor_sim_calculation_mode == CalcModeEnum.skip_mode:
            # Delete processing complete indicator
            DctMainCtl._delete_processing_complete(self._inductor_study_data.optimization_directory, SIMULATION_COMPLETE_FILE)
            # Perform inductor FEM simulation
            self._inductor_optimization.fem_simulation_handler(
                self._circuit_optimization.filter_data,
                toml_inductor.filter_distance.factor_dc_losses_min_max_list, debug=toml_debug)

            # Set processing complete indicator
            design_directory = os.path.join(self._inductor_study_data.study_name, CIRCUIT_INDUCTOR_FEM_LOSSES_FOLDER)
            DctMainCtl._set_processing_complete(self._inductor_study_data.optimization_directory,
                                                design_directory, SIMULATION_COMPLETE_FILE,
                                                self._circuit_optimization.filter_data.filtered_list_files)

        # --------------------------
        # Transformer FEM simulation
        # --------------------------
        logger.info("Start transformer FEM simulations.")

        # Check, if transformer FEM simulation is not to skip (cannot be skipped if circuit calculation mode is new)
        if not transformer_sim_calculation_mode == CalcModeEnum.skip_mode:
            # Delete processing complete indicator
            DctMainCtl._delete_processing_complete(self._transformer_study_data.optimization_directory, SIMULATION_COMPLETE_FILE)
            # Perform transformer FEM simulation
            self._transformer_optimization.fem_simulation_handler(
                self._circuit_optimization.filter_data,
                toml_inductor.filter_distance.factor_dc_losses_min_max_list, debug=toml_debug)
            # Set processing complete indicator
            design_directory = os.path.join(self._transformer_study_data.study_name, CIRCUIT_TRANSFORMER_FEM_LOSSES_FOLDER)
            DctMainCtl._set_processing_complete(self._transformer_study_data.optimization_directory,
                                                design_directory, SIMULATION_COMPLETE_FILE,
                                                self._circuit_optimization.filter_data.filtered_list_files)

        # --------------------------
        # Final summary calculation
        # --------------------------
        logger.info("Start final summary.")

        # Allocate summary data object
        self._summary_processing = SummaryProcessing()

        # Initialization thermal data
        if not self._summary_processing.init_thermal_configuration(toml_heat_sink):
            raise ValueError("Thermal data configuration not initialized!")
        # Create list of inductor and transformer study (ASA: Currently not implemented in configuration files)
        inductor_study_names = [self._inductor_study_data.study_name]
        stacked_transformer_study_names = [self._transformer_study_data.study_name]
        # Start summary processing by generating the DataFrame from calculated simulation results
        s_df = self._summary_processing.generate_result_database(
            self._inductor_study_data, self._transformer_study_data, summary_data,
            inductor_study_names, stacked_transformer_study_names, self._circuit_optimization.filter_data,
            self._capacitor_selection_data, self._capacitor_2_selection_data,
            capacitor_1_study_names, capacitor_2_study_names, is_pre_summary=False,
            r_th_per_unit_area_ind_heat_sink=self._summary_pre_processing.r_th_per_unit_area_ind_heat_sink,
            r_th_per_unit_area_xfmr_heat_sink=self._summary_pre_processing.r_th_per_unit_area_xfmr_heat_sink,
            heat_sink_boundary_conditions=self._summary_pre_processing.heat_sink_boundary_conditions
        )
        #  Select the needed heat sink configuration
        self._summary_processing.select_heat_sink_configuration(self._heat_sink_study_data, summary_data, s_df)

        # Check breakpoint
        self.check_breakpoint(toml_prog_flow.breakpoints.summary, "Calculation is complete")
        self.generate_zip_archive(toml_prog_flow)

        ParetoPlots.plot_circuit_results(self._circuit_optimization, summary_data.optimization_directory)
        ParetoPlots.plot_inductor_results(self._inductor_study_data, self._circuit_optimization.filter_data.filtered_list_files,
                                          summary_data.optimization_directory)
        ParetoPlots.plot_transformer_results(self._transformer_study_data, self._circuit_optimization.filter_data.filtered_list_files,
                                             summary_data.optimization_directory)
        ParetoPlots.plot_heat_sink_results(self._heat_sink_study_data, summary_data.optimization_directory)
        ParetoPlots.plot_summary(summary_data, self._circuit_optimization)

        # Stop runtime measurement for the optimization (never displayed due to stop of the server)
        self._total_time.stop_trigger()

        # Stop server
        srv_ctl.stop_dct_server()
        # Stop svr_response_thread
        srv_response_stop_flag = True


# Program flow control of DAB-optimization
if __name__ == "__main__":
    # Variable declaration
    arg1 = ""

    # Create a main control instance
    dct_mctl = DctMainCtl()
    # Read the command line
    arguments = sys.argv

    # Check on argument, which corresponds to the workspace file location
    if len(arguments) > 1:
        arg1 = arguments[1]
        # Check if this corresponds to the workspace path
        arg1 = os.path.join(arg1, "workspace")
        print(f"file path={arg1}")
        # Check if the path not exist (absolute or relative path)
        if not os.path.exists(arg1):
            # Consider it as relative path and create the absolute path
            arg1 = os.path.abspath(arg1)
            print(f"new file path={arg1}")
            # Check if the path does not exist
            if not os.path.exists(arg1):
                print(f"Provides argument {arguments[1]} does not corresponds to the path to subfolder 'workspace'.\n")
                print("This is neither the absolute nor the relative path. Program will use the default path!")
                # Reset path variable
                arg1 = ""

        # Convert it to the absolute path
        arg1 = os.path.abspath(arg1)
    # Execute program
    dct_mctl.run_optimization_from_toml_configurations(arg1)

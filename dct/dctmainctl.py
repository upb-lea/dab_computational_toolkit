"""Main control program to optimize the DAB converter."""
import logging.config

import datetime
# python libraries
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

# 3rd party libraries
import json


# own libraries
import dct
from dct import toml_checker as tc, ProgressStatus
from dct import server_ctl_dtos as srv_ctl_dtos
# Circuit, inductor, transformer and heat sink optimization class
from dct import CircuitOptimization
from dct import InductorOptimization
from dct import TransformerOptimization
from dct import HeatSinkOptimization
from dct import ParetoPlots
from dct import generate_logging_config
from dct.server_ctl_dtos import ConfigurationDataEntryDto, SummaryDataEntryDto
from dct.summary_processing import DctSummaryProcessing
from dct.summary_pre_processing import DctSummaryPreProcessing
from dct.server_ctl import DctServer as ServerCtl
from dct.server_ctl import ServerRequestData
from dct.server_ctl import RequestCmd
from dct.server_ctl import ParetoFrontSource
from dct.server_ctl_dtos import ProgressData
from dct.server_ctl_dtos import RunTimeMeasurement as RunTime

logger = logging.getLogger(__name__)

DEBUG: bool = True

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
        self._circuit_optimization: CircuitOptimization | None = None
        self._inductor_optimization: InductorOptimization | None = None
        self._transformer_optimization: TransformerOptimization | None = None
        self._heat_sink_optimization: HeatSinkOptimization | None = None
        self._summary_pre_processing: DctSummaryPreProcessing | None = None
        self._summary_processing: DctSummaryProcessing | None = None

        # Filtered point results in case of skip
        self._inductor_number_filtered_points_skip_list: list[int] = []
        self._transformer_number_filtered_points_skip_list: list[int] = []
        self._heat_sink_number_filtered_points_skip: list[int] = []

        # Key input variables
        self._key_input_lock: threading.Lock = threading.Lock()
        self._key_input_string: str = ""
        self._break_point_flag = False
        self._break_point_message: str = ""

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
        inductor_path = os.path.join(project_directory, toml_prog_flow.inductor.subdirectory)
        transformer_path = os.path.join(project_directory, toml_prog_flow.transformer.subdirectory)
        heat_sink_path = os.path.join(project_directory, toml_prog_flow.heat_sink.subdirectory)
        pre_summary_path = os.path.join(project_directory, toml_prog_flow.pre_summary.subdirectory)
        summary_path = os.path.join(project_directory, toml_prog_flow.summary.subdirectory)

        path_dict = {'circuit': circuit_path,
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
    def check_study_data(study_path: str, study_name: str) -> bool:
        """
        Verify if the study path and sqlite3-database file exists.

        Works for all types of studies (circuit, inductor, transformer, heat sink).
        :param study_path: drive location path to the study
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
            if act_toml_prog_flow.circuit.calculation_mode == "skip":
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
            if act_toml_prog_flow.inductor.calculation_mode == "skip":
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
            if act_toml_prog_flow.transformer.calculation_mode == "skip":
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
            if act_toml_prog_flow.heat_sink.calculation_mode == "skip":
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
            self._inductor_list[0].progress_data.number_of_filtered_points = self._inductor_number_filtered_points_skip_list[filtered_pt_id]
        # Transformer
        # Check if transformer is initialized,if not initialized progress data are valid
        if self._transformer_optimization is not None:
            self._transformer_list[0].progress_data = self._transformer_optimization.get_progress_data(filtered_pt_id)
        elif self._inductor_list[0].progress_data.progress_status == ProgressStatus.Skipped:
            self._transformer_list[0].progress_data.number_of_filtered_points = self._transformer_number_filtered_points_skip_list[filtered_pt_id]
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
                response_data.evaluation_info = f"Circuit configuration file: {self._circuit_list[c_configuration_index].configuration_name}"
                response_data.pareto_front_optuna = self._circuit_optimization.get_actual_pareto_html()
            elif not self._circuit_list[c_configuration_index].progress_data.progress_status == ProgressStatus.Idle:
                # Get Pareto front from file
                if self.check_study_data(self._circuit_study_data.optimization_directory,
                                         self._circuit_study_data.study_name):
                    response_data.evaluation_info = f"Circuit configuration: {self._circuit_list[c_configuration_index].configuration_name}"
                    response_data.pareto_front_optuna = self._circuit_optimization.get_pareto_html(
                        self._circuit_study_data.study_name,
                        os.path.join(self._circuit_study_data.optimization_directory,
                                     self._circuit_study_data.study_name+".sqlite3"))
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
                    if self.check_study_data(sqlite_file_path, self._inductor_study_data.study_name):
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
                    if self.check_study_data(sqlite_file_path, self._transformer_study_data.study_name):
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
                    if self.check_study_data(self._heat_sink_study_data.optimization_directory, self._heat_sink_study_data.study_name):
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

        # --------------------------
        # Logging
        # --------------------------
        # read logging for submodules
        logging_filename = os.path.join(workspace_path, "logging.conf")
        self.load_generate_logging_config(logging_filename)

        # --------------------------
        # Flow control
        # --------------------------
        logger.debug("Read flow control file")
        # Load the configuration for program flow and check the validity
        flow_control_loaded, dict_prog_flow = self.load_toml_file("progFlow.toml")
        toml_prog_flow = tc.FlowControl(**dict_prog_flow)

        if not flow_control_loaded:
            raise ValueError("Program flow toml file does not exist.")

        # Add absolute path to project data path
        workspace_path = os.path.abspath(workspace_path)
        toml_prog_flow.general.project_directory = os.path.join(workspace_path, toml_prog_flow.general.project_directory)

        self.set_up_folder_structure(toml_prog_flow)

        # -----------------------------------------
        # Introduce study data and filter data DTOs
        # -----------------------------------------

        project_directory = os.path.abspath(toml_prog_flow.general.project_directory)
        self._circuit_study_data = dct.StudyData(
            study_name=toml_prog_flow.configuration_data_files.circuit_configuration_file.replace(".toml", ""),
            optimization_directory=os.path.join(project_directory, toml_prog_flow.circuit.subdirectory,
                                                toml_prog_flow.configuration_data_files.circuit_configuration_file.replace(".toml", ""))
        )

        self._inductor_study_data = dct.StudyData(
            study_name=toml_prog_flow.configuration_data_files.inductor_configuration_file.replace(".toml", ""),
            optimization_directory=os.path.join(project_directory, toml_prog_flow.inductor.subdirectory,
                                                toml_prog_flow.configuration_data_files.circuit_configuration_file.replace(".toml", ""))
        )
        self._transformer_study_data = dct.StudyData(
            study_name=toml_prog_flow.configuration_data_files.transformer_configuration_file.replace(".toml", ""),
            optimization_directory=os.path.join(project_directory, toml_prog_flow.transformer.subdirectory,
                                                toml_prog_flow.configuration_data_files.circuit_configuration_file.replace(".toml", ""))
        )
        self._heat_sink_study_data = dct.StudyData(
            study_name=toml_prog_flow.configuration_data_files.heat_sink_configuration_file.replace(".toml", ""),
            optimization_directory=os.path.join(project_directory, toml_prog_flow.heat_sink.subdirectory,
                                                toml_prog_flow.configuration_data_files.heat_sink_configuration_file.replace(".toml", ""))
        )

        pre_summary_data = dct.StudyData(study_name="pre_summary",
                                         optimization_directory=os.path.join(project_directory, toml_prog_flow.pre_summary.subdirectory))
        summary_data = dct.StudyData(study_name="summary", optimization_directory=os.path.join(project_directory, toml_prog_flow.summary.subdirectory))

        filter_data = dct.FilterData(
            filtered_list_files=[],
            filtered_list_pathname=os.path.join(
                project_directory, toml_prog_flow.circuit.subdirectory,
                toml_prog_flow.configuration_data_files.circuit_configuration_file.replace(".toml", ""), "filtered_results"),
            circuit_study_name=toml_prog_flow.configuration_data_files.circuit_configuration_file.replace(".toml", "")
        )

        # Initialize the data for server monitoring (Only 1 circuit configuration is used, later to change)
        (self._circuit_list, self._inductor_main_list, self._inductor_list, self._transformer_main_list,
         self._transformer_list, self._heat_sink_list, self._summary_list) = self.get_initialization_queue_data(toml_prog_flow)

        # --------------------------
        # Circuit flow control
        # --------------------------
        logger.debug("Read circuit flow control")

        # Init circuit configuration
        is_circuit_loaded, dict_circuit = self.load_toml_file(toml_prog_flow.configuration_data_files.circuit_configuration_file)
        toml_circuit = tc.TomlCircuitParetoDabDesign(**dict_circuit)

        if not is_circuit_loaded:
            raise ValueError(f"Circuit configuration file: {toml_prog_flow.configuration_data_files.circuit_configuration_file} does not exist.")

        # Verify optimization parameter
        is_failed, issue_report = dct.CircuitOptimization.verify_optimization_parameter(toml_circuit)
        if is_failed:
            raise ValueError("Circuit optimization parameter in file ",
                             f"{toml_prog_flow.configuration_data_files.heat_sink_configuration_file} are inconsistent!\n", issue_report)

        # Check, if electrical optimization is to skip
        if toml_prog_flow.circuit.calculation_mode == "skip":
            # Check, if data are available (skip case)
            if not self.check_study_data(self._circuit_study_data.optimization_directory, self._circuit_study_data.study_name):
                raise ValueError(f"Study {self._circuit_study_data.study_name} in path {self._circuit_study_data.optimization_directory} does not exist. "
                                 f"No sqlite3-database found!")
            # Check, if data are available (skip case)
            # Check if filtered results folder exists
            if os.path.exists(filter_data.filtered_list_pathname):
                # Add filtered result list
                for filtered_circuit_result in os.listdir(filter_data.filtered_list_pathname):
                    if os.path.isfile(os.path.join(filter_data.filtered_list_pathname, filtered_circuit_result)):
                        filter_data.filtered_list_files.append(os.path.splitext(filtered_circuit_result)[0])
                        # Store list id for progress (Workaround)
                        self._filtered_list_files = filter_data.filtered_list_files
                if not filter_data.filtered_list_files:
                    raise ValueError(f"Filtered results folder {filter_data.filtered_list_pathname} is empty.")
            else:
                raise ValueError(f"Filtered circuit results folder {filter_data.filtered_list_pathname} does not exist.")

        # --------------------------
        # Inductor flow control
        # --------------------------
        logger.debug("Read inductor flow control")

        # Load the inductor-configuration parameter
        inductor_toml_filepath = toml_prog_flow.configuration_data_files.inductor_configuration_file
        is_inductor_loaded, inductor_dict = self.load_toml_file(toml_prog_flow.configuration_data_files.inductor_configuration_file)
        toml_inductor = dct.TomlInductor(**inductor_dict)

        if not is_inductor_loaded:
            raise ValueError(f"Inductor configuration file: {inductor_toml_filepath} does not exist.")

        # Verify optimization parameter
        is_failed, issue_report = dct.InductorOptimization.verify_optimization_parameter(toml_inductor)
        if is_failed:
            raise ValueError("Inductor optimization parameter in file ",
                             f"{toml_prog_flow.configuration_data_files.inductor_configuration_file} are inconsistent!\n", issue_report)

        # Check, if inductor optimization is to skip
        if toml_prog_flow.inductor.calculation_mode == "skip":
            # Initialize _inductor_number_filtered_points_skip_list
            self._inductor_number_filtered_points_skip_list = []
            # For loop to check, if all filtered values are available

            for id_entry in filter_data.filtered_list_files:
                # Assemble pathname
                inductor_results_datapath = os.path.join(self._inductor_study_data.optimization_directory,
                                                         str(id_entry), self._inductor_study_data.study_name)
                # Check, if data are available (skip case)
                if self.check_study_data(inductor_results_datapath, self._inductor_study_data.study_name):
                    self._inductor_number_filtered_points_skip_list.append(
                        self.get_number_of_pkl_files(os.path.join(inductor_results_datapath,
                                                                  "09_circuit_dtos_incl_inductor_losses")))
                else:
                    raise ValueError(
                        f"Study {self._inductor_study_data.study_name} in path {inductor_results_datapath} does not exist. No sqlite3-database found!")

        # --------------------------
        # Transformer flow control
        # --------------------------
        logger.debug("Read transformer flow control")

        # Load the transformer-configuration parameter
        transformer_toml_filepath = toml_prog_flow.configuration_data_files.transformer_configuration_file
        is_transformer_loaded, transformer_dict = self.load_toml_file(toml_prog_flow.configuration_data_files.transformer_configuration_file)
        toml_transformer = dct.TomlTransformer(**transformer_dict)

        if not is_transformer_loaded:
            raise ValueError(f"Transformer configuration file: {transformer_toml_filepath} does not exist.")

        # Verify optimization parameter
        is_failed, issue_report = dct.TransformerOptimization.verify_optimization_parameter(toml_transformer)
        if is_failed:
            raise ValueError("Transformer optimization parameter in file ",
                             f"{toml_prog_flow.configuration_data_files.transformer_configuration_file} are inconsistent!\n", issue_report)

        # Check, if transformer optimization is to skip
        if toml_prog_flow.transformer.calculation_mode == "skip":
            # Initialize _transformer_number_filtered_points_skip_list
            self._transformer_number_filtered_points_skip_list = []
            # For loop to check, if all filtered values are available
            for id_entry in filter_data.filtered_list_files:
                # Assemble pathname
                transformer_results_datapath = os.path.join(self._transformer_study_data.optimization_directory,
                                                            str(id_entry),
                                                            self._transformer_study_data.study_name)
                # Check, if data are available (skip case)
                if self.check_study_data(transformer_results_datapath, self._transformer_study_data.study_name):
                    self._transformer_number_filtered_points_skip_list.append(
                        self.get_number_of_pkl_files(os.path.join(transformer_results_datapath,
                                                                  "09_circuit_dtos_incl_transformer_losses")))
                else:
                    raise ValueError(
                        f"Study {self._transformer_study_data.study_name} in path {transformer_results_datapath}"
                        "does not exist. No sqlite3-database found!"
                    )

        # --------------------------
        # Heat sink flow control
        # --------------------------

        heat_sink_toml_filepath = toml_prog_flow.configuration_data_files.heat_sink_configuration_file
        is_heat_sink_loaded, heat_sink_dict = self.load_toml_file(heat_sink_toml_filepath)
        toml_heat_sink = dct.TomlHeatSink(**heat_sink_dict)
        if not is_heat_sink_loaded:
            raise ValueError(f"Heat sink configuration file: {heat_sink_toml_filepath} does not exist.")

        # Verify optimization parameter
        is_failed, issue_report = dct.HeatSinkOptimization.verify_optimization_parameter(toml_heat_sink)
        if is_failed:
            raise ValueError("Heat sink optimization parameter in file "
                             f"{toml_prog_flow.configuration_data_files.heat_sink_configuration_file} are inconsistent!\n", issue_report)

        # Check, if heat sink optimization is to skip
        if toml_prog_flow.heat_sink.calculation_mode == "skip":
            # Check, if data are available (skip case)
            if not self.check_study_data(self._heat_sink_study_data.optimization_directory, self._heat_sink_study_data.study_name):
                raise ValueError(
                    f"Study {self._heat_sink_study_data.study_name} in path "
                    f"{self._heat_sink_study_data.optimization_directory} does not exist. No sqlite3-database found!")

        # -- Start server  --------------------------------------------------------------------------------------------

        # Initialize the runtime timer
        self._circuit_progress_time = [RunTime()]
        self._inductor_progress_time = [RunTime()]
        self._transformer_progress_time = [RunTime()]
        self._heat_sink_progress_time = [RunTime()]
        self._summary_progress_time = [RunTime()]

        # Start the data exchange queue thread
        srv_response_stop_flag = False
        _srv_response_handler = threading.Thread(target=self._srv_response_queue,
                                                 args=(srv_request_queue, srv_response_queue), daemon=True)

        _srv_response_handler.start()

        # Start the server
        srv_ctl.start_dct_server(srv_request_queue, srv_response_queue, True)

        # Initialize key input handler
        self._key_input_handler = threading.Thread(target=self._key_input,
                                                   args=(srv_request_queue, srv_response_queue), daemon=True)

        # -- Start optimization  ----------------------------------------------------------------------------------------
        # --------------------------
        # Circuit optimization
        # --------------------------
        logger.info("Start circuit optimization.")

        # Check, if electrical optimization is not to skip
        if not toml_prog_flow.circuit.calculation_mode == "skip":

            # Allocate and initialize circuit configuration
            self._circuit_optimization = CircuitOptimization()
            self._circuit_optimization.initialize_circuit_optimization(toml_circuit, toml_prog_flow)

            # Check, if old study is to delete, if available
            if toml_prog_flow.circuit.calculation_mode == "new":
                # delete old circuit study data
                self.delete_study_content(self._circuit_study_data.optimization_directory, self._circuit_study_data.study_name)

                # Create the filtered result folder
                os.makedirs(filter_data.filtered_list_pathname, exist_ok=True)
                # Delete obsolete folders of inductor and transformer
                self.delete_study_content(self._inductor_study_data.optimization_directory)
                self.delete_study_content(self._transformer_study_data.optimization_directory)

            # Perform circuit optimization
            self._circuit_optimization.start_proceed_study(number_trials=toml_prog_flow.circuit.number_of_trials)

        # Check breakpoint
        self.check_breakpoint(toml_prog_flow.breakpoints.circuit_pareto, "Electric Pareto front calculated")

        # Check, if electrical optimization is not to skip
        if not toml_prog_flow.circuit.calculation_mode == "skip":

            # Check if _circuit_optimization is not allocated, what corresponds to a serious programming error
            if self._circuit_optimization is None:
                raise ValueError("Serious programming error. Please write an issue!")

            # Calculate the filtered results
            self._circuit_optimization.filter_study_results()
            # Get filtered result path

            # Add filtered result list
            for filtered_circuit_result in os.listdir(filter_data.filtered_list_pathname):
                if os.path.isfile(os.path.join(filter_data.filtered_list_pathname, filtered_circuit_result)):
                    filter_data.filtered_list_files.append(os.path.splitext(filtered_circuit_result)[0])

            # Workaround: Set filtered result id list here, later to handle in circuit_optimization
            self._filtered_list_files = filter_data.filtered_list_files

        # Set the number of calculations for the magnetic components
        self._inductor_main_list[0].number_calculations = len(self._filtered_list_files)
        self._transformer_main_list[0].number_calculations = len(self._filtered_list_files)

        # Check breakpoint
        self.check_breakpoint(toml_prog_flow.breakpoints.circuit_filtered, "Filtered value of electric Pareto front calculated")

        # --------------------------
        # Inductor reluctance model optimization
        # --------------------------
        logger.info("Start inductor reluctance model optimization.")

        # Start the inductor processing time measurement
        self._inductor_progress_time[0].reset_start_trigger()

        # Check, if inductor optimization is not to skip (cannot be skipped if circuit calculation mode is new)
        if not toml_prog_flow.inductor.calculation_mode == "skip":
            # Set the status to InProgress
            self._inductor_main_list[0].progress_data.progress_status = ProgressStatus.InProgress

            # Check, if old study is to delete, if available
            if toml_prog_flow.inductor.calculation_mode == "new":
                # Delete old inductor study
                self.delete_study_content(self._inductor_study_data.optimization_directory)

            # Allocate and initialize inductor configuration
            self._inductor_optimization = InductorOptimization()
            self._inductor_optimization.initialize_inductor_optimization_list(toml_inductor, self._inductor_study_data,
                                                                              filter_data)

            # Perform inductor optimization
            self._inductor_optimization.optimization_handler_reluctance_model(
                filter_data, toml_prog_flow.inductor.number_of_trials, toml_inductor.filter_distance.factor_dc_losses_min_max_list,
                debug=DEBUG)

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
        # Check, if transformer optimization is not to skip (cannot be skipped if circuit calculation mode is new)
        if not toml_prog_flow.transformer.calculation_mode == "skip":
            # Set the status to InProgress
            self._transformer_main_list[0].progress_data.progress_status = ProgressStatus.InProgress
            # Check, if old study is to delete, if available
            if toml_prog_flow.transformer.calculation_mode == "new":
                # Delete old transformer study
                self.delete_study_content(self._transformer_study_data.optimization_directory)

            # Allocate and initialize transformer configuration
            self._transformer_optimization = TransformerOptimization()
            self._transformer_optimization.initialize_transformer_optimization_list(toml_transformer, self._transformer_study_data,
                                                                                    filter_data)
            # Perform transformer optimization
            self._transformer_optimization.optimization_handler(
                filter_data, toml_prog_flow.transformer.number_of_trials, toml_transformer.filter_distance.factor_dc_losses_min_max_list)

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
        if not toml_prog_flow.heat_sink.calculation_mode == "skip":
            # Check, if old study is to delete, if available
            if toml_prog_flow.heat_sink.calculation_mode == "new":
                # Delete old heat sink study
                self.delete_study_content(self._heat_sink_study_data.optimization_directory, self._heat_sink_study_data.study_name)

            # Allocate and initialize heat sink configuration
            self._heat_sink_optimization = HeatSinkOptimization()
            self._heat_sink_optimization.initialize_heat_sink_optimization(toml_heat_sink, toml_prog_flow)

            # Perform heat sink optimization
            self._heat_sink_optimization.optimization_handler(toml_prog_flow.heat_sink.number_of_trials)

        # Check breakpoint
        self.check_breakpoint(toml_prog_flow.breakpoints.heat_sink, "Heat sink Pareto front calculated")

        # --------------------------
        # Pre-summary calculation
        # --------------------------
        logger.info("Start pre-summary.")

        # Allocate summary data object
        self._summary_pre_processing = DctSummaryPreProcessing()

        # Initialization thermal data
        if not self._summary_pre_processing.init_thermal_configuration(toml_heat_sink):
            raise ValueError("Thermal data configuration not initialized!")
        # Create list of inductor and transformer study (ASA: Currently not implemented in configuration files)
        inductor_study_names = [self._inductor_study_data.study_name]
        stacked_transformer_study_names = [self._transformer_study_data.study_name]
        # Start summary processing by generating the DataFrame from calculated simulation results
        s_df = self._summary_pre_processing.generate_result_database(
            self._inductor_study_data, self._transformer_study_data, pre_summary_data,
            inductor_study_names, stacked_transformer_study_names, filter_data)
        #  Select the needed heat sink configuration
        self._summary_pre_processing.select_heat_sink_configuration(self._heat_sink_study_data, pre_summary_data, s_df)

        # Check breakpoint
        self.check_breakpoint(toml_prog_flow.breakpoints.summary, "Calculation is complete")
        self.generate_zip_archive(toml_prog_flow)

        ParetoPlots.plot_circuit_results(toml_prog_flow, is_pre_summary=True)
        ParetoPlots.plot_inductor_results(toml_prog_flow, is_pre_summary=True)
        ParetoPlots.plot_transformer_results(toml_prog_flow, is_pre_summary=True)
        ParetoPlots.plot_heat_sink_results(toml_prog_flow, is_pre_summary=True)
        ParetoPlots.plot_summary(toml_prog_flow, is_pre_summary=True)

        # --------------------------
        # Inductor FEM simulation
        # --------------------------
        logger.info("Start inductor FEM simulations.")

        # Check, if inductor optimization is not to skip (cannot be skipped if circuit calculation mode is new)
        if not toml_prog_flow.inductor.calculation_mode == "skip":
            # Perform inductor optimization
            self._inductor_optimization.fem_simulation_handler(
                filter_data, toml_prog_flow.inductor.number_of_trials, toml_inductor.filter_distance.factor_dc_losses_min_max_list,
                debug=DEBUG)


        # --------------------------
        # Transformer FEM simulation
        # --------------------------
        logger.info("Start transformer FEM simulations.")

        # --------------------------
        # Final summary calculation
        # --------------------------
        logger.info("Start final summary.")

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

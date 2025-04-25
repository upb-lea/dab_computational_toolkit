"""Main control program to optimize the DAB converter."""
# python libraries
import os
import shutil
import sys
import tomllib

# 3rd party libraries
import json

# own libraries
import dct
import toml_checker as tc
import dct.circuit_optimization_dtos as p_dtos
# Circuit, inductor, transformer and heat sink optimization class
from dct import CircuitOptimization
from dct import InductorOptimization
from dct import TransformerOptimization
from dct import HeatSinkOptimization
from summary_processing import DctSummaryProcessing as spro

# logging.basicConfig(format='%(levelname)s,%(asctime)s:%(message)s', encoding='utf-8')
# logging.getLogger('pygeckocircuits2').setLevel(logging.DEBUG)

class DctMainCtl:
    """Main class for control dab-optimization."""

    @staticmethod
    def set_up_folder_structure(toml_prog_flow: tc.FlowControl) -> None:
        """
        Set up the folder structure for the subprojects.

        :param toml_prog_flow: Flow control toml file
        :type toml_prog_flow: tc.FlowControl
        """
        # ASA: Merge ginfo and set_up_folder_structure Fix file structure on top layer
        project_directory = os.path.abspath(toml_prog_flow.general.project_directory)
        circuit_path = os.path.join(project_directory, toml_prog_flow.circuit.subdirectory)
        inductor_path = os.path.join(project_directory, toml_prog_flow.inductor.subdirectory)
        transformer_path = os.path.join(project_directory, toml_prog_flow.transformer.subdirectory)
        heat_sink_path = os.path.join(project_directory, toml_prog_flow.heat_sink.subdirectory)

        path_dict = {'circuit': circuit_path,
                     'inductor': inductor_path,
                     'transformer': transformer_path,
                     'heat_sink': heat_sink_path}

        for _, value in path_dict.items():
            os.makedirs(value, exist_ok=True)

        json_filepath = os.path.join(project_directory, "filepath_config.json")

        with open(json_filepath, 'w', encoding='utf8') as json_file:
            json.dump(path_dict, json_file, ensure_ascii=False, indent=4)

    @staticmethod
    def load_toml_file(toml_file: str) -> tuple[bool, dict]:
        """
        Load the toml configuration data to a dictionary.

        :param toml_file : File name of the toml-file
        :type  toml_file : str
        :return: True, if the data could be loaded successful and the loaded dictionary
        :rtype: bool, dict
        """
        # return value init to false and tomlData to empty
        is_toml_file_existing = False

        # Separate filename and path
        toml_file_directory = os.path.dirname(toml_file)

        # check path
        if os.path.exists(toml_file_directory) or toml_file_directory == "":
            # check filename
            if os.path.isfile(toml_file):
                with open(toml_file, "rb") as f:
                    config = tomllib.load(f)
                is_toml_file_existing = True
            else:
                print(f"File {toml_file} does not exists!")
        else:
            print(f"Path {toml_file_directory} does not exists!")

        return is_toml_file_existing, config

    @staticmethod
    def generate_conf_file(path: str) -> bool:
        """
        Create and save the configuration file.

        Generate following default configuration files within the path:
        DabElectricConf.toml, DabInductorConf.toml, DabTransformerConf.toml and DabHeatSinkConf.toml,

        :param path : Location of the configuration
        :type  path : str
        :return: true, if the files are stored successfully
        :rtype: bool

        """
        return False

    @staticmethod
    def delete_study_content(folder_name: str, study_file_name: str = ""):
        """
        Delete the study files and the femmt folders.

        If a new study is to generate the old obsolete files and folders needs to be deleted.

        :param folder_name : Location of the study files
        :type  folder_name : str
        :param study_file_name : Name of the study files (without extension)
        :type  study_file_name : str
        """
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

    @staticmethod
    def user_input_break_point(break_point_key: str, info: str):
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
        # return value init to false
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
                print(f"File {target_file} does not exists!")
        else:
            print(f"Path {study_path} does not exists!")

        # True = study exists
        return is_study_existing

    @staticmethod
    def check_breakpoint(break_point_key: str, info: str):
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
            # Wait for keyboard entry
            while key_inp != "c" and key_inp != "C" and key_inp != "s" and key_inp != "S":
                key_inp = input()

            # Check result
            if key_inp == "s" or key_inp == "S":
                print("User stops the program!")
                # stop program
                sys.exit()
        else:
            pass

    @staticmethod
    def circuit_toml_2_dto(toml_circuit: tc.TomlCircuitParetoDabDesign, toml_prog_flow: tc.FlowControl) -> p_dtos.CircuitParetoDabDesign:
        """
        Circuit toml file to circuit_dto file.

        :param toml_circuit: toml file class for the circuit
        :type toml_circuit: tc.TomlCircuitParetoDabDesign
        :param toml_prog_flow: toml file class for the flow control
        :type toml_prog_flow: tc.FlowControl
        :return: circuit DTO
        :rtype: p_dtos.CircuitParetoDabDesign
        """
        design_space = p_dtos.CircuitParetoDesignSpace(
            f_s_min_max_list=toml_circuit.design_space.f_s_min_max_list,
            l_s_min_max_list=toml_circuit.design_space.l_s_min_max_list,
            l_1_min_max_list=toml_circuit.design_space.l_1_min_max_list,
            l_2__min_max_list=toml_circuit.design_space.l_2__min_max_list,
            n_min_max_list=toml_circuit.design_space.n_min_max_list,
            transistor_1_name_list=toml_circuit.design_space.transistor_1_name_list,
            transistor_2_name_list=toml_circuit.design_space.transistor_2_name_list,
            c_par_1=toml_circuit.design_space.c_par_1,
            c_par_2=toml_circuit.design_space.c_par_2
        )

        output_range = p_dtos.CircuitOutputRange(
            v_1_min_nom_max_list=toml_circuit.output_range.v_1_min_nom_max_list,
            v_2_min_nom_max_list=toml_circuit.output_range.v_2_min_nom_max_list,
            p_min_nom_max_list=toml_circuit.output_range.p_min_nom_max_list,
            steps_per_direction=toml_circuit.output_range.steps_per_direction)

        filter = p_dtos.CircuitFilter(
            number_filtered_designs=toml_circuit.filter_distance.number_filtered_designs,
            difference_percentage=toml_circuit.filter_distance.difference_percentage
        )

        circuit_dto = p_dtos.CircuitParetoDabDesign(
            circuit_study_name=toml_prog_flow.configuration_data_files.circuit_configuration_file.replace(".toml", ""),
            project_directory=toml_prog_flow.general.project_directory,
            design_space=design_space,
            output_range=output_range,
            filter=filter)

        return circuit_dto

    @staticmethod
    def run_optimization_from_toml_configs(workspace_path: str):
        """Perform the main program.

        This function corresponds to 'main', which is called after the instance of the class are created.

        :param  workspace_path: Path to subfolder 'workspace' (if empty default path '../<path to this file>' is used)
        :type   workspace_path: str
        """
        # Flag for re-simulation  (if False the summary will failed)
        enable_ind_re_simulation = True
        enable_trans_re_simulation = True

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
        # Flow control
        # --------------------------
        # Load the configuration for program flow and check the validity
        flow_control_loaded, dict_prog_flow = DctMainCtl.load_toml_file("progFlow.toml")
        toml_prog_flow = tc.FlowControl(**dict_prog_flow)

        if not flow_control_loaded:
            raise ValueError("Program flow toml file does not exist.")

        # Add absolute path to project data path (ASA: Later to remove because do not manipulate)
        workspace_path = os.path.abspath(workspace_path)
        toml_prog_flow.general.project_directory = os.path.join(workspace_path, toml_prog_flow.general.project_directory)

        DctMainCtl.set_up_folder_structure(toml_prog_flow)

        # -----------------------------------------
        # Introduce study data and filter data DTOs
        # -----------------------------------------

        project_directory = os.path.abspath(toml_prog_flow.general.project_directory)
        circuit_study_data = dct.StudyData(
            study_name=toml_prog_flow.configuration_data_files.circuit_configuration_file.replace(".toml", ""),
            optimization_directory=os.path.join(project_directory, toml_prog_flow.circuit.subdirectory,
                                                toml_prog_flow.configuration_data_files.circuit_configuration_file.replace(".toml", ""))
        )

        inductor_study_data = dct.StudyData(
            study_name=toml_prog_flow.configuration_data_files.inductor_configuration_file.replace(".toml", ""),
            optimization_directory=os.path.join(project_directory, toml_prog_flow.inductor.subdirectory,
                                                toml_prog_flow.configuration_data_files.circuit_configuration_file.replace(".toml", ""))
        )
        transformer_study_data = dct.StudyData(
            study_name=toml_prog_flow.configuration_data_files.transformer_configuration_file.replace(".toml", ""),
            optimization_directory=os.path.join(project_directory, toml_prog_flow.transformer.subdirectory,
                                                toml_prog_flow.configuration_data_files.circuit_configuration_file.replace(".toml", ""))
        )
        heat_sink_study_data = dct.StudyData(
            study_name=toml_prog_flow.configuration_data_files.heat_sink_configuration_file.replace(".toml", ""),
            optimization_directory=os.path.join(project_directory, toml_prog_flow.heat_sink.subdirectory,
                                                toml_prog_flow.configuration_data_files.heat_sink_configuration_file.replace(".toml", ""))
        )

        filter_data = dct.FilterData(
            filtered_list_id=[],
            filtered_list_pathname=os.path.join(
                project_directory, toml_prog_flow.circuit.subdirectory,
                toml_prog_flow.configuration_data_files.circuit_configuration_file.replace(".toml", ""), "filtered_results"),
            circuit_study_name=toml_prog_flow.configuration_data_files.circuit_configuration_file.replace(".toml", "")
        )

        # --------------------------
        # Circuit flow control
        # --------------------------

        # Init circuit configuration
        is_circuit_loaded, dict_circuit = DctMainCtl.load_toml_file(toml_prog_flow.configuration_data_files.circuit_configuration_file)
        toml_circuit = tc.TomlCircuitParetoDabDesign(**dict_circuit)
        config_circuit = DctMainCtl.circuit_toml_2_dto(toml_circuit, toml_prog_flow)

        if not is_circuit_loaded:
            raise ValueError(f"Circuit configuration file: {toml_prog_flow.configuration_data_files.circuit_configuration_file} does not exist.")

        # Check, if electrical optimization is to skip
        if toml_prog_flow.circuit.calculation_mode == "skip":
            if not DctMainCtl.check_study_data(circuit_study_data.optimization_directory, circuit_study_data.study_name):
                raise ValueError(f"Study {circuit_study_data.study_name} in path {circuit_study_data.optimization_directory} does not exist. "
                                 f"No sqlite3-database found!")
            # Check, if data are available (skip case)
            # Check if filtered results folder exists
            if os.path.exists(filter_data.filtered_list_pathname):
                # Add filtered result list
                for filtered_circuit_result in os.listdir(filter_data.filtered_list_pathname):
                    if os.path.isfile(os.path.join(filter_data.filtered_list_pathname, filtered_circuit_result)):
                        filter_data.filtered_list_id.append(int(os.path.splitext(filtered_circuit_result)[0]))
                if not filter_data.filtered_list_id:
                    raise ValueError(f"Filtered results folder {filter_data.filtered_list_pathname} is empty.")
            else:
                raise ValueError(f"Filtered circuit results folder {filter_data.filtered_list_pathname} does not exist.")

        # --------------------------
        # Inductor flow control
        # --------------------------

        # Load the inductor-configuration parameter
        inductor_toml_filepath = toml_prog_flow.configuration_data_files.inductor_configuration_file
        is_inductor_loaded, inductor_dict = DctMainCtl.load_toml_file(toml_prog_flow.configuration_data_files.inductor_configuration_file)
        toml_inductor = dct.TomlInductor(**inductor_dict)

        if not is_inductor_loaded:
            raise ValueError(f"Inductor configuration file: {inductor_toml_filepath} does not exist.")

        # Check, if inductor optimization is to skip
        if toml_prog_flow.inductor.calculation_mode == "skip":
            # For loop to check, if all filtered values are available

            for id_entry in filter_data.filtered_list_id:
                # Assemble pathname
                inductor_results_datapath = os.path.join(inductor_study_data.optimization_directory,
                                                         str(id_entry), inductor_study_data.study_name)
                # Check, if data are available (skip case)
                if not DctMainCtl.check_study_data(inductor_results_datapath, inductor_study_data.study_name):
                    raise ValueError(
                        f"Study {inductor_study_data.study_name} in path {inductor_results_datapath} does not exist. No sqlite3-database found!")

        # --------------------------
        # Transformer flow control
        # --------------------------

        # Load the transformer-configuration parameter
        transformer_toml_filepath = toml_prog_flow.configuration_data_files.transformer_configuration_file
        is_transformer_loaded, transformer_dict = DctMainCtl.load_toml_file(toml_prog_flow.configuration_data_files.transformer_configuration_file)
        toml_transformer = dct.TomlTransformer(**transformer_dict)

        if not is_transformer_loaded:
            raise ValueError(f"Transformer configuration file: {transformer_toml_filepath} does not exist.")

        # Check, if transformer optimization is to skip
        if toml_prog_flow.transformer.calculation_mode == "skip":
            # For loop to check, if all filtered values are available
            for id_entry in filter_data.filtered_list_id:
                # Assemble pathname
                transformer_results_datapath = os.path.join(transformer_study_data.optimization_directory,
                                                            str(id_entry),
                                                            transformer_study_data.study_name)
                # Check, if data are available (skip case)
                if not DctMainCtl.check_study_data(transformer_results_datapath, transformer_study_data.study_name):
                    raise ValueError(
                        f"Study {transformer_study_data.study_name} in path {transformer_results_datapath}"
                        "does not exist. No sqlite3-database found!"
                    )

        # --------------------------
        # Heat sink flow control
        # --------------------------

        heat_sink_toml_filepath = toml_prog_flow.configuration_data_files.heat_sink_configuration_file
        is_heat_sink_loaded, heat_sink_dict = DctMainCtl.load_toml_file(heat_sink_toml_filepath)
        toml_heat_sink = dct.TomlHeatSink(**heat_sink_dict)
        if not is_heat_sink_loaded:
            raise ValueError(f"Heat sink configuration file: {heat_sink_toml_filepath} does not exist.")

        # Check, if heat sink optimization is to skip
        if toml_prog_flow.heat_sink.calculation_mode == "skip":
            # Check, if data are available (skip case)
            if not DctMainCtl.check_study_data(heat_sink_study_data.optimization_directory, heat_sink_study_data.study_name):
                raise ValueError(
                    f"Study {heat_sink_study_data.study_name} in path {heat_sink_study_data.optimization_directory} does not exist. No sqlite3-database found!")

        # -- Start server  --------------------------------------------------------------------------------------------
        # Debug: Server switched off
        # srv_ctl.start_dct_server(histogram_data,False)

        # -- Start simulation  ----------------------------------------------------------------------------------------
        # --------------------------
        # Circuit optimization
        # --------------------------
        # Check, if electrical optimization is not to skip
        if not toml_prog_flow.circuit.calculation_mode == "skip":
            if not is_circuit_loaded:
                raise ValueError("Electrical configuration not initialized!")
            # Check, if old study is to delete, if available
            if toml_prog_flow.circuit.calculation_mode == "new":
                # delete old circuit study data
                DctMainCtl.delete_study_content(circuit_study_data.optimization_directory, circuit_study_data.study_name)

                # Create the filtered result folder
                os.makedirs(filter_data.filtered_list_pathname, exist_ok=True)
                # Delete obsolete folders of inductor and transformer
                DctMainCtl.delete_study_content(inductor_study_data.optimization_directory)
                DctMainCtl.delete_study_content(transformer_study_data.optimization_directory)

            # Start calculation
            dct.CircuitOptimization.start_proceed_study(config_circuit, number_trials=toml_prog_flow.circuit.number_of_trials)

        # Check breakpoint
        DctMainCtl.check_breakpoint(toml_prog_flow.breakpoints.circuit_pareto, "Electric Pareto front calculated")

        # Check, if electrical optimization is not to skip
        if not toml_prog_flow.circuit.calculation_mode == "skip":
            # Calculate the filtered results
            CircuitOptimization.filter_study_results(dab_config=config_circuit)
            # Get filtered result path

            # Add filtered result list
            for filtered_circuit_result in os.listdir(filter_data.filtered_list_pathname):
                if os.path.isfile(os.path.join(filter_data.filtered_list_pathname, filtered_circuit_result)):
                    filter_data.filtered_list_id.append(int(os.path.splitext(filtered_circuit_result)[0]))

        # Check breakpoint
        DctMainCtl.check_breakpoint(toml_prog_flow.breakpoints.circuit_filtered, "Filtered value of electric Pareto front calculated")

        # --------------------------
        # Inductor optimization
        # --------------------------

        # Check, if inductor optimization is not to skip (cannot be skipped if circuit calculation mode is new)
        if not toml_prog_flow.inductor.calculation_mode == "skip" or toml_prog_flow.circuit.calculation_mode == "new":
            # Check, if old study is to delete, if available
            if toml_prog_flow.inductor.calculation_mode == "new":
                # Delete old inductor study
                DctMainCtl.delete_study_content(inductor_study_data.optimization_directory)

            inductor_optimization = InductorOptimization(toml_inductor, inductor_study_data, filter_data)
            inductor_optimization.optimization_handler(
                filter_data, toml_prog_flow.inductor.number_of_trials, toml_inductor.filter_distance.factor_min_dc_losses,
                toml_inductor.filter_distance.factor_max_dc_losses, enable_ind_re_simulation)

        # Check breakpoint
        DctMainCtl.check_breakpoint(toml_prog_flow.breakpoints.inductor, "Inductor Pareto front calculated")

        # --------------------------
        # Transformer optimization
        # --------------------------

        # Check, if transformer optimization is not to skip (cannot be skipped if circuit calculation mode is new)
        if not toml_prog_flow.transformer.calculation_mode == "skip" or toml_prog_flow.circuit.calculation_mode == "new":
            # Check, if old study is to delete, if available
            if toml_prog_flow.transformer.calculation_mode == "new":
                # Delete old transformer study
                DctMainCtl.delete_study_content(transformer_study_data.optimization_directory)

            # Initialize transformer configuration
            transformer_optimization = TransformerOptimization(toml_transformer, transformer_study_data, filter_data)

            # Perform transformer optimization
            transformer_optimization.simulation_handler(
                filter_data, toml_prog_flow.transformer.number_of_trials, toml_transformer.filter_distance.factor_min_dc_losses,
                toml_transformer.filter_distance.factor_max_dc_losses, enable_trans_re_simulation)

        # Check breakpoint
        DctMainCtl.check_breakpoint(toml_prog_flow.breakpoints.transformer, "Transformer Pareto front calculated")

        # --------------------------
        # Heat sink optimization
        # --------------------------

        # Check, if heat sink optimization is to skip
        if not toml_prog_flow.heat_sink.calculation_mode == "skip":
            # Check, if old study is to delete, if available
            if toml_prog_flow.heat_sink.calculation_mode == "new":
                # Delete old heat sink study
                DctMainCtl.delete_study_content(heat_sink_study_data.optimization_directory, heat_sink_study_data.study_name)

            heat_sink_optimization = HeatSinkOptimization(toml_heat_sink, toml_prog_flow)
            # Perform heat sink optimization
            heat_sink_optimization.optimization_handler(toml_prog_flow.heat_sink.number_of_trials)

        # Check breakpoint
        DctMainCtl.check_breakpoint(toml_prog_flow.breakpoints.heat_sink, "Heat sink Pareto front calculated")

        # Initialisation thermal data
        if not spro.init_thermal_configuration(toml_heat_sink.thermal_resistance_data):
            raise ValueError("Thermal data configuration not initialized!")
        # Create list of inductor and transformer study (ASA: Currently not implemented in configuration files)
        inductor_study_names = [inductor_study_data.study_name]
        stacked_transformer_study_names = [transformer_study_data.study_name]
        # Start summary processing by generating the DataFrame from calculated simulation results
        s_df = spro.generate_result_database(circuit_study_data, inductor_study_data, transformer_study_data, heat_sink_study_data,
                                             inductor_study_names, stacked_transformer_study_names, filter_data)
        #  Select the needed heat sink configuration
        spro.select_heat_sink_configuration(heat_sink_study_data, s_df)
        # Check breakpoint
        DctMainCtl.check_breakpoint(toml_prog_flow.breakpoints.summary, "Calculation is complete")


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
    dct_mctl.run_optimization_from_toml_configs(arg1)

"""Main control program to optimise the DAB converter."""
# python libraries
import os
import sys
import tomllib
import json

# 3rd party libraries

# own libraries
import dct
import toml_checker as tc
import circuit_optimization_dtos as p_dtos
from dct import CircuitOptimization
from dct import InductorOptimization
from dct import TransformerOptimization
from dct import HeatSinkOptimization

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
        # ASA: TODO: Merge ginfo and set_up_folder_structure
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
        toml_file_direction = os.path.dirname(toml_file)

        # check path
        if os.path.exists(toml_file_direction) or toml_file_direction == "":
            # check filename
            if os.path.isfile(toml_file):
                with open(toml_file, "rb") as f:
                    config = tomllib.load(f)
                is_toml_file_existing = True
            else:
                print(f"File {toml_file} does not exists!")
        else:
            print(f"Path {toml_file_direction} does not exists!")

        return is_toml_file_existing, config

    @staticmethod
    def generate_conf_file(path: str) -> bool:
        """
        Create and save the configuration file.

        Generate following default configuration files within the path:
        DabElectricConf.toml, DabInductorConf.toml, DabTransformerConf.toml and DabHeatSinkConf.toml,

        :param path : Location of the configuration
        :type  path : str:
        :return: true, if the files are stored successfully
        :rtype: bool

        """
        return False

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
    def init_general_info(act_ginfo: dct.GeneralInformation, act_config_program_flow: tc.FlowControl):
        """
        Init the general information variable.

        :param act_ginfo: reference to the general information variable
        :type  act_ginfo: dct.GeneralInformation
        :param act_config_program_flow: toml data of the program flow
        :type  act_config_program_flow: dict
        """
        # Read the current directory name
        abs_path = os.getcwd()
        # Store project directory and study name
        act_ginfo.project_directory = act_config_program_flow.general.project_directory
        act_ginfo.circuit_study_name = act_config_program_flow.configuration_data_files.circuit_configuration_file.replace(".toml", "")
        # Create path names
        act_ginfo.circuit_study_path = os.path.join(abs_path, act_ginfo.project_directory,
                                                    act_config_program_flow.circuit.subdirectory, act_ginfo.circuit_study_name)
        act_ginfo.inductor_study_path = os.path.join(abs_path, act_ginfo.project_directory,
                                                     act_config_program_flow.inductor.subdirectory, act_ginfo.circuit_study_name)
        act_ginfo.transformer_study_path = os.path.join(abs_path, act_ginfo.project_directory,
                                                        act_config_program_flow.transformer.subdirectory, act_ginfo.circuit_study_name)

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
        if break_point_key == "Stop":
            print("Program stops cause by breakpoint at: '"+info+"'!")
            # stop program
            sys.exit()

        elif break_point_key == "Pause":
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
    def run_optimization_from_toml_configs(workspace_path: str) -> None:
        """Perform the main program.

        This function corresponds to 'main', which is called after the instance of the class are created.

        :param  workspace_path: Path to subfolder 'workspace' (if empty default path '../<path to this file>' is used)
        :type   workspace_path: str
        """
        # Variable declaration
        ginfo = dct.GeneralInformation
        # Inductor simulation
        isim = InductorOptimization
        # Transformer simulation
        tsim = TransformerOptimization
        # heat sink simulation
        hsim = HeatSinkOptimization
        # Flag for available filtered results
        filtered_circuit_result_folder_exists = False

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

        # Load the configuration for program flow and check the validity
        flow_control_loaded, dict_prog_flow = DctMainCtl.load_toml_file("progFlow.toml")
        toml_prog_flow = tc.FlowControl(**dict_prog_flow)

        if not flow_control_loaded:
            raise ValueError("Program flow toml file does not exist.")

        DctMainCtl.set_up_folder_structure(toml_prog_flow)

        # read study names from toml file names
        circuit_study_name = toml_prog_flow.configuration_data_files.circuit_configuration_file.replace(".toml", "")
        inductor_study_name = toml_prog_flow.configuration_data_files.inductor_configuration_file.replace(".toml", "")
        transformer_study_name = toml_prog_flow.configuration_data_files.transformer_configuration_file.replace(".toml", "")
        heat_sink_study_name = toml_prog_flow.configuration_data_files.heat_sink_configuration_file.replace(".toml", "")

        # --------------------------
        # Circuit flow control
        # --------------------------

        # Init circuit configuration
        is_circuit_loaded, dict_circuit = DctMainCtl.load_toml_file(toml_prog_flow.configuration_data_files.circuit_configuration_file)
        toml_circuit = tc.TomlCircuitParetoDabDesign(**dict_circuit)
        # generate
        config_circuit = DctMainCtl.circuit_toml_2_dto(toml_circuit, toml_prog_flow)

        if not is_circuit_loaded:
            raise ValueError(f"Circuit configuration file: {toml_prog_flow.configuration_data_files.circuit_configuration_file} does not exist.")

        # Add project directory and study name
        DctMainCtl.init_general_info(ginfo, toml_prog_flow)

        # Check, if electrical optimization is to skip
        if toml_prog_flow.circuit.calculation_mode == "skip":
            # Check, if data are available (skip case)
            if not DctMainCtl.check_study_data(ginfo.circuit_study_path, ginfo.circuit_study_name):
                raise ValueError(f"Study {ginfo.circuit_study_name} in path {ginfo.circuit_study_path} does not exist. No sqlite3-database found!")
            # Check if filtered results folder exists
            filtered_circuit_results_datapath = os.path.join(ginfo.circuit_study_path, "filtered_results")
            if os.path.exists(filtered_circuit_results_datapath):
                filtered_circuit_result_folder_exists = True
                # Add filtered result list
                for filtered_circuit_result in os.listdir(filtered_circuit_results_datapath):
                    if os.path.isfile(os.path.join(filtered_circuit_results_datapath, filtered_circuit_result)):
                        ginfo.filtered_list_id.append(os.path.splitext(filtered_circuit_result)[0])
                if not ginfo.filtered_list_id:
                    raise ValueError(f"Filtered results folder {filtered_circuit_results_datapath} is empty.")
            else:
                raise ValueError(f"Filtered circuit results folder {filtered_circuit_results_datapath} does not exist.")

        # --------------------------
        # Inductor flow control
        # --------------------------

        # Load the inductor-configuration parameter
        transformer_toml_filepath = toml_prog_flow.configuration_data_files.inductor_configuration_file
        is_inductor_loaded, inductor_dict = DctMainCtl.load_toml_file(toml_prog_flow.configuration_data_files.inductor_configuration_file)
        toml_inductor = dct.TomlInductor(**inductor_dict)

        if not is_inductor_loaded:
            raise ValueError(f"Inductor configuration file: {transformer_toml_filepath} does not exist.")

        # Check, if inductor optimization is to skip
        if toml_prog_flow.inductor.calculation_mode == "skip":
            # For loop to check, if all filtered values are available

            for id_entry in ginfo.filtered_list_id:
                # Assemble pathname
                inductor_results_datapath = os.path.join(toml_prog_flow.general.project_directory,
                                                         toml_prog_flow.inductor.subdirectory,
                                                         circuit_study_name,
                                                         id_entry,
                                                         inductor_study_name)
                # Check, if data are available (skip case)
                if not DctMainCtl.check_study_data(inductor_results_datapath, inductor_study_name):
                    raise ValueError(
                        f"Study {inductor_study_name} in path {inductor_results_datapath} does not exist. No sqlite3-database found!")

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
            for id_entry in ginfo.filtered_list_id:
                # Assemble pathname
                transformer_results_datapath = os.path.join(toml_prog_flow.general.project_directory,
                                                            toml_prog_flow.transformer.subdirectory,
                                                            circuit_study_name,
                                                            id_entry,
                                                            transformer_study_name)
                # Check, if data are available (skip case)
                if not DctMainCtl.check_study_data(transformer_results_datapath, transformer_study_name):
                    raise ValueError(
                        f"Study {transformer_study_name} in path {transformer_results_datapath} does not exist. No sqlite3-database found!")

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
            # Assemble pathname
            heat_sink_results_datapath = os.path.join(ginfo.heat_sink_study_path, heat_sink_study_name)
            # Check, if data are available (skip case)
            if not DctMainCtl.check_study_data(heat_sink_results_datapath, heat_sink_study_name):
                raise ValueError(
                    f"Study {heat_sink_study_name} in path {heat_sink_results_datapath} does not exist. No sqlite3-database found!")

        # Warning, no data are available
        # Check, if transformer optimization is to skip
        # Warning, no data are available
        # Check, if heat sink optimization is to skip
        # Warning, no data are available

        # -- Start simulation  ----------------------------------------------------------------------------------------
        # --------------------------
        # Circuit optimization
        # --------------------------
        # Check, if electrical optimization is not to skip
        if not toml_prog_flow.circuit.calculation_mode == "skip":
            # Load initialisation data of electrical simulation and initialize
            is_circuit_loaded, circuit_dict = DctMainCtl.load_toml_file(toml_prog_flow.configuration_data_files.circuit_configuration_file)
            toml_circuit = dct.TomlCircuitParetoDabDesign(**circuit_dict)
            config_circuit = DctMainCtl.circuit_toml_2_dto(toml_circuit, toml_prog_flow)

            if not is_circuit_loaded:
                raise ValueError("Electrical configuration not initialized!")
            # Check, if old study is to delete, if available
            if toml_prog_flow.circuit.calculation_mode == "new":
                # delete old study
                is_new_circuit_study = True
            else:
                # overtake the trails of the old study
                is_new_circuit_study = False

            # Start calculation
            dct.CircuitOptimization.start_proceed_study(config_circuit, number_trials=toml_prog_flow.circuit.number_of_trials,
                                                        delete_study=is_new_circuit_study)

        # Check breakpoint
        DctMainCtl.check_breakpoint(toml_prog_flow.breakpoints.circuit_pareto, "Electric Pareto front calculated")

        # Check if filter results are not available
        if not filtered_circuit_result_folder_exists:
            # Calculate the filtered results
            CircuitOptimization.filter_study_results(dab_config=config_circuit)
            # Get filtered result path
            filtered_circuit_results_datapath = os.path.join(ginfo.circuit_study_path, "filtered_results")
            # Add filtered result list
            for filtered_circuit_result in os.listdir(filtered_circuit_results_datapath):
                if os.path.isfile(os.path.join(filtered_circuit_results_datapath, filtered_circuit_result)):
                    ginfo.filtered_list_id.append(os.path.splitext(filtered_circuit_result)[0])

        # Check breakpoint
        DctMainCtl.check_breakpoint(toml_prog_flow.breakpoints.circuit_filtered, "Filtered value of electric Pareto front calculated")

        # --------------------------
        # Inductor optimization
        # --------------------------

        # Check, if inductor optimization is not to skip
        if not toml_prog_flow.inductor.calculation_mode == "skip":
            # Check, if old study is to delete, if available
            if toml_prog_flow.inductor.calculation_mode == "new":
                # delete old study
                is_new_inductor_study = True
            else:
                # overtake the trails of the old study
                is_new_inductor_study = False

            # Start simulation ASA: Filter_factor to correct
            isim.init_configuration(toml_inductor, toml_prog_flow, ginfo)
            isim.simulation_handler(ginfo, toml_prog_flow.inductor.number_of_trials, toml_inductor.filter_distance.factor_min_dc_losses,
                                    is_new_inductor_study)

        # Check breakpoint
        DctMainCtl.check_breakpoint(toml_prog_flow.breakpoints.inductor, "Inductor Pareto front calculated")

        # --------------------------
        # Transformer optimization
        # --------------------------

        # Check, if transformer optimization is not to skip
        if not toml_prog_flow.transformer.calculation_mode == "skip":
            # Check, if old study is to delete, if available
            if toml_prog_flow.transformer.calculation_mode == "new":
                # delete old study
                is_new_transformer_study = True
            else:
                # overtake the trails of the old study
                is_new_transformer_study = False

            tsim.init_configuration(toml_transformer, toml_prog_flow, ginfo)

            # Start simulation ASA: Filter_factor to correct
            tsim.simulation_handler(ginfo, toml_prog_flow.transformer.number_of_trials, toml_transformer.filter_distance.factor_min_dc_losses,
                                    toml_transformer.filter_distance.factor_max_dc_losses, is_new_transformer_study)

        # Check breakpoint
        DctMainCtl.check_breakpoint(toml_prog_flow.breakpoints.transformer, "Transformer Pareto front calculated")

        # --------------------------
        # Heat sink optimization
        # --------------------------

        # Check, if heat sink optimization is to skip
        if not toml_prog_flow.heat_sink.calculation_mode == "skip":
            # Check, if old study is to delete, if available
            if toml_prog_flow.heat_sink.calculation_mode == "new":
                # delete old study
                is_new_heat_sink_study = True
            else:
                # overtake the trails of the old study
                is_new_heat_sink_study = False

            hsim.init_configuration(toml_heat_sink, toml_prog_flow)

            hsim.optimization_handler(ginfo, toml_prog_flow.heat_sink.number_of_trials, is_new_heat_sink_study)

        # Check breakpoint
        DctMainCtl.check_breakpoint(toml_prog_flow.breakpoints.heat_sink, "Heat sink Pareto front calculated")

        # Calculate the combination of components inductor and transformer with same electrical pareto point
        # Filter the pareto front data of inductor and transformer
        # Create a setup of the three components
        # Define the heat sink
        # Add this to the summary pareto list (no optimization?)

        # Check, if electrical optimization is to skip
        # Initialize data
        # Start calculation
        # Filter the pareto front data

        pass


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

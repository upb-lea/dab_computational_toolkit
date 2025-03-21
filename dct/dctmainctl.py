"""Main control program to optimise the DAB converter."""
# python libraries
import os
import sys
import tomllib
import toml

# 3rd party libraries

# own libraries
import dct
# Inductor simulations class
import induct_sim as Inductsimclass
# import transf_sim
import transf_sim as Transfsimclass
# import heatsink_sim
import heatsink_sim as Heatsinksimclass
import toml_checker as tc
import pareto_dtos as p_dtos
from dct import CircuitOptimization


# logging.basicConfig(format='%(levelname)s,%(asctime)s:%(message)s', encoding='utf-8')
# logging.getLogger('pygeckocircuits2').setLevel(logging.DEBUG)

class DctMainCtl:
    """Main class for control dab-optimization."""

    # Fix folder names
    const_circuit_folder = "01_circuit"
    const_inductor_folder = "02_inductor"
    const_transformer_folder = "03_transformer"
    const_heat_sink_folder = "04_heat_sink"

    @staticmethod
    def load_conf_file(target_file: str) -> tuple[bool, dict]:
        """
        Load the toml configuration data to a dictionary.

        :param target_file : File name of the toml-file
        :type  target_file : str
        :return: True, if the data could be loaded successful and the loaded dictionary
        :rtype: bool, dict
        """
        # return value init to false and tomlData to empty
        toml_file_exists = False

        # Separate filename and path
        dirname = os.path.dirname(target_file)
        filename = os.path.basename(target_file)

        # check path
        if os.path.exists(dirname) or dirname == "":
            # check filename
            if os.path.isfile(target_file):
                with open(target_file, "rb") as f:
                    config = tomllib.load(f)
                toml_file_exists = True
            else:
                print("File does not exists!")
        else:
            print("Path does not exists!")

        return toml_file_exists, config

    @staticmethod
    def load_conf_file_deprecated(target_file: str, toml_data: dict) -> bool:
        """
        Load the toml configuration data to dict.

        :param target_file : File name of the toml-file
        :type  target_file : str
        :param toml_data: Reference to the variable for the toml-data
        :type  toml_data: bool
        :return: True, if the data could be loaded successful
        :rtype: bool

        """
        # return value init to false and tomlData to empty
        toml_file_exists = False

        # Separate filename and path
        dirname = os.path.dirname(target_file)
        filename = os.path.basename(target_file)

        # check path
        if os.path.exists(dirname) or dirname == "":
            # check filename
            if os.path.isfile(target_file):
                new_dict_data = toml.load(target_file)
                # Delete old data and copy new data to tomlData
                toml_data.clear()
                toml_data.update(new_dict_data)
                toml_file_exists = True
            else:
                print("File does not exists!")
        else:
            print("Path does not exists!")

        return {toml_file_exists}

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
        act_ginfo.circuit_study_name = act_config_program_flow.general.study_name
        # Create path names
        act_ginfo.circuit_study_path = os.path.join(abs_path, act_ginfo.project_directory,
                                                    DctMainCtl.const_circuit_folder, act_ginfo.circuit_study_name)
        act_ginfo.inductor_study_path = os.path.join(abs_path, act_ginfo.project_directory,
                                                     DctMainCtl.const_inductor_folder, act_ginfo.circuit_study_name)
        act_ginfo.transformer_study_path = os.path.join(abs_path, act_ginfo.project_directory,
                                                        DctMainCtl.const_transformer_folder, act_ginfo.circuit_study_name)
        # Check, if heatsink study name uses the circuit name
        if act_config_program_flow.heat_sink.circuit_study_name_flag == "True":
            act_ginfo.heat_sink_study_path = os.path.join(abs_path, act_ginfo.project_directory,
                                                          DctMainCtl.const_heat_sink_folder, act_ginfo.circuit_study_name)
        else:
            act_ginfo.heat_sink_study_path = os.path.join(abs_path, act_ginfo.project_directory,
                                                          DctMainCtl.const_heat_sink_folder)

    @staticmethod
    def check_study_data(study_path: str, study_name: str) -> bool:
        """
        Verify if the study path and sqlite3-database file exists.

        :param study_path: drive location path to the study
        :type  study_path: str
        :param study_name: Name of the study
        :type  study_name: str
        :return: True, if the optimization could be performed successful
        :rtype: bool
        """
        # return value init to false
        study_exists = False

        # check path
        if os.path.exists(study_path) or study_path == "":
            # Assemble file name
            study_name = study_name + ".sqlite3"
            target_file = os.path.join(study_path, study_name)
            # check filename
            if os.path.isfile(target_file):
                study_exists = True
            else:
                print(f"File {target_file} does not exists!")
        else:
            print(f"Path {study_path} does not exists!")
        # True = file exists
        return {study_exists}

    @staticmethod
    def load_inductor_config(act_ginfo: dct.GeneralInformation, act_config_inductor: dict, act_isim: Inductsimclass.InductorSim) -> bool:
        """
        Load and initialize the inductor optimization configuration.

        :param act_ginfo : General information about the study
        :type  act_ginfo : dct.GeneralInformation:
        :param act_config_inductor: actual inductor configuration information
        :type  act_config_inductor: dict: dictionary with the necessary configuration parameter
        :param act_isim: inductor optimization object reference
        :type  act_isim: Inductsimclass.Inductorsim:
        :return: True, if the configuration is successful
        :rtype: bool
        """
        #   Variable initialisation

        # design space
        designspace_dict = {"core_name_list": act_config_inductor["Designspace"]["core_name_list"],
                            "material_name_list": act_config_inductor["Designspace"]["material_name_list"],
                            "litz_wire_list": act_config_inductor["Designspace"]["litz_wire_list"]}

        #   Insulation Dto
        insulations_dict = {"primary_to_primary": act_config_inductor["InsulationData"]["primary_to_primary"],
                            "core_bot": act_config_inductor["InsulationData"]["core_bot"],
                            "core_top": act_config_inductor["InsulationData"]["core_top"],
                            "core_right": act_config_inductor["InsulationData"]["core_right"],
                            "core_left": act_config_inductor["InsulationData"]["core_left"]}

        # Initialize inductor optimization and return, if it was successful (true)
        return act_isim.init_configuration(act_config_inductor["InductorConfigName"]["inductor_config_name"], act_ginfo, designspace_dict, insulations_dict)

    @staticmethod
    def load_transformer_config(act_ginfo: dct.GeneralInformation, act_config_transformer: dict, act_tsim: Transfsimclass.Transfsim) -> bool:
        """
        Load and initialize the transformer optimization configuration.

        :param act_ginfo : General information about the study
        :type  act_ginfo : dct.GeneralInformation:
        :param act_config_transformer: actual inductor configuration information
        :type  act_config_transformer: dict: dictionary with the necessary configuration parameter
        :param act_tsim: transformer optimization object reference
        :type  act_tsim: Transformersimclass.Transfsim:
        :return: True, if the configuration is successful
        :rtype: bool
        """
        #   Variable initialisation

        # design space
        designspace_dict = {"core_name_list": act_config_transformer["Designspace"]["core_name_list"],
                            "material_name_list": act_config_transformer["Designspace"]["material_name_list"],
                            "core_inner_diameter_min_max_list": act_config_transformer["Designspace"]["core_inner_diameter_min_max_list"],
                            "window_w_min_max_list": act_config_transformer["Designspace"]["window_w_min_max_list"],
                            "window_h_bot_min_max_list": act_config_transformer["Designspace"]["window_h_bot_min_max_list"],
                            "primary_litz_wire_list": act_config_transformer["Designspace"]["primary_litz_wire_list"],
                            "secondary_litz_wire_list": act_config_transformer["Designspace"]["secondary_litz_wire_list"]}

        # Transformer data
        transformer_data_dict = {"max_transformer_total_height": act_config_transformer["TransformerData"]["max_transformer_total_height"],
                                 "max_core_volume": act_config_transformer["TransformerData"]["max_core_volume"],
                                 "n_p_top_min_max_list": act_config_transformer["TransformerData"]["n_p_top_min_max_list"],
                                 "n_p_bot_min_max_list": act_config_transformer["TransformerData"]["n_p_bot_min_max_list"],
                                 "iso_window_top_core_top": act_config_transformer["TransformerData"]["iso_window_top_core_top"],
                                 "iso_window_top_core_bot": act_config_transformer["TransformerData"]["iso_window_top_core_bot"],
                                 "iso_window_top_core_left": act_config_transformer["TransformerData"]["iso_window_top_core_left"],
                                 "iso_window_top_core_right": act_config_transformer["TransformerData"]["iso_window_top_core_right"],
                                 "iso_window_bot_core_top": act_config_transformer["TransformerData"]["iso_window_bot_core_top"],
                                 "iso_window_bot_core_bot": act_config_transformer["TransformerData"]["iso_window_bot_core_bot"],
                                 "iso_window_bot_core_left": act_config_transformer["TransformerData"]["iso_window_bot_core_left"],
                                 "iso_window_bot_core_right": act_config_transformer["TransformerData"]["iso_window_bot_core_right"],
                                 "iso_primary_to_primary": act_config_transformer["TransformerData"]["iso_primary_to_primary"],
                                 "iso_secondary_to_secondary": act_config_transformer["TransformerData"]["iso_secondary_to_secondary"],
                                 "iso_primary_to_secondary": act_config_transformer["TransformerData"]["iso_primary_to_secondary"],
                                 "fft_filter_value_factor": act_config_transformer["TransformerData"]["fft_filter_value_factor"],
                                 "mesh_accuracy": act_config_transformer["TransformerData"]["mesh_accuracy"]}

        # Initialize inductor optimization and return, if it was successful (true)
        return act_tsim.init_configuration(act_config_transformer["TransformerConfigName"]["transformer_config_name"],
                                           act_ginfo, designspace_dict, transformer_data_dict)

    @staticmethod
    def load_heat_sink_config(act_ginfo: dct.GeneralInformation, act_config_heat_sink: dict, act_hsim: Heatsinksimclass.HeatSinkSim) -> bool:
        """
        Load and initialize the transformer optimization configuration.

        :param act_ginfo : General information about the study
        :type  act_ginfo : dct.GeneralInformation:
        :param act_config_heat_sink: actual heat sink configuration information
        :type  act_config_heat_sink: dict: heat sink with the necessary configuration parameter
        :param act_hsim: heat sink optimization object reference
        :type  act_hsim: Heatsinksimclass.HeatSinkSim
        :return: True, if the configuration is successful
        :rtype: bool
        """
        # def init_configuration(act_hct_config_name: str, act_ginfo: dct.GeneralInformation, act_designspace_dict: dict,
        #                       act_hctdimension_dict: dict) -> bool:
        #   Variable initialisation

        # Get design space path
        design_space_path = act_config_heat_sink["Designspace"]["heatsink_designspace_path"]

        # Get heat sink dimension data
        hct_dimension_dict = {
            "height_c_list": act_config_heat_sink["DimensionData"]["height_c_list"],
            "width_b_list": act_config_heat_sink["DimensionData"]["width_b_list"],
            "length_l_list": act_config_heat_sink["DimensionData"]["length_l_list"],
            "height_d_list": act_config_heat_sink["DimensionData"]["height_d_list"],
            "number_fins_n_list": act_config_heat_sink["DimensionData"]["number_fins_n_list"],
            "thickness_fin_t_list": act_config_heat_sink["DimensionData"]["thickness_fin_t_list"],
            "t_ambient": act_config_heat_sink["DimensionData"]["t_ambient"],
            "area_min": act_config_heat_sink["DimensionData"]["area_min"],
            "number_directions": act_config_heat_sink["DimensionData"]["number_directions"],
            "factor_pcb_area_copper_coin": act_config_heat_sink["DimensionData"]["factor_pcb_area_copper_coin"],
            "factor_bottom_area_copper_coin": act_config_heat_sink["DimensionData"]["factor_bottom_area_copper_coin"],
            # W/(m*K)
            "thermal_conductivity_copper": act_config_heat_sink["DimensionData"]["thermal_conductivity_copper"]
        }

        # Initialize inductor optimization and return, if it was successful (true)
        return act_hsim.init_configuration(act_config_heat_sink["HeatsinkConfigName"]["heatsink_config_name"], act_ginfo, design_space_path, hct_dimension_dict)

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
        Circuit toml file to dto file.

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

        dto = p_dtos.CircuitParetoDabDesign(
            circuit_study_name=toml_prog_flow.general.study_name,
            project_directory=toml_prog_flow.general.project_directory,
            design_space=design_space,
            output_range=output_range)

        return dto

    @staticmethod
    def executeProgram(workspace_path: str):
        """Perform the main program.

        This function corresponds to 'main', which is called after the instance of the class are created.

        :param  workspace_path: Path to subfolder 'workspace' (if empty default path '../<path to this file>' is used)
        :type   workspace_path: str
        """
        # Variable declaration
        # General information
        ginfo = dct.GeneralInformation
        # program flow parameter
        toml_prog_flow = {}
        # Electric, inductor, transformer and heat sink configuration files
        config_electric = {}
        config_inductor = {}
        config_transformer = {}
        config_heat_sink = {}
        # Inductor simulation
        isim = Inductsimclass.InductorSim
        # Transformer simulation
        tsim = Transfsimclass.Transfsim
        # heat sink simulation
        hsim = Heatsinksimclass.HeatSinkSim
        # Flag for available filtered results
        filtered_resultFlag = False

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
        flow_control_loaded, dict_prog_flow = DctMainCtl.load_conf_file("progFlow.toml")
        toml_prog_flow = tc.FlowControl(**dict_prog_flow)

        if not flow_control_loaded:
            raise ValueError("Program flow toml file does not exist.")

        # Init circuit configuration
        circuit_loaded, dict_circuit = DctMainCtl.load_conf_file(toml_prog_flow.configuration_data_files.circuit_configuration_file)
        toml_circuit = tc.TomlCircuitParetoDabDesign(**dict_circuit)
        # generate
        config_circuit = DctMainCtl.circuit_toml_2_dto(toml_circuit, toml_prog_flow)

        if not circuit_loaded:
            raise ValueError(f"Electrical configuration file: {toml_prog_flow.configuration_data_files.circuit_configuration_file} does not exist.")

        # Add project directory and study name
        DctMainCtl.init_general_info(ginfo, toml_prog_flow)

        # Check, if electrical optimization is to skip
        if toml_prog_flow.circuit.re_calculation == "skip":
            # Check, if data are available (skip case)
            if not DctMainCtl.check_study_data(ginfo.circuit_study_path, ginfo.circuit_study_name):
                raise ValueError(f"Study {ginfo.circuit_study_name} in path {ginfo.circuit_study_path} does not exist. No sqlite3-database found!")
            # Check if filtered results folder exists
            datapath = os.path.join(ginfo.circuit_study_path, "filtered_results")
            if os.path.exists(datapath):
                # Set Flag to false
                filtered_resultFlag = True
                # Add filtered result list
                for pareto_entry in os.listdir(datapath):
                    if os.path.isfile(os.path.join(datapath, pareto_entry)):
                        ginfo.filtered_list_id.append(os.path.splitext(pareto_entry)[0])

            # Assemble pathname
            datapath = os.path.join(toml_prog_flow.general.project_directory,
                                    toml_prog_flow.general.subdirectory,
                                    toml_prog_flow.general.study_name)

            # Check, if data are available (skip case)
            if not DctMainCtl.check_study_data(datapath, toml_prog_flow.general.study_name):
                raise ValueError(
                    f"Study {toml_prog_flow.general.study_name} in path {datapath} "
                    "does not exist. No sqlite3-database found!"
                )

        # Load the inductor-configuration parameter
        target_file = toml_prog_flow.configuration_data_files.inductor_configuration_file
        if not DctMainCtl.load_conf_file_deprecated(target_file, config_inductor):
            raise ValueError(f"Inductor configuration file: {target_file} does not exist.")

        # Check, if inductor optimization is to skip
        if toml_prog_flow.inductor.re_calculation == "skip":
            # For loop to check, if all filtered values are available
            for id_entry in ginfo.filtered_list_id:
                # Assemble pathname
                datapath = os.path.join(toml_prog_flow.general.project_directory,
                                        toml_prog_flow.inductor.subdirectory,
                                        toml_prog_flow.general.study_name,
                                        id_entry,
                                        config_inductor["InductorConfigName"]["inductor_config_name"])
                # Check, if data are available (skip case)
                if not DctMainCtl.check_study_data(datapath, "inductor_01"):
                    raise ValueError(f"Study {toml_prog_flow.general.study_name} in path {datapath} does not exist. No sqlite3-database found!")

        # Load the transformer-configuration parameter
        target_file = toml_prog_flow.configuration_data_files.transformer_configuration_file
        if not DctMainCtl.load_conf_file_deprecated(target_file, config_transformer):
            raise ValueError(f"Transformer configuration file: {target_file} does not exist.")

        # Check, if transformer optimization is to skip
        if toml_prog_flow.transformer.re_calculation == "skip":
            # For loop to check, if all filtered values are available
            for id_entry in ginfo.filtered_list_id:
                # Assemble pathname
                datapath = os.path.join(toml_prog_flow.general.project_directory,
                                        toml_prog_flow.transformer.subdirectory,
                                        toml_prog_flow.general.study_name,
                                        id_entry,
                                        config_transformer["TransformerConfigName"]["transformer_config_name"])
                # Check, if data are available (skip case)
                if not DctMainCtl.check_study_data(datapath, "transformer_01"):
                    raise ValueError(f"Study {toml_prog_flow.general.study_name} in path {datapath} does not exist. No sqlite3-database found!")

        # Load the heat sink-configuration parameter
        target_file = toml_prog_flow.configuration_data_files.heat_sink_configuration_file
        if not DctMainCtl.load_conf_file_deprecated(target_file, config_heat_sink):
            raise ValueError(f"Heat sink configuration file: {target_file} does not exist.")

        # Check, if heat sink optimization is to skip
        if toml_prog_flow.heat_sink.re_calculation == "skip":
            # Assemble pathname
            datapath = os.path.join(ginfo.heat_sink_study_path, config_heat_sink["HeatsinkConfigName"]["heatsink_config_name"])
            # Check, if data are available (skip case)
            if not DctMainCtl.check_study_data(datapath, "heatsink_01"):
                raise ValueError(f"Study {toml_prog_flow.general.study_name} in path {datapath} does not exist. No sqlite3-database found!")

        # Warning, no data are available
        # Check, if transformer optimization is to skip
        # Warning, no data are available
        # Check, if heat sink optimization is to skip
        # Warning, no data are available

        # -- Start simulation  ----------------------------------------------------------------------------------------

        # Check, if electrical optimization is not to skip
        if not toml_prog_flow.circuit.re_calculation == "skip":
            # Load initialisation data of electrical simulation and initialize
            circuit_loaded, circuit_dict = DctMainCtl.load_conf_file(toml_prog_flow.configuration_data_files.circuit_configuration_file)
            toml_circuit = dct.TomlCircuitParetoDabDesign(**circuit_dict)
            # generate

            config_circuit = DctMainCtl.circuit_toml_2_dto(toml_circuit, toml_prog_flow)

            if not circuit_loaded:
                raise ValueError("Electrical configuration not initialized!")
            # Check, if old study is to delete, if available
            if toml_prog_flow.circuit.re_calculation == "new":
                # delete old study
                new_study_flag = True
            else:
                # overtake the trails of the old study
                new_study_flag = False

            # Start calculation
            dct.CircuitOptimization.start_proceed_study(config_circuit, number_trials=toml_prog_flow.circuit.number_of_trials, delete_study=new_study_flag)

        # Check breakpoint
        DctMainCtl.check_breakpoint(toml_prog_flow.breakpoints.circuit_pareto, "Electric Pareto front calculated")

        # Check if filter results are not available
        if not filtered_resultFlag:
            # Calculate the filtered results
            CircuitOptimization.filter_study_results(dab_config=config_circuit)
            # Get filtered result path
            datapath = os.path.join(ginfo.circuit_study_path, "filtered_results")
            # Add filtered result list
            for pareto_entry in os.listdir(datapath):
                if os.path.isfile(os.path.join(datapath, pareto_entry)):
                    ginfo.filtered_list_id.append(os.path.splitext(pareto_entry)[0])

        # Check breakpoint
        DctMainCtl.check_breakpoint(toml_prog_flow.breakpoints.circuit_filtered, "Filtered value of electric Pareto front calculated")

        # Check, if inductor optimization is not to skip
        if not toml_prog_flow.inductor.re_calculation == "skip":
            # Load initialisation data of inductor simulation and initialize
            if not DctMainCtl.load_inductor_config(ginfo, config_inductor, isim):
                raise ValueError("Inductor configuration not initialized!")
            # Check, if old study is to delete, if available
            if toml_prog_flow.inductor.re_calculation == "new":
                # delete old study
                new_study_flag = True
            else:
                # overtake the trails of the old study
                new_study_flag = False

            # Start simulation ASA: Filter_factor to correct
            isim.simulation_handler(ginfo, toml_prog_flow.inductor.number_of_trials, 1.0, new_study_flag)

        # Check breakpoint
        DctMainCtl.check_breakpoint(toml_prog_flow.breakpoints.inductor, "Inductor Pareto front calculated")

        # Check, if transformer optimization is not to skip
        if not toml_prog_flow.transformer.re_calculation == "skip":
            # Load initialisation data of transformer simulation and initialize
            if not DctMainCtl.load_transformer_config(ginfo, config_transformer, tsim):
                raise ValueError("Transformer configuration not initialized!")
            # Check, if old study is to delete, if available
            if toml_prog_flow.transformer.re_calculation == "new":
                # delete old study
                new_study_flag = True
            else:
                # overtake the trails of the old study
                new_study_flag = False

            # Start simulation ASA: Filter_factor to correct
            tsim.simulation_handler(ginfo, toml_prog_flow.transformer.number_of_trials, 1.0, new_study_flag)

        # Check breakpoint
        DctMainCtl.check_breakpoint(toml_prog_flow.breakpoints.transformer, "Transformer Pareto front calculated")

        # Check, if heat sink optimization is to skip
        if not toml_prog_flow.heat_sink.re_calculation == "skip":
            # Load initialisation data of heat sink simulation and initialize
            if not DctMainCtl.load_heat_sink_config(ginfo, config_heat_sink, hsim):
                raise ValueError("Heat sink configuration not initialized!")
            # Check, if old study is to delete, if available
            if toml_prog_flow.heat_sink.re_calculation == "new":
                # delete old study
                new_study_flag = True
            else:
                # overtake the trails of the old study
                new_study_flag = False

            # Start simulation ASA: Filter_factor to correct
            hsim.simulation_handler(ginfo, toml_prog_flow.heat_sink.number_of_trials, new_study_flag)

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
    dct_mctl.executeProgram(arg1)

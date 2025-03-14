"""Main control program to optimise the DAB converter."""
# python libraries
import os
import sys
import multiprocessing

# 3rd party libraries
import toml

# own libraries
import dct
# Electrical circuit simulations class
import circuit_sim as Elecsimclass
# Inductor simulations class
import induct_sim as Inductsimclass
# Import transf_sim
import transf_sim as Transfsimclass
# Import heatsink_sim
from heatsink_sim import HeatSinkSim as Heatsinksimclass
# Import server control class
import summary_processing as SumProcessing
# Import server control class

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
    def load_conf_file(target_file: str, toml_data: dict) -> bool:
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
        retval = False

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
                retval = True
            else:
                print("File does not exists!")
        else:
            print("Path does not exists!")

        return {retval}

    @staticmethod
    def generate_conf_file(path: str) -> bool:
        """
        Create and save the configuration file.

        Generate following default configuration files within the path:
        DabElectricConf.toml, DabInductorConf.toml, DabTransformerConf.toml and DabHeatsinkConf.toml,

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
    def init_general_info(act_ginfo: dct.GeneralInformation, act_config_program_flow: dict):
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
        act_ginfo.project_directory = act_config_program_flow["general"]["ProjectDirectory"]
        act_ginfo.circuit_study_name = act_config_program_flow["general"]["StudyName"]
        # Create path names
        act_ginfo.circuit_study_path = os.path.join(abs_path, act_ginfo.project_directory,
                                                    DctMainCtl.const_circuit_folder, act_ginfo.circuit_study_name)
        act_ginfo.inductor_study_path = os.path.join(abs_path, act_ginfo.project_directory,
                                                     DctMainCtl.const_inductor_folder, act_ginfo.circuit_study_name)
        act_ginfo.transformer_study_path = os.path.join(abs_path, act_ginfo.project_directory,
                                                        DctMainCtl.const_transformer_folder, act_ginfo.circuit_study_name)
        # Check, if heatsink study name uses the circuit name
        if act_config_program_flow["heatsink"]["CircuitStudyNameFlag"] == "True":
            act_ginfo.heatsink_study_path = os.path.join(abs_path, act_ginfo.project_directory,
                                                         DctMainCtl.const_heat_sink_folder, act_ginfo.circuit_study_name)
        else:
            act_ginfo.heatsink_study_path = os.path.join(abs_path, act_ginfo.project_directory,
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
        retval = False

        # check path
        if os.path.exists(study_path) or study_path == "":
            # Assemble file name
            study_name = study_name + ".sqlite3"
            target_file = os.path.join(study_path, study_name)
            # check filename
            if os.path.isfile(target_file):
                retval = True
            else:
                print(f"File {target_file} does not exists!")
        else:
            print(f"Path {study_path} does not exists!")
        # True = file exists
        return {retval}

    @staticmethod
    def load_elec_config(act_ginfo: dct.GeneralInformation, act_config_electric: dict, act_esim: Elecsimclass.CircuitSim) -> bool:
        """
        Load and initialize the electric circuit optimization configuration.

        :param act_ginfo : General information about the study
        :type  act_ginfo : dct.GeneralInformation
        :param act_config_electric: actual electric configuration information
        :type  act_config_electric: dict : dictionary with the necessary configuration parameter
        :param act_esim: electric optimization object reference
        :type  act_esim: Elecsimclass.Elecsim:
        :return: True, if the configuration is successful
        :rtype: bool
        """
        # Design space
        design_space = dct.CircuitParetoDesignSpace(
            f_s_min_max_list=act_config_electric["Designspace"]["f_s_min_max_list"],
            l_s_min_max_list=act_config_electric["Designspace"]["l_s_min_max_list"],
            l_1_min_max_list=act_config_electric["Designspace"]["l_1_min_max_list"],
            l_2__min_max_list=act_config_electric["Designspace"]["l_2__min_max_list"],
            n_min_max_list=act_config_electric["Designspace"]["n_min_max_list"],
            transistor_1_name_list=act_config_electric["Designspace"]["transistor_1_name_list"],
            transistor_2_name_list=act_config_electric["Designspace"]["transistor_2_name_list"],
            c_par_1=act_config_electric["Designspace"]["c_par_1"],
            c_par_2=act_config_electric["Designspace"]["c_par_2"]
        )
        # Output range
        output_range = dct.CircuitOutputRange(
            v_1_min_nom_max_list=act_config_electric["OutputRange"]["v_1_min_nom_max_list"],
            v_2_min_nom_max_list=act_config_electric["OutputRange"]["v_2_min_nom_max_list"],
            p_min_nom_max_list=act_config_electric["OutputRange"]["p_min_nom_max_list"],
            steps_per_direction=act_config_electric["OutputRange"]["steps_per_direction"]
        )

        # study name path
        dab_config = dct.CircuitParetoDabDesign(
            circuit_study_name=act_ginfo.circuit_study_name,
            project_directory=act_ginfo.project_directory,
            design_space=design_space,
            output_range=output_range
        )

        # Initialize configuration and return, if this is successful

        return act_esim.init_configuration(dab_config)

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

        # Designspace
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
    def load_heat_sink_config(act_ginfo: dct.GeneralInformation, act_config_heat_sink: dict, act_hsim: Heatsinksimclass) -> bool:
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

    @staticmethod  # (ginfo, config_heat_sink, spro)
    def init_summary_thermal_data(act_ginfo: dct.GeneralInformation, act_config_heat_sink: dict, act_spro: SumProcessing.DctSummmaryProcessing) -> bool:
        """
        Initialize thermal data for summary processing.

        :param act_ginfo : General information about the study
        :type  act_ginfo : dct.GeneralInformation:
        :param act_config_heat_sink: actual heat sink configuration information
        :type  act_config_heat_sink: dict: heat sink with the necessary configuration parameter
        :param act_spro: summary processing object reference
        :type  act_spro: SumProcessing.DctSummmaryProcessing
        :return: True, if the configuration is successful
        :rtype: bool
        """
        #   Variable initialisation

        # Get heat sink dimension data
        thermal_configuration_dict = {
            "transistor_b1_cooling": act_config_heat_sink["ThermalResistanceData"]["transistor_b1_cooling"],
            "transistor_b2_cooling": act_config_heat_sink["ThermalResistanceData"]["transistor_b2_cooling"],
            "inductor_cooling": act_config_heat_sink["ThermalResistanceData"]["inductor_cooling"],
            "transformer_cooling": act_config_heat_sink["ThermalResistanceData"]["transformer_cooling"],
            "heat_sink": act_config_heat_sink["ThermalResistanceData"]["heat_sink"],
        }

        # Initialize inductor optimization and return, if it was successful (true)
        return act_spro.init_thermal_configuration(thermal_configuration_dict)

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
        config_program_flow = {}
        # Electric, inductor, transformer and heat sink configuration files
        config_electric = {}
        config_inductor = {}
        config_transformer = {}
        config_heat_sink = {}
        # Electrical simulation
        esim = Elecsimclass.CircuitSim
        # Inductor simulation
        isim = Inductsimclass.InductorSim
        # Transformer simulation
        tsim = Transfsimclass.Transfsim
        # Heat sink simulation
        hsim = Heatsinksimclass
        # Summary processing
        spro = SumProcessing.DctSummmaryProcessing
        # Flag for available filtered results
        filtered_resultFlag = False
        # Shared Memory für das Histogramm und den Status
        histogram_data = multiprocessing.Array('i', [0] * 25)

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
        if not DctMainCtl.load_conf_file("progFlow.toml", config_program_flow):
            raise ValueError("Program flow toml file does not exist.")

        # Init dab-configuration
        # Load the dab-configuration and electrical parameter
        target_file = config_program_flow["configurationdatafiles"]["ElectricalConfFile"]
        if not DctMainCtl.load_conf_file(target_file, config_electric):
            raise ValueError(f"Electrical configuration file: {target_file} does not exist.")

        # Add project directory and study name
        DctMainCtl.init_general_info(ginfo, config_program_flow)

        # Check, if electrical optimization is to skip
        if config_program_flow["electrical"]["ReCalculation"] == "skip":
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
            datapath = os.path.join(config_program_flow["general"]["ProjectDirectory"],
                                    config_program_flow["electrical"]["Subdirectory"],
                                    config_program_flow["general"]["StudyName"])

            # Check, if data are available (skip case)
            if not DctMainCtl.check_study_data(datapath, config_program_flow["general"]["StudyName"]):
                raise ValueError(
                    f"Study {config_program_flow['general']['StudyName']} in path {datapath} "
                    "does not exist. No sqlite3-database found!"
                )

        # Load the inductor-configuration parameter
        target_file = config_program_flow["configurationdatafiles"]["InductorConfFile"]
        if not DctMainCtl.load_conf_file(target_file, config_inductor):
            raise ValueError(f"Inductor configuration file: {target_file} does not exist.")

        # Check, if inductor optimization is to skip
        if config_program_flow["inductor"]["ReCalculation"] == "skip":
            # For loop to check, if all filtered values are available
            for id_entry in ginfo.filtered_list_id:
                # Assemble pathname
                datapath = os.path.join(config_program_flow["general"]["ProjectDirectory"],
                                        config_program_flow["inductor"]["Subdirectory"],
                                        config_program_flow["general"]["StudyName"],
                                        id_entry,
                                        config_inductor["InductorConfigName"]["inductor_config_name"])
                # Check, if data are available (skip case)
                if not DctMainCtl.check_study_data(datapath, "inductor_01"):
                    raise ValueError(f"Study {config_program_flow['general']['StudyName']} in path {datapath} does not exist. No sqlite3-database found!")

        # Load the transformer-configuration parameter
        target_file = config_program_flow["configurationdatafiles"]["TransformerConfFile"]
        if not DctMainCtl.load_conf_file(target_file, config_transformer):
            raise ValueError(f"Transformer configuration file: {target_file} does not exist.")

        # Check, if transformer optimization is to skip
        if config_program_flow["transformer"]["ReCalculation"] == "skip":
            # For loop to check, if all filtered values are available
            for id_entry in ginfo.filtered_list_id:
                # Assemble pathname
                datapath = os.path.join(config_program_flow["general"]["ProjectDirectory"],
                                        config_program_flow["transformer"]["Subdirectory"],
                                        config_program_flow["general"]["StudyName"],
                                        id_entry,
                                        config_transformer["TransformerConfigName"]["transformer_config_name"])
                # Check, if data are available (skip case)
                if not DctMainCtl.check_study_data(datapath, "transformer_01"):
                    raise ValueError(f"Study {config_program_flow['general']['StudyName']} in path {datapath} does not exist. No sqlite3-database found!")

        # Load the heat sink-configuration parameter
        target_file = config_program_flow["configurationdatafiles"]["HeatsinkConfFile"]
        if not DctMainCtl.load_conf_file(target_file, config_heat_sink):
            raise ValueError(f"Heat sink configuration file: {target_file} does not exist.")

        # Check, if heat sink optimization is to skip
        if config_program_flow["heatsink"]["ReCalculation"] == "skip":
            # Assemble pathname
            datapath = os.path.join(ginfo.heatsink_study_path, config_heat_sink["HeatsinkConfigName"]["heatsink_config_name"])
            # Check, if data are available (skip case)
            if not DctMainCtl.check_study_data(datapath, "heatsink_01"):
                raise ValueError(f"Study {config_program_flow['general']['StudyName']} in path {datapath} does not exist. No sqlite3-database found!")

        # -- Start server  --------------------------------------------------------------------------------------------
        # Debug: Server switched off
        # srv_ctl.start_dct_server(histogram_data,False)

        # -- Start simulation  ----------------------------------------------------------------------------------------

        # Check, if electrical optimization is not to skip
        if not config_program_flow["electrical"]["ReCalculation"] == "skip":
            # Load initialisation data of electrical simulation and initialize
            if not DctMainCtl.load_elec_config(ginfo, config_electric, esim):
                raise ValueError("Electrical configuration not initialized!")
            # Check, if old study is to delete, if available
            if config_program_flow["electrical"]["ReCalculation"] == "new":
                # delete old study
                new_study_flag = True
            else:
                # overtake the trails of the old study
                new_study_flag = False

            # Start calculation
            esim.run_new_study(config_program_flow["electrical"]["NumberOfTrials"], new_study_flag)

        # Check breakpoint
        DctMainCtl.check_breakpoint(config_program_flow["breakpoints"]["Electrical_pareto"], "Electric Pareto front calculated")

        # Check if filter results are not available
        if not filtered_resultFlag:
            # Calculate the filtered results
            esim.filter_study_results_and_run_gecko()
            # Get filtered result path
            datapath = os.path.join(ginfo.circuit_study_path, "filtered_results")
            # Add filtered result list
            for pareto_entry in os.listdir(datapath):
                if os.path.isfile(os.path.join(datapath, pareto_entry)):
                    ginfo.filtered_list_id.append(os.path.splitext(pareto_entry)[0])

        # Check breakpoint
        DctMainCtl.check_breakpoint(config_program_flow["breakpoints"]["Electrical_filtered"], "Filtered value of electric Pareto front calculated")

        # Check, if inductor optimization is not to skip
        if not config_program_flow["inductor"]["ReCalculation"] == "skip":
            # Load initialisation data of inductor simulation and initialize
            if not DctMainCtl.load_inductor_config(ginfo, config_inductor, isim):
                raise ValueError("Inductor configuration not initialized!")
            # Check, if old study is to delete, if available
            if config_program_flow["inductor"]["ReCalculation"] == "new":
                # delete old study
                new_study_flag = True
            else:
                # overtake the trails of the old study
                new_study_flag = False

            # Start simulation ASA: Filter_factor to correct
            isim.simulation_handler(ginfo, config_program_flow["inductor"]["NumberOfTrials"], 1.0, new_study_flag)

        # Check breakpoint
        DctMainCtl.check_breakpoint(config_program_flow["breakpoints"]["Inductor"], "Inductor Pareto front calculated")

        # Check, if transformer optimization is not to skip
        if not config_program_flow["transformer"]["ReCalculation"] == "skip":
            # Load initialisation data of transformer simulation and initialize
            if not DctMainCtl.load_transformer_config(ginfo, config_transformer, tsim):
                raise ValueError("Transformer configuration not initialized!")
            # Check, if old study is to delete, if available
            if config_program_flow["transformer"]["ReCalculation"] == "new":
                # delete old study
                new_study_flag = True
            else:
                # overtake the trails of the old study
                new_study_flag = False

            # Start simulation ASA: Filter_factor to correct
            tsim.simulation_handler(ginfo, config_program_flow["transformer"]["NumberOfTrials"], 1.0, new_study_flag)

        # Check breakpoint
        DctMainCtl.check_breakpoint(config_program_flow["breakpoints"]["Transformer"], "Transformer Pareto front calculated")

        # Check, if heat sink optimization is to skip
        if not config_program_flow["heatsink"]["ReCalculation"] == "skip":
            # Load initialisation data of heat sink simulation and initialize
            if not DctMainCtl.load_heat_sink_config(ginfo, config_heat_sink, hsim):
                raise ValueError("Heat sink configuration not initialized!")
            # Check, if old study is to delete, if available
            if config_program_flow["heatsink"]["ReCalculation"] == "new":
                # delete old study
                new_study_flag = True
            else:
                # overtake the trails of the old study
                new_study_flag = False

            # Start simulation ASA: Filter_factor to correct
            hsim.simulation_handler(ginfo, config_program_flow["heatsink"]["NumberOfTrials"], new_study_flag)

        # Check breakpoint
        DctMainCtl.check_breakpoint(config_program_flow["breakpoints"]["Heatsink"], "Heat sink Pareto front calculated")

        # Initialisation thermal data
        if not DctMainCtl.init_summary_thermal_data(ginfo, config_heat_sink, spro):
            raise ValueError("Thermal data configuration not initialized!")
        # Create list of inductor and transformer study (ASA: Currently not implemented in configuration files)
        inductor_study_names = [config_inductor["InductorConfigName"]["inductor_config_name"]]
        stacked_transformer_study_names = [config_transformer["TransformerConfigName"]["transformer_config_name"]]
        # Start summary processing by generating the dataframe from calculated simmulation results
        s_df = spro.generate_result_database(ginfo, inductor_study_names, stacked_transformer_study_names)
        #  Select the needed heatsink configuration
        spro.select_heatsink_configuration(ginfo, config_heat_sink["HeatsinkConfigName"]["heatsink_config_name"], s_df)

        # Check breakpoint
        DctMainCtl.check_breakpoint(config_program_flow["breakpoints"]["Summary"], "Calculation is complete")

        # Join process if necessary
        esim.join_process()
        # Shut down server
        # Debug: Server switched off
        # srv_ctl.stop_dct_server()
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

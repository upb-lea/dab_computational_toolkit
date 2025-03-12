"""Main control program to optimise the DAB converter."""
# python libraries
import os
import sys
import base64
import multiprocessing
import random
import time
import io
import matplotlib.pyplot as plt
import uvicorn
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
import threading

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
import heatsink_sim as Heatsinksimclass
# Import server control class
import server_ctl as Serverctlclass


# logging.basicConfig(format='%(levelname)s,%(asctime)s:%(message)s', encoding='utf-8')
# logging.getLogger('pygeckocircuits2').setLevel(logging.DEBUG)

class DctMainCtl:
    """Main class for control dab-optimization."""

    # Fix folder names
    const_circuit_folder = "01_circuit"
    const_inductor_folder = "02_inductor"
    const_transformer_folder = "03_transformer"
    const_heatsink_folder = "04_heat_sink"

    @staticmethod
    def load_conf_file(targetfile: str, toml_data: dict) -> bool:
        """
        Load the toml configuration data to dict.

        :param targetfile : File name of the toml-file
        :type  targetfile : str:
        :param toml_data: Reference to the variable for the toml-data
        :type  toml_data: bool:
        :return: True, if the data could be loaded sucessfull
        :rtype: bool

        """
        # return value init to false and tomlData to empty
        retval = False

        # Separate filename and path
        dirname = os.path.dirname(targetfile)
        filename = os.path.basename(targetfile)

        # check path
        if os.path.exists(dirname) or dirname == "":
            # check filename
            if os.path.isfile(targetfile):
                new_dict_data = toml.load(targetfile)
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
    def userInput_BrkPt(breakpointkey: str, info: str):
        """
        Continue, wait for user input or stop the programm according breakpoint configuration.

        :param  breakpointkey: Breakpoint configuration keyword
        :type   breakpointkey: str
        :param info: Information text displayed at this breakpoint (if programm waits or stops).
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
                                                         DctMainCtl.const_heatsink_folder, act_ginfo.circuit_study_name)
        else:
            act_ginfo.heatsink_study_path = os.path.join(abs_path, act_ginfo.project_directory,
                                                         DctMainCtl.const_heatsink_folder)

    @staticmethod
    def check_study_data(studypath: str, studyname: str) -> bool:
        """
        Verify if the study path and sqlite3-databasefile exists.

        :param studypath: drive location path to the study
        :type  studypath: str
        :param studyname: Name of the study
        :type  studyname: str
        :return: True, if the optimization could be performed sucessfull
        :rtype: bool
        """
        # return value init to false
        retval = False

        # check path
        if os.path.exists(studypath) or studypath == "":
            # Assemble file name
            studyname = studyname+".sqlite3"
            targetfile = os.path.join(studypath, studyname)
            # check filename
            if os.path.isfile(targetfile):
                retval = True
            else:
                print(f"File {targetfile} does not exists!")
        else:
            print(f"Path {studypath} does not exists!")
        # True = file exists
        return {retval}

    @staticmethod
    def load_elec_config(act_ginfo: dct.GeneralInformation, act_config_electric: dict, act_esim: Elecsimclass.Elecsim) -> bool:
        """
        Load and initialize the electric circuit optimization configuration.

        :param act_ginfo : General information about the study
        :type  act_ginfo : dct.GeneralInformation
        :param act_config_electric: actual electric configuration information
        :type  act_config_electric: dict : dictionary with the necessary configuration parameter
        :param act_esim: electric optimization object reference
        :type  act_esim: Elecsimclass.Elecsim:
        :return: True, if the configuration is sucessfull
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

        # Initialize configuration and return, if this is successfull

        return act_esim.init_configuration(dab_config)

    @staticmethod
    def load_inductor_config(act_ginfo: dct.GeneralInformation, act_config_inductor: dict, act_isim: Inductsimclass.Inductorsim) -> bool:
        """
        Load and initialize the inductor optimization configuration.

        :param act_ginfo : General information about the study
        :type  act_ginfo : dct.GeneralInformation:
        :param act_config_inductor: actual inductor configuration information
        :type  act_config_inductor: dict: dictionary with the necessary configuration parameter
        :param act_isim: inductor optimization object reference
        :type  act_isim: Inductsimclass.Inductorsim:
        :return: True, if the configuration is sucessfull
        :rtype: bool
        """
        #   Variable initialisation

        # Designspace
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
        :return: True, if the configuration is sucessfull
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
    def load_heatsink_config(act_ginfo: dct.GeneralInformation, act_config_heatsink: dict, act_hsim: Heatsinksimclass.Heatsinksim) -> bool:
        """
        Load and initialize the transformer optimization configuration.

        :param act_ginfo : General information about the study
        :type  act_ginfo : dct.GeneralInformation:
        :param act_config_heatsink: actual heatsink configuration information
        :type  act_config_heatsink: dict: heatsink with the necessary configuration parameter
        :param act_hsim: heatsink optimization object reference
        :type  act_hsim: Heatsinksimclass.Heatsinksim
        :return: True, if the configuration is sucessfull
        :rtype: bool
        """
        # def init_configuration(act_hct_config_name: str, act_ginfo: dct.GeneralInformation, act_designspace_dict: dict,
        #                       act_hctdimension_dict: dict) -> bool:
        #   Variable initialisation

        # Get designspace path
        designspace_path = act_config_heatsink["Designspace"]["heatsink_designspace_path"]

        # Get heatsink dimension data
        hctdimension_dict = {"height_c_list": act_config_heatsink["DimensionData"]["height_c_list"],
                             "width_b_list": act_config_heatsink["DimensionData"]["width_b_list"],
                             "length_l_list": act_config_heatsink["DimensionData"]["length_l_list"],
                             "height_d_list": act_config_heatsink["DimensionData"]["height_d_list"],
                             "number_fins_n_list": act_config_heatsink["DimensionData"]["number_fins_n_list"],
                             "thickness_fin_t_list": act_config_heatsink["DimensionData"]["thickness_fin_t_list"],
                             "t_ambient": act_config_heatsink["DimensionData"]["t_ambient"],
                             "area_min": act_config_heatsink["DimensionData"]["area_min"],
                             "number_directions": act_config_heatsink["DimensionData"]["number_directions"],
                             "factor_pcb_area_copper_coin": act_config_heatsink["DimensionData"]["factor_pcb_area_copper_coin"],
                             "factor_bottom_area_copper_coin": act_config_heatsink["DimensionData"]["factor_bottom_area_copper_coin"],
                             # W/(m*K)
                             "thermal_conductivity_copper": act_config_heatsink["DimensionData"]["thermal_conductivity_copper"]}

        # Initialize inductor optimization and return, if it was successful (true)
        return act_hsim.init_configuration(act_config_heatsink["HeatsinkConfigName"]["heatsink_config_name"], act_ginfo, designspace_path, hctdimension_dict)

    @staticmethod
    def check_breakpoint(breakpointkey: str, info: str):
        """
        Continue, wait for user input or stop the programm according breakpoint configuration.

        :param  breakpointkey: Breakpoint configuration keyword
        :type  breakpointkey: str
        :param info: Information text displayed at this breakpoint (if programm waits or stops).
        :type  info: str
        """
        # Check if breakpoint stops the programm
        if breakpointkey == "Stop":
            print("Program stops cause by breakpoint at: '"+info+"'!")
            # stop program
            sys.exit()

        elif breakpointkey == "Pause":
            # Information
            print("Active breakpoint at: '"+info+"'!\n")
            print("'C'=continue, 'S'=stop the program. Please enter your choise")
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
    def start_dct_server(req_stop_server,stop_flag):
        """Starts the server to control and supervice simulation.

        :param req_stop_server: Shared memory flag to request server to stop
        :type  req_stop_server: multiprocessing.Value
        :param stop_flag: Shared memory flag which indicates that the server stops the measurment
        :type  stop_flag: multiprocessing.Value
        """
        # Mounten des Stylesheetpfades
        app.mount("/StyleSheets", StaticFiles(directory="htmltemplates/StyleSheets"), name="Stylesheets")

        # Start the server process
        server_process = multiprocessing.Process(target=srv_ctl.run_server, args=(req_stop_server, stop_flag))
        server_process.start();

    @staticmethod
    def stop_dct_server(req_stop_server):
        """Stop the server for the control and supervisuib of the simulation.

        :param req_stop_server: Shared memory flag to request server to stop
        :type  req_stop_server: multiprocessing.Value
        """

        # Request server to stop
        req_stop_server.value = 1
        # Wait for joined server process
        server_process.join(5)

    @staticmethod
    def executeProgram(workspace_path: str):
        """Perform the main programm.

        This function corresponds to 'main', which is called after the instance of the class are created.

        :param  workspace_path: Path to subfolder 'workspace' (if empty default path '../<path to this file>' is used)
        :type   workspace_path: str
        """
        # Variable declaration
        # General information
        ginfo = dct.GeneralInformation
        # program flow parameter
        config_program_flow = {}
        # Electric, inductor, transformer and heatsink configuration files
        config_electric = {}
        config_inductor = {}
        config_transformer = {}
        config_heatsink = {}
        # Electrical simulation
        esim = Elecsimclass.Elecsim
        # Inductor simulation
        isim = Inductsimclass.Inductorsim
        # Transformer simulation
        tsim = Transfsimclass.Transfsim
        # heatsink simulation
        hsim = Heatsinksimclass.Heatsinksim
        # Flag for available filtered results
        filtered_resultFlag = False
        # Server class to control the workflow
        srv_ctl = Serverctlclass
        # Shared Memory für das Histogramm und den Status
        # histogram_data = multiprocessing.Array('i', [0] * 25)
        req_stop_server = multiprocessing.Value('i', 0)
        stop_flag = multiprocessing.Value('i', 0)


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
        targetfile = config_program_flow["configurationdatafiles"]["ElectricalConfFile"]
        if not DctMainCtl.load_conf_file(targetfile, config_electric):
            raise ValueError(f"Electrical configuration file: {targetfile} does not exist.")

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
        targetfile = config_program_flow["configurationdatafiles"]["InductorConfFile"]
        if not DctMainCtl.load_conf_file(targetfile, config_inductor):
            raise ValueError(f"Inductor configuration file: {targetfile} does not exist.")

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
        targetfile = config_program_flow["configurationdatafiles"]["TransformerConfFile"]
        if not DctMainCtl.load_conf_file(targetfile, config_transformer):
            raise ValueError(f"Transformer configuration file: {targetfile} does not exist.")

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

        # Load the heatsink-configuration parameter
        targetfile = config_program_flow["configurationdatafiles"]["HeatsinkConfFile"]
        if not DctMainCtl.load_conf_file(targetfile, config_heatsink):
            raise ValueError(f"Heatsink configuration file: {targetfile} does not exist.")

        # Check, if heatsink optimization is to skip
        if config_program_flow["heatsink"]["ReCalculation"] == "skip":
            # Assemble pathname
            datapath = os.path.join(ginfo.heatsink_study_path, config_heatsink["HeatsinkConfigName"]["heatsink_config_name"])
            # Check, if data are available (skip case)
            if not DctMainCtl.check_study_data(datapath, "heatsink_01"):
                raise ValueError(f"Study {config_program_flow['general']['StudyName']} in path {datapath} does not exist. No sqlite3-database found!")

        # Warning, no data are available
        # Check, if transformer optimization is to skip
        # Warning, no data are available
        # Check, if heatsink optimization is to skip
        # Warning, no data are available
        # -- Start server  --------------------------------------------------------------------------------------------
        DctMainCtl.start_dct_server(req_stop_server,stop_flag)

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
        DctMainCtl.check_breakpoint(config_program_flow["breakpoints"]["Electrical_pareto"], "Electric paretofront calculated")

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
        DctMainCtl.check_breakpoint(config_program_flow["breakpoints"]["Electrical_filtered"], "Filtered value of electric paretofront calculated")

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
        DctMainCtl.check_breakpoint(config_program_flow["breakpoints"]["Inductor"], "Inductor paretofront calculated")

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
        DctMainCtl.check_breakpoint(config_program_flow["breakpoints"]["Transformer"], "Transformer paretofront calculated")

        # Check, if heatsink optimization is to skip
        if not config_program_flow["heatsink"]["ReCalculation"] == "skip":
            # Load initialisation data of heatsink simulation and initialize
            if not DctMainCtl.load_heatsink_config(ginfo, config_heatsink, hsim):
                raise ValueError("Heatsink configuration not initialized!")
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
        DctMainCtl.check_breakpoint(config_program_flow["breakpoints"]["Heatsink"], "Heatsink paretofront calculated")

        # Calculate the combination of components inductor and transformer with same electrical pareto point
        # Filter the pareto front data of inductor and transformer
        # Create a setup of the three components
        # Define the headsink
        # Add this to the summary pareto list (no optimization?)

        # Check, if electrical optimization is to skip
        # Initialize data
        # Start calculation
        # Filter the pareto front data

        # Join process if necessary
        esim.join_process()
        # Shut down server
        DctMainCtl.stop_dct_server()
        pass


# Program flow control of DAB-optimization
if __name__ == "__main__":
    # Variable declaration
    arg1 = ""

    # Create an mainctl-instance
    dct_mctl = DctMainCtl()
    # Read the command line
    arguments = sys.argv

    # Check on argument, which corresponds to the workspace file location
    if len(arguments) > 1:
        arg1 = arguments[1]
        # Check if this corresponds to the workspace path
        arg1 = os.path.join(arg1, "workspace")
        print(f"Pfad={arg1}")
        # Check if the path not exist (absolute or relative path)
        if not os.path.exists(arg1):
            # Consider it as relative path and create the absolute path
            arg1 = os.path.abspath(arg1)
            print(f"Neuer Pfad={arg1}")
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

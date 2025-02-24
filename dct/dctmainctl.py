"""Main control program to optimise the DAB converter."""
# python libraries
import os
import sys

# 3rd party libraries
import toml

# own libraries
import dct
# Electrical circuit simulations class
import circuit_sim as Elecsimclass
# Inductor simulations class
# import induct_sim as Inductsimclass
# import transf_sim


import logging
logging.basicConfig(format='%(levelname)s,%(asctime)s:%(message)s', encoding='utf-8')
logging.getLogger('pygeckocircuits2').setLevel(logging.DEBUG)

class DctMainCtl:
    """Main class for control dab-optimization."""

    # Fix folder names
    const_circuit_folder = "01_circuit"
    const_inductor_folder = "02_inductor"
    const_transformer_folder = "03_transformer"

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
        # Store project directory and study name
        act_ginfo.project_directory = act_config_program_flow["general"]["ProjectDirectory"]
        act_ginfo.circuit_study_name = act_config_program_flow["general"]["StudyName"]
        # Create path names
        act_ginfo.circuit_study_path = os.path.join(act_ginfo.project_directory,
                                                    DctMainCtl.const_circuit_folder, act_ginfo.circuit_study_name)
        act_ginfo.inductor_study_path = os.path.join(act_ginfo.project_directory,
                                                     DctMainCtl.const_inductor_folder, act_ginfo.circuit_study_name)
        act_ginfo.transformer_study_path = os.path.join(act_ginfo.project_directory,
                                                        DctMainCtl.const_transformer_folder, act_ginfo.circuit_study_name)

    @staticmethod
    def check_study_data(studypath: str, studyname: str) -> bool:
        """
        Verify if the study path exists.

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
    # ASA: Femmt will be installed later
    #def load_inductor_config(act_ginfo: dct.GeneralInformation, act_config_inductor: dict, act_isim: Inductsimclass.Inductorsim) -> bool:
    def load_inductor_config(act_ginfo: dct.GeneralInformation, act_config_inductor: dict, Nix: any) -> bool:
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
        """

        # ASA: Femmt will be installed later
        # return act_isim.init_configuration(act_config_inductor["InductorConfigName"]["inductor_config_name"], act_ginfo, designspace_dict, insulations_dict)
        return False

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
    def executeProgram():
        """Perform the main programm.

        This function corresponds to 'main', which is called after the instance of the class are created.
        """
        # Variable declaration
        # General information
        ginfo = dct.GeneralInformation
        # program flow parameter
        config_program_flow = {}
        # Electric, inductor, transformer and heatsink configuration files
        config_electric = {}
        config_inductor = {}
        # Electrical simulation
        esim = Elecsimclass.Elecsim
        # Inductor simulation
        # ASA: Femmt will be installed later
        # isim = Inductsimclass.Inductorsim
        isim = ""
        # Flag for available filtered results
        filtered_resultFlag = False

        # Change to workspace
        os.chdir("..")
        # change to folder 'workspace'
        try:
            os.chdir("workspace")
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

            # Check, if Optimisation is not skipped
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
                                        config_program_flow["inductor"]["LinkIdSubdirectory"])
                # Check, if data are available (skip case)
                if not DctMainCtl.check_study_data(datapath, "inductor_01"):
                    raise ValueError(f"Study {config_program_flow[general][StudyName]} in path {datapath} does not exist. No sqlite3-database found!")

        # Load the configuration for heatsink optimization

        # Warning, no data are available
        # Check, if transformer optimization is to skip
        # Warning, no data are available
        # Check, if heatsink optimization is to skip
        # Warning, no data are available

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
            # Load initialisation data of electrical simulation and initialize
            if not DctMainCtl.load_inductor_config(ginfo, config_inductor, isim):
                raise ValueError("Inductor configuration not initialized!")
            # Check, if old study is to delete, if available
            if config_program_flow["inductor"]["ReCalculation"] == "new":
                # delete old study
                NewStudyFlag = True
            else:
                # overtake the trails of the old study
                NewStudyFlag = False

            # Start simulation ASA: Comment out
            # isim.simulation_handler(ginfo, 100, 1.0, True)
        # Initialize data
        # Start calculation

        # Check, if transformer optimization is to skip
        # Initialize data
        # Start calculation

        # Check, if heatsink optimization is to skip
        # Perform heatsink calculation

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

        pass


# Program flow control of DAB-optimization
if __name__ == "__main__":
    # Create an mainctl-instance
    dct_mctl = DctMainCtl()
    # Execute program
    dct_mctl.executeProgram()

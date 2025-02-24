"""DAB-Circuit simulation class."""
# python libraries
import os
# Test Multiprocessing
import multiprocessing
# Debug

# 3rd party libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# own libraries
import dct

import logging
logging.basicConfig(format='%(levelname)s,%(asctime)s:%(message)s', encoding='utf-8')
logging.getLogger('pygeckocircuits2').setLevel(logging.DEBUG)


class Elecsim:
    """Optimation of the electrical circuit."""

    # declaration of static membervariables
    # DAB configuration
    _dab_config: dct.CircuitParetoDabDesign
    # Initflag
    _initFlag = False
    # filepath
    _folders: dct.ParetoFilePaths
    # process
    p = None

    @staticmethod
    def init_configuration(act_dab_config: dct.CircuitParetoDabDesign) -> bool:
        """
        Initialize the configuration.

        :param act_dab_config : actual configuration for the optimization
        :type  act_dab_config : dct.CircuitParetoDabDesign


        :return: True, if the configuration was sucessfull
        :rtype: bool
        """
        Elecsim._dab_config = act_dab_config
        # Initialisation are successfull
        Elecsim._initFlag = True        # dab config needs to be checked
        return Elecsim._initFlag

    @staticmethod
    def _fct_add_gecko_simulation_results(act_dto: any) -> bool:
        """
        Store the simulation results of geckocircuits.

        :param act_dto : actual configuration for the optimization
        :type  act_dto : any
        :return: True, if the configuration was sucessfull
        :rtype: bool

        """
        dto_directory = os.path.join(Elecsim._folders.circuit, Elecsim._dab_config.circuit_study_name, "filtered_results")
        result_list = {"DTO": dct.HandleDabDto.add_gecko_simulation_results(act_dto, get_waveforms=True), "Dir": dto_directory, "Name": act_dto.name}
        return True

    @staticmethod
    def run_new_study(no_of_trials: int, deleteFlag: bool = False) -> bool:
        """Run the circuit optimization.

        :param no_of_trials : Number of trials in the genetic optimization algorithm
        :type  no_of_trials : int
        :param deleteFlag: Indication, if the old study are to delete (True) or optimization shall be continued.
        :type  deleteFlag: bool
        :return: True, if the optimization could be performed sucessfull
        :rtype: bool

        """
        # Variable declaration
        retval = False
        # Check the number of trials
        if no_of_trials > 0:
            # Debug Test
            # Verbindung zur MySQL-Datenbank
            # storage_url = "mysql+pymysql://oaml_optuna:optuna@localhost/optuna_db"
            # Create storage-Objekt for Optuna
            # storage = optuna.storages.RDBStorage(storage_url)

            dct.Optimization.start_proceed_study(dab_config=Elecsim._dab_config, number_trials=no_of_trials, deleteStudyFlag=deleteFlag)
            # ASA: Comment out because GUI is not available
            # dct.Optimization.show_study_results(Elecsim._dab_config)

            # Set result value to True (Check of optimization is necessary
            retval = True
        # Return if function process without errors
        return retval

    @staticmethod
    def show_study_results(self):
        """Display the result of the study."""
        Elecsim._dab_config = dct.Optimization.load_config(Elecsim._dab_config.project_directory, Elecsim._dab_config.circuit_study_name)
        print(f"{Elecsim._dab_config.project_directory=}")
        Elecsim._dab_config.project_directory = Elecsim._dab_config.project_directory.replace("@uni-paderborn.de", "")
        print(f"{Elecsim._dab_config.project_directory=}")
        dct.Optimization.show_study_results(Elecsim._dab_config)
        df = dct.Optimization.study_to_df(Elecsim._dab_config)

    @staticmethod
    def filter_study_results_and_run_gecko():
        """Filter the study result and use geckocircuits for detailed calculation."""
        df = dct.Optimization.study_to_df(Elecsim._dab_config)
        df = df[df["values_0"] == 100]

        df_original = df.copy()

        smallest_dto_list = []
        df_smallest_all = df.nsmallest(n=1, columns=["values_1"])
        df_smallest = df.nsmallest(n=1, columns=["values_1"])

        smallest_dto_list.append(dct.Optimization.df_to_dab_dto_list(Elecsim._dab_config, df_smallest))
        print(f"{np.shape(df)=}")

        for count in np.arange(0, 3):
            print("------------------")
            print(f"{count=}")
            n_suggest = df_smallest['params_n_suggest'].item()
            f_s_suggest = df_smallest['params_f_s_suggest'].item()
            l_s_suggest = df_smallest['params_l_s_suggest'].item()
            l_1_suggest = df_smallest['params_l_1_suggest'].item()
            l_2__suggest = df_smallest['params_l_2__suggest'].item()
            transistor_1_name_suggest = df_smallest['params_transistor_1_name_suggest'].item()
            transistor_2_name_suggest = df_smallest['params_transistor_2_name_suggest'].item()

            # make sure to use parameters with minimum x % difference.
            difference = 0.05

            df = df.loc[
                ~((df["params_n_suggest"].ge(n_suggest * (1 - difference)) & df["params_n_suggest"].le(n_suggest * (1 + difference))) & \
                  (df["params_f_s_suggest"].ge(f_s_suggest * (1 - difference)) & df["params_f_s_suggest"].le(f_s_suggest * (1 + difference))) & \
                  (df["params_l_s_suggest"].ge(l_s_suggest * (1 - difference)) & df["params_l_s_suggest"].le(l_s_suggest * (1 + difference))) & \
                  (df["params_l_1_suggest"].ge(l_1_suggest * (1 - difference)) & df["params_l_1_suggest"].le(l_1_suggest * (1 + difference))) & \
                  (df["params_l_2__suggest"].ge(l_2__suggest * (1 - difference)) & df["params_l_2__suggest"].le(l_2__suggest * (1 + difference))) & \
                  df["params_transistor_1_name_suggest"].isin([transistor_1_name_suggest]) & \
                  df["params_transistor_2_name_suggest"].isin([transistor_2_name_suggest])
                  )]

            df_smallest = df.nsmallest(n=1, columns=["values_1"])
            df_smallest_all = pd.concat([df_smallest_all, df_smallest], axis=0)

        smallest_dto_list = dct.Optimization.df_to_dab_dto_list(Elecsim._dab_config, df_smallest_all)

        # join if necessary
        Elecsim.join_process()

        Elecsim.p = multiprocessing.Process(target=Elecsim.show_plot, args=(df_original, df_smallest_all))
        Elecsim.p.start()

        Elecsim._folders = dct.Optimization.load_filepaths(Elecsim._dab_config.project_directory)

        for dto in smallest_dto_list:
            print(f"{dto.name=}")
            dto_directory = os.path.join(Elecsim._folders.circuit, Elecsim._dab_config.circuit_study_name, "filtered_results")
            os.makedirs(dto_directory, exist_ok=True)
            dto = dct.HandleDabDto.add_gecko_simulation_results(dto, get_waveforms=True)
            dct.HandleDabDto.save(dto, dto.name, comment="", directory=dto_directory, timestamp=False)

        #    dto_list=[]
        #    for dto in smallest_dto_list:
        #        print(f"{dto.name=}")
        #        dto_directory = os.path.join(folders.circuit, dab_config.circuit_study_name, "filtered_results")
        #        dto_list.append( {"DTO": fct_add_gecko_simulation_results(dto),"Dir": dto_directory,"Name": dto.name})

        # Parallelization Test
        # with multiprocessing.Pool(processes=1) as pool:
        #     dto_list = pool.map(elecsim._fct_add_gecko_simulation_results, smallest_dto_list)

        # for dtox in dto_list:
        #     os.makedirs(dtox["Dir"], exist_ok=True)
        #     dct.HandleDabDto.save(dtox["DTO"], dtox["Name"], comment="", directory=dtox["Dir"], timestamp=False)

    @staticmethod
    def show_plot(act_df_original: any, act_df_smallest_all: any):
        """
        Plot the result.

        :param  act_df_original: ????Number of trials in the genetic optimization algorithm (ASA: Type is unclear)
        :type act_df_original: any
        :param act_df_smallest_all: ????Indication, (ASA: Type is unclear)
        :type act_df_smallest_all: any
        """
        # Display the Graphic
        # dct.global_plot_settings_font_latex()
        fig = plt.figure(figsize=(80/25.4, 80/25.4), dpi=350)

        plt.scatter(act_df_original["values_0"], act_df_original["values_1"], color=dct.colors()["blue"], label="Possible designs")
        plt.scatter(act_df_smallest_all["values_0"], act_df_smallest_all["values_1"], color=dct.colors()["red"], label="Non-similar designs")
        plt.xlabel(r"ZVS coverage / \%")
        plt.ylabel(r"$i_\mathrm{cost}$ / A²")

        plt.xticks(ticks=[100], labels=["100"])
        plt.legend()
        plt.grid()
        plt.tight_layout()
        # Show the graphic in a second process
        plt.show()

    @staticmethod
    def join_process():
        """Wait until all parallel processes are finalized."""
        # Check if p is still a process
        if Elecsim.p is not None:
            Elecsim.p.join()
            Elecsim.p = None

    @staticmethod
    def custom(self):
        """Perform customized code currently not used."""
        pass
    #    dab_config = dct.Optimization.load_config(elecsim.__dab_config.project_directory, elecsim.__dab_config.circuit_study_name)
    #    df = dct.Optimization.load_csv_to_df(os.path.join(dab_config.project_directory, "01_circuit", dab_config.circuit_study_name,
    #                                         f"{dab_config.circuit_study_name}.csv"))
    #    df = df[df["number"] == 79030]
    #    print(df.head())

    #    [dab_dto] = dct.Optimization.df_to_dab_dto_list(dab_config, df)

    #    i_cost = dab_dto.calc_currents.i_hf_1_rms ** 2 + dab_dto.calc_currents.i_hf_2_rms ** 2

    #    print(f"{np.mean(i_cost)=}")

    #    i_cost_matrix = dab_dto.calc_currents.i_hf_1_rms ** 2 + dab_dto.calc_currents.i_hf_2_rms ** 2
    #    i_cost_new = np.mean(i_cost_matrix)
    #    print(f"{i_cost_new=}")
    #    i_cost_original = np.mean(i_cost_matrix[~np.isnan(i_cost_matrix)])
    #    print(f"{i_cost_original=}")

    #    dab_dto = dct.HandleDabDto.add_gecko_simulation_results(dab_dto, False)

    #    i_cost_matrix_gecko = dab_dto.gecko_results.i_HF1 ** 2 + dab_dto.gecko_results.i_HF2 ** 2
    #    i_cost_gecko = np.mean(i_cost_matrix_gecko)
    #    print(f"{i_cost_gecko=}")

    #    error_matrix_hf_1 = np.abs((dab_dto.calc_currents.i_hf_1_rms - dab_dto.gecko_results.i_HF1) / dab_dto.calc_currents.i_hf_1_rms)
    #    print(f"{np.mean(error_matrix_hf_1)=}")
    #    error_matrix_hf_2 = np.abs((dab_dto.calc_currents.i_hf_2_rms - dab_dto.gecko_results.i_HF2) / dab_dto.calc_currents.i_hf_2_rms)
    #    print(f"{np.mean(error_matrix_hf_2)=}")

    #    dct.HandleDabDto.save(dab_dto, "results", "", "~/Downloads", False)

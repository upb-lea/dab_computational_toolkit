"""Pareto optimization classes and functions."""
# Python libraries
import os
import datetime
import logging
import json
import pickle

# 3rd party libraries
import optuna
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import deepdiff

# own libraries
import dct


class Optimization:
    """Optimize the DAB converter regarding maximum ZVS coverage and minimum conduction losses."""

    @staticmethod
    def set_up_folder_structure(config: dct.CircuitParetoDabDesign) -> None:
        """
        Set up the folder structure for the subprojects.

        :param config: configuration
        :type config: InductorOptimizationDTO
        """
        project_directory = os.path.abspath(config.project_directory)
        circuit_path = os.path.join(project_directory, "01_circuit")
        inductor_path = os.path.join(project_directory, "02_inductor")
        transformer_path = os.path.join(project_directory, "03_transformer")
        heat_sink_path = os.path.join(project_directory, "04_heat_sink")

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
    def load_filepaths(project_directory: str) -> dct.ParetoFilePaths:
        """
        Load file path of the subdirectories of the project.

        :param project_directory: project directory file path
        :type project_directory: str
        :return: File path in a DTO
        :rtype: dct.ParetoFilePaths
        """
        filepath_config = f"{project_directory}/filepath_config.json"
        if os.path.exists(filepath_config):
            with open(filepath_config, 'r', encoding='utf8') as json_file:
                loaded_file = json.load(json_file)
        else:
            raise ValueError("Project does not exist.")

        file_path_dto = dct.ParetoFilePaths(
            circuit=loaded_file["circuit"],
            transformer=loaded_file["transformer"],
            inductor=loaded_file["inductor"],
            heat_sink=loaded_file["heat_sink"]
        )
        return file_path_dto

    @staticmethod
    def save_config(config: dct.CircuitParetoDabDesign) -> None:
        """
        Save the configuration file as pickle file on the disk.

        :param config: configuration
        :type config: InductorOptimizationDTO
        """
        filepaths = Optimization.load_filepaths(config.project_directory)

        os.makedirs(config.project_directory, exist_ok=True)
        with open(f"{filepaths.circuit}/{config.circuit_study_name}/{config.circuit_study_name}.pkl", 'wb') as output:
            pickle.dump(config, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_config(circuit_project_directory: str, circuit_study_name: str) -> dct.CircuitParetoDabDesign:
        """
        Load pickle configuration file from disk.

        :param circuit_project_directory: project directory
        :type circuit_project_directory: str
        :param circuit_study_name: name of the circuit study
        :type circuit_study_name: str
        :return: Configuration file as dct.DabDesign
        :rtype: dct.CircuitParetoDabDesign
        """
        filepaths = Optimization.load_filepaths(circuit_project_directory)
        config_pickle_filepath = os.path.join(filepaths.circuit, circuit_study_name, f"{circuit_study_name}.pkl")

        with open(config_pickle_filepath, 'rb') as pickle_file_data:
            return pickle.load(pickle_file_data)

    @staticmethod
    def objective(trial: optuna.Trial, dab_config: dct.CircuitParetoDabDesign):
        """
        Objective function to optimize.

        :param dab_config: DAB optimization configuration file
        :type dab_config: dct.CircuitParetoDabDesign
        :param trial: optuna trial
        :type trial: optuna.Trial

        :return:
        """
        f_s_suggest = trial.suggest_int('f_s_suggest', dab_config.design_space.f_s_min_max_list[0], dab_config.design_space.f_s_min_max_list[1])
        l_s_suggest = trial.suggest_float('l_s_suggest', dab_config.design_space.l_s_min_max_list[0], dab_config.design_space.l_s_min_max_list[1])
        l_1_suggest = trial.suggest_float('l_1_suggest', dab_config.design_space.l_1_min_max_list[0], dab_config.design_space.l_1_min_max_list[1])
        l_2__suggest = trial.suggest_float('l_2__suggest', dab_config.design_space.l_2__min_max_list[0], dab_config.design_space.l_2__min_max_list[1])
        n_suggest = trial.suggest_float('n_suggest', dab_config.design_space.n_min_max_list[0], dab_config.design_space.n_min_max_list[1])
        transistor_1_name_suggest = trial.suggest_categorical('transistor_1_name_suggest', dab_config.design_space.transistor_1_list)
        transistor_2_name_suggest = trial.suggest_categorical('transistor_2_name_suggest', dab_config.design_space.transistor_2_list)

        dab_config = dct.HandleDabDto.init_config(
            name=dab_config.circuit_study_name,
            V1_nom=dab_config.output_range.v_1_min_nom_max_list[1],
            V1_min=dab_config.output_range.v_1_min_nom_max_list[0],
            V1_max=dab_config.output_range.v_1_min_nom_max_list[2],
            V1_step=dab_config.output_range.steps_per_direction,
            V2_nom=dab_config.output_range.v_2_min_nom_max_list[1],
            V2_min=dab_config.output_range.v_2_min_nom_max_list[0],
            V2_max=dab_config.output_range.v_2_min_nom_max_list[2],
            V2_step=dab_config.output_range.steps_per_direction,
            P_min=dab_config.output_range.p_min_nom_max_list[0],
            P_max=dab_config.output_range.p_min_nom_max_list[2],
            P_nom=dab_config.output_range.p_min_nom_max_list[1],
            P_step=dab_config.output_range.steps_per_direction,
            n=n_suggest,
            Ls=l_s_suggest,
            fs=f_s_suggest,
            Lc1=l_1_suggest,
            Lc2=l_2__suggest / n_suggest ** 2,
            c_par_1=dab_config.design_space.c_par_1,
            c_par_2=dab_config.design_space.c_par_2,
            transistor_name_1=transistor_1_name_suggest,
            transistor_name_2=transistor_2_name_suggest
        )

        if (np.any(np.isnan(dab_config.calc_modulation.phi)) or np.any(np.isnan(dab_config.calc_modulation.tau1)) \
                or np.any(np.isnan(dab_config.calc_modulation.tau2))):
            return float('nan'), float('nan')

        # Calculate the cost function. Mean for not-NaN values, as there will be too many NaN results.
        i_cost_matrix = dab_config.calc_currents.i_hf_1_rms ** 2 + dab_config.calc_currents.i_hf_2_rms ** 2
        i_cost = np.mean(i_cost_matrix[~np.isnan(i_cost_matrix)])

        return dab_config.calc_modulation.mask_zvs_coverage_notnan * 100, i_cost

    @staticmethod
    def start_proceed_study(dab_config: dct.CircuitParetoDabDesign, number_trials: int,
                            storage: str = 'sqlite',
                            sampler=optuna.samplers.NSGAIIISampler(),
                            ) -> None:
        """Proceed a study which is stored as sqlite database.

        :param dab_config: DAB optimization configuration file
        :type dab_config: dct.CircuitParetoDabDesign
        :param number_trials: Number of trials adding to the existing study
        :type number_trials: int
        :param storage: storage database, e.g. 'sqlite' or 'mysql'
        :type storage: str
        :param sampler: optuna.samplers.NSGAIISampler() or optuna.samplers.NSGAIIISampler(). Note about the brackets () !! Default: NSGAIII
        :type sampler: optuna.sampler-object
        """
        Optimization.set_up_folder_structure(dab_config)
        filepaths = Optimization.load_filepaths(dab_config.project_directory)

        circuit_study_working_directory = os.path.join(filepaths.circuit, dab_config.circuit_study_name)
        circuit_study_sqlite_database = os.path.join(circuit_study_working_directory, f"{dab_config.circuit_study_name}.sqlite3")

        if os.path.exists(circuit_study_sqlite_database):
            print("Existing study found. Proceeding.")
        else:
            os.makedirs(f"{filepaths.circuit}/{dab_config.circuit_study_name}", exist_ok=True)

        # introduce study in storage, e.g. sqlite or mysql
        if storage == 'sqlite':
            # Note: for sqlite operation, there needs to be three slashes '///' even before the path '/home/...'
            # Means, in total there are four slashes including the path itself '////home/.../database.sqlite3'
            storage = f"sqlite:///{circuit_study_sqlite_database}"
        elif storage == 'mysql':
            storage = "mysql://monty@localhost/mydb",

        # set logging verbosity: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.logging.set_verbosity.html#optuna.logging.set_verbosity
        # .INFO: all messages (default)
        # .WARNING: fails and warnings
        # .ERROR: only errors
        optuna.logging.set_verbosity(optuna.logging.ERROR)

        # check for differences with the old configuration file
        config_on_disk_filepath = f"{filepaths.circuit}/{dab_config.circuit_study_name}/{dab_config.circuit_study_name}.pkl"
        if os.path.exists(config_on_disk_filepath):
            config_on_disk = Optimization.load_config(dab_config.project_directory, dab_config.circuit_study_name)
            difference = deepdiff.DeepDiff(config_on_disk, dab_config, ignore_order=True, significant_digits=10)
            if difference:
                print("Configuration file has changed from previous simulation. Do you want to proceed?")
                print(f"Difference: {difference}")
                read_text = input("'1' or Enter: proceed, 'any key': abort\nYour choice: ")
                if read_text == str(1) or read_text == "":
                    print("proceed...")
                else:
                    print("abort...")
                    return None

        directions = ['maximize', 'minimize']

        func = lambda trial: dct.pareto.Optimization.objective(trial, dab_config)

        study_in_storage = optuna.create_study(study_name=dab_config.circuit_study_name,
                                               storage=storage,
                                               directions=directions,
                                               load_if_exists=True, sampler=sampler)

        study_in_memory = optuna.create_study(directions=directions, study_name=dab_config.circuit_study_name, sampler=sampler)
        print(f"Sampler is {study_in_memory.sampler.__class__.__name__}")
        study_in_memory.add_trials(study_in_storage.trials)
        study_in_memory.optimize(func, n_trials=number_trials, show_progress_bar=True)

        study_in_storage.add_trials(study_in_memory.trials[-number_trials:])
        print(f"Finished {number_trials} trials.")
        print(f"current time: {datetime.datetime.now()}")
        Optimization.save_config(dab_config)

    @staticmethod
    def show_study_results(dab_config: dct.CircuitParetoDabDesign) -> None:
        """Show the results of a study.

        A local .html file is generated under config.working_directory to store the interactive plotly plots on disk.

        :param dab_config: DAB optimization configuration file
        :type dab_config: dct.CircuitParetoDabDesign
        """
        filepaths = Optimization.load_filepaths(dab_config.project_directory)
        database_url = Optimization.create_sqlite_database_url(dab_config)
        study = optuna.create_study(study_name=dab_config.circuit_study_name,
                                    storage=database_url, load_if_exists=True)

        fig = optuna.visualization.plot_pareto_front(study, target_names=["ZVS coverage / %", r"i_\mathrm{cost}"])
        fig.update_layout(
            title=f"{dab_config.circuit_study_name}")
        fig.write_html(
            f"{filepaths.circuit}/{dab_config.circuit_study_name}/{dab_config.circuit_study_name}"
            f"_{datetime.datetime.now().isoformat(timespec='minutes')}.html")
        fig.show()

    @staticmethod
    def load_dab_dto_from_study(dab_config: dct.CircuitParetoDabDesign, trial_number: int | None = None):
        """
        Load a DAB-DTO from an optuna study.

        :param dab_config: DAB optimization configuration file
        :type dab_config: dct.CircuitParetoDabDesign
        :param trial_number: trial number to load to the DTO
        :type trial_number: int
        :return:
        """
        if trial_number is None:
            raise NotImplementedError("needs to be implemented")

        filepaths = Optimization.load_filepaths(dab_config.project_directory)
        database_url = Optimization.create_sqlite_database_url(dab_config)

        loaded_study = optuna.create_study(study_name=dab_config.circuit_study_name,
                                           storage=database_url, load_if_exists=True)
        logging.info(f"The study '{dab_config.circuit_study_name}' contains {len(loaded_study.trials)} trials.")
        trials_dict = loaded_study.trials[trial_number].params

        dab_dto = dct.HandleDabDto.init_config(
            name=str(trial_number),
            V1_nom=dab_config.output_range.v_1_min_nom_max_list[1],
            V1_min=dab_config.output_range.v_1_min_nom_max_list[0],
            V1_max=dab_config.output_range.v_1_min_nom_max_list[2],
            V1_step=dab_config.output_range.steps_per_direction,
            V2_nom=dab_config.output_range.v_2_min_nom_max_list[1],
            V2_min=dab_config.output_range.v_2_min_nom_max_list[0],
            V2_max=dab_config.output_range.v_2_min_nom_max_list[2],
            V2_step=dab_config.output_range.steps_per_direction,
            P_min=dab_config.output_range.p_min_nom_max_list[0],
            P_max=dab_config.output_range.p_min_nom_max_list[2],
            P_nom=dab_config.output_range.p_min_nom_max_list[1],
            P_step=dab_config.output_range.steps_per_direction,
            n=trials_dict["n_suggest"],
            Ls=trials_dict["l_s_suggest"],
            fs=trials_dict["f_s_suggest"],
            Lc1=trials_dict["l_1_suggest"],
            Lc2=trials_dict["l_2__suggest"] / trials_dict["n_suggest"] ** 2,
            c_par_1=dab_config.design_space.c_par_1,
            c_par_2=dab_config.design_space.c_par_2,
            transistor_name_1=trials_dict["transistor_1_name_suggest"],
            transistor_name_2=trials_dict["transistor_2_name_suggest"]
        )

        return dab_dto

    @staticmethod
    def df_to_dab_dto_list(dab_config: dct.CircuitParetoDabDesign, df: pd.DataFrame) -> list[dct.CircuitDabDTO]:
        """
        Load a DAB-DTO from an optuna study.

        :param dab_config: DAB optimization configuration file
        :type dab_config: dct.CircuitParetoDabDesign
        :param df: Pandas dataframe to convert to the DAB-DTO list
        :type df: pd.DataFrame
        :return:
        """
        logging.info(f"The study '{dab_config.circuit_study_name}' contains {len(df)} trials.")

        dab_dto_list = []

        for index, _ in df.iterrows():

            dab_dto = dct.HandleDabDto.init_config(
                name=str(df["number"][index].item()),
                V1_nom=dab_config.output_range.v_1_min_nom_max_list[1],
                V1_min=dab_config.output_range.v_1_min_nom_max_list[0],
                V1_max=dab_config.output_range.v_1_min_nom_max_list[2],
                V1_step=dab_config.output_range.steps_per_direction,
                V2_nom=dab_config.output_range.v_2_min_nom_max_list[1],
                V2_min=dab_config.output_range.v_2_min_nom_max_list[0],
                V2_max=dab_config.output_range.v_2_min_nom_max_list[2],
                V2_step=dab_config.output_range.steps_per_direction,
                P_min=dab_config.output_range.p_min_nom_max_list[0],
                P_max=dab_config.output_range.p_min_nom_max_list[2],
                P_nom=dab_config.output_range.p_min_nom_max_list[1],
                P_step=dab_config.output_range.steps_per_direction,
                n=df["params_n_suggest"][index].item(),
                Ls=df["params_l_s_suggest"][index].item(),
                fs=df["params_f_s_suggest"][index].item(),
                Lc1=df["params_l_1_suggest"][index].item(),
                Lc2=df["params_l_2__suggest"][index].item() / df["params_n_suggest"][index].item() ** 2,
                c_par_1=dab_config.design_space.c_par_1,
                c_par_2=dab_config.design_space.c_par_2,
                transistor_name_1=df["params_transistor_1_name_suggest"][index],
                transistor_name_2=df["params_transistor_2_name_suggest"][index]
            )
            dab_dto_list.append(dab_dto)

        return dab_dto_list

    @staticmethod
    def study_to_df(dab_config: dct.CircuitParetoDabDesign):
        """Create a dataframe from a study.

        :param dab_config: DAB optimization configuration file
        :type dab_config: dct.CircuitParetoDabDesign
        """
        filepaths = Optimization.load_filepaths(dab_config.project_directory)
        database_url = Optimization.create_sqlite_database_url(dab_config)
        loaded_study = optuna.create_study(study_name=dab_config.circuit_study_name, storage=database_url, load_if_exists=True)
        df = loaded_study.trials_dataframe()
        df.to_csv(f'{filepaths.circuit}/{dab_config.circuit_study_name}/{dab_config.circuit_study_name}.csv')
        return df

    @staticmethod
    def create_sqlite_database_url(dab_config: dct.CircuitParetoDabDesign) -> str:
        """
        Create the DAB circuit optimization sqlite URL.

        :param dab_config: DAB optimization configuration file
        :type dab_config: dct.CircuitParetoDabDesign
        :return: SQLite URL
        :rtype: str
        """
        filepaths = Optimization.load_filepaths(dab_config.project_directory)
        sqlite_storage_url = f"sqlite:///{filepaths.circuit}/{dab_config.circuit_study_name}/{dab_config.circuit_study_name}.sqlite3"
        return sqlite_storage_url

    @staticmethod
    def df_plot_pareto_front(df: pd.DataFrame, figure_size: tuple):
        """Plot an interactive Pareto diagram (losses vs. volume) to select the transformers to re-simulate.

        :param df: Dataframe, generated from an optuna study (exported by optuna)
        :type df: pd.Dataframe
        :param figure_size: figure size as x,y-tuple in mm, e.g. (160, 80)
        :type figure_size: tuple
        """
        print(df.head())

        names = df["number"].to_numpy()
        # plt.figure()
        fig, ax = plt.subplots(figsize=[x / 25.4 for x in figure_size] if figure_size is not None else None, dpi=80)
        sc = plt.scatter(df["values_0"], df["values_1"], s=10)

        annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        def update_annot(ind):
            pos = sc.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            text = f"{[names[n] for n in ind['ind']]}"
            annot.set_text(text)
            annot.get_bbox_patch().set_alpha(0.4)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)

        plt.xlabel(r'ZVS coverage in \%')
        plt.ylabel(r'$i_\mathrm{HF,1}^2 + i_\mathrm{HF,2}^2$ in A')
        plt.grid()
        plt.tight_layout()
        plt.show()

    @staticmethod
    def load_csv_to_df(csv_filepath: str) -> pd.DataFrame:
        """
        Load a csv file (previously stored from a Pandas dataframe) back to a Pandas dataframe.

        :param csv_filepath: File path of .csv file
        :type csv_filepath: str
        :return: loaded results from the given .csv file
        :rtype: pandas.DataFrame
        """
        df = pd.read_csv(csv_filepath, header=0, index_col=0)
        # reading a pandas dataframe seems to change a global variable in the c subsystem
        # after reading csv values, there are issues running onelab/gmsh, as gmsh writes ',' instead '.' to its own files
        # reading the file again with setting back the delimiter to ';', is a workaround for the mentioned problem.
        pd.read_csv(csv_filepath, header=0, index_col=0, delimiter=';')
        return df

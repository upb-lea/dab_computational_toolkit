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
import dct.datasets_dtos as d_dtos
import dct.pareto_dtos as p_dtos
import dct.datasets as d_sets


class Optimization:
    """Optimize the DAB converter regarding maximum ZVS coverage and minimum conduction losses."""

    @staticmethod
    def set_up_folder_structure(config: p_dtos.CircuitParetoDabDesign) -> None:
        """
        Set up the folder structure for the subprojects.

        :param config: configuration
        :type config: InductorOptimizationDTO
        """
        # ASA: xtodo: Merge ginfo and set_up_folder_structure
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
    def load_filepaths(project_directory: str) -> p_dtos.ParetoFilePaths:
        """
        Load file path of the subdirectories of the project.

        :param project_directory: project directory file path
        :type project_directory: str
        :return: File path in a DTO
        :rtype: p_dtos.ParetoFilePaths
        """
        # ASA: xtodo: Merge ginfo and set_up_folder_structure
        filepath_config = f"{project_directory}/filepath_config.json"
        if os.path.exists(filepath_config):
            with open(filepath_config, 'r', encoding='utf8') as json_file:
                loaded_file = json.load(json_file)
        else:
            raise ValueError("Project does not exist.")

        file_path_dto = p_dtos.ParetoFilePaths(
            circuit=loaded_file["circuit"],
            transformer=loaded_file["transformer"],
            inductor=loaded_file["inductor"],
            heat_sink=loaded_file["heat_sink"]
        )
        return file_path_dto

    @staticmethod
    def save_config(config: p_dtos.CircuitParetoDabDesign) -> None:
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
    def load_config(circuit_project_directory: str, circuit_study_name: str) -> p_dtos.CircuitParetoDabDesign:
        """
        Load pickle configuration file from disk.

        :param circuit_project_directory: project directory
        :type circuit_project_directory: str
        :param circuit_study_name: name of the circuit study
        :type circuit_study_name: str
        :return: Configuration file as p_dtos.DabDesign
        :rtype: p_dtos.CircuitParetoDabDesign
        """
        filepaths = Optimization.load_filepaths(circuit_project_directory)
        config_pickle_filepath = os.path.join(filepaths.circuit, circuit_study_name, f"{circuit_study_name}.pkl")

        with open(config_pickle_filepath, 'rb') as pickle_file_data:
            return pickle.load(pickle_file_data)

    @staticmethod
    def objective(trial: optuna.Trial, dab_config: p_dtos.CircuitParetoDabDesign, fixed_parameters: d_dtos.FixedParameters):
        """
        Objective function to optimize.

        :param dab_config: DAB optimization configuration file
        :type dab_config: p_dtos.CircuitParetoDabDesign
        :param trial: optuna trial
        :type trial: optuna.Trial
        :param fixed_parameters: fixed parameters (loaded transistor DTOs)
        :type fixed_parameters: d_dtos.FixedParameters

        :return:
        """
        f_s_suggest = trial.suggest_int('f_s_suggest', dab_config.design_space.f_s_min_max_list[0], dab_config.design_space.f_s_min_max_list[1])
        l_s_suggest = trial.suggest_float('l_s_suggest', dab_config.design_space.l_s_min_max_list[0], dab_config.design_space.l_s_min_max_list[1])
        l_1_suggest = trial.suggest_float('l_1_suggest', dab_config.design_space.l_1_min_max_list[0], dab_config.design_space.l_1_min_max_list[1])
        l_2__suggest = trial.suggest_float('l_2__suggest', dab_config.design_space.l_2__min_max_list[0], dab_config.design_space.l_2__min_max_list[1])
        n_suggest = trial.suggest_float('n_suggest', dab_config.design_space.n_min_max_list[0], dab_config.design_space.n_min_max_list[1])
        transistor_1_name_suggest = trial.suggest_categorical('transistor_1_name_suggest', dab_config.design_space.transistor_1_name_list)
        transistor_2_name_suggest = trial.suggest_categorical('transistor_2_name_suggest', dab_config.design_space.transistor_2_name_list)

        for _, transistor_dto in enumerate(fixed_parameters.transistor_1_dto_list):
            if transistor_dto.name == transistor_1_name_suggest:
                transistor_1_dto: p_dtos.TransistorDTO = transistor_dto

        for _, transistor_dto in enumerate(fixed_parameters.transistor_2_dto_list):
            if transistor_dto.name == transistor_2_name_suggest:
                transistor_2_dto: p_dtos.TransistorDTO = transistor_dto

        dab_config = d_sets.HandleDabDto.init_config(
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
            transistor_dto_1=transistor_1_dto,
            transistor_dto_2=transistor_2_dto
        )

        if (np.any(np.isnan(dab_config.calc_modulation.phi)) or np.any(np.isnan(dab_config.calc_modulation.tau1)) \
                or np.any(np.isnan(dab_config.calc_modulation.tau2))):
            return float('nan'), float('nan')

        # Calculate the cost function. Mean for not-NaN values, as there will be too many NaN results.
        i_cost_matrix = dab_config.calc_currents.i_hf_1_rms ** 2 + dab_config.calc_currents.i_hf_2_rms ** 2
        i_cost = np.mean(i_cost_matrix[~np.isnan(i_cost_matrix)])

        return dab_config.calc_modulation.mask_zvs_coverage * 100, i_cost

    @staticmethod
    def calculate_fix_parameters(dab_config: p_dtos.CircuitParetoDabDesign) -> d_dtos.FixedParameters:
        """
        Calculate time-consuming parameters which are same for every single simulation.

        :param dab_config: DAB circuit configuration
        :type dab_config: p_dtos.CircuitParetoDabDesign
        :return: Fix parameters (transistor DTOs)
        :rtype: d_dtos.FixedParameters
        """
        transistor_1_dto_list = []
        transistor_2_dto_list = []

        for transistor in dab_config.design_space.transistor_1_name_list:
            transistor_1_dto_list.append(d_sets.HandleTransistorDto.tdb_to_transistor_dto(transistor))

        for transistor in dab_config.design_space.transistor_2_name_list:
            transistor_2_dto_list.append(d_sets.HandleTransistorDto.tdb_to_transistor_dto(transistor))

        fix_parameters = d_dtos.FixedParameters(
            transistor_1_dto_list=transistor_1_dto_list,
            transistor_2_dto_list=transistor_2_dto_list,
        )
        return fix_parameters

    # Add for Parallelisation: Optimization function
    @staticmethod
    def run_optimization_SQLite(act_study: optuna.Study, act_study_name: str, act_number_trials: int, act_dab_config: p_dtos.CircuitParetoDabDesign,
                                act_fixed_parameters: d_dtos.FixedParameters):
        """Proceed a study which is stored as sqlite database.

        :param act_study: Study information configuration
        :type  act_study: optuna.Study
        :param act_study_name: Study information configuration
        :type  act_study_name: str
        :param act_number_trials: Number of trials adding to the existing study
        :type act_number_trials: int
        :param act_dab_config: DAB optimization configuration file
        :type act_dab_config: p_dtos.CircuitParetoDabDesign
        :param act_fixed_parameters: fix configuration parameters for the optimization process
        :type act_fixed_parameters: d_dtos.FixedParameters
        """
        # Function to execute
        func = lambda trial: Optimization.objective(trial, act_dab_config, act_fixed_parameters)

        try:
            act_study.optimize(func, n_trials=act_number_trials, n_jobs=1, show_progress_bar=True)
        except KeyboardInterrupt:
            pass

    @staticmethod
    def run_optimization_MySQL(act_storage_url: str, act_study_name: str, act_number_trials: int, act_dab_config: p_dtos.CircuitParetoDabDesign,
                               act_fixed_parameters: d_dtos.FixedParameters):
        """Proceed a study which is stored as sqlite database.

        :param act_storage_url: url-Name of the database path
        :type act_storage_url: str
        :param act_study_name: Study information configuration
        :type  act_study_name: str
        :param act_number_trials: Number of trials adding to the existing study
        :type  act_number_trials: int
        :param act_dab_config: DAB optimization configuration file
        :type act_dab_config: p_dtos.CircuitParetoDabDesign
        :param act_fixed_parameters: fix configuration parameters for the optimization process
        :type act_fixed_parameters: d_dtos.FixedParameters
        """
        # Function to execute
        func = lambda trial: Optimization.objective(trial, act_dab_config, act_fixed_parameters)

        # Each process create his own study instance with the same database and study name
        act_study = optuna.load_study(storage=act_storage_url, study_name=act_study_name)
        # Run optimization
        try:
            act_study.optimize(func, n_trials=act_number_trials, n_jobs=1, show_progress_bar=True)
        except KeyboardInterrupt:
            pass
        finally:
            # study_in_storage.add_trials(study_in_memory.trials[-number_trials:])
            print(f"Finished {act_number_trials} trials.")
            print(f"current time: {datetime.datetime.now()}")
            # Save methode from RAM-Disk to whereever (Currently opend by missing RAM-DISK)

    @staticmethod
    def start_proceed_study(dab_config: p_dtos.CircuitParetoDabDesign, number_trials: int,
                            # dbType: str = 'mysql',
                            dbType: str = 'sqlite',
                            sampler=optuna.samplers.NSGAIIISampler(),
                            deleteStudyFlag: bool = False
                            ):
        """Proceed a study which is stored as sqlite database.

        :param dab_config: DAB optimization configuration file
        :type dab_config: p_dtos.CircuitParetoDabDesign
        :param number_trials: Number of trials adding to the existing study
        :type number_trials: int
        :param dbType: storage database, e.g. 'sqlite' or 'mysql'
        :type  dbType: str
        :param sampler: optuna.samplers.NSGAIISampler() or optuna.samplers.NSGAIIISampler(). Note about the brackets () !! Default: NSGAIII
        :type sampler: optuna.sampler-object
        :param deleteStudyFlag: Indication, if the old study are to delete (True) or optimization shall be continued.
        :type  deleteStudyFlag: bool
        """
        Optimization.set_up_folder_structure(dab_config)
        filepaths = Optimization.load_filepaths(dab_config.project_directory)

        circuit_study_working_directory = os.path.join(filepaths.circuit, dab_config.circuit_study_name)
        circuit_study_sqlite_database = os.path.join(circuit_study_working_directory, f"{dab_config.circuit_study_name}.sqlite3")

        if os.path.exists(circuit_study_sqlite_database):
            print("Existing study found. Proceeding.")
        else:
            os.makedirs(f"{filepaths.circuit}/{dab_config.circuit_study_name}", exist_ok=True)

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

        fixed_parameters = Optimization.calculate_fix_parameters(dab_config)

        # introduce study in storage, e.g. sqlite or mysql
        if dbType == 'sqlite':
            # Note: for sqlite operation, there needs to be three slashes '///' even before the path '/home/...'
            # Means, in total there are four slashes including the path itself '////home/.../database.sqlite3'
            storage = f"sqlite:///{circuit_study_sqlite_database}"

            # Check the deleteStudyFlag
            if deleteStudyFlag and os.path.exists(circuit_study_sqlite_database):
                os.remove(circuit_study_sqlite_database)

            # Create study object in drive
            study_in_storage = optuna.create_study(study_name=dab_config.circuit_study_name,
                                                   storage=storage,
                                                   directions=directions,
                                                   load_if_exists=True, sampler=sampler)

            # Create study object in memory
            study_in_memory = optuna.create_study(study_name=dab_config.circuit_study_name, directions=directions, sampler=sampler)
            # If trials exists, add them to study_in_memory
            study_in_memory.add_trials(study_in_storage.trials)
            # Inform about sampler type
            print(f"Sampler is {study_in_storage.sampler.__class__.__name__}")
            # actual number of trials
            overtaken_no_trials = len(study_in_memory.trials)
            # Start optimization
            Optimization.run_optimization_SQLite(study_in_memory, dab_config.circuit_study_name, number_trials, dab_config, fixed_parameters)
            # Store memory to storage
            study_in_storage.add_trials(study_in_memory.trials[-number_trials:])
            print(f"Add {number_trials} new calculated trials to existing {overtaken_no_trials} trials = {len(study_in_memory.trials)} trials.")
            print(f"current time: {datetime.datetime.now()}")
            Optimization.save_config(dab_config)

        elif dbType == 'mysql':

            # Verbindung zur MySQL-Datenbank
            storage_url = "mysql+pymysql://oaml_optuna:optuna@localhost/optuna_db"

            # Create storage-Objekt for Optuna on drive (Later RAMDISK)
            storage = optuna.storages.RDBStorage(storage_url)
            # storage = "mysql://oaml_optuna:optuna@localhost/optuna_db"

            # Create study object in drive
            study_in_storage = optuna.create_study(study_name=dab_config.circuit_study_name,
                                                   storage=storage,
                                                   directions=directions,
                                                   load_if_exists=True, sampler=sampler)

            # Inform about sampler type
            print(f"Sampler is {study_in_storage.sampler.__class__.__name__}")
            # Start optimization
            Optimization.run_optimization_MySQL(storage_url, dab_config.circuit_study_name, number_trials, dab_config, fixed_parameters)

        # Parallelization Test with mysql
        # Number of processes
        #   num_processes = 1
        # Process list
        #    processes = []
        # Loop to start the processes
        #   for proc in range(num_processes):
        #       print(f"Process {proc} started")
        #       p = multiprocessing.Process(target=Optimization.run_optimization,
        #                                   args=(storage_url, dab_config.circuit_study_name,
        #                                         number_trials, dab_config,fixed_parameters
        #                                        )
        #                                  )
        #       p.start()
        #       processes.append(p)

        # Wait for joining
        #   for proc in processes:
            # wait until each process is joined
        #       p.join()
        #       print(f"Process {proc} joins")

#        Old approach
#        study_in_memory = optuna.create_study(directions=directions, study_name=dab_config.circuit_study_name, sampler=sampler)
#        print(f"Sampler is {study_in_memory.sampler.__class__.__name__}")
#        study_in_memory.add_trials(study_in_storage.trials)
#        try:
#            study_in_memory.optimize(func, n_trials=number_trials, n_jobs=1, show_progress_bar=True)
#        except KeyboardInterrupt:
#            pass
#        finally:
#            study_in_storage.add_trials(study_in_memory.trials[-number_trials:])
#            print(f"Finished {number_trials} trials.")
#            print(f"current time: {datetime.datetime.now()}")
#            Optimization.save_config(dab_config)

    @staticmethod
    def show_study_results(dab_config: p_dtos.CircuitParetoDabDesign) -> None:
        """Show the results of a study.

        A local .html file is generated under config.working_directory to store the interactive plotly plots on disk.

        :param dab_config: DAB optimization configuration file
        :type dab_config: p_dtos.CircuitParetoDabDesign
        """
        filepaths = Optimization.load_filepaths(dab_config.project_directory)
        database_url = Optimization.create_sqlite_database_url(dab_config)
        study = optuna.create_study(study_name=dab_config.circuit_study_name, storage=database_url, load_if_exists=True)

        fig = optuna.visualization.plot_pareto_front(study, target_names=["ZVS coverage / %", r"i_\mathrm{cost}"])
        fig.update_layout(title=f"{dab_config.circuit_study_name} <br><sup>{dab_config.project_directory}</sup>")
        fig.write_html(
            f"{filepaths.circuit}/{dab_config.circuit_study_name}/{dab_config.circuit_study_name}"
            f"_{datetime.datetime.now().isoformat(timespec='minutes')}.html")
        fig.show()

    @staticmethod
    def load_dab_dto_from_study(dab_config: p_dtos.CircuitParetoDabDesign, trial_number: int | None = None):
        """
        Load a DAB-DTO from an optuna study.

        :param dab_config: DAB optimization configuration file
        :type dab_config: p_dtos.CircuitParetoDabDesign
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

        dab_dto = d_sets.HandleDabDto.init_config(
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
            transistor_dto_1=trials_dict["transistor_1_name_suggest"],
            transistor_dto_2=trials_dict["transistor_2_name_suggest"]
        )

        return dab_dto

    @staticmethod
    def df_to_dab_dto_list(dab_config: p_dtos.CircuitParetoDabDesign, df: pd.DataFrame) -> list[d_dtos.CircuitDabDTO]:
        """
        Load a DAB-DTO from an optuna study.

        :param dab_config: DAB optimization configuration file
        :type dab_config: p_dtos.CircuitParetoDabDesign
        :param df: Pandas dataframe to convert to the DAB-DTO list
        :type df: pd.DataFrame
        :return:
        """
        logging.info(f"The study '{dab_config.circuit_study_name}' contains {len(df)} trials.")

        dab_dto_list = []

        for index, _ in df.iterrows():
            transistor_dto_1 = d_sets.HandleTransistorDto.tdb_to_transistor_dto(df["params_transistor_1_name_suggest"][index])
            transistor_dto_2 = d_sets.HandleTransistorDto.tdb_to_transistor_dto(df["params_transistor_2_name_suggest"][index])

            dab_dto = d_sets.HandleDabDto.init_config(
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
                transistor_dto_1=transistor_dto_1,
                transistor_dto_2=transistor_dto_2
            )
            dab_dto_list.append(dab_dto)

        return dab_dto_list

    @staticmethod
    def study_to_df(dab_config: p_dtos.CircuitParetoDabDesign):
        """Create a dataframe from a study.

        :param dab_config: DAB optimization configuration file
        :type dab_config: p_dtos.CircuitParetoDabDesign
        """
        filepaths = Optimization.load_filepaths(dab_config.project_directory)
        database_url = Optimization.create_sqlite_database_url(dab_config)
        loaded_study = optuna.create_study(study_name=dab_config.circuit_study_name, storage=database_url, load_if_exists=True)
        df = loaded_study.trials_dataframe()
        df.to_csv(f'{filepaths.circuit}/{dab_config.circuit_study_name}/{dab_config.circuit_study_name}.csv')
        return df

    @staticmethod
    def create_sqlite_database_url(dab_config: p_dtos.CircuitParetoDabDesign) -> str:
        """
        Create the DAB circuit optimization sqlite URL.

        :param dab_config: DAB optimization configuration file
        :type dab_config: p_dtos.CircuitParetoDabDesign
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

    @staticmethod
    def is_pareto_efficient(costs: np.array, return_mask: bool = True):
        """
        Find the pareto-efficient points.

        :param costs: An (n_points, n_costs) array
        :type costs: np.array
        :param return_mask: True to return a mask
        :type return_mask: bool
        :return: An array of indices of pareto-efficient points.
            If return_mask is True, this will be an (n_points, ) boolean array
            Otherwise it will be a (n_efficient_points, ) integer array of indices.
        :rtype: np.array
        """
        is_efficient = np.arange(costs.shape[0])
        n_points = costs.shape[0]
        next_point_index = 0  # Next index in the is_efficient array to search for
        while next_point_index < len(costs):
            nondominated_point_mask = np.any(costs < costs[next_point_index], axis=1)
            nondominated_point_mask[next_point_index] = True
            is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
            costs = costs[nondominated_point_mask]
            next_point_index = np.sum(nondominated_point_mask[:next_point_index]) + 1
        if return_mask:
            is_efficient_mask = np.zeros(n_points, dtype=bool)
            is_efficient_mask[is_efficient] = True
            return is_efficient_mask
        else:
            return is_efficient

    @staticmethod
    def pareto_front_from_df(df: pd.DataFrame, x: str = "values_0", y: str = "values_1") -> pd.DataFrame:
        """
        Calculate the Pareto front from a Pandas dataframe. Return a Pandas dataframe.

        :param df: Pandas dataframe
        :type df: pd.DataFrame
        :param x: Name of x-parameter from df to show in Pareto plane
        :type x: str
        :param y: Name of y-parameter from df to show in Pareto plane
        :type y: str
        :return: Pandas dataframe with pareto efficient points
        :rtype: pd.DataFrame
        """
        x_vec = df[x][~np.isnan(df[x])]
        y_vec = df[y][~np.isnan(df[x])]
        numpy_zip = np.column_stack((x_vec, y_vec))
        pareto_tuple_mask_vec = Optimization.is_pareto_efficient(numpy_zip)
        pareto_df = df[~np.isnan(df[x])][pareto_tuple_mask_vec]
        return pareto_df

    @staticmethod
    def filter_df(df: pd.DataFrame, x: str = "values_0", y: str = "values_1", factor_min_dc_losses: float = 1.2,
                  factor_max_dc_losses: float = 10) -> pd.DataFrame:
        """
        Remove designs with too high losses compared to the minimum losses.

        :param df: pandas dataframe with study results
        :type df: pd.DataFrame
        :param x: x-value name for Pareto plot filtering
        :type x: str
        :param y: y-value name for Pareto plot filtering
        :type y: str
        :param factor_min_dc_losses: filter factor for the minimum dc losses
        :type factor_min_dc_losses: float
        :param factor_max_dc_losses: dc_max_loss = factor_max_dc_losses * min_available_dc_losses_in_pareto_front
        :type factor_max_dc_losses: float
        :returns: pandas dataframe with Pareto front near points
        :rtype: pd.DataFrame
        """
        # figure out pareto front
        # pareto_volume_list, pareto_core_hyst_list, pareto_dto_list = self.pareto_front(volume_list, core_hyst_loss_list, valid_design_list)

        pareto_df: pd.DataFrame = Optimization.pareto_front_from_df(df, x, y)

        vector_to_sort = np.array([pareto_df[x], pareto_df[y]])

        # sorting 2d array by 1st row
        # https://stackoverflow.com/questions/49374253/sort-a-numpy-2d-array-by-1st-row-maintaining-columns
        sorted_vector = vector_to_sort[:, vector_to_sort[0].argsort()]
        x_pareto_vec = sorted_vector[0]
        y_pareto_vec = sorted_vector[1]

        total_losses_list = df[y][~np.isnan(df[y])].to_numpy()

        min_total_dc_losses = total_losses_list[np.argmin(total_losses_list)]
        loss_offset = factor_min_dc_losses * min_total_dc_losses

        ref_loss_max = np.interp(df[x], x_pareto_vec, y_pareto_vec) + loss_offset
        # clip losses to a maximum of the minimum losses
        ref_loss_max = np.clip(ref_loss_max, a_min=-1, a_max=factor_max_dc_losses * min_total_dc_losses)

        pareto_df_offset = df[df[y] < ref_loss_max]

        return pareto_df_offset

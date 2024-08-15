"""Pareto optimization classes and functions."""
# Python libraries
import os
import datetime

# 3rd party libraries
import optuna
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# own libraries
import dct


class Optimization:
    """Optimize the DAB converter regarding maximum ZVS coverage and minimum conduction losses."""

    @staticmethod
    def objective(trial: optuna.Trial, design_space: dct.DesignSpace, work_area: dct.WorkArea):
        """
        Objective function to optimize.

        :param design_space: Component design space
        :type design_space: DesignSpace
        :param work_area: DAB operating area
        :type work_area: WorkArea
        :param trial: optuna trial
        :type trial: optuna.Trial

        :return:
        """
        f_s_suggest = trial.suggest_int('f_s_suggest', design_space.f_s_min_max_list[0], design_space.f_s_min_max_list[1])
        l_s_suggest = trial.suggest_float('l_s_suggest', design_space.l_s_min_max_list[0], design_space.l_s_min_max_list[1])
        l_1_suggest = trial.suggest_float('l_1_suggest', design_space.l_1_min_max_list[0], design_space.l_1_min_max_list[1])
        l_2_suggest = trial.suggest_float('l_2_suggest', design_space.l_2_min_max_list[0], design_space.l_2_min_max_list[1])
        n_suggest = trial.suggest_float('n_suggest', design_space.n_min_max_list[0], design_space.n_min_max_list[1])
        transistor_1_name_suggest = trial.suggest_categorical('transistor_1_name_suggest', design_space.transistor_1_list)
        transistor_2_name_suggest = trial.suggest_categorical('transistor_2_name_suggest', design_space.transistor_2_list)

        dab_config = dct.HandleDabDto.init_config(
            V1_nom=work_area.v_1_min_nom_max_list[1],
            V1_min=work_area.v_1_min_nom_max_list[0],
            V1_max=work_area.v_1_min_nom_max_list[2],
            V1_step=work_area.steps_per_direction,
            V2_nom=work_area.v_2_min_nom_max_list[1],
            V2_min=work_area.v_2_min_nom_max_list[0],
            V2_max=work_area.v_2_min_nom_max_list[2],
            V2_step=work_area.steps_per_direction,
            P_min=work_area.p_min_nom_max_list[0],
            P_max=work_area.p_min_nom_max_list[2],
            P_nom=work_area.p_min_nom_max_list[1],
            P_step=work_area.steps_per_direction,
            n=n_suggest,
            Ls=l_s_suggest,
            fs=f_s_suggest,
            Lc1=l_1_suggest,
            Lc2=l_2_suggest,
            c_par_1=16e-12,
            c_par_2=16e-12,
            transistor_name_1=transistor_1_name_suggest,
            transistor_name_2=transistor_2_name_suggest
        )

        # Calculate the cost function. Mean for not-NaN values, as there will be too many NaN results.
        i_cost_matrix = dab_config.calc_currents.i_hf_1_rms ** 2 + dab_config.calc_currents.i_hf_2_rms ** 2
        i_cost = np.mean(i_cost_matrix[~np.isnan(i_cost_matrix)])

        return dab_config.calc_modulation.mask_zvs_coverage * 100, i_cost

    @staticmethod
    def start_proceed_study(study_name: str, design_space: dct.DesignSpace,
                            work_area: dct.WorkArea, number_trials: int,
                            storage: str = 'sqlite',
                            sampler=optuna.samplers.NSGAIIISampler(),
                            ) -> None:
        """Proceed a study which is stored as sqlite database.

        :param study_name: Name of the study
        :type study_name: str
        :param design_space: Simulation configuration
        :type design_space: ItoSingleInputConfig
        :param number_trials: Number of trials adding to the existing study
        :type number_trials: int
        :param storage: storage database, e.g. 'sqlite' or 'mysql'
        :type storage: str
        :param sampler: optuna.samplers.NSGAIISampler() or optuna.samplers.NSGAIIISampler(). Note about the brackets () !! Default: NSGAIII
        :type sampler: optuna.sampler-object
        :param work_area: work area
        :type work_area: WorkArea
        """
        if os.path.exists(f"{design_space.working_directory}/study_{study_name}.sqlite3"):
            print("Existing study found. Proceeding.")

        # introduce study in storage, e.g. sqlite or mysql
        if storage == 'sqlite':
            # Note: for sqlite operation, there needs to be three slashes '///' even before the path '/home/...'
            # Means, in total there are four slashes including the path itself '////home/.../database.sqlite3'
            storage = f"sqlite:///{design_space.working_directory}/study_{study_name}.sqlite3"
        elif storage == 'mysql':
            storage = "mysql://monty@localhost/mydb",

        # set logging verbosity: https://optuna.readthedocs.io/en/stable/reference/generated/optuna.logging.set_verbosity.html#optuna.logging.set_verbosity
        # .INFO: all messages (default)
        # .WARNING: fails and warnings
        # .ERROR: only errors
        optuna.logging.set_verbosity(optuna.logging.ERROR)

        directions = ['maximize', 'minimize']

        func = lambda trial: dct.pareto.Optimization.objective(trial, design_space, work_area)

        study_in_storage = optuna.create_study(study_name=study_name,
                                               storage=storage,
                                               directions=directions,
                                               load_if_exists=True, sampler=sampler)

        study_in_memory = optuna.create_study(directions=directions, study_name=study_name, sampler=sampler)
        print(f"Sampler is {study_in_memory.sampler.__class__.__name__}")
        study_in_memory.add_trials(study_in_storage.trials)
        study_in_memory.optimize(func, n_trials=number_trials, show_progress_bar=True)

        study_in_storage.add_trials(study_in_memory.trials[-number_trials:])
        print(f"Finished {number_trials} trials.")
        print(f"current time: {datetime.datetime.now()}")

    @staticmethod
    def show_study_results(study_name: str, config: dct.DesignSpace,) -> None:
        """Show the results of a study.

        A local .html file is generated under config.working_directory to store the interactive plotly plots on disk.

        :param study_name: Name of the study
        :type study_name: str
        :param config: Integrated transformer configuration file
        :type config: ItoSingleInputConfig
        """
        study = optuna.create_study(study_name=study_name,
                                    storage=f"sqlite:///{config.working_directory}/study_{study_name}.sqlite3",
                                    load_if_exists=True)

        fig = optuna.visualization.plot_pareto_front(study, target_names=["ZVS coverage / %", r"i_\mathrm{cost}"])
        fig.update_layout(
            title=f"{study_name}")
        fig.write_html(
            f"{config.working_directory}/{study_name}"
            f"_{datetime.datetime.now().isoformat(timespec='minutes')}.html")
        fig.show()

    @staticmethod
    def load_dab_dto_from_study(study_name: str, design_space: dct.DesignSpace, work_area: dct.WorkArea, trial_number: int | None = None):
        """
        Load a DAB-DTO from an optuna study.

        :param study_name: study name to load
        :type study_name: str
        :param design_space: design space
        :type design_space: DesignSpace
        :param work_area: work area
        :type work_area: WorkArea
        :param trial_number: trial number to load to the DTO
        :type trial_number: int
        :return:
        """
        if trial_number is None:
            raise NotImplementedError("needs to be implemented")

        loaded_study = optuna.create_study(study_name=study_name,
                                           storage=f"sqlite:///{design_space.working_directory}/study_{study_name}.sqlite3",
                                           load_if_exists=True)

        trials_dict = loaded_study.trials[trial_number].params

        dab_dto = dct.HandleDabDto.init_config(
            V1_nom=work_area.v_1_min_nom_max_list[1],
            V1_min=work_area.v_1_min_nom_max_list[0],
            V1_max=work_area.v_1_min_nom_max_list[2],
            V1_step=work_area.steps_per_direction,
            V2_nom=work_area.v_2_min_nom_max_list[1],
            V2_min=work_area.v_2_min_nom_max_list[0],
            V2_max=work_area.v_2_min_nom_max_list[2],
            V2_step=work_area.steps_per_direction,
            P_min=work_area.p_min_nom_max_list[0],
            P_max=work_area.p_min_nom_max_list[2],
            P_nom=work_area.p_min_nom_max_list[1],
            P_step=work_area.steps_per_direction,
            n=trials_dict["n_suggest"],
            Ls=trials_dict["l_s_suggest"],
            fs=trials_dict["f_s_suggest"],
            Lc1=trials_dict["l_1_suggest"],
            Lc2=trials_dict["l_2_suggest"],
            c_par_1=16e-12,
            c_par_2=16e-12,
            transistor_name_1=trials_dict["transistor_1_name_suggest"],
            transistor_name_2=trials_dict["transistor_2_name_suggest"]
        )

        return dab_dto

    @staticmethod
    def study_to_df(study_name: str, database_url: str):
        """Create a dataframe from a study.

        :param study_name: name of study
        :type study_name: str
        :param database_url: url of database
        :type database_url: str
        """
        loaded_study = optuna.create_study(study_name=study_name, storage=database_url, load_if_exists=True)
        df = loaded_study.trials_dataframe()
        df.to_csv(f'{study_name}.csv')
        return df

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

"""Plot the optimization results."""

# python libraries
import os
import pickle

# 3rd party libraries
from matplotlib import pyplot as plt
import pandas as pd

# own libraries
import dct.generalplotsettings as gps
from dct.datasets_dtos import StudyData
import hct
import femmt as fmt
from dct.datasets_dtos import PlotData
from dct.topology.circuit_optimization_base import CircuitOptimizationBase
from dct.constant_path import (DF_SUMMARY_FINAL, PARETO_PLOT_PDF_FOLDER, PARETO_PLOT_PNG_FOLDER, PARETO_PLOT_PKL_FOLDER,
                               CAPACITOR_RESULTS, CAPACITOR_RESULTS_FILTERED, FEMMT_FEM_RESULTS_FOLDER)
from dct.constants import FACTOR_M3_TO_CM3, FACTOR_M2_TO_CM2


class ParetoPlots:
    """Generate PDF plots to see the results of single Pareto steps (circuit, inductor, transformer, heat sink)."""

    @staticmethod
    def generate_pareto_plot(x_values_list: list, y_values_list: list, color_list: list, alpha_list: list[float],
                             x_label: str, y_label: str, label_list: list[str | None], fig_name_path: str,
                             xlim: list | None = None, ylim: list | None = None) -> None:
        """
        Generate multiple Pareto plot in one PDF file.

        :param x_values_list: list of different Pareto plot x values
        :type x_values_list: list
        :param y_values_list: list of different Pareto plot y values
        :type y_values_list: list
        :param color_list:
        :param alpha_list: The alpha blending value, between 0 (transparent) and 1 (opaque) for each sequence in a list
        :type alpha_list: list[float]
        :param x_label: x label of the Pareto plot
        :type x_label: str
        :param y_label: y label of the Pareto plot
        :type y_label: str
        :param label_list: list of different Pareto plot labels in a legend
        :type label_list: list[str | None]
        :param fig_name_path: filename, will be saved as pdf
        :type fig_name_path: str
        :param xlim: x-axis limitation [x_min, x_max]
        :type xlim: list[float]
        :param ylim: y-axis limitation [y_min, y_max]
        :type ylim: list[float]
        """
        # set font to LaTeX font
        gps.global_plot_settings_font_latex()

        # generate plot
        fig = plt.figure(figsize=(80 / 25.4, 80 / 25.4), dpi=350)
        for count, _ in enumerate(x_values_list):
            x_values = x_values_list[count]
            y_values = y_values_list[count]
            color = color_list[count]
            legend = label_list[count]
            plt.scatter(x_values, y_values, color=color, alpha=alpha_list[count], label=legend)

        plt.legend()
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.grid()
        plt.tight_layout()
        # make sure to not generate a filename.pdf.pdf (twice ".pdf").
        fig_name_path = fig_name_path.replace(".pdf", "")
        path, fig_name = os.path.split(fig_name_path)

        if not os.path.exists(f"{path}/{PARETO_PLOT_PDF_FOLDER}"):
            os.mkdir(f"{path}/{PARETO_PLOT_PDF_FOLDER}")
        if not os.path.exists(f"{path}/{PARETO_PLOT_PNG_FOLDER}"):
            os.mkdir(f"{path}/{PARETO_PLOT_PNG_FOLDER}")
        if not os.path.exists(f"{path}/{PARETO_PLOT_PKL_FOLDER}"):
            os.mkdir(f"{path}/{PARETO_PLOT_PKL_FOLDER}")
        plt.savefig(f"{path}/{PARETO_PLOT_PDF_FOLDER}/{fig_name}.pdf")
        plt.savefig(f"{path}/{PARETO_PLOT_PNG_FOLDER}/{fig_name}.png")
        # Save the figure as pickle file type, for later view
        with open(f"{path}/{PARETO_PLOT_PKL_FOLDER}/{fig_name}.pkl", "wb") as f:
            pickle.dump(fig, f)

    @staticmethod
    def plot_circuit_results(circuit_optimization: CircuitOptimizationBase, summary_directory: str) -> None:
        """
        Plot the results of the circuit optimization in the Pareto plane.

        :param circuit_optimization: circuit optimization class
        :type  circuit_optimization: CircuitOptimizationBase
        :param summary_directory: Path of the summary directory (pre-summary or summary directory)
        :type  summary_directory: str
        """
        # Set the target directory
        fig_name = os.path.join(summary_directory, "circuit")

        # Read data from circuit
        plot_data: PlotData = circuit_optimization.get_circuit_plot_data(circuit_optimization.circuit_study_data)

        ParetoPlots.generate_pareto_plot(plot_data.x_values_list, plot_data.y_values_list, color_list=plot_data.color_list,
                                         alpha_list=plot_data.alpha_list, x_label=plot_data.x_label, y_label=plot_data.y_label,
                                         label_list=plot_data.label_list, fig_name_path=fig_name)

    @staticmethod
    def plot_inductor_results(inductor_study_data: StudyData, filtered_list_files: list[str], summary_directory: str) -> None:
        """
        Plot the results of the inductor optimization in the Pareto plane.

        :param inductor_study_data: Information about the inductor study name and study path
        :type  inductor_study_data: StudyData
        :param filtered_list_files: List of filtered circuit design names
        :type  filtered_list_files: list[str]
        :param summary_directory: Path of the summary directory (pre-summary or summary directory)
        :type  summary_directory: str
        """
        volume_key = "values_0"
        loss_key = "values_1"
        # Loop over all filtered circuit designs
        for circuit_number in filtered_list_files:
            # Assemble file path for actual circuit design
            file_path = os.path.join(inductor_study_data.optimization_directory, circuit_number,
                                     inductor_study_data.study_name)
            # Assemble pkl-file name
            config_pickle_filepath = os.path.join(file_path, f"{inductor_study_data.study_name}.pkl")
            # Load data from pkl-file
            config = fmt.optimization.InductorOptimization.ReluctanceModel.load_config(config_pickle_filepath)
            df = fmt.optimization.InductorOptimization.ReluctanceModel.study_to_df(config)

            # m³ -> cm³
            df[volume_key] = df[volume_key] * FACTOR_M3_TO_CM3

            fem_results_folder_path = os.path.join(file_path, FEMMT_FEM_RESULTS_FOLDER)
            df_filtered = fmt.InductorOptimization.ReluctanceModel.filter_loss_list_df(df, factor_min_dc_losses=0.2, factor_max_dc_losses=100)
            df_fem_reluctance = fmt.InductorOptimization.FemSimulation.fem_logs_to_df(df_filtered, fem_results_folder_path)

            # all fem simulation points
            fem_loss_results = df_fem_reluctance["fem_p_loss_winding"] + df_fem_reluctance["fem_eddy_core"] + df_fem_reluctance["user_attrs_p_hyst"]

            x_values_list = [df[volume_key], df_filtered[volume_key], df_fem_reluctance[volume_key]]
            y_values_list = [df[loss_key], df_filtered[loss_key], fem_loss_results]
            label_list: list[str | None] = ["RM all", "RM filtered", "FEM"]

            x_scale_min = 0.9 * df_filtered[volume_key].min()
            x_scale_max = 1.1 * df_filtered[volume_key].max()

            y_scale_min = 0.9 * df_filtered[loss_key].min()
            y_scale_max = 1.1 * df_filtered[loss_key].max()

            # Set the target directory
            fig_name = os.path.join(summary_directory, f"inductor_c{circuit_number}_{inductor_study_data.study_name}")

            ParetoPlots.generate_pareto_plot(x_values_list, y_values_list, color_list=["black", "red", "green"], alpha_list=[0.5, 0.5, 0.5],
                                             x_label=r'$V_\mathrm{ind}$ / cm³', y_label=r'$P_\mathrm{ind}$ / W', label_list=label_list,
                                             fig_name_path=fig_name, xlim=[x_scale_min, x_scale_max], ylim=[y_scale_min, y_scale_max])

    @staticmethod
    def plot_transformer_results(transformer_study_data: StudyData, filtered_list_files: list[str], summary_directory: str) -> None:
        """
        Plot the results of the transformer optimization in the Pareto plane.

        :param transformer_study_data: Information about the transformer study name and study path
        :type  transformer_study_data: StudyData
        :param filtered_list_files: List of filtered circuit design names
        :type  filtered_list_files: list[str]
        :param summary_directory: Path of the summary directory (pre-summary or summary directory)
        :type  summary_directory: str
        """
        volume_key = "values_0"
        loss_key = "values_1"
        # Loop over all filtered circuit designs
        for circuit_number in filtered_list_files:
            # Assemble file path for actual circuit design
            file_path = os.path.join(transformer_study_data.optimization_directory, circuit_number,
                                     transformer_study_data.study_name)
            # Assemble pkl-file name
            config_pickle_filepath = os.path.join(file_path, f"{transformer_study_data.study_name}.pkl")
            # Load data from pkl-file
            config = fmt.optimization.StackedTransformerOptimization.ReluctanceModel.load_config(config_pickle_filepath)
            df = fmt.optimization.StackedTransformerOptimization.ReluctanceModel.study_to_df(config)

            # m³ -> cm³
            df[volume_key] = df[volume_key] * FACTOR_M3_TO_CM3

            fem_results_folder_path = os.path.join(file_path, FEMMT_FEM_RESULTS_FOLDER)
            df_filtered = fmt.StackedTransformerOptimization.ReluctanceModel.filter_loss_list_df(df, factor_min_dc_losses=0.2, factor_max_dc_losses=10)
            df_fem_reluctance = fmt.StackedTransformerOptimization.FemSimulation.fem_logs_to_df(df_filtered, fem_results_folder_path)

            # all fem simulation points
            fem_loss_results = df_fem_reluctance["fem_p_loss_winding"] + df_fem_reluctance["fem_eddy_core"] + df_fem_reluctance["user_attrs_p_hyst"]

            x_values_list = [df[volume_key], df_filtered[volume_key], df_fem_reluctance[volume_key]]
            y_values_list = [df[loss_key], df_filtered[loss_key], fem_loss_results]
            label_list: list[str | None] = ["RM all", "RM filtered", "FEM"]

            x_scale_min = 0.9 * df_filtered[volume_key].min()
            x_scale_max = 1.1 * df_filtered[volume_key].max()

            y_scale_min = 0.9 * df_filtered[loss_key].min()
            y_scale_max = 1.1 * df_filtered[loss_key].max()

            # Set the target directory
            fig_name = os.path.join(summary_directory, f"transformer_c{circuit_number}_{transformer_study_data.study_name}")

            ParetoPlots.generate_pareto_plot(x_values_list, y_values_list, color_list=["black", "red", "green"], alpha_list=[0.5, 0.5, 0.5],
                                             x_label=r'$V_\mathrm{tr}$ / cm³', y_label=r'$P_\mathrm{tr}$ / W', label_list=label_list,
                                             fig_name_path=fig_name, xlim=[x_scale_min, x_scale_max], ylim=[y_scale_min, y_scale_max])

    @staticmethod
    def plot_capacitor_results(capacitor_study_data: StudyData, filtered_list_files: list[str], summary_directory: str) -> None:
        """
        Plot the results of the transformer optimization in the Pareto plane.

        :param capacitor_study_data: Information about the capacitor study name and study path
        :type  capacitor_study_data: StudyData
        :param filtered_list_files: List of filtered circuit design names
        :type  filtered_list_files: list[str]
        :param summary_directory: Path of the summary directory (pre-summary or summary directory)
        :type  summary_directory: str
        """
        volume_key = "volume_total"
        loss_key = "power_loss_total"

        # Loop over all filtered circuit designs
        for circuit_number in filtered_list_files:
            # Assemble file path for actual circuit design
            file_path = os.path.join(capacitor_study_data.optimization_directory, circuit_number,
                                     capacitor_study_data.study_name)

            # Assemble pkl-file name
            df_capacitors_directory = os.path.join(file_path, CAPACITOR_RESULTS)
            df_capacitors_filtered_directory = os.path.join(file_path, CAPACITOR_RESULTS_FILTERED)

            df_capacitors = pd.read_csv(df_capacitors_directory)
            df_capacitors_filtered = pd.read_csv(df_capacitors_filtered_directory)

            df_capacitors[volume_key] = df_capacitors[volume_key] * FACTOR_M3_TO_CM3
            df_capacitors_filtered[volume_key] = df_capacitors_filtered[volume_key] * FACTOR_M3_TO_CM3

            x_values_list = [df_capacitors[volume_key], df_capacitors_filtered[volume_key]]
            y_values_list = [df_capacitors[loss_key], df_capacitors_filtered[loss_key]]
            label_list: list[str | None] = ["all", "filtered"]

            x_scale_min = 0.9 * df_capacitors_filtered[volume_key].min()
            x_scale_max = 1.1 * df_capacitors_filtered[volume_key].max()

            y_scale_min = 0.9 * df_capacitors_filtered[loss_key].min()
            y_scale_max = 1.1 * df_capacitors_filtered[loss_key].max()

            # Set the target directory
            fig_name = os.path.join(summary_directory, f"capacitor_c{circuit_number}_{capacitor_study_data.study_name}")

            ParetoPlots.generate_pareto_plot(x_values_list, y_values_list, color_list=["black", "red"], alpha_list=[0.5, 0.5], x_label=r'$V_\mathrm{tr}$ / cm³',
                                             y_label=r'$P_\mathrm{capacitor}$ / W', label_list=label_list,
                                             fig_name_path=fig_name, xlim=[x_scale_min, x_scale_max], ylim=[y_scale_min, y_scale_max])

    @staticmethod
    def plot_heat_sink_results(heat_sink_study_data: StudyData, summary_directory: str) -> None:
        """
        Plot the results of the heat sink optimization in the Pareto plane.

        :param heat_sink_study_data: Information about the heat sink study name and study path
        :type  heat_sink_study_data: StudyData
        :param summary_directory: Path of the summary directory (pre-summary or summary directory)
        :type  summary_directory: str
        """
        volume_key = "values_0"
        loss_key = "values_1"
        area_key = "values_2"
        # target color list and different heat sink areas to plot (split 3d plot into several 2d plots)
        color_list = [gps.colors()["black"], gps.colors()["red"], gps.colors()["blue"], gps.colors()["green"]]
        a_min_m2_list = [0.002, 0.003, 0.005]

        # Assemble heat sink pkl-file name
        heat_sink_pkl_path = os.path.join(heat_sink_study_data.optimization_directory,
                                          f"{heat_sink_study_data.study_name}.pkl")
        # Load data from pkl-file
        config = hct.Optimization.load_config(heat_sink_pkl_path)
        df_heat_sink = hct.Optimization.study_to_df(config)

        df_heat_sink[volume_key] = df_heat_sink[volume_key] * FACTOR_M3_TO_CM3
        df_heat_sink[area_key] = df_heat_sink[area_key] * FACTOR_M2_TO_CM2

        x_values_list = []
        y_values_list = []
        legend_list: list[str | None] = []

        # filter for different heat sink surface areas
        for area_min in a_min_m2_list:

            df_a_min = df_heat_sink.loc[df_heat_sink[area_key] > area_min * FACTOR_M2_TO_CM2]

            x_values_list.append(df_a_min[volume_key])
            y_values_list.append(df_a_min[loss_key])
            legend_list.append(f"{int(area_min * FACTOR_M2_TO_CM2)} cm²")

        # Set the target directory
        fig_name = os.path.join(summary_directory, "heat_sink")

        # plot all the different heat sink areas
        ParetoPlots.generate_pareto_plot(x_values_list, y_values_list, color_list, alpha_list=[0.5, 0.5, 0.5],
                                         x_label=r'$V_\mathrm{HS}$ / cm³', y_label=r'$R_\mathrm{th,HS}$ / (K/W)',
                                         label_list=legend_list,
                                         fig_name_path=fig_name)

    @staticmethod
    def plot_summary(summary_study_data: StudyData, circuit_optimization: CircuitOptimizationBase, combination_id: int = 0) -> None:
        """
        Plot the combined results of circuit, inductor, transformer and heat sink in the Pareto plane.

        :param summary_study_data: Information about the summary study name and study path
        :type  summary_study_data: StudyData
        :param circuit_optimization: circuit optimization class
        :type  circuit_optimization: CircuitOptimizationBase
        :param combination_id: combination ID to highlight in the Pareto plane
        :type combination_id: int
        """
        total_volume_key = "total_volume"
        total_mean_loss_key = "total_mean_loss"

        # Assemble summary data csv-file name
        summary_data_csv_file = os.path.join(summary_study_data.optimization_directory, DF_SUMMARY_FINAL)
        # Load data frame from csv-file
        df = pd.read_csv(summary_data_csv_file)

        df_filtered = circuit_optimization.filter_df(df, x=total_volume_key, y=total_mean_loss_key,
                                                     factor_min_dc_losses=0.001, factor_max_dc_losses=10)

        gps.global_plot_settings_font_latex()
        fig = plt.figure(figsize=(80/25.4, 60/25.4), dpi=1000)
        x_values_list = [df[total_volume_key] * FACTOR_M3_TO_CM3, df_filtered[total_volume_key] * FACTOR_M3_TO_CM3]
        y_values_list = [df[total_mean_loss_key], df_filtered[total_mean_loss_key]]
        label_list: list[str | None] = ["Design", "Best designs"]

        if combination_id != 0:
            volume = df.loc[df["combination_id"] == combination_id][total_volume_key].values[0] * FACTOR_M3_TO_CM3
            loss = df.loc[df["combination_id"] == combination_id][total_mean_loss_key].values[0]
            x_values_list.append(volume)
            y_values_list.append(loss)
            label_list.append(str(combination_id))

        # Set the target directory
        fig_name = os.path.join(summary_study_data.optimization_directory, "summary")

        x_scale_min = 0.9 * df_filtered[total_volume_key].min() * FACTOR_M3_TO_CM3
        x_scale_max = 1.1 * df_filtered[total_volume_key].max() * FACTOR_M3_TO_CM3

        y_scale_min = 0.9 * df_filtered[total_mean_loss_key].min()
        y_scale_max = 1.1 * df_filtered[total_mean_loss_key].max()

        ParetoPlots.generate_pareto_plot(x_values_list, y_values_list, label_list=label_list, color_list=["black", "red", "green"], alpha_list=[0.5, 0.5, 1],
                                         x_label=r"$V_\mathrm{Converter}$ / cm³", y_label=r"$P_\mathrm{Converter,mean}$ / W",
                                         fig_name_path=fig_name, xlim=[x_scale_min, x_scale_max], ylim=[y_scale_min, y_scale_max])

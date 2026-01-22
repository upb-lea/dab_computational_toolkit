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
from dct.constant_path import DF_SUMMARY_FINAL


class ParetoPlots:
    """Generate PDF plots to see the results of single Pareto steps (circuit, inductor, transformer, heat sink)."""

    @staticmethod
    def generate_pareto_plot(x_values_list: list, y_values_list: list, color_list: list, alpha: float,
                             x_label: str, y_label: str, label_list: list[str | None], fig_name_path: str,
                             xlim: list | None = None, ylim: list | None = None) -> None:
        """
        Generate multiple Pareto plot in one PDF file.

        :param x_values_list: list of different Pareto plot x values
        :type x_values_list: list
        :param y_values_list: list of different Pareto plot y values
        :type y_values_list: list
        :param color_list:
        :param alpha: The alpha blending value, between 0 (transparent) and 1 (opaque).
        :type alpha: float
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
            plt.scatter(x_values, y_values, color=color, alpha=alpha, label=legend)

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

        if not os.path.exists(f"{path}/pdf"):
            os.mkdir(f"{path}/pdf")
        if not os.path.exists(f"{path}/png"):
            os.mkdir(f"{path}/png")
        if not os.path.exists(f"{path}/pkl"):
            os.mkdir(f"{path}/pkl")
        plt.savefig(f"{path}/pdf/{fig_name}.pdf")
        plt.savefig(f"{path}/png/{fig_name}.png")
        # Save the figure as pickle file type, for later view
        with open(f"{path}/pkl/{fig_name}.pkl", "wb") as f:
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
                                         alpha=plot_data.alpha, x_label=plot_data.x_label, y_label=plot_data.y_label,
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
            factor_m3_cm3 = 1e6
            df["values_0"] = df["values_0"] * factor_m3_cm3

            fem_results_folder_path = os.path.join(file_path, "02_fem_simulation_results")
            df_filtered = fmt.InductorOptimization.ReluctanceModel.filter_loss_list_df(df, factor_min_dc_losses=0.2, factor_max_dc_losses=100)
            df_fem_reluctance = fmt.InductorOptimization.FemSimulation.fem_logs_to_df(df_filtered, fem_results_folder_path)

            # all fem simulation points
            fem_loss_results = df_fem_reluctance["fem_p_loss_winding"] + df_fem_reluctance["fem_eddy_core"] + df_fem_reluctance["user_attrs_p_hyst"]

            x_values_list = [df["values_0"], df_filtered["values_0"], df_fem_reluctance["values_0"]]
            y_values_list = [df["values_1"], df_filtered["values_1"], fem_loss_results]
            label_list: list[str | None] = ["RM all", "RM filtered", "FEM"]

            x_scale_min = 0.9 * df_filtered["values_0"].min()
            x_scale_max = 1.1 * df_filtered["values_0"].max()

            y_scale_min = 0.9 * df_filtered["values_1"].min()
            y_scale_max = 1.1 * df_filtered["values_1"].max()

            # Set the target directory
            fig_name = os.path.join(summary_directory, f"inductor_c{circuit_number}")

            ParetoPlots.generate_pareto_plot(x_values_list, y_values_list, color_list=["black", "red", "green"], alpha=0.5, x_label=r'$V_\mathrm{ind}$ / cm³',
                                             y_label=r'$P_\mathrm{ind}$ / W', label_list=label_list,
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
            factor_m3_cm3 = 1e6
            df["values_0"] = df["values_0"] * factor_m3_cm3

            fem_results_folder_path = os.path.join(file_path, "02_fem_simulation_results")
            df_filtered = fmt.StackedTransformerOptimization.ReluctanceModel.filter_loss_list_df(df, factor_min_dc_losses=0.2, factor_max_dc_losses=10)
            df_fem_reluctance = fmt.StackedTransformerOptimization.FemSimulation.fem_logs_to_df(df_filtered, fem_results_folder_path)

            # all fem simulation points
            fem_loss_results = df_fem_reluctance["fem_p_loss_winding"] + df_fem_reluctance["fem_eddy_core"] + df_fem_reluctance["user_attrs_p_hyst"]

            x_values_list = [df["values_0"], df_filtered["values_0"], df_fem_reluctance["values_0"]]
            y_values_list = [df["values_1"], df_filtered["values_1"], fem_loss_results]
            label_list: list[str | None] = ["RM all", "RM filtered", "FEM"]

            x_scale_min = 0.9 * df_filtered["values_0"].min()
            x_scale_max = 1.1 * df_filtered["values_0"].max()

            y_scale_min = 0.9 * df_filtered["values_1"].min()
            y_scale_max = 1.1 * df_filtered["values_1"].max()

            # Set the target directory
            fig_name = os.path.join(summary_directory, f"transformer_c{circuit_number}")

            ParetoPlots.generate_pareto_plot(x_values_list, y_values_list, color_list=["black", "red", "green"], alpha=0.5, x_label=r'$V_\mathrm{ind}$ / cm³',
                                             y_label=r'$P_\mathrm{ind}$ / W', label_list=label_list,
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
        # factor definitions
        factor_m2_cm2 = 10000
        factor_m3_cm3 = 1e6

        # target color list and different heat sink areas to plot (split 3d plot into several 2d plots)
        color_list = [gps.colors()["black"], gps.colors()["red"], gps.colors()["blue"], gps.colors()["green"]]
        a_min_m2_list = [0.002, 0.003, 0.005]

        # Assemble heat sink pkl-file name
        heat_sink_pkl_path = os.path.join(heat_sink_study_data.optimization_directory,
                                          f"{heat_sink_study_data.study_name}.pkl")
        # Load data from pkl-file
        config = hct.Optimization.load_config(heat_sink_pkl_path)
        df_heat_sink = hct.Optimization.study_to_df(config)

        df_heat_sink["values_0"] = df_heat_sink["values_0"] * factor_m3_cm3
        df_heat_sink["values_2"] = df_heat_sink["values_2"] * factor_m2_cm2

        x_values_list = []
        y_values_list = []
        legend_list: list[str | None] = []

        # filter for different heat sink surface areas
        for area_min in a_min_m2_list:

            df_a_min = df_heat_sink.loc[df_heat_sink["values_2"] > area_min * factor_m2_cm2]

            x_values_list.append(df_a_min["values_0"])
            y_values_list.append(df_a_min["values_1"])
            legend_list.append(f"{int(area_min * factor_m2_cm2)} cm²")

        # Set the target directory
        fig_name = os.path.join(summary_directory, "heat_sink")

        # plot all the different heat sink areas
        ParetoPlots.generate_pareto_plot(x_values_list, y_values_list, color_list, alpha=0.5,
                                         x_label=r'$V_\mathrm{HS}$ / cm³', y_label=r'$R_\mathrm{th,HS}$ / (K/W)',
                                         label_list=legend_list,
                                         fig_name_path=fig_name)

    @staticmethod
    def plot_summary(summary_study_data: StudyData, circuit_optimization: CircuitOptimizationBase) -> None:
        """
        Plot the combined results of circuit, inductor, transformer and heat sink in the Pareto plane.

        :param summary_study_data: Information about the summary study name and study path
        :type  summary_study_data: StudyData
        :param circuit_optimization: circuit optimization class
        :type  circuit_optimization: CircuitOptimizationBase
        """
        # Assemble summary data csv-file name
        summary_data_csv_file = os.path.join(summary_study_data.optimization_directory, DF_SUMMARY_FINAL)
        # Load data frame from csv-file
        df = pd.read_csv(summary_data_csv_file)

        df_filtered = circuit_optimization.filter_df(df, x="total_volume", y="total_mean_loss",
                                                     factor_min_dc_losses=0.001, factor_max_dc_losses=10)

        gps.global_plot_settings_font_latex()
        fig = plt.figure(figsize=(80/25.4, 60/25.4), dpi=1000)
        x_values_list = [df["total_volume"] * 1e6, df_filtered["total_volume"] * 1e6]
        y_values_list = [df["total_mean_loss"], df_filtered["total_mean_loss"]]
        label_list: list[str | None] = ["Design", "Best designs"]

        # Set the target directory
        fig_name = os.path.join(summary_study_data.optimization_directory, "summary")

        x_scale_min = 0.9 * df_filtered["total_volume"].min() * 1e6
        x_scale_max = 1.1 * df_filtered["total_volume"].max() * 1e6

        y_scale_min = 0.9 * df_filtered["total_mean_loss"].min()
        y_scale_max = 1.1 * df_filtered["total_mean_loss"].max()

        ParetoPlots.generate_pareto_plot(x_values_list, y_values_list, label_list=label_list, color_list=["black", "red"], alpha=0.5,
                                         x_label=r"$V_\mathrm{DAB}$ / cm³", y_label=r"$P_\mathrm{DAB,mean}$ / W",
                                         fig_name_path=fig_name, xlim=[x_scale_min, x_scale_max], ylim=[y_scale_min, y_scale_max])

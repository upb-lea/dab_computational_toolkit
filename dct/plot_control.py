"""Plot the optimization results."""

# python libraries
import os

# 3rd party libraries
from matplotlib import pyplot as plt
import pandas as pd

# own libraries
import dct
import hct
import femmt as fmt

class ParetoPlots:
    """Generate PDF plots to see the results of single Pareto steps (circuit, inductor, transformer, heat sink)."""

    @staticmethod
    def generate_pdf_pareto(x_values_list: list, y_values_list: list, color_list: list, alpha: float,
                            x_label: str, y_label: str, label_list: list[str | None], fig_name: str,
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
        :param fig_name: filename, will be saved as pdf
        :type fig_name: str
        :param xlim: x-axis limitation [x_min, x_max]
        :type xlim: list[float]
        :param ylim: y-axis limitation [y_min, y_max]
        :type ylim: list[float]
        """
        # set font to LaTeX font
        dct.global_plot_settings_font_latex()

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
        fig_name = fig_name.replace(".pdf", "")
        plt.savefig(f"{fig_name}.pdf")
        plt.savefig(f"{fig_name}.png")

    @staticmethod
    def read_circuit_numbers_from_filestructure(toml_prog_flow: dct.FlowControl) -> list[str]:
        """
        Get the filtered circuit numbers from the "filtered_results" folder.

        :param toml_prog_flow: Flow control toml file
        :type toml_prog_flow: tc.FlowControl
        :return: List with number of filtered circuit simulations
        :rtype: list[int]
        """
        for _, _, files in os.walk(os.path.join(toml_prog_flow.general.project_directory, toml_prog_flow.circuit.subdirectory,
                                   toml_prog_flow.configuration_data_files.circuit_configuration_file.replace(".toml", ""), "filtered_results")):
            files = [file.replace(".pkl", "") for file in files]
        return files

    @staticmethod
    def plot_circuit_results(toml_prog_flow: dct.FlowControl, is_pre_summary: bool = False) -> None:
        """
        Plot the results of the circuit optimization in the Pareto plane.

        :param toml_prog_flow: Flow control toml file
        :type toml_prog_flow: tc.FlowControl
        :param is_pre_summary: True to store the results in the pre_summary directory
        :type is_pre_summary: bool
        """
        # load circuit configuration file
        dab_config = dct.CircuitOptimization.load_stored_config(
            toml_prog_flow.general.project_directory, toml_prog_flow.configuration_data_files.circuit_configuration_file.replace(
                ".toml", ""))
        # generate circuit dataframe
        df_circuit = dct.CircuitOptimization.study_to_df(dab_config)

        if is_pre_summary:
            fig_name = os.path.join(toml_prog_flow.general.project_directory, toml_prog_flow.pre_summary.subdirectory, "circuit")
        else:
            fig_name = os.path.join(toml_prog_flow.general.project_directory, toml_prog_flow.summary.subdirectory, "circuit")

        ParetoPlots.generate_pdf_pareto([df_circuit["values_0"]], [df_circuit["values_1"]], color_list=[dct.colors()["black"]], alpha=0.5,
                                        x_label=r"$\mathcal{L}_\mathrm{v}$ / \%", y_label=r"$\mathcal{L}_\mathrm{i}$ / A²",
                                        label_list=[None], fig_name=fig_name)

    @staticmethod
    def plot_inductor_results(toml_prog_flow: dct.FlowControl, is_pre_summary: bool = False) -> None:
        """
        Plot the results of the inductor optimization in the Pareto plane.

        :param toml_prog_flow: Flow control toml file
        :type toml_prog_flow: tc.FlowControl
        :param is_pre_summary: True to store the results in the pre_summary directory
        :type is_pre_summary: bool
        """
        circuit_numbers = ParetoPlots.read_circuit_numbers_from_filestructure(toml_prog_flow)

        project_name = toml_prog_flow.general.project_directory
        circuit_study_name = toml_prog_flow.configuration_data_files.circuit_configuration_file.replace(".toml", "")
        inductor_study_name = toml_prog_flow.configuration_data_files.inductor_configuration_file.replace(".toml", "")

        for circuit_number in circuit_numbers:

            file_path = f"/{project_name}/{toml_prog_flow.inductor.subdirectory}/{circuit_study_name}/{circuit_number}/{inductor_study_name}"
            config_pickle_filepath = os.path.join(file_path, f"{inductor_study_name}.pkl")
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

            if is_pre_summary:
                fig_name = os.path.join(toml_prog_flow.general.project_directory, toml_prog_flow.pre_summary.subdirectory, f"inductor_c{circuit_number}")
            else:
                fig_name = os.path.join(toml_prog_flow.general.project_directory, toml_prog_flow.summary.subdirectory, f"inductor_c{circuit_number}")

            ParetoPlots.generate_pdf_pareto(x_values_list, y_values_list, color_list=["black", "red", "green"], alpha=0.5, x_label=r'$V_\mathrm{ind}$ / cm³',
                                            y_label=r'$P_\mathrm{ind}$ / W', label_list=label_list,
                                            fig_name=fig_name, xlim=[x_scale_min, x_scale_max], ylim=[y_scale_min, y_scale_max])

    @staticmethod
    def plot_transformer_results(toml_prog_flow: dct.FlowControl, is_pre_summary: bool = False) -> None:
        """
        Plot the results of the transformer optimization in the Pareto plane.

        :param toml_prog_flow: Flow control toml file
        :type toml_prog_flow: tc.FlowControl
        :param is_pre_summary: True to store the results in the pre_summary directory
        :type is_pre_summary: bool
        """
        circuit_numbers = ParetoPlots.read_circuit_numbers_from_filestructure(toml_prog_flow)

        project_name = toml_prog_flow.general.project_directory
        circuit_study_name = toml_prog_flow.configuration_data_files.circuit_configuration_file.replace(".toml", "")
        transformer_study_name = toml_prog_flow.configuration_data_files.transformer_configuration_file.replace(".toml", "")

        for circuit_number in circuit_numbers:
            file_path = f"/{project_name}/{toml_prog_flow.transformer.subdirectory}/{circuit_study_name}/{circuit_number}/{transformer_study_name}"
            config_pickle_filepath = os.path.join(file_path, f"{transformer_study_name}.pkl")
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

            if is_pre_summary:
                fig_name = os.path.join(toml_prog_flow.general.project_directory, toml_prog_flow.pre_summary.subdirectory, f"transformer_c{circuit_number}")
            else:
                fig_name = os.path.join(toml_prog_flow.general.project_directory, toml_prog_flow.summary.subdirectory, f"transformer_c{circuit_number}")

            ParetoPlots.generate_pdf_pareto(x_values_list, y_values_list, color_list=["black", "red", "green"], alpha=0.5, x_label=r'$V_\mathrm{ind}$ / cm³',
                                            y_label=r'$P_\mathrm{ind}$ / W', label_list=label_list,
                                            fig_name=fig_name, xlim=[x_scale_min, x_scale_max], ylim=[y_scale_min, y_scale_max])

    @staticmethod
    def plot_heat_sink_results(toml_prog_flow: dct.FlowControl, is_pre_summary: bool = False) -> None:
        """
        Plot the results of the heat sink optimization in the Pareto plane.

        :param toml_prog_flow: Flow control toml file
        :type toml_prog_flow: tc.FlowControl
        :param is_pre_summary: True to store the results in the pre_summary directory
        :type is_pre_summary: bool
        """
        # factor definitions
        factor_m2_cm2 = 10000
        factor_m3_cm3 = 1e6

        # target color list and different heat sink areas to plot (split 3d plot into several 2d plots)
        color_list = [dct.colors()["black"], dct.colors()["red"], dct.colors()["blue"], dct.colors()["green"]]
        a_min_m2_list = [0.002, 0.003, 0.005]

        study_name = toml_prog_flow.configuration_data_files.heat_sink_configuration_file.replace(".toml", "")
        heat_sink_pkl_path = os.path.join(toml_prog_flow.general.project_directory, toml_prog_flow.heat_sink.subdirectory,
                                          study_name, f"{study_name}.pkl")

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

        if is_pre_summary:
            fig_name = os.path.join(toml_prog_flow.general.project_directory, toml_prog_flow.pre_summary.subdirectory, "heat_sink")
        else:
            fig_name = os.path.join(toml_prog_flow.general.project_directory, toml_prog_flow.summary.subdirectory, "heat_sink")

        # plot all the different heat sink areas
        ParetoPlots.generate_pdf_pareto(x_values_list, y_values_list, color_list, alpha=0.5,
                                        x_label=r'$V_\mathrm{HS}$ / cm³', y_label=r'$R_\mathrm{th,HS}$ / (K/W)',
                                        label_list=legend_list,
                                        fig_name=fig_name)

    @staticmethod
    def plot_summary(toml_prog_flow: dct.FlowControl, is_pre_summary: bool = False) -> None:
        """
        Plot the combined results of circuit, inductor, transformer and heat sink in the Pareto plane.

        :param toml_prog_flow: Flow control toml file
        :type toml_prog_flow: tc.FlowControl
        :param is_pre_summary: True to store the results in the pre_summary directory
        :type is_pre_summary: bool
        """
        if is_pre_summary:
            df = pd.read_csv(f"{toml_prog_flow.general.project_directory}/{toml_prog_flow.pre_summary.subdirectory}/df_w_hs.csv")
        else:
            df = pd.read_csv(f"{toml_prog_flow.general.project_directory}/{toml_prog_flow.summary.subdirectory}/df_w_hs.csv")

        df_filtered = dct.CircuitOptimization.filter_df(df, x="total_volume", y="total_mean_loss",
                                                        factor_min_dc_losses=0.001, factor_max_dc_losses=10)

        dct.global_plot_settings_font_latex()
        fig = plt.figure(figsize=(80/25.4, 60/25.4), dpi=1000)
        x_values_list = [df["total_volume"] * 1e6, df_filtered["total_volume"] * 1e6]
        y_values_list = [df["total_mean_loss"], df_filtered["total_mean_loss"]]
        label_list: list[str | None] = ["Design", "Best designs"]

        if is_pre_summary:
            fig_name = os.path.join(toml_prog_flow.general.project_directory, toml_prog_flow.pre_summary.subdirectory, "summary")
        else:
            fig_name = os.path.join(toml_prog_flow.general.project_directory, toml_prog_flow.summary.subdirectory, "summary")

        x_scale_min = 0.9 * df_filtered["total_volume"].min() * 1e6
        x_scale_max = 1.1 * df_filtered["total_volume"].max() * 1e6

        y_scale_min = 0.9 * df_filtered["total_mean_loss"].min()
        y_scale_max = 1.1 * df_filtered["total_mean_loss"].max()

        ParetoPlots.generate_pdf_pareto(x_values_list, y_values_list, label_list=label_list, color_list=["red", "green"], alpha=0.5,
                                        x_label=r"$V_\mathrm{DAB}$ / cm³", y_label=r"$P_\mathrm{DAB,mean}$ / W",
                                        fig_name=fig_name, xlim=[x_scale_min, x_scale_max], ylim=[y_scale_min, y_scale_max])

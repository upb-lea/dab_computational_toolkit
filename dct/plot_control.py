"""Plot the optimization results."""
import os.path

# python libraries
import hct

# 3rd party libraries
from matplotlib import pyplot as plt

# own libraries
import dct

class ParetoPlots:
    """Generate PDF plots to see the results of single Pareto steps (circuit, inductor, transformer, heat sink)."""

    @staticmethod
    def generate_pdf_pareto(x_values_list: list, y_values_list: list, color_list: list, alpha: float,
                            x_label: str, y_label: str, label_list: list[str | None], fig_name: str):
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
        plt.grid()
        plt.tight_layout()
        # make sure to not generate a filename.pdf.pdf (twice ".pdf").
        fig_name = fig_name.replace(".pdf", "")
        plt.savefig(f"{fig_name}.pdf")

    @staticmethod
    def plot_circuit_results(toml_prog_flow: dct.FlowControl) -> None:
        """
        Plot the results of the circuit optimization in the Pareto plane.

        :param toml_prog_flow: Flow control toml file
        :type toml_prog_flow: tc.FlowControl
        """
        # load circuit configuration file
        dab_config = dct.CircuitOptimization.load_config(
            toml_prog_flow.general.project_directory, toml_prog_flow.configuration_data_files.circuit_configuration_file.replace(".toml", ""))
        # generate circuit dataframe
        df_circuit = dct.CircuitOptimization.study_to_df(dab_config)

        ParetoPlots.generate_pdf_pareto([df_circuit["values_0"]], [df_circuit["values_1"]], color_list=[dct.colors()["black"]], alpha=0.5,
                                        x_label=r"$\mathcal{L}_\mathrm{v}$ / \%", y_label=r"$\mathcal{L}_\mathrm{i}$ / A²",
                                        label_list=[None], fig_name="circuit")

    @staticmethod
    def plot_heat_sink_results(toml_prog_flow: dct.FlowControl) -> None:
        """
        Plot the results of the heat sink optimization in the Pareto plane.

        :param toml_prog_flow: Flow control toml file
        :type toml_prog_flow: tc.FlowControl
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

        # plot all the different heat sink areas
        ParetoPlots.generate_pdf_pareto(x_values_list, y_values_list, color_list, alpha=0.5,
                                        x_label=r'$V_\mathrm{HS}$ / cm³', y_label=r'$R_\mathrm{th,HS}$ / (K/W)',
                                        label_list=legend_list, fig_name="heat_sink")

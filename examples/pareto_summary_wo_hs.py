"""Heat sink Pareto optimization."""
# python libraries
import os
import pickle

# 3rd party libraries
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# own libraries
import dct

# specify input parameters
project_name = "2025-03-12_debug"
circuit_study_name = "circuit_01"
inductor_study_name_list = ["inductor_01"]
stacked_transformer_study_name_list = ["transformer_01"]

df = pd.DataFrame()

# load project file paths
filepaths = dct.Optimization.load_filepaths(os.path.abspath(os.path.join(os.curdir, project_name)))

circuit_filepath_results = os.path.join(filepaths.circuit, circuit_study_name, "filtered_results")
circuit_objects = os.scandir(circuit_filepath_results)
circuit_numbers = [entry.name.split(os.extsep)[0] for entry in circuit_objects]

print(circuit_numbers)

transistor_b1_cooling = dct.TransistorCooling(
    tim_thickness=1e-3,
    tim_conductivity=12,
)

transistor_b2_cooling = dct.TransistorCooling(
    tim_thickness=1e-3,
    tim_conductivity=12,
)


inductor_cooling = dct.InductiveElementCooling(
    tim_conductivity=12,
    tim_thickness=1e-3
)

transformer_cooling = dct.InductiveElementCooling(
    tim_conductivity=12,
    tim_thickness=1e-3
)


heat_sink = dct.HeatSinkTemp(
    t_ambient=40,
    t_hs_max=90,
)

def df_plot_final_pareto_front(df: pd.DataFrame, figure_size: tuple | None = None):
    """
    Plot an interactive Pareto diagram (losses vs. volume).

    :param df: Dataframe
    :type df: pd.Dataframe
    :param figure_size: figures size as a x/y-tuple in mm, e.g. (160, 80)
    :type figure_size: tuple
    """
    df["name_column"] = "c:" + df["circuit_number"] + " i: " + df["inductor_number"] + " t: " + df["transformer_number"]
    names = df["name_column"].to_numpy()

    fig, ax = plt.subplots(figsize=[x / 25.4 for x in figure_size] if figure_size is not None else None, dpi=80)
    sc = plt.scatter(df["total_volume"], df["total_mean_loss"], s=10)

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

    plt.xlabel(r'Volume in m³')
    plt.ylabel(r'$R_\mathrm{th}$ in K/W')
    plt.grid()
    plt.tight_layout()
    plt.show()


# iterate circuit numbers
for circuit_number in circuit_numbers:
    circuit_filepath_results = os.path.join(filepaths.circuit, circuit_study_name, "filtered_results")
    circuit_filepath_number = os.path.join(circuit_filepath_results, f"{circuit_number}.pkl")

    # Get circuit results
    circuit_dto = dct.HandleDabDto.load_from_file(circuit_filepath_number)

    print(f"{circuit_number=}")

    # iterate inductor study
    for inductor_study_name in inductor_study_name_list:
        inductor_filepath_results = os.path.join(filepaths.inductor, circuit_study_name, circuit_number,
                                                 inductor_study_name, "09_circuit_dtos_incl_inductor_losses")
        if os.path.exists(inductor_filepath_results):

            inductor_objects = os.scandir(inductor_filepath_results)
            inductor_numbers = [entry.name.split(os.extsep)[0] for entry in inductor_objects]

            # iterate inductor numbers
            for inductor_number in inductor_numbers:
                inductor_filepath_number = os.path.join(inductor_filepath_results, f"{inductor_number}.pkl")

                # Get inductor results
                with open(inductor_filepath_number, 'rb') as pickle_file_data:
                    inductor_dto = pickle.load(pickle_file_data)

                if int(inductor_dto.circuit_trial_number) != int(circuit_number):
                    raise ValueError(f"{inductor_dto.circuit_trial_number=} != {circuit_number}")
                if int(inductor_dto.inductor_trial_number) != int(inductor_number):
                    raise ValueError(f"{inductor_dto.inductor_trial_number=} != {inductor_number}")

                inductance_loss_matrix = inductor_dto.p_combined_losses

                for stacked_transformer_study_name in stacked_transformer_study_name_list:
                    stacked_transformer_filepath_results = os.path.join(filepaths.transformer, circuit_study_name, circuit_number,
                                                                        stacked_transformer_study_name, "09_circuit_dtos_incl_transformer_losses")

                    if os.path.exists(stacked_transformer_filepath_results):

                        stacked_transformer_objects = os.scandir(stacked_transformer_filepath_results)
                        stacked_transformer_numbers = [entry.name.split(os.extsep)[0] for entry in stacked_transformer_objects]

                        for stacked_transformer_number in stacked_transformer_numbers:
                            stacked_transformer_filepath_number = os.path.join(stacked_transformer_filepath_results, f"{stacked_transformer_number}.pkl")

                            # get transformer results
                            with open(stacked_transformer_filepath_number, 'rb') as pickle_file_data:
                                transformer_dto = pickle.load(pickle_file_data)

                            if int(transformer_dto.circuit_trial_number) != int(circuit_number):
                                raise ValueError(f"{transformer_dto.circuit_trial_number=} != {circuit_number}")
                            if int(transformer_dto.stacked_transformer_trial_number) != int(stacked_transformer_number):
                                raise ValueError(f"{transformer_dto.stacked_transformer_trial_number=} != {stacked_transformer_number}")

                            transformer_loss_matrix = transformer_dto.p_combined_losses

                            # get transistor results
                            total_transistor_cond_loss_matrix = 4 * (circuit_dto.calc_losses.p_m1_conduction + circuit_dto.calc_losses.p_m2_conduction)

                            b1_transistor_cond_loss_matrix = circuit_dto.calc_losses.p_m1_conduction

                            b2_transistor_cond_loss_matrix = circuit_dto.calc_losses.p_m2_conduction

                            total_loss_matrix = (inductor_dto.p_combined_losses + total_transistor_cond_loss_matrix + \
                                                 transformer_dto.p_combined_losses)

                            # maximum loss indices
                            max_loss_all_index = np.unravel_index(total_loss_matrix.argmax(), np.shape(total_loss_matrix))
                            max_loss_circuit_1_index = np.unravel_index(b1_transistor_cond_loss_matrix.argmax(),
                                                                        np.shape(b1_transistor_cond_loss_matrix))
                            max_loss_circuit_2_index = np.unravel_index(b2_transistor_cond_loss_matrix.argmax(),
                                                                        np.shape(b2_transistor_cond_loss_matrix))
                            max_loss_inductor_index = np.unravel_index(inductance_loss_matrix.argmax(), np.shape(inductance_loss_matrix))
                            max_loss_transformer_index = np.unravel_index(transformer_loss_matrix.argmax(), np.shape(transformer_loss_matrix))

                            # get all the losses in a matrix
                            r_th_copper_coin_1, copper_coin_area_1 = dct.calculate_r_th_copper_coin(circuit_dto.input_config.transistor_dto_1.cooling_area)
                            r_th_copper_coin_2, copper_coin_area_2 = dct.calculate_r_th_copper_coin(circuit_dto.input_config.transistor_dto_2.cooling_area)

                            circuit_r_th_tim_1 = dct.calculate_r_th_tim(copper_coin_area_1, transistor_b1_cooling)
                            circuit_r_th_tim_2 = dct.calculate_r_th_tim(copper_coin_area_2, transistor_b2_cooling)

                            circuit_r_th_1_jhs = circuit_dto.input_config.transistor_dto_1.r_th_jc + r_th_copper_coin_1 + circuit_r_th_tim_1
                            circuit_r_th_2_jhs = circuit_dto.input_config.transistor_dto_2.r_th_jc + r_th_copper_coin_2 + circuit_r_th_tim_2

                            circuit_heat_sink_max_1_matrix = (
                                circuit_dto.input_config.transistor_dto_1.t_j_max_op - circuit_r_th_1_jhs * b1_transistor_cond_loss_matrix)
                            circuit_heat_sink_max_2_matrix = (
                                circuit_dto.input_config.transistor_dto_2.t_j_max_op - circuit_r_th_2_jhs * b2_transistor_cond_loss_matrix)

                            r_th_ind_heat_sink = 1 / inductor_cooling.tim_conductivity * inductor_cooling.tim_thickness / inductor_dto.area_to_heat_sink
                            temperature_inductor_heat_sink_max_matrix = 125 - r_th_ind_heat_sink * inductance_loss_matrix

                            r_th_xfmr_heat_sink = (1 / transformer_cooling.tim_conductivity * \
                                                   transformer_cooling.tim_thickness / transformer_dto.area_to_heat_sink)
                            temperature_xfmr_heat_sink_max_matrix = 125 - r_th_xfmr_heat_sink * transformer_loss_matrix

                            # maximum heat sink temperatures (minimum of all the maximum temperatures of single components)
                            t_min_matrix = np.minimum(circuit_heat_sink_max_1_matrix, circuit_heat_sink_max_2_matrix)
                            t_min_matrix = np.minimum(t_min_matrix, temperature_inductor_heat_sink_max_matrix)
                            t_min_matrix = np.minimum(t_min_matrix, temperature_xfmr_heat_sink_max_matrix)
                            t_min_matrix = np.minimum(t_min_matrix, heat_sink.t_hs_max)

                            # maximum delta temperature over the heat sink
                            delta_t_max_heat_sink_matrix = t_min_matrix - heat_sink.t_ambient

                            r_th_heat_sink_target_matrix = delta_t_max_heat_sink_matrix / total_loss_matrix

                            r_th_target = r_th_heat_sink_target_matrix.min()

                            data = {
                                # circuit
                                "circuit_number": circuit_number,
                                "circuit_mean_loss": np.mean(total_transistor_cond_loss_matrix),
                                "circuit_max_all_loss": total_transistor_cond_loss_matrix[max_loss_all_index],
                                "circuit_max_circuit_ib_loss": total_transistor_cond_loss_matrix[max_loss_circuit_1_index],
                                "circuit_max_circuit_ob_loss": total_transistor_cond_loss_matrix[max_loss_circuit_2_index],
                                "circuit_max_inductor_loss": total_transistor_cond_loss_matrix[max_loss_inductor_index],
                                "circuit_max_transformer_loss": total_transistor_cond_loss_matrix[max_loss_transformer_index],
                                "circuit_t_j_max_1": circuit_dto.input_config.transistor_dto_1.t_j_max_op,
                                "circuit_t_j_max_2": circuit_dto.input_config.transistor_dto_2.t_j_max_op,
                                "circuit_r_th_ib_jhs_1": circuit_r_th_1_jhs,
                                "circuit_r_th_ib_jhs_2": circuit_r_th_2_jhs,
                                "circuit_heat_sink_temperature_max_1": circuit_heat_sink_max_1_matrix[max_loss_circuit_1_index],
                                "circuit_heat_sink_temperature_max_2": circuit_heat_sink_max_2_matrix[max_loss_circuit_2_index],
                                "circuit_area": 4 * (copper_coin_area_1 + copper_coin_area_2),
                                # inductor
                                "inductor_study_name": inductor_study_name,
                                "inductor_number": inductor_number,
                                "inductor_volume": inductor_dto.volume,
                                "inductor_mean_loss": np.mean(inductance_loss_matrix),
                                "inductor_max_all_loss": inductance_loss_matrix[max_loss_all_index],
                                "inductor_max_circuit_ib_loss": inductance_loss_matrix[max_loss_circuit_1_index],
                                "inductor_max_circuit_ob_loss": inductance_loss_matrix[max_loss_circuit_2_index],
                                "inductor_max_inductor_loss": inductance_loss_matrix[max_loss_inductor_index],
                                "inductor_max_transformer_loss": inductance_loss_matrix[max_loss_transformer_index],
                                "inductor_t_max": 0,
                                "inductor_heat_sink_temperature_max": temperature_inductor_heat_sink_max_matrix[max_loss_inductor_index],
                                "inductor_area": inductor_dto.area_to_heat_sink,
                                # transformer
                                "transformer_study_name": stacked_transformer_study_name,
                                "transformer_number": stacked_transformer_number,
                                "transformer_volume": transformer_dto.volume,
                                "transformer_mean_loss": np.mean(transformer_dto.p_combined_losses),
                                "transformer_max_all_loss": transformer_loss_matrix[max_loss_all_index],
                                "transformer_max_circuit_ib_loss": transformer_loss_matrix[max_loss_circuit_1_index],
                                "transformer_max_circuit_ob_loss": transformer_loss_matrix[max_loss_circuit_2_index],
                                "transformer_max_inductor_loss": transformer_loss_matrix[max_loss_inductor_index],
                                "transformer_max_transformer_loss": transformer_loss_matrix[max_loss_transformer_index],
                                "transformer_t_max": 0,
                                "transformer_heat_sink_temperature_max": temperature_xfmr_heat_sink_max_matrix[max_loss_transformer_index],
                                "transformer_area": transformer_dto.area_to_heat_sink,

                                # summary
                                "total_losses": total_loss_matrix[max_loss_all_index],

                                # heat sink
                                "r_th_heat_sink": r_th_target
                            }
                            local_df = pd.DataFrame([data])

                            df = pd.concat([df, local_df], axis=0)

df["total_area"] = df["circuit_area"] + df["inductor_area"] + df["transformer_area"]
df["total_mean_loss"] = df["circuit_mean_loss"] + df["inductor_mean_loss"] + df["transformer_mean_loss"]
df["volume_wo_heat_sink"] = df["transformer_volume"] + df["inductor_volume"]
df.to_csv(f"{filepaths.heat_sink}/result_df.csv")

fig, ax = plt.subplots(3, 1)
ax[0].scatter(df["volume_wo_heat_sink"], df["total_mean_loss"])
ax[0].set_xlabel(r"$V_\mathrm{HS}$ / m³")
ax[0].set_ylabel(r"$P_\mathrm{DAB,mean}$ / W")
ax[0].grid()

ax[1].scatter(df["total_area"], df["total_mean_loss"])
ax[1].set_xlabel(r"$A_\mathrm{HS}$ / m²")
ax[1].set_ylabel("$P_\mathrm{DAB,mean}$ / W")
ax[1].grid()

ax[2].scatter(df["volume_wo_heat_sink"], df["r_th_heat_sink"])
ax[2].set_xlabel(r"$V_\mathrm{HS}$ / m³")
ax[2].set_ylabel(r"R_\mathrm{th,max} / (K/W)")
ax[2].grid()

plt.tight_layout()
plt.show()

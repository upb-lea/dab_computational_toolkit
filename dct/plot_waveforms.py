"""Plot waveforms from calculation and from simulation."""
# python libraries

# own libraries
from dct import DabDTO

# 3rd party libraries
import numpy as np
from matplotlib import pyplot as plt

def plot_calc_waveforms(dab_dto: DabDTO, compare_gecko_waveforms: bool = False):
    """
    Plot calculated current waveforms for Ls, Lc1, Lc2.

    :param dab_dto: DAB DTO
    :type dab_dto: DabDTO
    :param compare_gecko_waveforms: True to compare calculation with simulated waveforms (GeckoCIRCUITS)
    :type compare_gecko_waveforms: bool
    """
    print(f"{np.shape(dab_dto.gecko_waveforms.i_Ls)=}")
    print(f"{type(dab_dto.gecko_waveforms.time)=}")

    for vec_vvp in np.ndindex(dab_dto.calc_modulation.phi.shape):

        # set simulation parameters and convert tau to degree for Gecko
        sorted_angles = np.transpose(dab_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))[vec_vvp]
        unsorted_angles = np.transpose(dab_dto.calc_currents.angles_rad_unsorted, (1, 2, 3, 0))[vec_vvp]
        i_l_s_sorted = np.transpose(dab_dto.calc_currents.i_l_s_sorted, (1, 2, 3, 0))[vec_vvp]
        i_l_1_sorted = np.transpose(dab_dto.calc_currents.i_l_1_sorted, (1, 2, 3, 0))[vec_vvp]
        i_l_2_sorted = np.transpose(dab_dto.calc_currents.i_l_2_sorted, (1, 2, 3, 0))[vec_vvp]

        sorted_total_angles = np.append(sorted_angles, np.pi + sorted_angles)
        sorted_i_l_s_total = np.append(i_l_s_sorted, -1 * i_l_s_sorted)
        sorted_i_l_1_total = np.append(i_l_1_sorted, -1 * i_l_1_sorted)
        sorted_i_l_2_total = np.append(i_l_2_sorted, -1 * i_l_2_sorted)
        sorted_total_angles = np.append(np.array([0]), sorted_total_angles)
        sorted_i_l_s_total = np.append(sorted_i_l_s_total[-1], sorted_i_l_s_total)
        sorted_i_l_1_total = np.append(sorted_i_l_1_total[-1], sorted_i_l_1_total)
        sorted_i_l_2_total = np.append(sorted_i_l_2_total[-1], sorted_i_l_2_total)

        # plot arrays with elements only (neglect nan-arrays)
        if np.all(~np.isnan(sorted_i_l_s_total)):

            timebase = '2pi'

            if compare_gecko_waveforms:
                if timebase == '2pi':
                    gecko_time = (dab_dto.gecko_waveforms.time * 2 * np.pi * dab_dto.input_config.fs - \
                                  dab_dto.gecko_additional_params.number_pre_sim_periods * 2 * np.pi - \
                                  dab_dto.gecko_additional_params.t_dead1 * 2 * np.pi * dab_dto.input_config.fs)
                elif timebase == 'time':
                    gecko_time = dab_dto.gecko_waveforms.time - dab_dto.gecko_additional_params.number_pre_sim_periods / dab_dto.input_config.fs - \
                        dab_dto.gecko_additional_params.t_dead1
                    sorted_total_angles = sorted_total_angles / 2 / np.pi / dab_dto.input_config.fs

                # this index corrects the display of the waveforms
                # so waveforms start at zero and will end exactly at the end of the period.
                # without, there is a small shift of the dead time (e.g. 50 ns)
                corr_index = int(np.round(dab_dto.gecko_additional_params.t_dead1 / dab_dto.gecko_additional_params.timestep, decimals=0))

            ax1 = plt.subplot(311)
            plt.plot(sorted_total_angles, sorted_i_l_s_total, label='Calculation')
            if compare_gecko_waveforms:
                plt.plot(gecko_time[corr_index:-corr_index], dab_dto.gecko_waveforms.i_Ls[vec_vvp][corr_index:-corr_index], label='GeckoCIRCUITS')
            plt.ylabel('i_L_s in A')
            plt.grid()
            plt.legend()
            plot_info = (f", P= {dab_dto.calc_config.mesh_P[vec_vvp]} W "
                         f"v1={dab_dto.calc_config.mesh_V1[vec_vvp]} V, v2={dab_dto.calc_config.mesh_V2[vec_vvp]} V,"
                         f"f={dab_dto.input_config.fs=}")

            if dab_dto.calc_modulation.mask_IIIm1[vec_vvp]:
                plt.title("IIIm1" + plot_info)
            if dab_dto.calc_modulation.mask_IIm2[vec_vvp]:
                plt.title("IIm2" + plot_info)
            if dab_dto.calc_modulation.mask_Im2[vec_vvp]:
                plt.title("Im2" + plot_info)

            plt.subplot(312, sharex=ax1)
            plt.plot(sorted_total_angles, sorted_i_l_1_total, label='calculation')
            if compare_gecko_waveforms:
                plt.plot(gecko_time[corr_index:-corr_index], dab_dto.gecko_waveforms.i_Lc1[vec_vvp][corr_index:-corr_index], label='GeckoCIRCUITS')
            plt.legend()
            plt.grid()
            plt.ylabel('i_L_1 in A')

            plt.subplot(313, sharex=ax1)
            plt.plot(sorted_total_angles, sorted_i_l_2_total, label='calculation')
            if compare_gecko_waveforms:
                plt.plot(gecko_time[corr_index:-corr_index], dab_dto.gecko_waveforms.i_Lc2[vec_vvp][corr_index:-corr_index], label='GeckoCIRCUITS')
            plt.ylabel('i_L_2 in A')
            if timebase == '2pi':
                plt.xlabel('t in rad')
            elif timebase == 'time':
                plt.xlabel('time in s')
            plt.legend()
            plt.grid()

            plt.tight_layout()
            plt.show()

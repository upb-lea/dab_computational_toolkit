"""Plot waveforms from calculation and from simulation."""
# python libraries

# own libraries
from dct.topology.dab import dab_datasets_dtos as d_dtos
from dct.topology.dab import dab_functions_waveforms as fw
from dct.components.component_dtos import InductorRequirements, TransformerRequirements

# 3rd party libraries
import numpy as np
from matplotlib import pyplot as plt

def plot_calc_waveforms(dab_dto: d_dtos.DabCircuitDTO, compare_gecko_waveforms: bool = False) -> None:
    """
    Plot calculated current waveforms for Ls, Lc1, Lc2.

    :param dab_dto: DAB DTO
    :type dab_dto: DabCircuitDTO
    :param compare_gecko_waveforms: True to compare calculation with simulated waveforms (GeckoCIRCUITS)
    :type compare_gecko_waveforms: bool
    """
    if not isinstance(dab_dto.gecko_waveforms, d_dtos.GeckoWaveforms):
        raise TypeError(f"{dab_dto.gecko_waveforms} is not of Type GeckoWaveforms.")

    for vec_vvp in np.ndindex(dab_dto.calc_modulation.phi.shape):

        # set simulation parameters and convert tau to degree for Gecko
        sorted_angles = np.transpose(dab_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))[vec_vvp]
        unsorted_angles = np.transpose(dab_dto.calc_currents.angles_rad_unsorted, (1, 2, 3, 0))[vec_vvp]
        i_l_s_sorted = np.transpose(dab_dto.calc_currents.i_l_s_sorted, (1, 2, 3, 0))[vec_vvp]
        i_l_1_sorted = np.transpose(dab_dto.calc_currents.i_l_1_sorted, (1, 2, 3, 0))[vec_vvp]
        i_l_2_sorted = np.transpose(dab_dto.calc_currents.i_l_2_sorted, (1, 2, 3, 0))[vec_vvp]

        sorted_total_angles = fw.full_angle_waveform_from_angles(sorted_angles)
        sorted_i_l_s_total = fw.full_current_waveform_from_currents(i_l_s_sorted)
        sorted_i_l_1_total = fw.full_current_waveform_from_currents(i_l_1_sorted)
        sorted_i_l_2_total = fw.full_current_waveform_from_currents(i_l_2_sorted)

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
            plot_info = (f", P= {dab_dto.input_config.mesh_p[vec_vvp]} W "
                         f"v1={dab_dto.input_config.mesh_v1[vec_vvp]} V, v2={dab_dto.input_config.mesh_v2[vec_vvp]} V,"
                         f"f={dab_dto.input_config.fs=}")

            if dab_dto.calc_modulation.mask_zvs[vec_vvp]:
                color = "green"
            else:
                color = "red"

            if dab_dto.calc_modulation.mask_IIIm1[vec_vvp]:
                plt.title("IIIm1" + plot_info, color=color)
            if dab_dto.calc_modulation.mask_IIm2[vec_vvp]:
                plt.title("IIm2" + plot_info, color=color)
            if dab_dto.calc_modulation.mask_Im2[vec_vvp]:
                plt.title("Im2" + plot_info, color=color)

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

def plot_calc_i_hf_waveforms(dab_dto: d_dtos.DabCircuitDTO, compare_gecko_waveforms: bool = False) -> None:
    """
    Plot calculated current waveforms for i_hf_1 and i_hf_2.

    :param dab_dto: DAB DTO
    :type dab_dto: DabCircuitDTO
    :param compare_gecko_waveforms: True to compare calculation with simulated waveforms (GeckoCIRCUITS)
    :type compare_gecko_waveforms: bool
    """
    if not isinstance(dab_dto.gecko_waveforms, d_dtos.GeckoWaveforms):
        raise TypeError(f"{dab_dto.gecko_waveforms} is not of Type GeckoWaveforms.")

    for vec_vvp in np.ndindex(dab_dto.calc_modulation.phi.shape):
        # set simulation parameters and convert tau to degree for Gecko
        sorted_angles = np.transpose(dab_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))[vec_vvp]
        unsorted_angles = np.transpose(dab_dto.calc_currents.angles_rad_unsorted, (1, 2, 3, 0))[vec_vvp]
        i_hf_1_sorted = np.transpose(dab_dto.calc_currents.i_hf_1_sorted, (1, 2, 3, 0))[vec_vvp]
        i_hf_2_sorted = np.transpose(dab_dto.calc_currents.i_hf_2_sorted, (1, 2, 3, 0))[vec_vvp]

        sorted_total_angles = fw.full_angle_waveform_from_angles(sorted_angles)
        sorted_i_hf_1_total = fw.full_current_waveform_from_currents(i_hf_1_sorted)
        sorted_i_hf_2_total = fw.full_current_waveform_from_currents(i_hf_2_sorted)

        # plot arrays with elements only (neglect nan-arrays)
        if np.all(~np.isnan(sorted_i_hf_1_total)):

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

            ax1 = plt.subplot(211)
            plt.plot(sorted_total_angles, sorted_i_hf_1_total, label='Calculation')
            if compare_gecko_waveforms:
                plt.plot(gecko_time[corr_index:-corr_index], dab_dto.gecko_waveforms.i_HF1[vec_vvp][corr_index:-corr_index], label='GeckoCIRCUITS')
            plt.ylabel('i_hf_1 in A')
            ax1.grid()
            ax1.legend()
            plot_info = (f", P= {dab_dto.input_config.mesh_p[vec_vvp]} W "
                         f"v1={dab_dto.input_config.mesh_v1[vec_vvp]} V, v2={dab_dto.input_config.mesh_v2[vec_vvp]} V,"
                         f"f={dab_dto.input_config.fs=}")

            if dab_dto.calc_modulation.mask_zvs[vec_vvp]:
                color = "green"
            else:
                color = "red"

            if dab_dto.calc_modulation.mask_IIIm1[vec_vvp]:
                plt.title("IIIm1" + plot_info, color=color)
            if dab_dto.calc_modulation.mask_IIm2[vec_vvp]:
                plt.title("IIm2" + plot_info, color=color)
            if dab_dto.calc_modulation.mask_Im2[vec_vvp]:
                plt.title("Im2" + plot_info, color=color)

            ax2 = plt.subplot(212, sharex=ax1)
            plt.plot(sorted_total_angles, sorted_i_hf_2_total, label='calculation')
            if compare_gecko_waveforms:
                plt.plot(gecko_time[corr_index:-corr_index], dab_dto.gecko_waveforms.i_HF2[vec_vvp][corr_index:-corr_index], label='GeckoCIRCUITS')
            ax2.legend()
            ax2.grid()
            ax2.set_ylabel('i_hf_2 in A')
            if timebase == '2pi':
                ax2.set_xlabel('t in rad')
            elif timebase == 'time':
                ax2.set_xlabel('time in s')

            plt.tight_layout()
            plt.show()

def plot_calc_vs_requirements(dab_dto: d_dtos.DabCircuitDTO) -> None:
    """
    Verify the component requirement waveforms against the calculated ones.

    :param dab_dto: DAB DTO
    :type dab_dto: DabCircuitDTO
    """
    for vec_vvp in np.ndindex(dab_dto.calc_modulation.phi.shape):
        # set simulation parameters and convert tau to degree for Gecko
        sorted_angles = np.transpose(dab_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))[vec_vvp]
        unsorted_angles = np.transpose(dab_dto.calc_currents.angles_rad_unsorted, (1, 2, 3, 0))[vec_vvp]
        i_hf_1_sorted = np.transpose(dab_dto.calc_currents.i_hf_1_sorted, (1, 2, 3, 0))[vec_vvp]
        i_hf_2_sorted = np.transpose(dab_dto.calc_currents.i_hf_2_sorted, (1, 2, 3, 0))[vec_vvp]

        sorted_total_angles = fw.full_angle_waveform_from_angles(sorted_angles)
        sorted_i_hf_1_total = fw.full_current_waveform_from_currents(i_hf_1_sorted)
        sorted_i_hf_2_total = fw.full_current_waveform_from_currents(i_hf_2_sorted)

        # set simulation parameters and convert tau to degree for Gecko
        i_l_s_sorted = np.transpose(dab_dto.calc_currents.i_l_s_sorted, (1, 2, 3, 0))[vec_vvp]
        i_l_1_sorted = np.transpose(dab_dto.calc_currents.i_l_1_sorted, (1, 2, 3, 0))[vec_vvp]
        i_l_2_sorted = np.transpose(dab_dto.calc_currents.i_l_2_sorted, (1, 2, 3, 0))[vec_vvp]

        sorted_total_angles = fw.full_angle_waveform_from_angles(sorted_angles)
        sorted_i_l_s_total = fw.full_current_waveform_from_currents(i_l_s_sorted)
        sorted_i_l_1_total = fw.full_current_waveform_from_currents(i_l_1_sorted)
        sorted_i_l_2_total = fw.full_current_waveform_from_currents(i_l_2_sorted)

        if dab_dto.component_requirements is None:
            raise TypeError(f"{dab_dto.component_requirements} is None, but must be of type InductorRequirements.")

        # from component requirements
        i_l_1_requirement_time_vec = dab_dto.component_requirements.inductor_requirements[0].time_array[vec_vvp]
        i_l_1_requirement_current_vec = dab_dto.component_requirements.inductor_requirements[0].current_array[vec_vvp]
        i_l_s_requirement_time_vec = dab_dto.component_requirements.transformer_requirements[0].time_array[vec_vvp]
        i_l_s_requirement_current_vec = dab_dto.component_requirements.transformer_requirements[0].current_1_array[vec_vvp]
        i_l_hf2_requirement_time_vec = dab_dto.component_requirements.transformer_requirements[0].time_array[vec_vvp]
        i_l_hf2_requirement_current_vec = dab_dto.component_requirements.transformer_requirements[0].current_2_array[vec_vvp]

        sorted_total_angles = sorted_total_angles / 2 / np.pi / dab_dto.input_config.fs

        # plot arrays with elements only (neglect nan-arrays)
        if np.all(~np.isnan(sorted_i_hf_1_total)):

            fig, axs = plt.subplots(3, 1)

            axs[0].plot(i_l_1_requirement_time_vec, i_l_1_requirement_current_vec, label="Inductor requirement")
            axs[0].plot(sorted_total_angles, sorted_i_hf_1_total - sorted_i_l_s_total, label="Calculation", linestyle='--')
            axs[0].legend()
            axs[0].grid()
            axs[0].set_ylabel("i_l_1 / A")

            axs[1].plot(i_l_s_requirement_time_vec, i_l_s_requirement_current_vec, label="Transformer i1 requirement")
            axs[1].plot(sorted_total_angles, sorted_i_l_s_total, label="Calculation i1", linestyle="--")
            axs[1].legend()
            axs[1].grid()
            axs[1].set_ylabel("i_l_s / A")

            axs[2].plot(i_l_hf2_requirement_time_vec, i_l_hf2_requirement_current_vec, label="Transformer i2 requirement")
            axs[2].plot(sorted_total_angles, sorted_i_hf_2_total, label="Calculation i2", linestyle="--")
            axs[2].legend()
            axs[2].grid()
            axs[2].set_ylabel("i_hf2 / A")

            plt.show()

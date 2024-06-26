"""Plot waveforms from calculation and from simulation."""
# python libraries

# own libraries
from dct import DabDTO

# 3rd party libraries
import numpy as np
from matplotlib import pyplot as plt

def plot_calc_waveforms(dab_dto: DabDTO):
    """Plot calculated current waveforms for Ls, Lc1, Lc2."""
    for vec_vvp in np.ndindex(dab_dto.calc_modulation.phi.shape):

        # set simulation parameters and convert tau to degree for Gecko
        sorted_angles = np.transpose(dab_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))[vec_vvp]
        i_l_s_sorted = np.transpose(dab_dto.calc_currents.i_l_s_sorted, (1, 2, 3, 0))[vec_vvp]
        i_l_1_sorted = np.transpose(dab_dto.calc_currents.i_l_1_sorted, (1, 2, 3, 0))[vec_vvp]
        i_l_2_sorted = np.transpose(dab_dto.calc_currents.i_l_2_sorted, (1, 2, 3, 0))[vec_vvp]

        sorted_total_angles = np.append(sorted_angles, np.pi + sorted_angles)
        sorted_i_l_s_total = np.append(i_l_s_sorted, -1 * i_l_s_sorted)
        sorted_total_angles = np.append(np.array([0]), sorted_total_angles)
        sorted_i_l_s_total = np.append(sorted_i_l_s_total[-1], sorted_i_l_s_total)

        print(f"{sorted_angles=}")
        print(f"{i_l_s_sorted=}")

        plt.subplot(311)
        plt.plot(sorted_total_angles, sorted_i_l_s_total)
        plt.ylabel('i_L_s in A')

        plt.subplot(312)
        plt.plot(sorted_angles, i_l_1_sorted)
        plt.ylabel('i_L_1 in A')

        plt.subplot(313)
        plt.plot(sorted_angles, i_l_2_sorted)
        plt.ylabel('i_L_2 in A')

        plt.tight_layout()
        plt.show()

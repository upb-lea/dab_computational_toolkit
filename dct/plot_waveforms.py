"""Plot waveforms from calculation and from simulation."""
# python libraries
import os

# own libraries
from dct.debug_tools import Log, timeit, info
from dct import DabDTO
from dct import Plot_DAB

# 3rd party libraries
import numpy as np
from matplotlib import pyplot as plt

def plot_calc_waveforms(dab_dto: DabDTO):
    for vec_vvp in np.ndindex(dab_dto.calc_modulation.phi.shape):

        # set simulation parameters and convert tau to degree for Gecko
        sorted_angles = np.transpose(dab_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))[vec_vvp]
        i_l_s_sorted = np.transpose(dab_dto.calc_currents.i_l_s_sorted, (1, 2, 3, 0))[vec_vvp]
        i_l_1_sorted = np.transpose(dab_dto.calc_currents.i_l_1_sorted, (1, 2, 3, 0))[vec_vvp]
        i_l_2_sorted = np.transpose(dab_dto.calc_currents.i_l_2_sorted, (1, 2, 3, 0))[vec_vvp]

        print(f"{sorted_angles=}")
        print(f"{i_l_s_sorted=}")

        plt.subplot(311)
        plt.plot(sorted_angles, i_l_s_sorted)
        plt.ylabel('i_L_s in A')

        plt.subplot(312)
        plt.plot(sorted_angles, i_l_1_sorted)
        plt.ylabel('i_L_1 in A')

        plt.subplot(313)
        plt.plot(sorted_angles, i_l_2_sorted)
        plt.ylabel('i_L_2 in A')

        plt.tight_layout()
        plt.show()

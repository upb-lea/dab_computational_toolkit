"""Show modes of the converter in some plots."""
# python libraries

# 3rd party libraries
from matplotlib import pyplot as plt
import numpy as np

# own libraries
import dct

def plot_mode_overview(dab_config: dct.CircuitDabDTO) -> None:
    """
    Mode overview of the converter. This function is for debugging.

    :param dab_config: DAB configuration file
    :type dab_config: dct.CircuitDabDTO
    """
    fig, axs = plt.subplots(2, 3, sharex=True, sharey=True)

    n = 1

    # plot phi for all modes
    axs[0, 0].contourf(dab_config.calc_config.mesh_P[:, n, :], dab_config.calc_config.mesh_V2[:, n, :], dab_config.calc_modulation.phi[:, n, :])
    axs[0, 0].set_title("all modes")

    # plot phi for mode 1+ (high positive power flow)
    axs[0, 1].contourf(dab_config.calc_config.mesh_P[:, n, :], dab_config.calc_config.mesh_V2[:, n, :], dab_config.calc_modulation.mask_m1p[:, n, :])
    axs[0, 1].set_title("Mode 1+")

    # plot phi for mode 2 (small negative and positive power flow)
    axs[1, 0].contourf(dab_config.calc_config.mesh_P[:, n, :], dab_config.calc_config.mesh_V2[:, n, :],
                       np.bitwise_or(dab_config.calc_modulation.mask_Im2[:, n, :], dab_config.calc_modulation.mask_IIm2[:, n, :]))
    axs[1, 0].set_title("Mode 2")

    # plot phi for mode 1- (high negative power flow)
    axs[1, 1].contourf(dab_config.calc_config.mesh_P[:, n, :], dab_config.calc_config.mesh_V2[:, n, :], dab_config.calc_modulation.mask_m1n[:, n, :])
    axs[1, 1].set_title("Mode 1-")

    # plot zvs coverage
    axs[0, 2].contourf(dab_config.calc_config.mesh_P[:, n, :], dab_config.calc_config.mesh_V2[:, n, :], dab_config.calc_modulation.mask_zvs[:, n, :])
    axs[0, 2].set_title("ZVS coverage")
    plt.show()

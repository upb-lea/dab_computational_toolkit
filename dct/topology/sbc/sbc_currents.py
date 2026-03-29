"""Current calculations, like RMS currents and currents for certain angles."""

# own libraries
from dct.topology.sbc.sbc_datasets_dtos import CircuitConfig

# 3rd party libraries
import numpy as np

def calc_rms_currents(config: CircuitConfig) -> tuple:
    """
    Calculate the RMS currents in l_s, l_1 and l_2 for the given input values.

    Works with 2- and 3-dimensional input arrays (2D e.g. v_2 and power, 3D e.g. v_1, v_2 and power)
    everything must be numpy!

    :param config: design configuration DTO
    :type  config: CircuitConfig
    :return: i_ripple current, i_rms current in A
    :rtype: tuple
    """
    # i_ripple=(V_in-V_out)*D/(L*fs)  = V_in*D(1-D)/(L*fs)

    # Calculate the duty cycle term  D(1-D)
    # vector_one = np.full((config.mesh_duty_cycle.size, 1), 1)
    duty_cycle_term = config.mesh_duty_cycle * (1 - config.mesh_duty_cycle)

    # Calculate the ripple current and rms current
    i_ripple = config.mesh_v1 * duty_cycle_term / (config.Ls * config.fs)

    # Calculate the current mean square
    i_ms = config.mesh_i2 ** 2 + (i_ripple / 12) ** 2
    # Calculate the current root mean square
    i_rms = np.sqrt(i_ms)

    return i_ripple, i_ms, i_rms

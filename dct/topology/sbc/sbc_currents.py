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
    i_ripple = config.mesh_v * duty_cycle_term / (config.Ls * config.fs)

    # Calculate the current mean square
    i_ms = config.mesh_i ** 2 + (i_ripple / 12) ** 2
    # Calculate the current root mean square
    i_rms = np.sqrt(i_ms)

    return i_ripple, i_ms, i_rms

def calc_volume_inductor_proxy(config: CircuitConfig, i_ripple: np.ndarray, i_rms: np.ndarray) -> np.ndarray:
    """
    Calculate volume inductor proxy based on current and inductance.

    :param config: design configuration DTO
    :type  config: CircuitConfig
    :param i_ripple: Ripple current in A
    :type  i_ripple: np.ndarray
    :param i_rms: Root means square current A
    :type  i_rms: np.ndarray
    :return: volume inductor proxy
    :rtype:  np.ndarray
    """
    # Calculate the volume inductor proxy with different currents
    # Parameter setting
    # Core volume constant
    # k_core = 1.3e-5
    kcore: float = 4.5e-3
    # Core volume constant
    # k_electric = 7.5e-9
    kel: float = 4.50e-6
    # Debug
    # k_electric= 0
    # B max (material dependent) 0.25 - 0.35 Tesla
    bsquare_max: float = 0.25 ** 2
    # Volume proxy = 1.3e-6 * L * i_ripple² / B³max + 7.5e-9 * I * Sqrt(L)
    # Calculate the ripple current and rms current
    volume_inductor_proxy: np.ndarray = kcore / bsquare_max * (i_ripple ** 2) * config.Ls + kel * (i_rms) * (config.Ls ** (1/2))
    # Debug
    # volume_inductor_proxy = (config.mesh_i / config.mesh_i) * config.Ls ** (2/3)

    return volume_inductor_proxy

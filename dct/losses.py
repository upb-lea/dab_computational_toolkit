"""Calculate the transistor losses."""

# 3rd party libraries
import numpy as np
from numpy.typing import NDArray

# own modules
import dct.datasets_dtos as dtos

def transistor_conduction_loss(transistor_ms_current: NDArray[np.float64], transistor_dto: dtos.TransistorDTO) -> np.ndarray:
    """
    Calculate the transistor conduction losses.

    :param transistor_ms_current: transistor mean square current in A²
    :type transistor_ms_current: np.ndarray
    :param transistor_dto: transistor DTO (Data transfer object)
    :type transistor_dto: dct.TransistorDTO
    :return: transistor conduction loss in W
    :rtype: np.ndarray[np.float64]
    """
    return transistor_dto.r_channel * transistor_ms_current


def transistor_switch_loss(mesh_v1: np.ndarray, i_rms: np.ndarray, transistor_dto: dtos.TransistorDTO, fs: float) -> np.ndarray:
    """
    Calculate the transistor conduction losses.

    :param mesh_v1: transistor drain-source voltage in V
    :type  mesh_v1:  np.ndarray[np.float64]
    :param i_rms: current mean root square in A
    :type  i_rms:  np.ndarray[np.float64]
    :param transistor_dto: transistor DTO (Data transfer object) containing selected transistors
    :type  transistor_dto:  dtos.TransistorDTO
    :param fs: Switching frequency
    :type  fs: float
    :return: transistor conduction loss in W
    :rtype:  np.ndarray
    """
    # Variable declaration later to read from transistor database
    # R-gate from schematic
    r_gate_dummy: float = 10.0
    # c_i_s_s from transistor database: CREE_C3M0065100J.json no high dependency on voltage
    # c_i_s_s_dummy: float =  7.7833e-10
    # Debug value for impact to Pareto front
    c_iss_dummy: float = 7.7833e-10
    # c_o_s_s from transistor database: CREE_C3M0065100J.json at 31.575V
    c_oss_dummy: float = 3.3366e-10

    # Switch time depends on gate capacity: t_sw = r_gate * c_i_s_s
    t_sw_dummy = r_gate_dummy * c_iss_dummy
    # p_overlap = fs * 1 / 2 * U_DS * I peak * t_switch
    p_overlap = fs * 0.5 * mesh_v1 * i_rms * t_sw_dummy
    # p_cross = 1 / 2 * U_DS * Qoss = U_DS²*C_oss
    p_cross = fs * 0.5 * (mesh_v1 ** 2) * c_oss_dummy

    # Calculate the sum of transistor switching losses
    return np.array(p_overlap + p_cross)

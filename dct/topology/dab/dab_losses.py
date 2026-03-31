"""Calculate the transistor losses."""

# 3rd party libraries
import numpy as np

# own modules
from dct.topology.dab import dab_datasets_dtos as dtos
from dct.topology.dab.dab_mod_zvs import _integrate_c_oss

def transistor_conduction_loss(transistor_rms_current: float, transistor_dto: dtos.TransistorDTO) -> np.ndarray:
    """
    Calculate the transistor conduction losses.

    :param transistor_rms_current: transistor RMS current in A
    :type transistor_rms_current: float
    :param transistor_dto: transistor DTO (Data transfer object)
    :type transistor_dto: dct.TransistorDTO
    :return: transistor conduction loss in W
    :rtype: float
    """
    return transistor_dto.r_channel * transistor_rms_current ** 2


def transistor_turn_off_loss(transistor_turn_off_current: float, transistor_dto: dtos.TransistorDTO, v_dc: np.ndarray,
                             f_s: float, c_par: float) -> float:
    """
    Calculate the transistor turn-off losses.

    :param transistor_turn_off_current: Current in A at turn-off
    :type transistor_turn_off_current: float
    :param transistor_dto: transistor DTO (Data transfer object)
    :type transistor_dto: dct.TransistorDTO
    :param v_dc: dc-link voltage in V
    :type v_dc: np.ndarray
    :param f_s: switching frequency in Hz
    :type f_s: float
    :param c_par: parallel PCB capacitance in F
    :type c_par: float
    :return: Switching losses in W
    :rtype: float
    """
    # Calculate the charge in the MOSFETs capacitance in parallel with the parasitic PCB capacitance at the given voltage
    c_oss_par = transistor_dto.c_oss + c_par
    q_oss_par = _integrate_c_oss(c_oss_par, v_dc)
    # Calculate the energy stored in the MOSFETs capacitance at the given
    energy_in_capacitance_at_voltage_1 = 0.5 * q_oss_par * v_dc

    # correct the given turn-off energy according to the switching voltage
    turn_off_energy_corrected_ved = (transistor_dto.turn_off_energy_vec - energy_in_capacitance_at_voltage_1) * v_dc / transistor_dto.turn_off_at_voltage

    # interpolate the switching energy
    turn_off_energy = np.interp(transistor_turn_off_current, transistor_dto.turn_off_current_vec, turn_off_energy_corrected_ved)

    # estimate the switching loss
    turn_off_power = turn_off_energy * f_s

    return float(turn_off_power)

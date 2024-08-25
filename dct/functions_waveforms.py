"""Waveform calculations."""

# 3rd party libraries
import numpy as np

def full_angle_waveform_from_angles(sorted_angles: np.array) -> np.array:
    """
    Generate the full 2pi-periodic angle waveform from the four sorted angles [alpha, beta, gamma, delta].

    :param sorted_angles: [alpha, beta, gamma, delta], but in sorted order
    :type sorted_angles: np.array
    :return: 2pi-periodic angles for a full waveform
    :rtype: np.array
    """
    sorted_angles = np.append(sorted_angles, np.pi + sorted_angles)
    sorted_angles = np.append(np.array([0]), sorted_angles)

    return sorted_angles

def full_current_waveform_from_currents(sorted_currents: np.array) -> np.array:
    """
    Generate the full 2pi-periodic current waveform from the four current values [i_alpha, i_beta, i_gamma, i_delta].

    :param sorted_currents: [i_alpha, i_beta, i_gamma, i_delta], but sorted
    :type sorted_currents: np.array
    :return: 2pi-periodic current waveform
    :rtype: np.array
    """
    sorted_currents = np.append(sorted_currents, -1 * sorted_currents)
    sorted_currents = np.append(sorted_currents[-1], sorted_currents)
    return sorted_currents

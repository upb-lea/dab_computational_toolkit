"""Waveform calculations."""

# 3rd party libraries
import numpy as np

def full_angle_waveform_from_angles(sorted_angles: np.ndarray) -> np.ndarray:
    """
    Generate the full 2pi-periodic angle waveform from the four sorted angles [alpha, beta, gamma, delta].

    :param sorted_angles: [alpha, beta, gamma, delta], but in sorted order
    :type sorted_angles: np.ndarray
    :return: 2pi-periodic angles for a full waveform
    :rtype: np.ndarray
    """
    sorted_angles = np.append(sorted_angles, np.pi + sorted_angles)
    sorted_angles = np.append(np.array([0]), sorted_angles)

    return sorted_angles

def full_current_waveform_from_currents(sorted_currents: np.ndarray) -> np.ndarray:
    """
    Generate the full 2pi-periodic current waveform from the four current values [i_alpha, i_beta, i_gamma, i_delta].

    :param sorted_currents: [i_alpha, i_beta, i_gamma, i_delta], but sorted
    :type sorted_currents: np.ndarray
    :return: 2pi-periodic current waveform
    :rtype: np.ndarray
    """
    sorted_currents = np.append(sorted_currents, -1 * sorted_currents)
    sorted_currents = np.append(sorted_currents[-1], sorted_currents)
    return sorted_currents

def full_waveforms_from_angles_currents(angles_rad_sorted: np.ndarray, *sorted_currents: np.ndarray) -> tuple[np.ndarray]:
    """
    Generate the full 2pi-periodic time and current waveform from the four time and current values at [alpha, beta, gamma, delta].

    Multiple current inputs possible. Sorts out same time values, e.g [0, 2, 3.14, 3,14] -> [0, 2, 3.14]

    :param angles_rad_sorted: [alpha, beta, gamma, delta], but in sorted order. Unit is radiant.
    :type angles_rad_sorted: np.ndarray
    :param sorted_currents: [i_alpha, i_beta, i_gamma, i_delta], but sorted
    :type sorted_currents: np.ndarray
    :return: 2pi-periodic time and current waveforms
    :rtype: np.ndarray
    """
    sorted_angles_full_waveform, unique_indices = np.unique(full_angle_waveform_from_angles(angles_rad_sorted), return_index=True)

    sorted_currents_full_waveform = sorted_angles_full_waveform,
    for sorted_current in sorted_currents:
        sorted_current_full_waveform = full_current_waveform_from_currents(sorted_current)[unique_indices]
        sorted_currents_full_waveform = sorted_currents_full_waveform + (sorted_current_full_waveform,)  # type: ignore

    return sorted_currents_full_waveform

"""Current calculations, like RMS currents and currents for certain angles."""

# own libraries
from dct.datasets import Config, CalcFromConfig, CalcModulation

# 3rd party libraries
import numpy as np

def calc_l_s_mode_2_currents(phi_rad, tau_1_rad, tau_2_rad, v_1, d, f_s, l_s):
    """
    Calculate currents in l_s for the given angles alpha_rad, beta_rad, gamma_rad and delta_rad.

    Note: Convention for mode 1+ and mode 2 according to the following source paper.
    Source Paper: Optimal ZVS Modulation of Single-Phase Single-Stage Bidirectional DAB AC–DC Converters
    Also note, that there are multiple mode definitions of the same author in different papers.

    :param phi_rad: phi in rad
    :param tau_1_rad: tau_1 in rad
    :param tau_2_rad: tau_2 in rad
    :param v_1: DC voltage v_1
    :param d: v_2_reflected / v_1 = (n_1 / n_2) * v_2 / v_1
    :param f_s: switching frequency
    :param l_s: series inductance
    :return: i_l_s_alpha, i_l_s_beta, i_l_s_gamma, i_l_s_delta
    """
    denominator = (2 * np.pi * f_s * l_s)
    i_l_s_alpha = v_1 * (d * tau_2_rad / 2 - tau_1_rad / 2) / denominator
    i_l_s_beta = v_1 * (d * tau_2_rad / 2 + tau_1_rad / 2 - tau_2_rad + phi_rad) / denominator
    i_l_s_gamma = v_1 * (-d * tau_2_rad / 2 + tau_1_rad / 2) / denominator
    i_l_s_delta = v_1 * (-d * tau_2_rad / 2 + tau_1_rad + phi_rad) / denominator

    return i_l_s_alpha, i_l_s_beta, i_l_s_gamma, i_l_s_delta

def calc_l_1_mode_2_currents(phi_rad, tau_1_rad, tau_2_rad, v_1, f_s, l_1):
    """
    Calculate currents in l_1 for the given angles alpha_rad, beta_rad, gamma_rad and delta_rad.

    Note: Convention for mode 1+ and mode 2 according to the following source paper.
    Source Paper: Optimal ZVS Modulation of Single-Phase Single-Stage Bidirectional DAB AC–DC Converters
    Also note, that there are multiple mode definitions of the same author in different papers.

    :param phi_rad: phi in rad
    :param tau_1_rad: tau_1 in rad
    :param tau_2_rad: tau_2 in rad
    :param v_1: DC voltage v_1
    :param f_s: switching frequency
    :param l_1: commutation inductance l_1
    :return: i_l_1_alpha, i_l_1_beta, i_l_1_gamma, i_l_1_delta
    """
    denominator = 2 * np.pi * f_s * l_1
    i_l_1_alpha = -v_1 * tau_1_rad / 2 / denominator
    i_l_1_beta = v_1 * (tau_1_rad / 2 - tau_2_rad + phi_rad) / denominator
    i_l_1_gamma = v_1 * tau_1_rad / 2 / denominator
    i_l_1_delta = v_1 * (tau_1_rad / 2 + phi_rad) / denominator

    return i_l_1_alpha, i_l_1_beta, i_l_1_gamma, i_l_1_delta

def calc_l_2_mode_2_currents(tau_2_rad, v_2, f_s, l_2):
    """
    Calculate currents in l_2 for the given angles alpha_rad, beta_rad, gamma_rad and delta_rad.

    This formular works for the following cases:
     - input: l_2, v_2, returns i_l_2
     - input: l_2_, v_2_, returns i_l_2_

    Note: Convention for mode 1+ and mode 2 according to the following source paper.
    Source Paper: Optimal ZVS Modulation of Single-Phase Single-Stage Bidirectional DAB AC–DC Converters
    Also note, that there are multiple mode definitions of the same author in different papers.

    :param tau_2_rad: tau_2 in rad
    :param v_2: DC voltage v_2
    :param f_s: switching frequency
    :param l_2: commutation inductance l_2
    :return: i_l_2_alpha_rad, i_l_2_beta, i_l_2_gamma, i_l_2_delta
    """
    denominator = 2 * np.pi * f_s * l_2
    i_l_s_alpha = -v_2 * tau_2_rad / 2 / denominator
    i_l_2_beta = -v_2 * tau_2_rad / 2 / denominator
    i_l_2_gamma = v_2 * tau_2_rad / 2 / denominator
    i_l_2_delta = v_2 * tau_2_rad / 2 / denominator

    return i_l_s_alpha, i_l_2_beta, i_l_2_gamma, i_l_2_delta

def calc_l_s_mode_1_plus_currents(phi_rad, tau_1_rad, tau_2_rad, d, v_1, f_s, l_s):
    """
    Calculate currents in l_s for the given angles alpha_rad, beta_rad, gamma_rad and delta_rad.

    Note: Convention for mode 1+ and mode 2 according to the following source paper.
    Source Paper: Optimal ZVS Modulation of Single-Phase Single-Stage Bidirectional DAB AC–DC Converters
    Also note, that there are multiple mode definitions of the same author in different papers.

    :param phi_rad: phi in rad
    :param tau_1_rad: tau_1 in rad
    :param tau_2_rad: tau_2 in rad
    :param v_1: DC voltage v_1
    :param d: v_2_reflected / v_1 = (n_1 / n_2) * v_2 / v_1
    :param f_s: switching frequency
    :param l_s: series inductance
    :return: i_l_s_alpha_rad, i_l_s_beta_rad, i_l_s_gamma_rad, i_l_s_delta_rad
    """
    denominator = 2 * np.pi * f_s * l_s
    i_l_s_alpha_rad = v_1 * (d * (-tau_1_rad + tau_2_rad / 2 - phi_rad + np.pi) - tau_1_rad / 2) / denominator
    i_l_s_beta_rad = v_1 * (d * tau_2_rad / 2 + tau_1_rad / 2 - tau_2_rad + phi_rad) / denominator
    i_l_s_gamma_rad = v_1 * (d * (-tau_2_rad / 2 + phi_rad) + tau_1_rad / 2) / denominator
    i_l_s_delta_rad = v_1 * (-d * tau_2_rad / 2 - tau_1_rad / 2 - phi_rad + np.pi) / denominator

    return i_l_s_alpha_rad, i_l_s_beta_rad, i_l_s_gamma_rad, i_l_s_delta_rad

def calc_l_1_mode_1_plus_currents(phi_rad, tau_1_rad, tau_2_rad, v_1, f_s, l_1):
    """
    Calculate currents in l_1 for the given angles alpha_rad, beta_rad, gamma_rad and delta_rad.

    Note: Convention for mode 1+ and mode 2 according to the following source paper.
    Source Paper: Optimal ZVS Modulation of Single-Phase Single-Stage Bidirectional DAB AC–DC Converters
    Also note, that there are multiple mode definitions of the same author in different papers.

    :param phi_rad: phi in rad
    :param tau_1_rad: tau_1 in rad
    :param tau_2_rad: tau_2 in rad
    :param v_1: DC voltage v_1
    :param f_s: switching frequency
    :param l_1: commutation inductance l_1
    :return: i_l_1_alpha, i_l_1_beta, i_l_1_gamma, i_l_1_delta
    """
    denominator = 2 * np.pi * f_s * l_1
    i_l_1_alpha = -v_1 * tau_1_rad / 2 / denominator
    i_l_1_beta = v_1 * (tau_1_rad / 2 - tau_2_rad + phi_rad) / denominator
    i_l_1_gamma = v_1 * tau_1_rad / 2 / denominator
    i_l_1_delta = v_1 * (-tau_1_rad / 2 - phi_rad + np.pi) / denominator

    return i_l_1_alpha, i_l_1_beta, i_l_1_gamma, i_l_1_delta

def calc_l_2_mode_1_plus_currents(phi_rad, tau_1_rad, tau_2_rad, v_2, f_s, l_2):
    """
    Calculate currents in l_2 for the given angles alpha_rad, beta_rad, gamma_rad and delta_rad.

    This formular works for the following cases:
     - input: l_2, v_2, returns i_l_2
     - input: l_2_, v_2_, returns i_l_2_

    Note: Convention for mode 1+ and mode 2 according to the following source paper.
    Source Paper: Optimal ZVS Modulation of Single-Phase Single-Stage Bidirectional DAB AC–DC Converters
    Also note, that there are multiple mode definitions of the same author in different papers.

    :param phi_rad: phi in rad
    :param tau_1_rad: tau_1 in rad
    :param tau_2_rad: tau_2 in rad
    :param v_2: DC voltage v_2
    :param f_s: switching frequency
    :param l_2: commutation inductance l_2
    :return: i_l_2_alpha, i_l_2_beta, i_l_2_gamma, i_l_2_delta
    """
    denominator = 2 * np.pi * f_s * l_2
    i_l_2_alpha = v_2 * (tau_1_rad - tau_2_rad / 2 + phi_rad - np.pi) / denominator
    i_l_2_beta = -v_2 * tau_2_rad / 2 / denominator
    i_l_2_gamma = v_2 * (tau_2_rad / 2 - phi_rad) / denominator
    i_l_2_delta = v_2 * tau_2_rad / 2 / denominator

    return i_l_2_alpha, i_l_2_beta, i_l_2_gamma, i_l_2_delta

def int_square_line_between_angles(angle_1_rad, angle_2_rad, y_1, y_2):
    """
    Calculate the integral of the square for a simple line, described with two points.

    Helper function to calculate a part of the RMS value (mean and square, no root).
    Everything must be numpy!
    :param angle_1_rad: angle 1 in rad
    :param angle_2_rad: angle 2 in rad
    :param y_1: y1
    :param y_2: y2
    :return:
    """
    inan = ~np.isnan(angle_1_rad)
    int_square = np.full_like(angle_1_rad, np.nan)
    int_square[inan] = (y_1[inan] ** 2 + y_1[inan] * y_2[inan] + y_2[inan] ** 2) * (angle_2_rad[inan] - angle_1_rad[inan]) / 3 / 2 / np.pi

    return int_square

def calc_rms(alpha_rad, beta_rad, gamma_rad, delta_rad, i_alpha, i_beta, i_gamma, i_delta):
    """
    Calculate a single RMS current for some single points, with a linear current between them.

    Angles alpha to delta according to "closed-form solution for efficient ZVS modulation of DAB converters".
    Function works with multidimensional arrays.

    Interval is symmetric by pi. Starts from zero, ends with pi. Some angles will be higher than pi, needs to be
    mirrored to values smaller than pi.

    Everything must be numpy!

    :param alpha_rad: angle alpha in rad
    :param beta_rad: angle beta in rad
    :param gamma_rad: angle gamma in rad
    :param delta_rad: angle delta in rad
    :param i_alpha: current at angle alpha
    :param i_beta: current at angle beta
    :param i_gamma: current at angle gamma
    :param i_delta: current at angle delta
    :return: rms current
    """
    if not (isinstance(alpha_rad, np.ndarray) | isinstance(beta_rad, np.ndarray) | isinstance(gamma_rad, np.ndarray) | isinstance(delta_rad, np.ndarray)):
        raise TypeError("Input must be numpy!")
    if not (isinstance(i_alpha, np.ndarray) | isinstance(i_beta, np.ndarray) | isinstance(i_gamma, np.ndarray) | isinstance(i_delta, np.ndarray)):
        raise TypeError("Input must be numpy!")

    # as delta can be smaller than gamma, sort the angles.
    angles_vec = np.array([alpha_rad, beta_rad, gamma_rad, delta_rad])
    currents_vec = np.array([i_alpha, i_beta, i_gamma, i_delta])

    angles_zeros = np.zeros_like(alpha_rad)

    # if angle > pi, subtract pi and invert current. Current first, as angle will be always < pi after modification
    currents_vec[angles_vec > np.pi] = -currents_vec[angles_vec > np.pi]
    angles_vec[angles_vec > np.pi] = angles_vec[angles_vec > np.pi] - np.pi

    # if angle < 0, add pi and invert current. Current first, as angle will be always > 0 after modification
    currents_vec[angles_vec < 0] = -currents_vec[angles_vec < 0]
    angles_vec[angles_vec < 0] = angles_vec[angles_vec < 0] + np.pi

    # sort the angles and currents according to the angle order.
    # https://stackoverflow.com/questions/52121635/looking-up-index-of-value-in-numpy-3d-arrays
    sorted_indices = np.argsort(angles_vec, axis=0)
    angles_sorted = np.take_along_axis(angles_vec, sorted_indices, axis=0)
    currents_sorted = np.take_along_axis(currents_vec, sorted_indices, axis=0)

    int_square_part_00 = int_square_line_between_angles(angles_zeros, angles_sorted[0], -currents_sorted[3], currents_sorted[0])
    int_square_part_01 = int_square_line_between_angles(angles_sorted[0], angles_sorted[1], currents_sorted[0], currents_sorted[1])
    int_square_part_12 = int_square_line_between_angles(angles_sorted[1], angles_sorted[2], currents_sorted[1], currents_sorted[2])
    int_square_part_23 = int_square_line_between_angles(angles_sorted[2], angles_sorted[3], currents_sorted[2], currents_sorted[3])

    rms = np.sqrt(2 * (int_square_part_00 + int_square_part_01 + int_square_part_12 + int_square_part_23))

    return rms, angles_sorted, currents_sorted

def calc_rms_currents(config: Config, calc_from_config: CalcFromConfig, calc_modulation: CalcModulation):
    """
    Calculate the RMS currents in l_s, l_1 and l_2 for the given input values.

    Works with 2- and 3-dimensional input arrays (2D e.g. v_2 and power, 3D e.g. v_1, v_2 and power)
    everything must be numpy!

    :param config: design configuration DTO
    :return: i_l_s_rms, i_l_1_rms, i_l_2_rms
    """
    alpha_rad = calc_modulation.phi - calc_modulation.tau1
    beta_rad = np.pi + calc_modulation.phi - calc_modulation.tau2
    gamma_rad = np.full_like(alpha_rad, np.pi)
    delta_rad = np.pi + calc_modulation.phi

    # define the full mask for mode 2, made of interval 1 and interval 2
    mode_2_mask = np.bitwise_or(calc_modulation.mask_Im2, calc_modulation.mask_IIm2)

    # calculate current values for l_s depend on angles. Modulation modes are taken into account
    d = config.n * calc_from_config.mesh_V2 / calc_from_config.mesh_V1

    # currents in l_s for mode 2 and mode 1+
    m2_i_l_s_alpha, m2_i_l_s_beta, m2_i_l_s_gamma, m2_i_l_s_delta = calc_l_s_mode_2_currents(
        calc_modulation.phi, calc_modulation.tau1, calc_modulation.tau2, calc_from_config.mesh_V1, d, config.fs, config.Ls)
    m1_i_l_s_alpha, m1_i_l_s_beta, m1_i_l_s_gamma, m1_i_l_s_delta = calc_l_s_mode_1_plus_currents(
        calc_modulation.phi, calc_modulation.tau1, calc_modulation.tau2, d, calc_from_config.mesh_V1, config.fs, config.Ls)

    # currents in l_1 for mode 2 and mode 1+
    m2_i_l_1_alpha, m2_i_l_1_beta, m2_i_l_1_gamma, m2_i_l_1_delta = calc_l_1_mode_2_currents(
        calc_modulation.phi, calc_modulation.tau1, calc_modulation.tau2, calc_from_config.mesh_V1, config.fs, config.Lc1)
    m1_i_l_1_alpha, m1_i_l_1_beta, m1_i_l_1_gamma, m1_i_l_1_delta = calc_l_1_mode_1_plus_currents(
        calc_modulation.phi, calc_modulation.tau1, calc_modulation.tau2, calc_from_config.mesh_V1, config.fs, config.Lc1)

    # currents in l_2 for mode 2 and mode 1+
    m2_i_l_2_alpha, m2_i_l_2_beta, m2_i_l_2_gamma, m2_i_l_2_delta = calc_l_2_mode_2_currents(
        calc_modulation.tau2, calc_from_config.mesh_V2, config.fs, config.Lc2)
    m1_i_l_2_alpha, m1_i_l_2_beta, m1_i_l_2_gamma, m1_i_l_2_delta = calc_l_2_mode_1_plus_currents(
        calc_modulation.phi, calc_modulation.tau1, calc_modulation.tau2, calc_from_config.mesh_V2, config.fs, config.Lc2)

    # generate the output current for l_s, distinguish between mode 2 and mode 1+
    i_l_s_alpha = np.full_like(m1_i_l_s_alpha, np.nan)
    i_l_s_alpha[calc_modulation.mask_IIIm1] = m1_i_l_s_alpha[calc_modulation.mask_IIIm1]
    i_l_s_alpha[mode_2_mask] = m2_i_l_s_alpha[mode_2_mask]

    i_l_s_beta = np.full_like(m1_i_l_s_beta, np.nan)
    i_l_s_beta[calc_modulation.mask_IIIm1] = m1_i_l_s_beta[calc_modulation.mask_IIIm1]
    i_l_s_beta[mode_2_mask] = m2_i_l_s_beta[mode_2_mask]

    i_l_s_gamma = np.full_like(m1_i_l_s_gamma, np.nan)
    i_l_s_gamma[calc_modulation.mask_IIIm1] = m1_i_l_s_gamma[calc_modulation.mask_IIIm1]
    i_l_s_gamma[mode_2_mask] = m2_i_l_s_gamma[mode_2_mask]

    i_l_s_delta = np.full_like(m1_i_l_s_delta, np.nan)
    i_l_s_delta[calc_modulation.mask_IIIm1] = m1_i_l_s_delta[calc_modulation.mask_IIIm1]
    i_l_s_delta[mode_2_mask] = m2_i_l_s_delta[mode_2_mask]

    # generate the output current for l_1, distinguish between mode 2 and mode 1+
    i_l_1_alpha = np.full_like(m1_i_l_1_alpha, np.nan)
    i_l_1_alpha[calc_modulation.mask_IIIm1] = m1_i_l_1_alpha[calc_modulation.mask_IIIm1]
    i_l_1_alpha[mode_2_mask] = m2_i_l_1_alpha[mode_2_mask]

    i_l_1_beta = np.full_like(m1_i_l_1_beta, np.nan)
    i_l_1_beta[calc_modulation.mask_IIIm1] = m1_i_l_1_beta[calc_modulation.mask_IIIm1]
    i_l_1_beta[mode_2_mask] = m2_i_l_1_beta[mode_2_mask]

    i_l_1_gamma = np.full_like(m1_i_l_1_gamma, np.nan)
    i_l_1_gamma[calc_modulation.mask_IIIm1] = m1_i_l_1_gamma[calc_modulation.mask_IIIm1]
    i_l_1_gamma[mode_2_mask] = m2_i_l_1_gamma[mode_2_mask]

    i_l_1_delta = np.full_like(m1_i_l_1_delta, np.nan)
    i_l_1_delta[calc_modulation.mask_IIIm1] = m1_i_l_1_delta[calc_modulation.mask_IIIm1]
    i_l_1_delta[mode_2_mask] = m2_i_l_1_delta[mode_2_mask]

    # generate the output current for l_2, distinguish between mode 2 and mode 1+
    i_l_2_alpha = np.full_like(m1_i_l_2_alpha, np.nan)
    i_l_2_alpha[calc_modulation.mask_IIIm1] = m1_i_l_2_alpha[calc_modulation.mask_IIIm1]
    i_l_2_alpha[mode_2_mask] = m2_i_l_2_alpha[mode_2_mask]

    i_l_2_beta = np.full_like(m1_i_l_2_beta, np.nan)
    i_l_2_beta[calc_modulation.mask_IIIm1] = m1_i_l_2_beta[calc_modulation.mask_IIIm1]
    i_l_2_beta[mode_2_mask] = m2_i_l_2_beta[mode_2_mask]

    i_l_2_gamma = np.full_like(m1_i_l_2_gamma, np.nan)
    i_l_2_gamma[calc_modulation.mask_IIIm1] = m1_i_l_2_gamma[calc_modulation.mask_IIIm1]
    i_l_2_gamma[mode_2_mask] = m2_i_l_2_gamma[mode_2_mask]

    i_l_2_delta = np.full_like(m1_i_l_2_delta, np.nan)
    i_l_2_delta[calc_modulation.mask_IIIm1] = m1_i_l_2_delta[calc_modulation.mask_IIIm1]
    i_l_2_delta[mode_2_mask] = m2_i_l_2_delta[mode_2_mask]

    # calculate rms currents for l_s, l_1, l_2
    i_l_s_rms, angles_sorted, i_l_s_sorted = calc_rms(alpha_rad, beta_rad, gamma_rad, delta_rad, i_l_s_alpha, i_l_s_beta, i_l_s_gamma, i_l_s_delta)
    i_l_1_rms, _, i_l_1_sorted = calc_rms(alpha_rad, beta_rad, gamma_rad, delta_rad, i_l_1_alpha, i_l_1_beta, i_l_1_gamma, i_l_1_delta)
    i_l_2_rms, _, i_l_2_sorted = calc_rms(alpha_rad, beta_rad, gamma_rad, delta_rad, i_l_2_alpha, i_l_2_beta, i_l_2_gamma, i_l_2_delta)

    # print(f"{np.shape(angles_sorted)=}")
    # print(f"{np.shape(i_l_s_sorted)=}")
    # print(f"{angles_sorted=}")
    # print(f"{i_l_s_sorted=}")

    return i_l_s_rms, i_l_1_rms, i_l_2_rms, angles_sorted, i_l_s_sorted, i_l_1_sorted, i_l_2_sorted

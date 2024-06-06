"""Pytests for the current calculation."""
# python libraries

# own libraries
import dct

# 3rd party libraries
from pytest import approx
import numpy as np

def test_rms_1d():
    """
    Test RMS current calculation with a 1D-array.

    All results have been verified using GeckoCIRCUITS.
    """
    alpha_rad = np.array([1.5])
    beta_rad = np.array([2.0])
    gamma_rad = np.array([3.0])
    delta_rad = np.array([3.14])

    i_alpha = np.array([0.0])
    i_beta = np.array([3.0])
    i_gamma = np.array([1.0])
    i_delta = np.array([0.0])

    i_rms = dct.calc_rms(alpha_rad, beta_rad, gamma_rad, delta_rad, i_alpha, i_beta, i_gamma, i_delta)

    assert i_rms == approx(1.364, rel=5e-3)

def test_rms_2d():
    """
    Test RMS current calculation with a 2D-array.

    All results have been verified using GeckoCIRCUITS.
    """
    # first: same as 1d
    # second: same as 1d, but mixed in angles to check the sort algorithm
    # third: random current
    # fourth: random current

    alpha_rad = np.array([[1.5, 2.0], [1.5, 1.5]])
    beta_rad = np.array([[2.0, 1.5], [2.0, 2.0]])
    gamma_rad = np.array([[3.0, 3.14], [3.0, 3.0]])
    delta_rad = np.array([[3.14, 3.0], [3.14, 3.14]])

    i_alpha = np.array([[0.0, 3.0], [2.0, 2.0]])
    i_beta = np.array([[3.0, 0.0], [3.0, -3.0]])
    i_gamma = np.array([[1.0, 0.0], [1.0, -1.0]])
    i_delta = np.array([[0.0, 1.0], [-2.0, -2.0]])

    i_rms = dct.calc_rms(alpha_rad, beta_rad, gamma_rad, delta_rad, i_alpha, i_beta, i_gamma, i_delta)

    assert i_rms == approx(np.array([[1.364, 1.364], [2.08, 1.94]]), rel=5e-3)

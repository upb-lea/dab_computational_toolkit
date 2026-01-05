"""Calculate the transistor losses."""

# 3rd party libraries
import numpy as np
from scipy.interpolate import RegularGridInterpolator as RGI
from numpy.typing import NDArray

# own modules
import dct.topology.sbc.sbc_datasets_dtos as dtos

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


def transistor_switch_loss(mesh_v1: np.ndarray, i_rms: np.ndarray, tr_data_dto: dtos.TransistorDTO, fs: float) -> np.ndarray:
    """
    Calculate the transistor conduction losses.

    :param mesh_v1: transistor drain-source voltage in V
    :type  mesh_v1:  np.ndarray[np.float64]
    :param i_rms: current mean root square in A
    :type  i_rms:  np.ndarray[np.float64]
    :param tr_data_dto: transistor DTO (Data transfer object) containing selected transistors
    :type  tr_data_dto:  dtos.TransistorDTO
    :param fs: Switching frequency
    :type  fs: float
    :return: transistor conduction loss in W
    :rtype:  np.ndarray
    """
    # Transform the mesh to mesh-points
    mesh_points = np.vstack((mesh_v1, i_rms)).T

    # Initialize interpolation object for e-on
    e_on_losses_obj = RGI((tr_data_dto.switch_e_on_data.voltage_parameter, tr_data_dto.switch_e_on_data.current_data),
                          tr_data_dto.switch_e_on_data.loss_data)

    # Calculate the loss results for switch on
    e_on_losses = e_on_losses_obj(mesh_points)

    # Initialize interpolation object for e-off
    e_off_losses_obj = RGI((tr_data_dto.switch_e_off_data.voltage_parameter, tr_data_dto.switch_e_off_data.current_data),
                           tr_data_dto.switch_e_off_data.loss_data)

    # Calculate the loss results for switch on
    e_off_losses = e_off_losses_obj(mesh_points)

    # Add both energies and calculate the power loss
    p_switch_loss = (e_on_losses + e_off_losses) * fs

    # Calculate the sum of transistor switching losses
    return p_switch_loss

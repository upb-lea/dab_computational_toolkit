"""Calculate the transistor losses."""

# 3rd party libraries
import numpy as np

# own modules
from dct.topology.dab import dab_datasets_dtos as dtos

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

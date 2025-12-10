"""Component requirements from the circuit class."""

# python libraries
import dataclasses

# 3rd party libraries
import numpy as np

@dataclasses.dataclass(init=False)
class CapacitorRequirements:
    """
    Requirements for a single capacitor design from the circuit results.

    The current waveform needs to be given:
     - angle_vec (angle based 0...360°) and time_vec (time based) are close.
     - the current_vec gives the amplitude of the current at given time/angle
    """

    # exact a single current waveform to optimize the capacitor
    time_vec: np.ndarray
    current_vec: np.ndarray

    # all current waveforms for calculation the capacitor loss for a single (optimized) capacitor
    time_array: np.ndarray
    current_array: np.ndarray

    # maximum dc link voltage
    v_dc_max: float

    study_name: str

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


@dataclasses.dataclass(init=False)
class ComponentRequirements:
    """DTO contains the requirements for all the additional DAB components."""

    capacitor_requirements: list[CapacitorRequirements]
    # inductor_requirements: list[InductorRequirements]
    # transformer_requirements: list[TransformerRequirements]

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

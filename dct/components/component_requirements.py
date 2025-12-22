"""Component requirements from the circuit class."""

# python libraries
import dataclasses

# 3rd party libraries
import numpy as np

@dataclasses.dataclass(init=False)
class CapacitorRequirements:
    """
    Requirements for a single capacitor design from the circuit results.

    The current waveform needs to be given for a full period:
     - time_vec (time based)
     - the current_vec gives the amplitude of the current at given time
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
class InductorRequirements:
    """
    Requirements for a single inductor design from the circuit results.

    The current waveform needs to be given for a full period:
     - time_vec (time based)
     - the current_vec gives the amplitude of the current at given time
    """

    target_inductance: float

    # exact a single current waveform to optimize the inductor
    time_vec: np.ndarray
    current_vec: np.ndarray

    # all current waveforms for calculation the inductor loss for a single (optimized) inductor
    time_array: np.ndarray
    current_array: np.ndarray

    study_name: str

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


@dataclasses.dataclass(init=False)
class TransformerRequirements:
    """
    Requirements for a single transformer design from the circuit results.

    The current waveform needs to be given for a full period:
     - time_vec (time based)
     - the current_vec gives the amplitude of the current at given time
    """

    l_s12_target: float
    l_h_target: float
    n_target: float
    temperature: float

    # exact a single current waveform to optimize the transformer
    time_vec: np.ndarray
    current_1_vec: np.ndarray
    current_2_vec: np.ndarray

    # all current waveforms for calculation the transformer loss for a single (optimized) transformer
    time_array: np.ndarray
    current_1_array: np.ndarray
    current_2_array: np.ndarray

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
    inductor_requirements: list[InductorRequirements]
    transformer_requirements: list[TransformerRequirements]

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

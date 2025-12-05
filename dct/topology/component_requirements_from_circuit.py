"""Component requirements from the circuit class."""

# python libraries
import dataclasses

# 3rd party libraries
import numpy as np

@dataclasses.dataclass(init=False)
class CapacitorRequirements:
    """Requirements for a single capacitor design from the circuit results."""

    sorted_max_rms_angles: np.ndarray
    i_max_rms_current_waveform: np.ndarray
    time: np.ndarray
    v_max: float

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

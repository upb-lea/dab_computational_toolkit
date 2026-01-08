"""DTOs for the datasets."""

# python libraries
import dataclasses

# 3rd party libraries
import numpy as np

# own libraries
from dct.components.component_requirements import ComponentRequirements

@dataclasses.dataclass
class LossDataGrid:
    """Contains data to compute the loss of a specific operation point.

    voltage_parameter: Source-drain voltage
    current_data: Current through the transistor (x-axis value)
    loss_data: Loss value (y-axis value)
    """

    voltage_parameter: np.ndarray
    current_data: np.ndarray
    loss_data: np.ndarray


@dataclasses.dataclass
class TransistorDTO:
    """Contains constant transistor information."""

    name: str
    t_j_max_op: np.ndarray
    c_oss: np.ndarray
    q_oss: np.ndarray
    switch_e_on_data: LossDataGrid
    switch_e_off_data: LossDataGrid
    housing_area: np.float64
    cooling_area: np.float64
    r_th_jc: np.ndarray
    r_channel: np.ndarray

@dataclasses.dataclass
class FixedParameters:
    """Fixed parameters for the circuit optimization."""

    transistor_1_dto_list: list[TransistorDTO]
    transistor_2_dto_list: list[TransistorDTO]
    mesh_v1: np.ndarray
    mesh_duty_cycle: np.ndarray
    mesh_i: np.ndarray
    mesh_weights: np.ndarray

@dataclasses.dataclass
class Sampling:
    """Sampling."""

    sampling_method: str
    sampling_points: int
    v1_additional_user_point_list: list[float]
    duty_cycle_additional_user_point_list: list[float]
    i_additional_user_point_list: list[float]
    additional_user_weighting_point_list: list[float]

@dataclasses.dataclass(init=False)
class CircuitConfig:
    """Input configuration DTO for the SBC converter."""

    mesh_v1: np.ndarray
    mesh_duty_cycle: np.ndarray
    mesh_i: np.ndarray
    sampling: Sampling
    Ls: np.float64
    fs: np.float64
    transistor_dto_1: TransistorDTO
    transistor_dto_2: TransistorDTO

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

@dataclasses.dataclass(init=False)
class CalcFromCircuitConfig:
    """DTO calculates parameters for the next simulations, which can be derived from the input values."""

    Lc2_: np.ndarray
    t_j_1: np.float64
    t_j_2: np.float64
    c_oss_par_1: np.ndarray
    c_oss_par_2: np.ndarray
    c_oss_1: np.ndarray
    c_oss_2: np.ndarray
    q_oss_1: np.ndarray
    q_oss_2: np.ndarray

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

@dataclasses.dataclass()
class CalcCurrents:
    """DTO contains calculated RMS currents."""

    # RMS value
    i_rms: np.ndarray
    i_ms: np.ndarray

    # ripple current
    i_ripple: np.ndarray

@dataclasses.dataclass(init=False)
class CalcLosses:
    """DTO contains the calculated losses."""

    p_hs_conduction: np.ndarray
    p_ls_conduction: np.ndarray
    p_hs_switch: np.ndarray
    p_ls_switch: np.ndarray
    p_sbc_total: np.ndarray

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

@dataclasses.dataclass(init=False)
class InductorResults:
    """DTO contains the inductor losses."""

    p_combined_losses: np.ndarray
    volume: float
    area_to_heat_sink: float
    circuit_trial_file: str
    inductor_trial_number: int

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

@dataclasses.dataclass
class SbcCircuitDTO:
    """Main SbcDTO containing all input parameters, calculations and simulation results."""

    timestamp: np.ndarray | None
    circuit_id: str
    metadata: np.ndarray | None
    input_config: CircuitConfig
    calc_config: CalcFromCircuitConfig | None
    calc_volume_inductor_proxy: np.ndarray
    calc_currents: CalcCurrents
    calc_losses: CalcLosses
    component_requirements: ComponentRequirements | None
    inductor_results: InductorResults | None

@dataclasses.dataclass
class StudyData:
    """Data class containing all general information to perform a study."""

    study_name: str
    optimization_directory: str

@dataclasses.dataclass
class FilterData:
    """Information about the filtered circuit designs."""

    filtered_list_files: list[str]
    filtered_list_pathname: str
    circuit_study_name: str

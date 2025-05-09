"""DTOs for the datasets."""

# python libraries
import dataclasses

# 3rd party libraries
import numpy as np

@dataclasses.dataclass
class TransistorDTO:
    """Contains constant transistor information."""

    name: str
    t_j_max_op: np.ndarray
    c_oss: np.ndarray
    q_oss: np.ndarray
    housing_area: np.float64
    cooling_area: np.float64
    r_th_jc: np.ndarray
    r_channel: np.ndarray

@dataclasses.dataclass
class FixedParameters:
    """Fixed parameters for the circuit optimization."""

    transistor_1_dto_list: list[TransistorDTO]
    transistor_2_dto_list: list[TransistorDTO]

@dataclasses.dataclass(init=False)
class CircuitConfig:
    """Input configuration DTO for the DAB converter."""

    V1_nom: np.float64
    V1_min: np.float64
    V1_max: np.float64
    V1_step: np.float64
    V2_nom: np.float64
    V2_min: np.float64
    V2_max: np.float64
    V2_step: np.float64
    P_min: np.float64
    P_max: np.float64
    P_nom: np.float64
    P_step: np.float64
    n: np.float64
    Ls: np.float64
    Lc1: np.float64
    Lc2: np.float64
    fs: np.float64
    transistor_dto_1: TransistorDTO
    transistor_dto_2: TransistorDTO
    c_par_1: np.float64
    c_par_2: np.float64

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


@dataclasses.dataclass(init=False)
class GeckoAdditionalParameters:
    """Additional parameters for the GeckoCIRCUITS simulation, like simulation time or some file paths."""

    t_dead1: np.ndarray
    t_dead2: np.ndarray
    timestep: np.float64
    number_sim_periods: int
    timestep_pre: np.float64
    number_pre_sim_periods: int
    simfilepath: str
    lossfilepath: str

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


@dataclasses.dataclass(init=False)
class CalcFromCircuitConfig:
    """DTO calculates parameters for the next simulations, which can be derived from the input values."""

    mesh_V1: np.ndarray
    mesh_V2: np.ndarray
    mesh_P: np.ndarray
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

@dataclasses.dataclass(init=False)
class CalcModulation:
    """DTO contains calculated modulation parameters."""

    phi: np.ndarray
    tau1: np.ndarray
    tau2: np.ndarray
    mask_zvs: np.ndarray
    mask_Im2: np.ndarray
    mask_IIm2: np.ndarray
    mask_IIIm1: np.ndarray
    mask_zvs_coverage: np.ndarray
    mask_zvs_coverage_notnan: np.ndarray
    mask_m1n: np.ndarray
    mask_m1p: np.ndarray

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


@dataclasses.dataclass(init=False)
class CalcCurrents:
    """DTO contains calculated RMS currents."""

    # RMS values
    i_l_s_rms: np.ndarray
    i_l_1_rms: np.ndarray
    i_l_2_rms: np.ndarray
    i_hf_1_rms: np.ndarray
    i_hf_2_rms: np.ndarray
    i_m1_rms: np.ndarray
    i_m2_rms: np.ndarray

    # sorted values: angles (alpha, beta, gamma, delta) and currents.
    angles_rad_sorted: np.ndarray
    angles_rad_unsorted: np.ndarray
    i_l_s_sorted: np.ndarray
    i_l_1_sorted: np.ndarray
    i_l_2_sorted: np.ndarray
    i_hf_1_sorted: np.ndarray
    i_hf_2_sorted: np.ndarray

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

@dataclasses.dataclass(init=False)
class CalcLosses:
    """DTO contains the calculated losses."""

    p_m1_conduction: np.ndarray
    p_m2_conduction: np.ndarray
    p_dab_conduction: np.ndarray

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
    circuit_trial_number: int
    inductor_trial_number: int

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

@dataclasses.dataclass(init=False)
class StackedTransformerResults:
    """DTO contains the stacked transformer losses."""

    p_combined_losses: np.ndarray
    volume: float
    area_to_heat_sink: float
    circuit_trial_number: int
    stacked_transformer_trial_number: int

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


@dataclasses.dataclass(init=False)
class GeckoResults:
    """DTO contains the result of the GeckoCIRCUITS simulation."""

    p_dc1: np.ndarray
    p_dc2: np.ndarray
    S11_p_sw: np.ndarray
    S11_p_cond: np.ndarray
    S12_p_sw: np.ndarray
    S12_p_cond: np.ndarray
    S23_p_sw: np.ndarray
    S23_p_cond: np.ndarray
    S24_p_sw: np.ndarray
    S24_p_cond: np.ndarray
    v_dc1: np.ndarray
    i_dc1: np.ndarray
    v_dc2: np.ndarray
    i_dc2: np.ndarray
    p_sw1: np.ndarray
    p_cond1: np.ndarray
    p_sw2: np.ndarray
    p_cond2: np.ndarray
    i_HF1: np.ndarray
    i_HF2: np.ndarray
    i_Ls: np.ndarray
    i_Lc1: np.ndarray
    i_Lc2: np.ndarray
    i_C11: np.ndarray
    i_C12: np.ndarray
    i_C23: np.ndarray
    i_C24: np.ndarray
    i_S11: np.ndarray
    i_S12: np.ndarray
    i_S23: np.ndarray
    i_S24: np.ndarray
    v_ds_S11_sw_on: np.ndarray
    v_ds_S23_sw_on: np.ndarray
    i_HF1_S11_sw_on: np.ndarray
    i_HF2_S23_sw_on: np.ndarray
    power_deviation: np.ndarray
    zvs_coverage: np.ndarray
    zvs_coverage1: np.ndarray
    zvs_coverage2: np.ndarray
    zvs_coverage_notnan: np.ndarray
    zvs_coverage1_notnan: np.ndarray
    zvs_coverage2_notnan: np.ndarray
    i_HF1_total_mean: np.ndarray
    I1_squared_total_mean: np.ndarray

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


@dataclasses.dataclass(init=False)
class GeckoWaveforms:
    """DTO contains the waveform result of the GeckoCIRCUITS simulation."""

    time: np.ndarray
    i_Ls: np.ndarray
    i_Lc1: np.ndarray
    i_Lc2: np.ndarray
    i_HF1: np.ndarray
    i_HF2: np.ndarray

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

@dataclasses.dataclass
class CircuitDabDTO:
    """Main DabDTO containing all input parameters, calculations and simulation results."""

    timestamp: np.ndarray | None
    name: str
    metadata: np.ndarray | None
    input_config: CircuitConfig
    calc_config: CalcFromCircuitConfig
    calc_modulation: CalcModulation
    calc_currents: CalcCurrents
    calc_losses: CalcLosses | None
    gecko_additional_params: GeckoAdditionalParameters
    gecko_results: GeckoResults | None
    gecko_waveforms: GeckoWaveforms | None
    inductor_results: InductorResults | None
    stacked_transformer_results: StackedTransformerResults | None

@dataclasses.dataclass
class TransformerTargetParameters:
    """Target transformer parameters for the optimization."""

    l_s12_target: float
    l_h_target: float
    n_target: float

    # operating point: current waveforms and temperature
    time_current_1_vec: np.ndarray
    time_current_2_vec: np.ndarray
    temperature: float

@dataclasses.dataclass
class StudyData:
    """Data class containing all general information to perform a study."""

    study_name: str
    optimization_directory: str

@dataclasses.dataclass
class FilterData:
    """Information about the filtered circuit designs."""

    filtered_list_id: list[int]
    filtered_list_pathname: str
    circuit_study_name: str

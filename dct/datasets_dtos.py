"""DTOs for the datasets."""

# python libraries
import dataclasses

# 3rd party libraries
import numpy as np

@dataclasses.dataclass
class TransistorDTO:
    """Contains constant transistor information."""

    name: str
    t_j: np.array
    c_oss: np.array
    q_oss: np.array

@dataclasses.dataclass
class FixedParameters:
    """Fixed parameters for the circuit optimization."""

    transistor_1_dto_list: list[TransistorDTO]
    transistor_2_dto_list: list[TransistorDTO]

@dataclasses.dataclass(init=False)
class CircuitConfig:
    """Input configuration DTO for the DAB converter."""

    V1_nom: np.array
    V1_min: np.array
    V1_max: np.array
    V1_step: np.array
    V2_nom: np.array
    V2_min: np.array
    V2_max: np.array
    V2_step: np.array
    P_min: np.array
    P_max: np.array
    P_nom: np.array
    P_step: np.array
    n: np.array
    Ls: np.array
    Lc1: np.array
    Lc2: np.array
    fs: np.array
    transistor_dto_1: TransistorDTO
    transistor_dto_2: TransistorDTO
    c_par_1: np.array
    c_par_2: np.array

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


@dataclasses.dataclass(init=False)
class GeckoAdditionalParameters:
    """Additional parameters for the GeckoCIRCUITs simulation, like simulation time or some file paths."""

    t_dead1: np.array
    t_dead2: np.array
    timestep: np.array
    number_sim_periods: np.array
    timestep_pre: np.array
    number_pre_sim_periods: np.array
    simfilepath: np.array
    lossfilepath: np.array

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


@dataclasses.dataclass(init=False)
class CalcFromCircuitConfig:
    """DTO calculates parameters for the next simulations, which can be derived from the input values."""

    mesh_V1: np.array
    mesh_V2: np.array
    mesh_P: np.array
    Lc2_: np.array
    t_j_1: np.array
    t_j_2: np.array
    c_oss_par_1: np.array
    c_oss_par_2: np.array
    c_oss_1: np.array
    c_oss_2: np.array
    q_oss_1: np.array
    q_oss_2: np.array

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

@dataclasses.dataclass(init=False)
class CalcModulation:
    """DTO contains calculated modulation parameters."""

    phi: np.array
    tau1: np.array
    tau2: np.array
    mask_zvs: np.array
    mask_Im2: np.array
    mask_IIm2: np.array
    mask_IIIm1: np.array
    mask_zvs_coverage: np.array
    mask_zvs_coverage_notnan: np.array

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


@dataclasses.dataclass(init=False)
class CalcCurrents:
    """DTO contains calculated RMS currents."""

    # RMS values
    i_l_s_rms: np.array
    i_l_1_rms: np.array
    i_l_2_rms: np.array
    i_hf_1_rms: np.array
    i_hf_2_rms: np.array

    # sorted values: angles (alpha, beta, gamma, delta) and currents.
    angles_rad_sorted: np.array
    angles_rad_unsorted: np.array
    i_l_s_sorted: np.array
    i_l_1_sorted: np.array
    i_l_2_sorted: np.array

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

@dataclasses.dataclass(init=False)
class CalcLosses:
    """DTO contains te calculated losses."""

    p_1_tbd: np.array

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

@dataclasses.dataclass(init=False)
class InductorResults:
    """DTO contains the inductor losses."""

    p_combined_losses: np.array

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

@dataclasses.dataclass(init=False)
class StackedTransformerResults:
    """DTO contains the stacked transformer losses."""

    p_combined_losses: np.array

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


@dataclasses.dataclass(init=False)
class GeckoResults:
    """DTO contains the result of the GeckoCIRCUITS simulation."""

    p_dc1: np.array
    p_dc2: np.array
    S11_p_sw: np.array
    S11_p_cond: np.array
    S12_p_sw: np.array
    S12_p_cond: np.array
    S23_p_sw: np.array
    S23_p_cond: np.array
    S24_p_sw: np.array
    S24_p_cond: np.array
    v_dc1: np.array
    i_dc1: np.array
    v_dc2: np.array
    i_dc2: np.array
    p_sw1: np.array
    p_cond1: np.array
    p_sw2: np.array
    p_cond2: np.array
    i_HF1: np.array
    i_HF2: np.array
    i_Ls: np.array
    i_Lc1: np.array
    i_Lc2: np.array
    i_C11: np.array
    i_C12: np.array
    i_C23: np.array
    i_C24: np.array
    v_ds_S11_sw_on: np.array
    v_ds_S23_sw_on: np.array
    i_HF1_S11_sw_on: np.array
    i_HF2_S23_sw_on: np.array
    power_deviation: np.array
    zvs_coverage: np.array
    zvs_coverage1: np.array
    zvs_coverage2: np.array
    zvs_coverage_notnan: np.array
    zvs_coverage1_notnan: np.array
    zvs_coverage2_notnan: np.array
    i_HF1_total_mean: np.array
    I1_squared_total_mean: np.array

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


@dataclasses.dataclass(init=False)
class GeckoWaveforms:
    """DTO contains the waveform result of the GeckoCIRCUITS simulation."""

    time: np.array
    i_Ls: np.array
    i_Lc1: np.array
    i_Lc2: np.array

    def __init__(self, **kwargs):
        names = set([f.name for f in dataclasses.fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)

@dataclasses.dataclass
class CircuitDabDTO:
    """Main DabDTO containing all input parameters, calculations and simulation results."""

    timestamp: np.array
    name: str
    metadata: np.array
    input_config: CircuitConfig
    calc_config: CalcFromCircuitConfig
    calc_modulation: CalcModulation
    calc_currents: CalcCurrents
    calc_losses: CalcLosses | None
    gecko_additional_params: GeckoAdditionalParameters
    gecko_results: GeckoResults | None
    gecko_waveforms: GeckoWaveforms | None
    inductor_results: InductorResults | None
    # stacked_transformer_losses: StackedTransformerLosses | None

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

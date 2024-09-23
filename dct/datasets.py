"""Describe the dataclasses how to store the input parameters and results."""

# python libraries
import os
import dataclasses
import datetime
import logging

# 3rd party libraries
import numpy as np
import transistordatabase as tdb
from matplotlib import pyplot as plt

# own libraries
import dct


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
    transistor_name_1: np.array
    transistor_name_2: np.array
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
class InductorLosses:
    """DTO contains the inductor losses."""

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
    inductor_losses: InductorLosses | None

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

class HandleDabDto:
    """Class to handle the DabDTO, e.g. save and load the files."""

    @staticmethod
    def init_config(name: str, V1_nom: np.array, V1_min: np.array, V1_max: np.array, V1_step: np.array, V2_nom: np.array, V2_min: np.array,
                    V2_max: np.array, V2_step: np.array, P_min: np.array, P_max: np.array, P_nom: np.array, P_step: np.array,
                    n: np.array, Ls: np.array, Lc1: np.array, Lc2: np.array, fs: np.array,
                    transistor_name_1: str, transistor_name_2: str, c_par_1, c_par_2):
        """
        Initialize the DAB structure.

        :param name: name of the simulation
        :type name: str
        :param V1_nom: V1 nominal voltage
        :type V1_nom: np.array
        :param V1_min: V1 minimum voltage
        :type V1_min: np.array
        :param V1_max: V1 maximum voltage
        :type V1_max: np.array
        :param V1_step: V1 voltage steps
        :type V1_step: np.array
        :param V2_nom: V2 nominal voltage
        :type V2_nom: np.array
        :param V2_min: V2 minimum voltage
        :type V2_min: np.array
        :param V2_max: V2 maximum voltage
        :type V2_max: np.array
        :param V2_step: V2 voltage steps
        :type V2_step: np.array
        :param P_min: P minimum power
        :type P_min: np.array
        :param P_max: P maximum power
        :type P_max: np.array
        :param P_nom: P nominal power
        :type P_nom: np.array
        :param P_step: P power steps
        :type P_step:
        :param n: transformer transfer ratio
        :type n: np.array
        :param Ls: series inductance
        :type Ls: np.array
        :param Lc1: Commutation inductance Lc1
        :type Lc1: np.array
        :param Lc2: Commutation inductance Lc2
        :type Lc2: np.array
        :param fs: Switching frequency
        :type fs: np.array
        :param transistor_name_1: Transistor name for transistor bridge 1. Must match with transistordatbase available transistors.
        :type transistor_name_1: str
        :param transistor_name_2: Transistor name for transistor bridge 2. Must match with transistordatbase available transistors.
        :type transistor_name_2: str
        :param c_par_1: Parasitic PCB capacitance per transistor footprint of bridge 1
        :type c_par_1: np.array
        :param c_par_2: Parasitic PCB capacitance per transistor footprint of bridge 2
        :type c_par_2: np.array
        :return:
        """
        input_configuration = CircuitConfig(V1_nom=np.array(V1_nom),
                                            V1_min=np.array(V1_min),
                                            V1_max=np.array(V1_max),
                                            V1_step=np.array(V1_step),
                                            V2_nom=np.array(V2_nom),
                                            V2_min=np.array(V2_min),
                                            V2_max=np.array(V2_max),
                                            V2_step=np.array(V2_step),
                                            P_min=np.array(P_min),
                                            P_max=np.array(P_max),
                                            P_nom=np.array(P_nom),
                                            P_step=np.array(P_step),
                                            n=np.array(n),
                                            Ls=np.array(Ls),
                                            Lc1=np.array(Lc1),
                                            Lc2=np.array(Lc2),
                                            fs=np.array(fs),
                                            transistor_name_1=np.asarray(transistor_name_1),
                                            transistor_name_2=np.asarray(transistor_name_2),
                                            c_par_1=c_par_1,
                                            c_par_2=c_par_2,
                                            )
        calc_config = HandleDabDto.calculate_from_configuration(config=input_configuration)
        modulation_parameters = HandleDabDto.calculate_modulation(input_configuration, calc_config)

        i_l_s_rms, i_l_1_rms, i_l_2_rms, angles_rad_sorted, i_l_s_sorted, i_l_1_sorted, i_l_2_sorted, angles_rad_unsorted = dct.calc_rms_currents(
            input_configuration, calc_config, modulation_parameters)

        i_hf_1_rms, i_hf_2_rms = dct.calc_hf_rms_currents(angles_rad_sorted, i_l_s_sorted, i_l_1_sorted, i_l_2_sorted, input_configuration.n)

        calc_currents = CalcCurrents(**{'i_l_s_rms': i_l_s_rms, 'i_l_1_rms': i_l_1_rms, 'i_l_2_rms': i_l_2_rms, 'angles_rad_sorted': angles_rad_sorted,
                                        'angles_rad_unsorted': angles_rad_unsorted,
                                        'i_l_s_sorted': i_l_s_sorted, 'i_l_1_sorted': i_l_1_sorted, 'i_l_2_sorted': i_l_2_sorted,
                                        'i_hf_1_rms': i_hf_1_rms, 'i_hf_2_rms': i_hf_2_rms},)

        gecko_additional_params = GeckoAdditionalParameters(
            t_dead1=50e-9, t_dead2=50e-9, timestep=1e-9,
            number_sim_periods=2, timestep_pre=25e-9, number_pre_sim_periods=0,
            simfilepath=os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..', 'circuits', 'DAB_MOSFET_Modulation_v8.ipes')),
            lossfilepath=os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..', 'circuits')))

        dab_dto = CircuitDabDTO(
            name=name,
            timestamp=None,
            metadata=None,
            input_config=input_configuration,
            calc_config=calc_config,
            calc_modulation=modulation_parameters,
            calc_currents=calc_currents,
            calc_losses=None,
            gecko_additional_params=gecko_additional_params,
            gecko_results=None,
            gecko_waveforms=None,
            inductor_losses=None)
        return dab_dto

    @staticmethod
    def add_gecko_simulation_results(dab_dto: CircuitDabDTO, get_waveforms: bool = False) -> CircuitDabDTO:
        """
        Add GeckoCIRCUITS simulation results to the given DTO.

        :param dab_dto: DabDTO
        :param get_waveforms: Read back GeckoCIRCUITS simulation waveforms (high memory consumption). Default to False.
        :type get_waveforms: bool
        :return: DabDTO
        """
        gecko_results, gecko_waveforms = dct.start_gecko_simulation(
            mesh_V1=dab_dto.calc_config.mesh_V1, mesh_V2=dab_dto.calc_config.mesh_V2,
            mesh_P=dab_dto.calc_config.mesh_P, mod_phi=dab_dto.calc_modulation.phi,
            mod_tau1=dab_dto.calc_modulation.tau1, mod_tau2=dab_dto.calc_modulation.tau2,
            t_dead1=dab_dto.gecko_additional_params.t_dead1, t_dead2=dab_dto.gecko_additional_params.t_dead2,
            fs=dab_dto.input_config.fs, Ls=dab_dto.input_config.Ls, Lc1=dab_dto.input_config.Lc1,
            Lc2=dab_dto.input_config.Lc2, n=dab_dto.input_config.n,
            t_j_1=dab_dto.calc_config.t_j_1, t_j_2=dab_dto.calc_config.t_j_2,
            simfilepath=dab_dto.gecko_additional_params.simfilepath,
            lossfilepath=dab_dto.gecko_additional_params.lossfilepath,
            timestep=dab_dto.gecko_additional_params.timestep,
            number_sim_periods=dab_dto.gecko_additional_params.number_sim_periods,
            timestep_pre=dab_dto.gecko_additional_params.timestep_pre,
            number_pre_sim_periods=dab_dto.gecko_additional_params.number_pre_sim_periods, geckoport=43036,
            c_par_1=dab_dto.input_config.c_par_1, c_par_2=dab_dto.input_config.c_par_2,
            transistor_1_name=dab_dto.input_config.transistor_name_1,
            transistor_2_name=dab_dto.input_config.transistor_name_2, get_waveforms=get_waveforms,
            i_ls_start=dab_dto.calc_currents.i_l_s_sorted[0],
            i_lc1_start=dab_dto.calc_currents.i_l_1_sorted[0],
            i_lc2_start=dab_dto.calc_currents.i_l_2_sorted[0])

        # add GeckoCIRCUITS simulation results to the result DTO.
        dab_dto.gecko_results = GeckoResults(**gecko_results)

        dab_dto.gecko_waveforms = GeckoWaveforms(**gecko_waveforms)
        return dab_dto

    @staticmethod
    def calculate_from_configuration(config: CircuitConfig) -> CalcFromCircuitConfig:
        """
        Calculate logical parameters which can be calculated from the input parameters.

        :param config: DAB configuration
        :type config: CircuitConfig
        :return: CalcFromConfig
        :rtype: CalcFromCircuitConfig
        """
        mesh_V1, mesh_V2, mesh_P = np.meshgrid(
            np.linspace(config.V1_min, config.V1_max, int(config.V1_step)),
            np.linspace(config.V2_min, config.V2_max, int(config.V2_step)), np.linspace(config.P_min, config.P_max, int(config.P_step)),
            sparse=False)

        Lc2_ = config.Lc2 * config.n ** 2

        db = tdb.DatabaseManager()
        db.set_operation_mode_json()

        transistor_1 = db.load_transistor(config.transistor_name_1)
        transistor_2 = db.load_transistor(config.transistor_name_2)

        if transistor_1.type != "MOSFET" and transistor_1.type != "SiC-MOSFET":
            raise ValueError(f"Transistor 1: {transistor_1.name} is of non-allowed type {transistor_1.type}. "
                             f"Allowed types are MOSFET, SiC-MOSFET.")
        if transistor_2.type != "MOSFET" and transistor_2.type != "SiC-MOSFET":
            raise ValueError(f"Transistor 2: {transistor_2.name} is of non-allowed type {transistor_2.type}. "
                             f"Allowed types are MOSFET, SiC-MOSFET.")

        t_j_1 = transistor_1.switch.t_j_max - 25
        t_j_2 = transistor_2.switch.t_j_max - 25

        c_oss_1, q_oss_1 = HandleDabDto.get_c_oss_from_tdb(transistor_1, margin_factor=1.2)
        c_oss_2, q_oss_2 = HandleDabDto.get_c_oss_from_tdb(transistor_2, margin_factor=1.2)

        # export c_oss files for GeckoCIRCUITS
        path_to_save_c_oss_files = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'circuits')
        if not os.path.exists(os.path.join(path_to_save_c_oss_files, f"{transistor_1.name}_c_oss.nlc")):
            transistor_1.export_geckocircuits_coss(filepath=path_to_save_c_oss_files, margin_factor=1.2)
        if not os.path.exists(os.path.join(path_to_save_c_oss_files, f"{transistor_2.name}_c_oss.nlc")):
            transistor_2.export_geckocircuits_coss(filepath=path_to_save_c_oss_files, margin_factor=1.2)

        calc_from_config = CalcFromCircuitConfig(
            mesh_V1=mesh_V1,
            mesh_V2=mesh_V2,
            mesh_P=mesh_P,
            Lc2_=Lc2_,
            t_j_1=t_j_1,
            t_j_2=t_j_2,
            c_oss_par_1=c_oss_1 + config.c_par_1,
            c_oss_par_2=c_oss_2 + config.c_par_2,
            c_oss_1=c_oss_1,
            c_oss_2=c_oss_2,
            q_oss_1=q_oss_1,
            q_oss_2=q_oss_2
        )

        return calc_from_config

    @staticmethod
    def calculate_modulation(config: CircuitConfig, calc_config: CalcFromCircuitConfig) -> CalcModulation:
        """
        Calculate the modulation parameters like phi, tau1, tau, ...

        :param config: DAB input configuration
        :param calc_config: calculated parameters from the input configuration
        :return: Modulation parameters.
        """
        result_dict = dct.calc_modulation(config.n, config.Ls, config.Lc1, config.Lc2, config.fs, Coss1=calc_config.c_oss_par_1, Coss2=calc_config.c_oss_par_2,
                                          V1=calc_config.mesh_V1, V2=calc_config.mesh_V2, P=calc_config.mesh_P)

        return CalcModulation(**result_dict)

    @staticmethod
    def get_c_oss_from_tdb(transistor: tdb.Transistor, margin_factor: float = 1.2) -> tuple:
        """
        Import the transistor Coss(Vds) capacitance from the transistor database (TDB).

        Note we assume V_ds in Volt and C_oss in F. If this is not the case, scale your data accordingly!

        :param transistor: transistor database object
        :type transistor: transistordatabase.Transistor
        :param margin_factor: factor for margin. [1.0 = no margin], [1.2 = 20 % margin: DEFAULT]
        :type margin_factor: float
        :return: c_oss, q_oss (in 1 V steps)
        :rtype: tuple
        """
        csv_data = transistor.c_oss[0].graph_v_c.T

        # Maybe check if data is monotonically
        # Check if voltage is monotonically rising
        if not np.all(csv_data[1:, 0] >= csv_data[:-1, 0], axis=0):
            logging.warning("The voltage in csv file is not monotonically rising!")
        # Check if Coss is monotonically falling
        if not np.all(csv_data[1:, 1] <= csv_data[:-1, 1], axis=0):
            logging.warning("The C_oss in csv file is not monotonically falling!")

        # Rescale and interpolate the csv data to have a nice 1V step size from 0V to v_max
        # A first value with zero volt will be added
        v_max = int(np.round(csv_data[-1, 0]))
        v_interp = np.arange(v_max + 1)

        # The margin is considered here as a factor of the original capacitance value
        coss_interp = margin_factor * np.interp(v_interp, csv_data[:, 0], csv_data[:, 1])
        # Since we now have an evenly spaced vector where x corespond to the element-number of the vector
        # we don't have to store x (v_interp) with it.
        # To get Coss(V) just get the array element coss_interp[V]

        # return coss_interp
        c_oss = coss_interp
        q_oss = HandleDabDto._integrate_c_oss(coss_interp)
        return c_oss, q_oss

    @staticmethod
    def _integrate_c_oss(coss):
        """
        Integrate Coss for each voltage from 0 to V_max.

        :param coss: MOSFET Coss(Vds) curve from Vds=0V to >= V1_max. Just one row with Coss data and index = Vds.
        :return: Qoss(Vds) as one row of data and index = Vds.
        """

        # Integrate from 0 to v
        def integrate(v):
            v_interp = np.arange(v + 1)
            coss_v = np.interp(v_interp, np.arange(coss.shape[0]), coss)
            return np.trapz(coss_v)

        coss_int = np.vectorize(integrate)
        # get the qoss vector that has the resolution 1V from 0 to V_max
        v_vec = np.arange(coss.shape[0])
        # get the qoss vector that fits the mesh_V scale
        # v_vec = np.linspace(V_min, V_max, int(V_step))
        qoss = coss_int(v_vec)

        return qoss

    @staticmethod
    def save(dab_dto: CircuitDabDTO, name: str, comment: str, directory: str, timestamp: bool = True):
        """
        Save the DabDTO-class to a npz file.

        :param dab_dto: Class to store
        :type dab_dto: CircuitDabDTO
        :param name: Filename
        :type name: str
        :param comment: Comment
        :type comment: str
        :param directory: Directory to store the results
        :type directory: str
        :param timestamp: [True] to add a timestamp to the file name.
        :type timestamp: bool
        """
        # Add some descriptive data to the file
        # Adding a timestamp, it may be useful
        dab_dto.timestamp = np.asarray(datetime.datetime.now().isoformat())
        # Adding a comment to the file, hopefully a descriptive one
        if comment:
            dab_dto.comment = np.asarray(comment)

        # Adding a timestamp to the filename if requested
        if timestamp:
            if name:
                filename = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "_" + name
            else:
                filename = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        else:
            if name:
                filename = name
            else:
                # set some default non-empty filename
                filename = "dab_dataset"

        if directory:
            directory = os.path.expanduser(directory)
            directory = os.path.expandvars(directory)
            directory = os.path.abspath(directory)
            if os.path.isdir(directory):
                file = os.path.join(directory, filename)
            else:
                logging.warning("Directory does not exist!", stacklevel=2)
                file = os.path.join(filename)
        else:
            file = os.path.join(filename)

        # prepare a all-in-one parallel file
        input_dict_to_store = dataclasses.asdict(dab_dto.input_config)
        calc_config_dict_to_store = dataclasses.asdict(dab_dto.calc_config)
        calc_modulation_dict_to_store = dataclasses.asdict(dab_dto.calc_modulation)
        calc_currents_dict_to_store = dataclasses.asdict(dab_dto.calc_currents)
        calc_losses_dict_to_store = dataclasses.asdict(dab_dto.calc_losses) if isinstance(dab_dto.calc_losses, CalcLosses) else None
        gecko_additional_params_dict_to_store = dataclasses.asdict(dab_dto.gecko_additional_params)
        gecko_results_dict_to_store = dataclasses.asdict(dab_dto.gecko_results) if isinstance(dab_dto.gecko_results, GeckoResults) else None
        inductor_losses_dict_to_store = dataclasses.asdict(dab_dto.inductor_losses) if isinstance(dab_dto.inductor_losses, InductorLosses) else None

        dict_to_store = {}
        dict_to_store["timestamp"] = dab_dto.timestamp
        dict_to_store["name"] = dab_dto.name
        dict_to_store["metadata"] = dab_dto.metadata
        dict_to_store.update(input_dict_to_store)
        dict_to_store.update(calc_config_dict_to_store)
        dict_to_store.update(calc_modulation_dict_to_store)
        dict_to_store.update(calc_currents_dict_to_store)
        if isinstance(dab_dto.calc_losses, CalcLosses):
            dict_to_store.update(calc_losses_dict_to_store)
        dict_to_store.update(gecko_additional_params_dict_to_store)
        if isinstance(dab_dto.gecko_results, GeckoResults):
            dict_to_store.update(gecko_results_dict_to_store)
        if isinstance(dab_dto.inductor_losses, InductorLosses):
            dict_to_store.update(inductor_losses_dict_to_store)

        np.savez_compressed(**dict_to_store, file=file)

    @staticmethod
    def load_from_file(file: str) -> CircuitDabDTO:
        """
        Load everything from the given .npz file.

        :param file: a .nps filename or file-like object, string, or pathlib.Path
        :return: two objects with type DAB_Specification and DAB_Results
        """
        # Check for filename extension
        file_name, file_extension = os.path.splitext(file)
        if not file_extension:
            file += '.npz'
        file = os.path.expanduser(file)
        file = os.path.expandvars(file)
        file = os.path.abspath(file)
        # Open the file and parse the data
        # with np.load(file) as data:

        decoded_data = np.load(file, allow_pickle=True)
        keys_of_gecko_result_dto = [field.name for field in dataclasses.fields(GeckoResults)]
        keys_of_gecko_waveform_dto = [field.name for field in dataclasses.fields(GeckoWaveforms)]
        keys_of_inductor_losses_dto = [field.name for field in dataclasses.fields(InductorLosses)]

        # if loaded results have all keys that are mandatory for the GeckoResults-Class:
        if len(set(keys_of_gecko_result_dto) & set(list(decoded_data.keys()))) == len(keys_of_gecko_result_dto):
            gecko_results = GeckoResults(**decoded_data)
        else:
            gecko_results = None

        # if loaded results have all keys that are mandatory for the GeckoWaveform-Class:
        if len(set(keys_of_gecko_waveform_dto) & set(list(decoded_data.keys()))) == len(keys_of_gecko_waveform_dto):
            gecko_waveforms = GeckoWaveforms(**decoded_data)
        else:
            gecko_waveforms = None

        # if loaded results have all keys that are mandatory for the GeckoResults-Class:
        if len(set(keys_of_inductor_losses_dto) & set(list(decoded_data.keys()))) == len(keys_of_inductor_losses_dto):
            inductor_losses = InductorLosses(**decoded_data)
        else:
            inductor_losses = None

        dab_dto = CircuitDabDTO(name=str(decoded_data["name"]),
                                timestamp=decoded_data["timestamp"],
                                metadata=decoded_data["metadata"],
                                input_config=CircuitConfig(**decoded_data),
                                calc_config=CalcFromCircuitConfig(**decoded_data),
                                calc_modulation=CalcModulation(**decoded_data),
                                calc_currents=CalcCurrents(**decoded_data),
                                calc_losses=None,
                                gecko_additional_params=GeckoAdditionalParameters(**decoded_data),
                                gecko_results=gecko_results,
                                gecko_waveforms=gecko_waveforms,
                                inductor_losses=inductor_losses)

        return dab_dto

    @staticmethod
    def get_max_peak_waveform_transformer(dab_dto: CircuitDabDTO, plot: bool = False) -> tuple[np.array, np.array, np.array]:
        """
        Get the transformer waveform with the maximum current peak out of the three-dimensional simulation array (v_1, v_2, P).

        :param dab_dto: DAB data transfer object (DTO)
        :type dab_dto: CircuitDabDTO
        :param plot: True to plot the results, mostly for understanding and debugging
        :type plot: bool
        :return: sorted_max_angles, i_l_s_max_current_waveform, i_hf_2_max_current_waveform. All as a numpy array.
        :rtype: List[np.array]
        """
        i_hf_2_sorted = np.transpose(dab_dto.calc_currents.i_l_s_sorted * dab_dto.input_config.n - dab_dto.calc_currents.i_l_2_sorted, (1, 2, 3, 0))
        angles_rad_sorted = np.transpose(dab_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))

        max_index = (0, 0, 0)
        for vec_vvp in np.ndindex(dab_dto.calc_modulation.phi.shape):
            max_index = vec_vvp if np.max(i_hf_2_sorted[vec_vvp]) > np.max(i_hf_2_sorted[max_index]) else max_index
            if plot:
                plt.plot(dct.functions_waveforms.full_angle_waveform_from_angles(angles_rad_sorted[vec_vvp]),
                         dct.functions_waveforms.full_current_waveform_from_currents(i_hf_2_sorted[vec_vvp]), color='grey')

        i_hf_2_max_current_waveform = i_hf_2_sorted[max_index]
        i_l_s_max_current_waveform = np.transpose(dab_dto.calc_currents.i_l_s_sorted, (1, 2, 3, 0))[max_index]

        sorted_max_angles = dct.functions_waveforms.full_angle_waveform_from_angles(angles_rad_sorted[max_index])
        i_l_s_max_current_waveform = dct.functions_waveforms.full_current_waveform_from_currents(i_l_s_max_current_waveform)
        i_hf_2_max_current_waveform = dct.functions_waveforms.full_current_waveform_from_currents(i_hf_2_max_current_waveform)

        if plot:
            plt.plot(sorted_max_angles, i_hf_2_max_current_waveform, color='red', label='peak current full waveform')
            plt.grid()
            plt.xlabel('Angle in rad')
            plt.ylabel('Current in A')
            plt.legend()
            plt.show()

        return sorted_max_angles, i_l_s_max_current_waveform, i_hf_2_max_current_waveform

    @staticmethod
    def get_max_peak_waveform_inductor(dab_dto: CircuitDabDTO, plot: bool = False) -> tuple[np.array, np.array]:
        """
        Get the inductor waveform with the maximum current peak out of the three-dimensional simulation array (v_1, v_2, P).

        :param dab_dto: DAB data transfer object (DTO)
        :type dab_dto: CircuitDabDTO
        :param plot: True to plot the results, mostly for understanding and debugging
        :type plot: bool
        :return: sorted_max_angles, i_l_s_max_current_waveform, i_l1_max_current_waveform. All as a numpy array.
        :rtype: List[np.array]
        """
        i_l1_sorted = np.transpose(dab_dto.calc_currents.i_l_1_sorted, (1, 2, 3, 0))
        angles_rad_sorted = np.transpose(dab_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))

        max_index = (0, 0, 0)
        for vec_vvp in np.ndindex(dab_dto.calc_modulation.phi.shape):
            max_index = vec_vvp if np.max(i_l1_sorted[vec_vvp]) > np.max(i_l1_sorted[max_index]) else max_index
            if plot:
                plt.plot(dct.functions_waveforms.full_angle_waveform_from_angles(angles_rad_sorted[vec_vvp]),
                         dct.functions_waveforms.full_current_waveform_from_currents(i_l1_sorted[vec_vvp]), color='grey')

        i_l1_max_current_waveform = i_l1_sorted[max_index]

        sorted_max_angles = dct.functions_waveforms.full_angle_waveform_from_angles(angles_rad_sorted[max_index])
        i_l1_max_current_waveform = dct.functions_waveforms.full_current_waveform_from_currents(i_l1_max_current_waveform)

        if plot:
            plt.plot(sorted_max_angles, i_l1_max_current_waveform, color='red', label='peak current full waveform')
            plt.grid()
            plt.xlabel('Angle in rad')
            plt.ylabel('Current in A')
            plt.legend()
            plt.show()

        return sorted_max_angles, i_l1_max_current_waveform

    @staticmethod
    def export_transformer_target_parameters_dto(dab_dto: CircuitDabDTO) -> TransformerTargetParameters:
        """
        Export the optimization parameters for the transformer optimization (inside FEMMT).

        Note: the current counting system is adapted to FEMMT! The secondary current is counted negative!

        :param dab_dto: DAB DTO
        :type dab_dto: CircuitDabDTO
        :return: DTO for the transformer optimization using FEMMT
        :rtype: TransformerTargetParameters
        """
        # calculate the full 2pi waveform from the four given DAB currents
        sorted_max_angles, i_l_s_max_current_waveform, i_hf_2_max_current_waveform = dct.HandleDabDto.get_max_peak_waveform_transformer(dab_dto, plot=False)
        # transfer 2pi periodic time into a real time
        time = sorted_max_angles / 2 / np.pi / dab_dto.input_config.fs

        return TransformerTargetParameters(
            l_s12_target=dab_dto.input_config.Ls,
            l_h_target=dab_dto.input_config.Lc2,
            n_target=dab_dto.input_config.n,
            time_current_1_vec=np.array([time, i_l_s_max_current_waveform]),
            time_current_2_vec=np.array([time, -i_hf_2_max_current_waveform]),
            temperature=100)

    @staticmethod
    def add_inductor_results(dab_dto: CircuitDabDTO, inductor_losses: dict) -> CircuitDabDTO:
        """Add inductor results to the CirucitDabDTO.

        :param dab_dto: Dual-active bridge DTO
        :type dab_dto: CircuitDabDTO
        :param inductor_losses: inductor losses dictionary
        :type inductor_losses: dict
        :return: Dual-active bridge DTO including the inductor losses
        :rtype: CircuitDabDTO
        """
        dab_dto.gecko_waveforms = InductorLosses(**inductor_losses)
        return dab_dto

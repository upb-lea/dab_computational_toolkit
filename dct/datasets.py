"""Describe the dataclasses how to store the input parameters and results."""

# python libraries
import os
import datetime
import logging
import pickle

# 3rd party libraries
import numpy as np
import transistordatabase as tdb
from matplotlib import pyplot as plt

# own libraries
import dct.datasets_dtos as d_dtos
import dct.functions_waveforms as d_waveforms
import dct.mod_zvs as mod
import dct.currents as dct_currents
import dct.geckosimulation as dct_gecko
import dct.losses as dct_loss
import dct.sampling as sampling
from dct.circuit_optimization_dtos import CircuitSampling

logger = logging.getLogger(__name__)

class HandleDabDto:
    """Class to handle the DabDTO, e.g. save and load the files."""

    @staticmethod
    def init_config(name: str, v1_min: float, v1_max: float, v2_min: float, v2_max: float, p_min: float, p_max: float,
                    sampling: CircuitSampling, n: float, ls: float, lc1: float, lc2: float, fs: float,
                    transistor_dto_1: d_dtos.TransistorDTO, transistor_dto_2: d_dtos.TransistorDTO, c_par_1: float, c_par_2: float) -> d_dtos.CircuitDabDTO:
        """
        Initialize the DAB structure.

        :param name: name of the simulation
        :type name: str
        :param v1_min: V1 minimum voltage
        :type v1_min: float
        :param v1_max: V1 maximum voltage
        :type v1_max: float
        :param v2_min: V2 minimum voltage
        :type v2_min: float
        :param v2_max: V2 maximum voltage
        :type v2_max: float
        :param p_min: P minimum power
        :type p_min: float
        :param p_max: P maximum power
        :type p_max: float
        :param sampling: Sampling parameters
        :type sampling: d_dtos.Sampling
        :param n: transformer transfer ratio
        :type n: float
        :param ls: series inductance
        :type ls: float
        :param lc1: Commutation inductance Lc1
        :type lc1: float
        :param lc2: Commutation inductance Lc2
        :type lc2: float
        :param fs: Switching frequency
        :type fs: float
        :param transistor_dto_1: Transistor DTO for transistor bridge 1. Must match with transistordatabase available transistors.
        :type transistor_dto_1: TransistorDTO
        :param transistor_dto_2: Transistor DTO for transistor bridge 2. Must match with transistordatabase available transistors.
        :type transistor_dto_2: TransistorDTO
        :param c_par_1: Parasitic PCB capacitance per transistor footprint of bridge 1
        :type c_par_1: float
        :param c_par_2: Parasitic PCB capacitance per transistor footprint of bridge 2
        :type c_par_2: float
        :return:
        """
        input_configuration = d_dtos.CircuitConfig(V1_min=np.array(v1_min),
                                                   V1_max=np.array(v1_max),
                                                   V2_min=np.array(v2_min),
                                                   V2_max=np.array(v2_max),
                                                   P_min=np.array(p_min),
                                                   P_max=np.array(p_max),
                                                   sampling=sampling,
                                                   n=np.array(n),
                                                   Ls=np.array(ls),
                                                   Lc1=np.array(lc1),
                                                   Lc2=np.array(lc2),
                                                   fs=np.array(fs),
                                                   transistor_dto_1=transistor_dto_1,
                                                   transistor_dto_2=transistor_dto_2,
                                                   c_par_1=c_par_1,
                                                   c_par_2=c_par_2,
                                                   )
        calc_config = HandleDabDto.calculate_from_configuration(config=input_configuration)
        modulation_parameters = HandleDabDto.calculate_modulation(input_configuration, calc_config)

        i_l_s_rms, i_l_1_rms, i_l_2_rms, angles_rad_sorted, i_l_s_sorted, i_l_1_sorted, i_l_2_sorted, angles_rad_unsorted = dct_currents.calc_rms_currents(
            input_configuration, calc_config, modulation_parameters)

        i_hf_1_rms, i_hf_2_rms, i_hf_1_sorted, i_hf_2_sorted = dct_currents.calc_hf_currents(
            angles_rad_sorted, i_l_s_sorted, i_l_1_sorted, i_l_2_sorted, input_configuration.n)

        i_m1_rms = dct_currents.calc_transistor_rms_currents(i_hf_1_rms)
        i_m2_rms = dct_currents.calc_transistor_rms_currents(i_hf_2_rms)

        calc_currents = d_dtos.CalcCurrents(**{'i_l_s_rms': i_l_s_rms, 'i_l_1_rms': i_l_1_rms, 'i_l_2_rms': i_l_2_rms, 'angles_rad_sorted': angles_rad_sorted,
                                               'angles_rad_unsorted': angles_rad_unsorted, 'i_l_s_sorted': i_l_s_sorted, 'i_l_1_sorted': i_l_1_sorted,
                                               'i_l_2_sorted': i_l_2_sorted, 'i_hf_1_rms': i_hf_1_rms, 'i_hf_2_rms': i_hf_2_rms,
                                               'i_m1_rms': i_m1_rms, 'i_m2_rms': i_m2_rms, 'i_hf_1_sorted': i_hf_1_sorted, 'i_hf_2_sorted': i_hf_2_sorted})
        p_m1_cond = dct_loss.transistor_conduction_loss(i_m1_rms, transistor_dto_1)
        p_m2_cond = dct_loss.transistor_conduction_loss(i_m2_rms, transistor_dto_2)

        calc_losses = d_dtos.CalcLosses(**{'p_m1_conduction': p_m1_cond,
                                           'p_m2_conduction': p_m2_cond,
                                           'p_dab_conduction': 4 * (p_m1_cond + p_m2_cond)})

        gecko_additional_params = d_dtos.GeckoAdditionalParameters(
            t_dead1=50e-9, t_dead2=50e-9, timestep=1e-9,
            number_sim_periods=2, timestep_pre=25e-9, number_pre_sim_periods=0,
            simfilepath=os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..', 'circuits', 'DAB_MOSFET_Modulation_v8.ipes')),
            lossfilepath=os.path.abspath(os.path.join(os.path.abspath(__file__), '..', '..', 'circuits')))

        dab_dto = d_dtos.CircuitDabDTO(
            name=name,
            timestamp=None,
            metadata=None,
            input_config=input_configuration,
            calc_config=calc_config,
            calc_modulation=modulation_parameters,
            calc_currents=calc_currents,
            calc_losses=calc_losses,
            gecko_additional_params=gecko_additional_params,
            gecko_results=None,
            gecko_waveforms=None,
            inductor_results=None,
            stacked_transformer_results=None
        )
        return dab_dto

    @staticmethod
    def add_gecko_simulation_results(dab_dto: d_dtos.CircuitDabDTO, get_waveforms: bool = False) -> d_dtos.CircuitDabDTO:
        """
        Add GeckoCIRCUITS simulation results to the given DTO.

        :param dab_dto: DabDTO
        :param get_waveforms: Read back GeckoCIRCUITS simulation waveforms (high memory consumption). Default to False.
        :type get_waveforms: bool
        :return: DabDTO
        """
        gecko_results, gecko_waveforms = dct_gecko.start_gecko_simulation(
            mesh_v1=dab_dto.calc_config.mesh_V1, mesh_v2=dab_dto.calc_config.mesh_V2,
            mesh_p=dab_dto.calc_config.mesh_P, mod_phi=dab_dto.calc_modulation.phi,
            mod_tau1=dab_dto.calc_modulation.tau1, mod_tau2=dab_dto.calc_modulation.tau2,
            t_dead1=dab_dto.gecko_additional_params.t_dead1, t_dead2=dab_dto.gecko_additional_params.t_dead2,
            fs=dab_dto.input_config.fs, ls=dab_dto.input_config.Ls, lc1=dab_dto.input_config.Lc1,
            lc2=dab_dto.input_config.Lc2, n=dab_dto.input_config.n,
            t_j_1=dab_dto.calc_config.t_j_1, t_j_2=dab_dto.calc_config.t_j_2,
            simfilepath=dab_dto.gecko_additional_params.simfilepath,
            lossfilepath=dab_dto.gecko_additional_params.lossfilepath,
            timestep=dab_dto.gecko_additional_params.timestep,
            number_sim_periods=dab_dto.gecko_additional_params.number_sim_periods,
            timestep_pre=dab_dto.gecko_additional_params.timestep_pre,
            number_pre_sim_periods=dab_dto.gecko_additional_params.number_pre_sim_periods, geckoport=43036,
            c_par_1=dab_dto.input_config.c_par_1, c_par_2=dab_dto.input_config.c_par_2,
            transistor_1_name=dab_dto.input_config.transistor_dto_1.name,
            transistor_2_name=dab_dto.input_config.transistor_dto_2.name, get_waveforms=get_waveforms,
            i_ls_start=dab_dto.calc_currents.i_l_s_sorted[0],
            i_lc1_start=dab_dto.calc_currents.i_l_1_sorted[0],
            i_lc2_start=dab_dto.calc_currents.i_l_2_sorted[0])

        # add GeckoCIRCUITS simulation results to the result DTO.
        dab_dto.gecko_results = d_dtos.GeckoResults(**gecko_results)

        dab_dto.gecko_waveforms = d_dtos.GeckoWaveforms(**gecko_waveforms)
        return dab_dto

    @staticmethod
    def calculate_from_configuration(config: d_dtos.CircuitConfig) -> d_dtos.CalcFromCircuitConfig:
        """
        Calculate logical parameters which can be calculated from the input parameters.

        :param config: DAB configuration
        :type config: CircuitConfig
        :return: CalcFromConfig
        :rtype: CalcFromCircuitConfig
        """
        # choose sampling method
        if config.sampling.sampling_method == "meshgrid":
            steps_per_dimension = np.ceil(np.power(config.sampling.sampling_points, 1 / 3))
            logger.info(f"number of sampling points has been updated from {config.sampling.sampling_points} to {steps_per_dimension ** 3}.")
            logger.info("Note: meshgrid sampling does not take user-given operating points into account")
            v1_operating_points, v2_operating_points, p_operating_points = np.meshgrid(
                np.linspace(config.V1_min, config.V1_max, steps_per_dimension),
                np.linspace(config.V2_min, config.V2_max, steps_per_dimension),
                np.linspace(config.P_min, config.P_max, steps_per_dimension),
                sparse=False)
        elif config.sampling.sampling_method == "latin_hypercube":
            v1_operating_points, v2_operating_points, p_operating_points = sampling.latin_hypercube(
                config.V1_min, config.V1_max, config.V2_min, config.V2_max, config.P_min, config.P_max,
                total_number_points=config.sampling.sampling_points,
                dim_1_user_given_points_list=config.sampling.v_1_additional_user_point_list,
                dim_2_user_given_points_list=config.sampling.v_2_additional_user_point_list,
                dim_3_user_given_points_list=config.sampling.p_additional_user_point_list)
        elif config.sampling.sampling_method == "poisson_disk_sampling":
            raise NotImplementedError("Not implemented yet.")
        else:
            raise ValueError(f"sampling_method '{config.sampling.sampling_method}' not available.")

        Lc2_ = config.Lc2 * config.n ** 2

        calc_from_config = d_dtos.CalcFromCircuitConfig(
            mesh_V1=np.atleast_3d(v1_operating_points),
            mesh_V2=np.atleast_3d(v2_operating_points),
            mesh_P=np.atleast_3d(p_operating_points),
            Lc2_=Lc2_,
            t_j_1=config.transistor_dto_1.t_j_max_op,
            t_j_2=config.transistor_dto_2.t_j_max_op,
            c_oss_par_1=config.transistor_dto_1.c_oss + config.c_par_1,
            c_oss_par_2=config.transistor_dto_2.c_oss + config.c_par_2,
            c_oss_1=config.transistor_dto_1.c_oss,
            c_oss_2=config.transistor_dto_2.c_oss,
            q_oss_1=config.transistor_dto_1.q_oss,
            q_oss_2=config.transistor_dto_2.q_oss
        )

        return calc_from_config

    @staticmethod
    def calculate_modulation(config: d_dtos.CircuitConfig, calc_config: d_dtos.CalcFromCircuitConfig) -> d_dtos.CalcModulation:
        """
        Calculate the modulation parameters like phi, tau1, tau, ...

        :param config: DAB input configuration
        :param calc_config: calculated parameters from the input configuration
        :return: Modulation parameters.
        """
        result_dict = mod.calc_modulation_params(config.n, config.Ls, config.Lc1, config.Lc2, config.fs, c_oss_1=calc_config.c_oss_par_1,
                                                 c_oss_2=calc_config.c_oss_par_2, v1=calc_config.mesh_V1, v2=calc_config.mesh_V2, power=calc_config.mesh_P)

        return d_dtos.CalcModulation(**result_dict)

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
            logger.warning("The voltage in csv file is not monotonically rising!")
        # Check if Coss is monotonically falling
        if not np.all(csv_data[1:, 1] <= csv_data[:-1, 1], axis=0):
            logger.warning("The C_oss in csv file is not monotonically falling!")

        # Rescale and interpolate the csv data to have a nice 1V step size from 0V to v_max
        # A first value with zero volt will be added
        v_max = int(np.round(csv_data[-1, 0]))
        v_interp = np.arange(v_max + 1)

        # The margin is considered here as a factor of the original capacitance value
        coss_interp = margin_factor * np.interp(v_interp, csv_data[:, 0], csv_data[:, 1])
        # Since we now have an evenly spaced vector where x correspond to the element-number of the vector
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
            return np.trapezoid(coss_v)

        coss_int = np.vectorize(integrate)
        # get the qoss vector that has the resolution 1V from 0 to V_max
        v_vec = np.arange(coss.shape[0])
        qoss = coss_int(v_vec)

        return qoss

    @staticmethod
    def save(dab_dto: d_dtos.CircuitDabDTO, name: str, directory: str, timestamp: bool = True) -> None:
        """
        Save the DabDTO-class to a npz file.

        :param dab_dto: Class to store
        :type dab_dto: d_dtos.CircuitDabDTO
        :param name: Filename
        :type name: str
        :param directory: Directory to store the results
        :type directory: str
        :param timestamp: [True] to add a timestamp to the file name.
        :type timestamp: bool
        """
        # Add some descriptive data to the file
        # Adding a timestamp, it may be useful
        dab_dto.timestamp = np.asarray(datetime.datetime.now().isoformat())

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
                logger.warning("Directory does not exist!", stacklevel=2)
                file = os.path.join(filename)
        else:
            file = os.path.join(filename)

        with open(f"{file}.pkl", 'wb') as output:
            pickle.dump(dab_dto, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_from_file(file: str) -> d_dtos.CircuitDabDTO:
        """
        Load everything from the given .npz file.

        :param file: a .nps filename or file-like object, string, or pathlib.Path
        :return: two objects with type DAB_Specification and DAB_Results
        """
        # Check for filename extension
        file_name, file_extension = os.path.splitext(file)
        if not file_extension:
            file += '.pkl'
        file = os.path.expanduser(file)
        file = os.path.expandvars(file)
        file = os.path.abspath(file)

        with open(file, 'rb') as pickle_file_data:
            loaded_circuit_dto = pickle.load(pickle_file_data)
            if not isinstance(loaded_circuit_dto, d_dtos.CircuitDabDTO):
                raise TypeError(f"Loaded pickle file {loaded_circuit_dto} not of type CircuitDabDTO.")
            return loaded_circuit_dto

    @staticmethod
    def get_max_peak_waveform_transformer(dab_dto: d_dtos.CircuitDabDTO, plot: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get the transformer waveform with the maximum current peak out of the three-dimensional simulation array (v_1, v_2, P).

        :param dab_dto: DAB data transfer object (DTO)
        :type dab_dto: d_dtos.CircuitDabDTO
        :param plot: True to plot the results, mostly for understanding and debugging
        :type plot: bool
        :return: sorted_max_angles, i_l_s_max_current_waveform, i_hf_2_max_current_waveform. All as a numpy array.
        :rtype: List[np.ndarray]
        """
        i_hf_2_sorted = np.transpose(dab_dto.calc_currents.i_l_s_sorted * dab_dto.input_config.n - dab_dto.calc_currents.i_l_2_sorted, (1, 2, 3, 0))
        angles_rad_sorted = np.transpose(dab_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))

        max_index = (0, 0, 0)
        for vec_vvp in np.ndindex(dab_dto.calc_modulation.phi.shape):
            max_index = vec_vvp if np.max(i_hf_2_sorted[vec_vvp]) > np.max(i_hf_2_sorted[max_index]) else max_index  # type: ignore
            if plot:
                plt.plot(d_waveforms.full_angle_waveform_from_angles(angles_rad_sorted[vec_vvp]),
                         d_waveforms.full_current_waveform_from_currents(i_hf_2_sorted[vec_vvp]), color='gray')

        i_hf_2_max_current_waveform = i_hf_2_sorted[max_index]
        i_l_s_max_current_waveform = np.transpose(dab_dto.calc_currents.i_l_s_sorted, (1, 2, 3, 0))[max_index]

        sorted_max_angles, unique_indices = np.unique(d_waveforms.full_angle_waveform_from_angles(angles_rad_sorted[max_index]), return_index=True)
        i_l_s_max_current_waveform = d_waveforms.full_current_waveform_from_currents(i_l_s_max_current_waveform)[unique_indices]
        i_hf_2_max_current_waveform = d_waveforms.full_current_waveform_from_currents(i_hf_2_max_current_waveform)[unique_indices]

        if plot:
            plt.plot(sorted_max_angles, i_hf_2_max_current_waveform, color='red', label='peak current full waveform')
            plt.grid()
            plt.xlabel('Angle in rad')
            plt.ylabel('Current in A')
            plt.legend()
            plt.show()

        return sorted_max_angles, i_l_s_max_current_waveform, i_hf_2_max_current_waveform

    @staticmethod
    def get_max_peak_waveform_inductor(dab_dto: d_dtos.CircuitDabDTO, plot: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the inductor waveform with the maximum current peak out of the three-dimensional simulation array (v_1, v_2, P).

        :param dab_dto: DAB data transfer object (DTO)
        :type dab_dto: d_dtos.CircuitDabDTO
        :param plot: True to plot the results, mostly for understanding and debugging
        :type plot: bool
        :return: sorted_max_angles, i_l_s_max_current_waveform, i_l1_max_current_waveform. All as a numpy array.
        :rtype: List[np.ndarray]
        """
        i_l1_sorted = np.transpose(dab_dto.calc_currents.i_l_1_sorted, (1, 2, 3, 0))
        angles_rad_sorted = np.transpose(dab_dto.calc_currents.angles_rad_sorted, (1, 2, 3, 0))

        max_index = (0, 0, 0)
        for vec_vvp in np.ndindex(dab_dto.calc_modulation.phi.shape):
            max_index = vec_vvp if np.max(i_l1_sorted[vec_vvp]) > np.max(i_l1_sorted[max_index]) else max_index  # type: ignore
            if plot:
                plt.plot(d_waveforms.full_angle_waveform_from_angles(angles_rad_sorted[vec_vvp]),
                         d_waveforms.full_current_waveform_from_currents(i_l1_sorted[vec_vvp]), color='gray')

        i_l1_max_current_waveform = i_l1_sorted[max_index]

        sorted_max_angles, unique_indices = np.unique(d_waveforms.full_angle_waveform_from_angles(angles_rad_sorted[max_index]), return_index=True)
        i_l1_max_current_waveform = d_waveforms.full_current_waveform_from_currents(i_l1_max_current_waveform)[unique_indices]

        if plot:
            plt.plot(sorted_max_angles, i_l1_max_current_waveform, color='red', label='peak current full waveform')
            plt.grid()
            plt.xlabel('Angle in rad')
            plt.ylabel('Current in A')
            plt.legend()
            plt.show()

        return sorted_max_angles, i_l1_max_current_waveform

    @staticmethod
    def export_transformer_target_parameters_dto(dab_dto: d_dtos.CircuitDabDTO) -> d_dtos.TransformerTargetParameters:
        """
        Export the optimization parameters for the transformer optimization (inside FEMMT).

        Note: the current counting system is adapted to FEMMT! The secondary current is counted negative!

        :param dab_dto: DAB DTO
        :type dab_dto: d_dtos.CircuitDabDTO
        :return: DTO for the transformer optimization using FEMMT
        :rtype: TransformerTargetParameters
        """
        # calculate the full 2pi waveform from the four given DAB currents
        sorted_max_angles, i_l_s_max_current_waveform, i_hf_2_max_current_waveform = HandleDabDto.get_max_peak_waveform_transformer(dab_dto, plot=False)
        # transfer 2pi periodic time into a real time
        time = sorted_max_angles / 2 / np.pi / dab_dto.input_config.fs

        return d_dtos.TransformerTargetParameters(
            l_s12_target=dab_dto.input_config.Ls,
            l_h_target=dab_dto.input_config.Lc2 * dab_dto.input_config.n ** 2,
            n_target=dab_dto.input_config.n,
            time_current_1_vec=np.array([time, i_l_s_max_current_waveform]),
            time_current_2_vec=np.array([time, -i_hf_2_max_current_waveform]),
            temperature=100)

    @staticmethod
    def add_inductor_results(dab_dto: d_dtos.CircuitDabDTO, inductor_results: d_dtos.InductorResults) -> d_dtos.CircuitDabDTO:
        """Add inductor results to the CircuitDabDTO.

        :param dab_dto: Dual-active bridge DTO
        :type dab_dto: d_dtos.CircuitDabDTO
        :param inductor_results: inductor losses
        :type inductor_results: InductorResults
        :return: Dual-active bridge DTO including the inductor losses
        :rtype: d_dtos.CircuitDabDTO
        """
        dab_dto.inductor_results = inductor_results

        return dab_dto

    @staticmethod
    def add_stacked_transformer_results(dab_dto: d_dtos.CircuitDabDTO, transformer_results: d_dtos.StackedTransformerResults) -> d_dtos.CircuitDabDTO:
        """Add stacked transformer results to the CircuitDabDTO.

        :param dab_dto: Dual-active bridge DTO
        :type dab_dto: d_dtos.CircuitDabDTO
        :param transformer_results: stacked transformer results
        :type transformer_results: StackedTransformerResults
        :return: Dual-active bridge DTO including the inductor losses
        :rtype: d_dtos.CircuitDabDTO
        """
        dab_dto.stacked_transformer_results = transformer_results
        return dab_dto

class HandleTransistorDto:
    """Handle the transistor DTOs."""

    @staticmethod
    def tdb_to_transistor_dto(transistor_name: str, c_oss_margin_factor: float = 1.2) -> d_dtos.TransistorDTO:
        """
        Load a transistor from the transistor database and transfer the important parameters to a TransistorDTO.

        :param transistor_name: transistor name, must be same name as in the transistor database
        :type transistor_name: str
        :param c_oss_margin_factor: margin for c_oss
        :type c_oss_margin_factor: float
        :return: Transistor DTO
        :rtype: d_dtosTransistorDTO
        """
        db = tdb.DatabaseManager()
        db.set_operation_mode_json()

        transistor: tdb.Transistor = db.load_transistor(transistor_name)

        if transistor.type != "MOSFET" and transistor.type != "SiC-MOSFET":
            raise ValueError(f"Transistor 1: {transistor.name} is of non-allowed type {transistor.type}. "
                             f"Allowed types are MOSFET, SiC-MOSFET.")

        t_j_recommended = transistor.switch.t_j_max - 25

        c_oss, q_oss = HandleDabDto.get_c_oss_from_tdb(transistor, margin_factor=c_oss_margin_factor)

        # export c_oss files for GeckoCIRCUITS
        path_to_save_c_oss_files = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'circuits')
        if not os.path.exists(os.path.join(path_to_save_c_oss_files, f"{transistor.name}_c_oss.nlc")):
            transistor.export_geckocircuits_coss(filepath=path_to_save_c_oss_files, margin_factor=c_oss_margin_factor)

        transistor.quickstart_wp()

        transistor_dto = d_dtos.TransistorDTO(
            name=transistor.name,
            t_j_max_op=t_j_recommended,
            c_oss=c_oss,
            q_oss=q_oss,
            r_th_jc=transistor.switch.thermal_foster.r_th_total,
            cooling_area=transistor.cooling_area,
            housing_area=transistor.housing_area,
            r_channel=transistor.wp.switch_r_channel
        )

        return transistor_dto

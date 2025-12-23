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
import dct.topology.sbc.sbc_datasets_dtos as d_dtos
import dct.topology.sbc.sbc_currents as dct_currents
import dct.topology.sbc.sbc_losses as dct_loss
from dct.topology.sbc.sbc_circuit_topology_dtos import CircuitSampling

logger = logging.getLogger(__name__)

class HandleSbcDto:
    """Class to handle the SbcDTO, e.g. save and load the files."""

    @staticmethod
    def init_config(name: str, mesh_v1: np.ndarray, mesh_duty_cycle: np.ndarray, mesh_i: np.ndarray,
                    sampling: CircuitSampling, ls: float, fs: float,
                    transistor_dto_1: d_dtos.TransistorDTO, transistor_dto_2: d_dtos.TransistorDTO) -> d_dtos.SbcCircuitDTO:
        """
        Initialize the SBC structure.

        :param name: name of the simulation
        :type  name: str
        :param mesh_v1: mesh or hypercube sampling for v1
        :type  mesh_v1: np.ndarray
        :param mesh_duty_cycle: mesh or hypercube sampling for duty cycle
        :type  mesh_duty_cycle: np.ndarray
        :param mesh_i: mesh or hypercube sampling for current
        :type  mesh_i: np.ndarray
        :param sampling: Sampling parameters
        :type  sampling: d_dtos.Sampling
        :param ls: series inductance
        :type  ls: float
        :param fs: Switching frequency
        :type  fs: float
        :param transistor_dto_1: Transistor DTO for transistor bridge 1. Must match with transistordatabase available transistors.
        :type  transistor_dto_1: TransistorDTO
        :param transistor_dto_2: Transistor DTO for transistor bridge 1. Must match with transistordatabase available transistors.
        :type  transistor_dto_2: TransistorDTO
        :return:
        """
        input_configuration = d_dtos.CircuitConfig(mesh_v1=mesh_v1,
                                                   mesh_duty_cycle=mesh_duty_cycle,
                                                   mesh_i=mesh_i,
                                                   sampling=sampling,
                                                   Ls=np.array(ls),
                                                   fs=np.array(fs),
                                                   transistor_dto_1=transistor_dto_1,
                                                   transistor_dto_2=transistor_dto_2)

        # ASA Remove calc_config calculation

        # Design space:
        # fs, L M (HS and LS same type, later to replace by variable)
        #
        # Chose: fs, L M -> p_loss_transistor, ripple current, volume proxy L
        # Next iteration: Change...
        #
        # P_Loss=P_cond+P_switch=I_rms²*R_D_S_on=(I_out²+I_Ripple¹/12)*R_D_S_on
        # I_Ripple=(V_in-V_out)*D/(L*fs)  = V_in*D(1-D)/(L*fs)
        # P_switch= 0.5⋅V_in⋅I_pk⋅(t_raise+t_fall)⋅f (First assumption for the simulation, later calculation over switch energy)

        # Calculate the ripple current and rms current
        i_ripple, i_ms, i_rms = dct_currents.calc_rms_currents(input_configuration)

        # Calculate the Volume proxy
        calc_volume_inductor_proxy = dct_currents.calc_volume_inductor_proxy(input_configuration, i_ripple, i_rms)

        calc_currents = d_dtos.CalcCurrents(i_rms=i_rms, i_ms=i_ms, i_ripple=i_ripple)

        # Calculate the transistor conduction losses p_hs_cond = (I_out²+I_Ripple²/12)*R_D_S_on
        p_hs_cond = dct_loss.transistor_conduction_loss(i_ms * input_configuration.mesh_duty_cycle, transistor_dto_1)
        p_ls_cond = dct_loss.transistor_conduction_loss(i_ms * (1 - input_configuration.mesh_duty_cycle), transistor_dto_1)
        p_hs_switch = dct_loss.transistor_switch_loss(input_configuration.mesh_v1, i_rms,
                                                      input_configuration.transistor_dto_1, input_configuration.fs)
        p_ls_switch = dct_loss.transistor_switch_loss(input_configuration.mesh_v1, i_rms,
                                                      input_configuration.transistor_dto_2, input_configuration.fs)

        p_loss = d_dtos.CalcLosses(**{'p_hs_conduction': p_hs_cond,
                                      'p_ls_conduction': p_ls_cond,
                                      'p_hs_switch': p_hs_switch,
                                      'p_ls_switch': p_ls_switch,
                                      'p_sbc_total': p_hs_cond + p_ls_cond + p_hs_switch + p_ls_switch})

        sbc_dto = d_dtos.SbcCircuitDTO(
            name=name,
            timestamp=None,
            metadata=None,
            input_config=input_configuration,
            # Later to remove
            calc_config=None,
            # End Later to remove
            calc_currents=calc_currents,
            calc_losses=p_loss,
            component_requirements=None,
            calc_volume_inductor_proxy=calc_volume_inductor_proxy,
            inductor_results=None
        )

        return sbc_dto

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
        q_oss = HandleSbcDto._integrate_c_oss(coss_interp)
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
    def save(sbc_dto: d_dtos.SbcCircuitDTO, name: str, directory: str, timestamp: bool = True) -> None:
        """
        Save the SbcDTO-class to a npz file.

        :param sbc_dto: Class to store
        :type sbc_dto: d_dtos.SbcCircuitDTO
        :param name: Filename
        :type name: str
        :param directory: Directory to store the results
        :type directory: str
        :param timestamp: [True] to add a timestamp to the file name.
        :type timestamp: bool
        """
        # Add some descriptive data to the file
        # Adding a timestamp, it may be useful
        sbc_dto.timestamp = np.asarray(datetime.datetime.now().isoformat())

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
                filename = "sbc_dataset"

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
            pickle.dump(sbc_dto, output, pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load_from_file(file: str) -> d_dtos.SbcCircuitDTO:
        """
        Load everything from the given .npz file.

        :param file: a .nps filename or file-like object, string, or pathlib.Path
        :return: two objects with type SBC_Specification and SBC_Results
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
            if not isinstance(loaded_circuit_dto, d_dtos.SbcCircuitDTO):
                raise TypeError(f"Loaded pickle file {loaded_circuit_dto} not of type CircuitSbcDTO.")
            return loaded_circuit_dto

    @staticmethod
    def get_max_peak_waveform_inductor(sbc_dto: d_dtos.SbcCircuitDTO, plot: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """
        Get the inductor waveform with the maximum current peak out of the three-dimensional simulation array (v_1, v_2, P).

        :param sbc_dto: SBC data transfer object (DTO)
        :type sbc_dto: d_dtos.SbcCircuitDTO
        :param plot: True to plot the results, mostly for understanding and debugging
        :type plot: bool
        :return: sorted_max_angles, i_l_s_max_current_waveform, i_l1_max_current_waveform. All as a numpy array.
        :rtype: List[np.ndarray]
        """
        # Variable declaration
        i_rms_current: np.ndarray
        i_rms_max_current: np.ndarray
        time_array: np.ndarray

        # Later modulation are load/generator
        i_rms_current = np.squeeze(sbc_dto.calc_currents.i_ripple)
        i_rms_max_current = np.array([-0.5, +0.5, -0.5])
        i_rms_max_current = i_rms_max_current * i_rms_current.max()
        # FEMMT issue: DC current leads to problems-> Next line is commented out
        # i_rms_max_current = i_rms_max_current + np.squeeze(sbc_dto.calc_currents.i_rms)
        # ASA: Duty cycle worst case=0.5. This is to replace by suitable result out of mesh duty cycle
        time_array = np.array([0, 0.5, 1]) * 1/sbc_dto.input_config.fs

        if plot:
            plt.plot(time_array, i_rms_max_current, color='red', label='peak current full waveform')
            plt.grid()
            plt.xlabel('time in us')
            plt.ylabel('Current in A')
            plt.legend()
            plt.show()

        return time_array, i_rms_max_current

    @staticmethod
    def add_inductor_results(sbc_dto: d_dtos.SbcCircuitDTO, inductor_results: d_dtos.InductorResults) -> d_dtos.SbcCircuitDTO:
        """Add inductor results to the CircuitSbcDTO.

        :param sbc_dto: Dual-active bridge DTO
        :type sbc_dto: d_dtos.SbcCircuitDTO
        :param inductor_results: inductor losses
        :type inductor_results: InductorResults
        :return: Dual-active bridge DTO including the inductor losses
        :rtype: d_dtos.SbcCircuitDTO
        """
        sbc_dto.inductor_results = inductor_results

        return sbc_dto

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

        c_oss, q_oss = HandleSbcDto.get_c_oss_from_tdb(transistor, margin_factor=c_oss_margin_factor)

        # export c_oss files for GeckoCIRCUITS
        # if not os.path.exists(os.path.join(HandleSbcDto.c_oss_storage_directory, f"{transistor.name}_c_oss.nlc")):
        # transistor.export_geckocircuits_coss(filepath=HandleSbcDto.c_oss_storage_directory, margin_factor=c_oss_margin_factor)

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

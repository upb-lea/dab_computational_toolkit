"""Validate the results using GeckoCIRCUITS."""
# python libraries
import math
import os.path
import socket
import random

# own libraries
from dct.debug_tools import *

# 3rd party libraries
import numpy as np
from collections import defaultdict
from tqdm import tqdm
import pygeckocircuits2 as pgc
import pandas as pd

def start_gecko_simulation(mesh_V1: np.ndarray, mesh_V2: np.ndarray, mesh_P: np.ndarray,
                           mod_phi: np.ndarray, mod_tau1: np.ndarray, mod_tau2: np.ndarray,
                           t_dead1: float | np.ndarray, t_dead2: float | np.ndarray, fs: int | np.ndarray | np.float64,
                           Ls: float, Lc1: float, Lc2: float, n: float,
                           t_j_1: float, t_j_2: float,
                           simfilepath: str, timestep: float, number_sim_periods: int, transistor_1_name: str, transistor_2_name: str, lossfilepath: str,
                           i_ls_start: np.ndarray, i_lc1_start: np.ndarray, i_lc2_start: np.ndarray,
                           timestep_pre: float = 0, number_pre_sim_periods: int = 0, geckoport: int = 43036,
                           c_par_1: float | None = None, c_par_2: float | None = None,
                           get_waveforms: bool = False) -> tuple[dict, defaultdict]:
    """
    Start the GeckoCIRCUITS simulation.

    :param mesh_V1: mesh of voltage v1 in V
    :type mesh_V1: np.array
    :param mesh_V2: mesh of voltage v2 in V
    :type mesh_V2: np.array
    :param mesh_P: mesh of the power P in W
    :type mesh_P: np.array
    :param mod_phi: matrix with modulation parameter phi
    :type mod_phi: np.array
    :param mod_tau1: matrix with modulation parameter tau_1
    :type mod_tau1: np.array
    :param mod_tau2: matrix with modulation parameter tau_2
    :type mod_tau2: np.array
    :param t_dead1: dead time for bridge 1 in seconds
    :type t_dead1: float
    :param t_dead2: dead time for bridge 2 in seconds
    :type t_dead2: float
    :param fs: switching frequency in Hz
    :type fs: float
    :param Ls: series inductance in H
    :type Ls: float
    :param Lc1: Commutation inductance for bridge 1 in H
    :type Lc1: float
    :param Lc2: Commutation inductance for bridge 2 in H
    :type Lc2: float
    :param n: transfer ratio
    :type n: float
    :param t_j_1: MOSFET junction temperature for bridge 1 in degree Celsius
    :type t_j_1: float
    :param t_j_2: MOSFET junction temperature for bridge 2 in degree Celsius
    :type t_j_2: float
    :param simfilepath: simulation file filepath
    :param timestep: timestep in seconds, e.g. 5e-9
    :type timestep: float
    :param number_sim_periods: simulation periods
    :type number_sim_periods: int
    :type simfilepath: float
    :param timestep_pre: time-steps of pre-simulation, e.g. 50e-9
    :type timestep_pre: float
    :param number_pre_sim_periods: pre-simulation periods (not recorded)
    :type number_pre_sim_periods: int
    :param geckoport: port of GeckoCIRCUITS to connect
    :type geckoport: int
    :param c_par_1: parasitic capacitance for one single MOSFET of bridge 1
    :type c_par_1: float
    :param c_par_2: parasitic capacitance for one single MOSFET of bridge 2
    :type c_par_2: float
    :param get_waveforms: True to return i_Ls, i_Lc1 and i_Lc2. Defaults to False.
    :type get_waveforms: bool
    :param lossfilepath: file path of loss files for transistors
    :type lossfilepath: str
    :param transistor_1_name: Name of transistor 1
    :type transistor_1_name: str
    :param transistor_2_name: Name of transistor 2
    :type transistor_2_name: str
    :param i_ls_start: start current of L_s in A
    :type i_ls_start: np.array
    :param i_lc1_start: start current of L_c1 in A
    :type i_lc1_start: np.array
    :param i_lc2_start: start current of the L_c2 in A (on secondary side)
    :type i_lc2_start: np.array
    """
    # Broadcast possible scalar values to mesh size
    _ones = np.ones_like(mod_phi)
    mesh_t_dead1 = _ones * t_dead1
    mesh_t_dead2 = _ones * t_dead2
    mesh_fs = _ones * fs

    # Transform Lc2 to side 1
    Lc2_ = Lc2 * n ** 2

    # values we want to get from the simulation
    l_means_keys = ['p_dc1', 'p_dc2', 'S11_p_sw', 'S11_p_cond', 'S12_p_sw', 'S12_p_cond',
                    'S23_p_sw', 'S23_p_cond', 'S24_p_sw', 'S24_p_cond',
                    'v_dc1', 'i_dc1', 'v_dc2', 'i_dc2',
                    'p_sw1', 'p_cond1', 'p_sw2', 'p_cond2']
    l_rms_keys = ['i_HF1', 'i_Ls', 'i_Lc1', 'i_Lc2', 'i_C11', 'i_C12', 'i_C23', 'i_C24',
                  'i_S11', 'i_S12', 'i_S23', 'i_S24', 'i_HF2']
    l_min_keys = ['v_ds_S11_sw_on', 'v_ds_S23_sw_on', 'i_HF1_S11_sw_on', 'i_HF2_S23_sw_on']
    # values calculated from simulation results (only for overview of return dict keys)
    l_calc_keys = ['power_deviation', 'zvs_coverage', 'zvs_coverage1', 'zvs_coverage2',
                   'zvs_coverage_notnan', 'zvs_coverage1_notnan', 'zvs_coverage2_notnan',
                   'i_HF1_total_mean', 'I1_squared_total_mean']

    waveform_keys = ['i_Ls', 'i_Lc1', 'i_Lc2', 'i_HF1', 'i_HF2']

    # Set a reasonable low zvs voltage limit below we assume zvs operation
    zvs_vlimit = 50

    # init gecko waveform simulation
    result_df = pd.DataFrame()
    gecko_waveforms_single_simulation = dict()
    gecko_waveforms_multiple_simulations: defaultdict = defaultdict(dict)

    # Init arrays to store simulation results
    da_sim_results: dict[str, np.ndarray] = dict()
    for k in l_means_keys:
        da_sim_results[k] = np.full_like(mod_phi, np.nan)
    for k in l_rms_keys:
        da_sim_results[k] = np.full_like(mod_phi, np.nan)
    for k in l_min_keys:
        da_sim_results[k] = np.full_like(mod_phi, np.nan)

    # Progressbar init, calc total number of iterations to simulate
    pbar = tqdm(total=mod_phi.size)

    # Find a free port if zero is given as port
    if geckoport == 0:
        geckoport = get_free_port()

    simtime_pre = number_pre_sim_periods / fs
    simtime = number_sim_periods / fs

    gecko_dab_converter = pgc.GeckoSimulation(simfilepath=simfilepath, geckoport=geckoport, timestep=timestep,
                                              simtime=simtime + t_dead1, timestep_pre=timestep_pre, simtime_pre=simtime_pre)

    for vec_vvp in np.ndindex(mod_phi.shape):
        # set simulation parameters and convert tau to degree for Gecko
        sim_params = {
            'v_dc1': mesh_V1[vec_vvp].item(),
            'v_dc2': mesh_V2[vec_vvp].item(),
            'phi': mod_phi[vec_vvp].item() * 180 / np.pi,
            'tau1': mod_tau1[vec_vvp].item() * 180 / np.pi,
            'tau2': mod_tau2[vec_vvp].item() * 180 / np.pi,
            't_dead1': mesh_t_dead1[vec_vvp].item(),
            't_dead2': mesh_t_dead2[vec_vvp].item(),
            'fs': mesh_fs[vec_vvp].item(),
            'Ls': float(Ls),
            'Lc1': float(Lc1),
            'Lc2_': float(Lc2_),
            'n': float(n),
            't_j_1': float(t_j_1),
            't_j_2': float(t_j_2),
            'C_par_1': c_par_1,
            'C_par_2': c_par_2,
            'i_Ls_start': i_ls_start[vec_vvp].item(),
            'i_Lc1_start': i_lc1_start[vec_vvp].item(),
            'i_Lc2__start': i_lc2_start[vec_vvp].item() / n,
        }

        # Only run simulation if all params are valid
        if not np.any(np.isnan(list(sim_params.values()))):
            # start simulation for this operation point
            c_oss_1_file = os.path.join(lossfilepath, f'{transistor_1_name}_c_oss.nlc')
            c_oss_2_file = os.path.join(lossfilepath, f'{transistor_2_name}_c_oss.nlc')

            gecko_dab_converter.set_global_parameters(sim_params)
            gecko_dab_converter.set_nonlinear_file(['C11', 'C12', 'C13', 'C14'], c_oss_1_file)
            gecko_dab_converter.set_nonlinear_file(['C21', 'C22', 'C23', 'C24'], c_oss_2_file)

            # Start the simulation and get the results. Do not set times here because it is set while init Gecko.
            gecko_dab_converter.run_simulation()
            values_mean = gecko_dab_converter.get_values(
                nodes=l_means_keys,
                operations=['mean'],
                # Just use the last part, because the beginning is garbage
                # simtime must be greater than 2*Ts
                # Use only second half of simtime and make it multiple of Ts
                range_start_stop=[math.ceil(simtime * mesh_fs[vec_vvp].item() / 2) * 1 / mesh_fs[vec_vvp].item(),
                                  'end']
            )
            values_rms = gecko_dab_converter.get_values(
                nodes=l_rms_keys,
                operations=['rms'],
                # Just use the last part, because the beginning is garbage
                # simtime must be greater than 2*Ts
                # Use only second half of simtime and make it multiple of Ts
                range_start_stop=[math.ceil(simtime * mesh_fs[vec_vvp].item() / 2) * 1 / mesh_fs[vec_vvp].item(),
                                  'end']
            )
            values_min = gecko_dab_converter.get_values(
                nodes=l_min_keys,
                operations=['min'],
                # Just use the last part, because the beginning is garbage
                # Use the smallest possible timerange at the end
                range_start_stop=[2 * timestep, 'end']
            )

            if get_waveforms:
                result_df = gecko_dab_converter.get_scope_data(waveform_keys, "results")

                for key in waveform_keys:
                    gecko_waveforms_single_simulation[key] = result_df[key].to_numpy()
                gecko_waveforms_single_simulation['time'] = result_df['time'].to_numpy()

        else:
            # When params are not valid return NaN values
            values_mean = defaultdict(dict)
            values_rms = defaultdict(dict)
            values_min = defaultdict(dict)
            gecko_waveforms_single_simulation = dict()
            for k in l_means_keys:
                values_mean['mean'][k] = np.nan
            for k in l_rms_keys:
                values_rms['rms'][k] = np.nan
            for k in l_min_keys:
                values_min['min'][k] = np.nan
            for k in waveform_keys:
                gecko_waveforms_single_simulation[k] = np.nan
            gecko_waveforms_single_simulation['time'] = np.nan

        # save single simulation results in arrays (multidimensional / 3D) for the final results.
        for k in l_means_keys:
            da_sim_results[k][vec_vvp] = values_mean['mean'][k]
        for k in l_rms_keys:
            da_sim_results[k][vec_vvp] = values_rms['rms'][k]
        for k in l_min_keys:
            da_sim_results[k][vec_vvp] = values_min['min'][k]
        if get_waveforms:
            for k in waveform_keys:
                gecko_waveforms_multiple_simulations[k][vec_vvp] = gecko_waveforms_single_simulation[k]
                gecko_waveforms_multiple_simulations['time'] = gecko_waveforms_single_simulation['time']
        else:
            for k in waveform_keys:
                gecko_waveforms_multiple_simulations[k][vec_vvp] = None
                gecko_waveforms_multiple_simulations['time'] = None

        # Progressbar update, default increment +1
        pbar.update()

    if not __debug__:
        # Gecko Basics
        gecko_dab_converter.__del__()

    # Calculate results for the entire Simulation

    # Calc power deviation from expected power target
    da_sim_results['power_deviation'] = da_sim_results['p_dc2'] / (mesh_P + 0.01)
    # Show ZVS coverage based on simulation
    da_sim_results['zvs_coverage'] = np.count_nonzero(
        np.less_equal(da_sim_results['v_ds_S11_sw_on'], zvs_vlimit) & np.less_equal(da_sim_results['v_ds_S23_sw_on'],
                                                                                    zvs_vlimit)) / np.size(
        da_sim_results['v_ds_S11_sw_on'])
    da_sim_results['zvs_coverage1'] = np.count_nonzero(
        np.less_equal(da_sim_results['v_ds_S11_sw_on'], zvs_vlimit)) / np.size(
        da_sim_results['v_ds_S11_sw_on'])
    da_sim_results['zvs_coverage2'] = np.count_nonzero(
        np.less_equal(da_sim_results['v_ds_S23_sw_on'], zvs_vlimit)) / np.size(
        da_sim_results['v_ds_S23_sw_on'])
    # Only non NaN areas:
    # Show ZVS coverage based on simulation
    da_sim_results['zvs_coverage_notnan'] = np.count_nonzero(
        np.less_equal(da_sim_results['v_ds_S11_sw_on'], zvs_vlimit) & np.less_equal(da_sim_results['v_ds_S23_sw_on'],
                                                                                    zvs_vlimit)) / np.size(
        da_sim_results['v_ds_S11_sw_on'][~np.isnan(mod_tau1)])
    da_sim_results['zvs_coverage1_notnan'] = np.count_nonzero(
        np.less_equal(da_sim_results['v_ds_S11_sw_on'][~np.isnan(mod_tau1)], zvs_vlimit)) / np.size(
        da_sim_results['v_ds_S11_sw_on'][~np.isnan(mod_tau1)])
    da_sim_results['zvs_coverage2_notnan'] = np.count_nonzero(
        np.less_equal(da_sim_results['v_ds_S23_sw_on'][~np.isnan(mod_tau1)], zvs_vlimit)) / np.size(
        da_sim_results['v_ds_S23_sw_on'][~np.isnan(mod_tau1)])
    # Total i_HF1 mean
    da_sim_results['i_HF1_total_mean'] = np.nanmean(da_sim_results['i_HF1'])
    # Square I_rms before mean, because we need the relation P ~ R*I^2
    da_sim_results['I1_squared_total_mean'] = np.nanmean(da_sim_results['i_HF1'] ** 2)

    # Progressbar end
    pbar.close()

    return da_sim_results, gecko_waveforms_multiple_simulations


def get_free_port(start: int = 43047, stop: int = 50000) -> int:
    """
    Get a random free port in Range start <= port < stop.

    :param start: start port
    :type start: int
    :param stop: stop port
    :type stop: int
    :return:
    """
    # ss -tulpn example for a gecko run:
    # tcp   LISTEN   0  50    *:43036   *:*   users:(("java",pid=1532749,fd=57))

    random.seed()
    port = random.randrange(start, stop)
    port_in_use = 0
    while not port_in_use:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port_in_use = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        if port_in_use == 0:
            port = random.randint(start, stop)
    return port

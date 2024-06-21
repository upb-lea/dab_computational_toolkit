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

def start_gecko_simulation(mesh_V1: np.ndarray, mesh_V2: np.ndarray, mesh_P: np.ndarray,
                           mod_phi: np.ndarray, mod_tau1: np.ndarray, mod_tau2: np.ndarray,
                           t_dead1: float | np.ndarray, t_dead2: float | np.ndarray, fs: int | np.ndarray,
                           Ls: float, Lc1: float, Lc2: float, n: float,
                           temp: float,
                           simfilepath: str, timestep: float, simtime: float,
                           timestep_pre: float = 0, simtime_pre: float = 0, geckoport: int = 43036, gdebug: bool = False,
                           c_par_1: float = None, c_par_2: float = None, transistor_1_name: str = None, transistor_2_name: str = None,
                           lossfilepath: str = None) -> dict:
    """
    Start the GeckoCIRCUITS simulation.

    :param mesh_V1:
    :param mesh_V2:
    :param mesh_P:
    :param mod_phi:
    :param mod_tau1:
    :param mod_tau2:
    :param t_dead1:
    :param t_dead2:
    :param fs:
    :param Ls:
    :param Lc1:
    :param Lc2:
    :param n:
    :param temp:
    :param simfilepath:
    :param timestep:
    :param simtime:
    :param timestep_pre:
    :param simtime_pre:
    :param geckoport:
    :param gdebug:
    :param c_par_1:
    :param c_par_2:
    :return:
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
    l_rms_keys = ['i_HF1', 'i_Ls', 'i_Lc1', 'i_Lc2', 'i_C11', 'i_C12', 'i_C23', 'i_C24']
    l_min_keys = ['v_ds_S11_sw_on', 'v_ds_S23_sw_on', 'i_HF1_S11_sw_on', 'i_HF2_S23_sw_on']
    # values calculated from simulation results (only for overview of return dict keys)
    l_calc_keys = ['power_deviation', 'zvs_coverage', 'zvs_coverage1', 'zvs_coverage2',
                   'zvs_coverage_notnan', 'zvs_coverage1_notnan', 'zvs_coverage2_notnan',
                   'i_HF1_total_mean', 'I1_squared_total_mean']
    # Set a reasonable low zvs voltage limit below we assume zvs operation
    zvs_vlimit = 50

    # Init arrays to store simulation results
    da_sim_results = dict()
    for k in l_means_keys:
        da_sim_results[k] = np.full_like(mod_phi, np.nan)
    for k in l_rms_keys:
        da_sim_results[k] = np.full_like(mod_phi, np.nan)
    for k in l_min_keys:
        da_sim_results[k] = np.full_like(mod_phi, np.nan)

    # Progressbar init, calc total number of iterations to simulate
    pbar = tqdm(total=mod_phi.size)

    if not __debug__:
        # Find a free port if zero is given as port
        if geckoport == 0:
            geckoport = get_free_port()
        # Gecko Basics
        gecko_dab_converter = pgc.GeckoSimulation(simfilepath=simfilepath, geckoport=geckoport, debug=gdebug,
                                                  timestep=timestep, simtime=simtime, timestep_pre=timestep_pre, simtime_pre=simtime_pre)

    for vec_vvp in np.ndindex(mod_phi.shape):
        # set simulation parameters and convert tau to degree for Gecko
        sim_params = {
            'v_dc1': mesh_V1[vec_vvp].item(),
            'v_dc2': mesh_V2[vec_vvp].item(),
            'phi': mod_phi[vec_vvp].item() / np.pi * 180,
            'tau1': mod_tau1[vec_vvp].item() / np.pi * 180,
            'tau2': mod_tau2[vec_vvp].item() / np.pi * 180,
            't_dead1': mesh_t_dead1[vec_vvp].item(),
            't_dead2': mesh_t_dead2[vec_vvp].item(),
            'fs': mesh_fs[vec_vvp].item(),
            'Ls': float(Ls),
            'Lc1': float(Lc1),
            'Lc2_': float(Lc2_),
            'n': float(n),
            'temp': float(temp),
            'C_par_1': c_par_1,
            'C_par_2': c_par_2,
        }

        # Only run simulation if all params are valid
        if not np.any(np.isnan(list(sim_params.values()))):
            # start simulation for this operation point
            c_oss_1_file = os.path.join(lossfilepath, f'{transistor_1_name}_c_oss.nlc')
            c_oss_2_file = os.path.join(lossfilepath, f'{transistor_2_name}_c_oss.nlc')

            if not __debug__:
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
            else:
                # generate some fake data for debugging
                values_mean = defaultdict(dict)
                values_rms = defaultdict(dict)
                values_min = defaultdict(dict)
                for k in l_means_keys:
                    values_mean['mean'][k] = np.random.uniform(0.0, 100)
                for k in l_rms_keys:
                    values_rms['rms'][k] = np.random.uniform(0.0, 1)
                for k in l_min_keys:
                    values_min['min'][k] = np.random.uniform(-5, 50)
        else:
            # When params are not valid return NaN values
            values_mean = defaultdict(dict)
            values_rms = defaultdict(dict)
            values_min = defaultdict(dict)
            for k in l_means_keys:
                values_mean['mean'][k] = np.nan
            for k in l_rms_keys:
                values_rms['rms'][k] = np.nan
            for k in l_min_keys:
                values_min['min'][k] = np.nan

        # save simulation results in arrays
        for k in l_means_keys:
            da_sim_results[k][vec_vvp] = values_mean['mean'][k]
        for k in l_rms_keys:
            da_sim_results[k][vec_vvp] = values_rms['rms'][k]
        for k in l_min_keys:
            da_sim_results[k][vec_vvp] = values_min['min'][k]
        # info(values_mean)

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

    return da_sim_results


def get_free_port(start=43047, stop=50000) -> int:
    """
    Get a random free port in Range start <= port < stop.

    :param start: start port
    :param stop: stop port
    :return:
    """
    # ss -tulpn example for a gecko run:
    # tcp   LISTEN   0  50    *:43036   *:*   users:(("java",pid=1532749,fd=57))

    random.seed()
    # port = random.randint(start, stop)
    port = random.randrange(start, stop)
    port_in_use = 0
    while not port_in_use:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        port_in_use = sock.connect_ex(('127.0.0.1', port))
        sock.close()
        if port_in_use == 0:
            port = random.randint(start, stop)
    return port

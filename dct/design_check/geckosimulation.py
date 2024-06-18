"""Validate the results using GeckoCIRCUITS."""
# python libraries
import math
import socket
import random

# own libraries
from dct.debug_tools import *

# 3rd party libraries
import numpy as np
from collections import defaultdict
# For threads: run parallel in single process
# from threading import Thread, Lock
import threading as td
# For processes: run parallel on multiple cpu's
# from multiprocessing import Process, Lock
import multiprocessing as mp
# Status bar
from tqdm import tqdm
import pygeckocircuits2 as pgc

class GeckoSimulation:
    """
    This class is intended to run multiple Gecko instances in threads oder multiple processes.

    As of now the concept kind of works and is very promising but there are some Java errors
    that prevent running the Java process in parallel.
    """

    # mean values we want to get from the simulation
    l_means_keys = ['p_dc1', 'p_dc2', 'S11_p_sw', 'S11_p_cond', 'S12_p_sw', 'S12_p_cond', 'S21_p_sw', 'S21_p_cond',
                    'S22_p_sw', 'S22_p_cond']
    l_rms_keys = ['i_Ls', 'i_Lm', 'v_dc1', 'i_dc1', 'v_dc2', 'i_dc2', 'i_C11', 'i_C12', 'i_C21', 'i_C22']
    mutex: td.Lock = None
    pbar: tqdm = None

    def __init__(self):
        # Number of threads or processes in parallel
        self.thread_count = 3
        # Init dict to store simulation result arrays
        self.da_sim_results = dict()

    @timeit
    def start_sim_threads(self, mesh_V1: np.ndarray, mesh_V2: np.ndarray,
                          mod_phi: np.ndarray, mod_tau1: np.ndarray, mod_tau2: np.ndarray,
                          simfilepath: str, timestep: float = None, simtime: float = None,
                          timestep_pre: float = 0, simtime_pre: float = 0, geckoport: int = 43036,
                          gdebug: bool = False) -> dict:
        """Start threads for parallel GeckoCIRCUITS simulations."""
        # Init arrays to store simulation results
        for k in self.l_means_keys:
            self.da_sim_results[k] = np.full_like(mod_phi, np.nan)
        for k in self.l_rms_keys:
            self.da_sim_results[k] = np.full_like(mod_phi, np.nan)

        # Progressbar init
        # Calc total number of iterations to simulate
        it_total = mod_phi.size
        self.pbar = tqdm(total=it_total)

        # ************ Gecko Start **********

        self.mutex = td.Lock()
        threads = []
        # Start the worker threads
        for i in range(self.thread_count):
            kwargs = {'mesh_V1': mesh_V1, 'mesh_V2': mesh_V2, 'mod_phi': mod_phi, 'mod_tau1': mod_tau1,
                      'mod_tau2': mod_tau2, 'simfilepath': simfilepath, 'timestep': timestep,
                      'simtime': simtime, 'timestep_pre': timestep_pre, 'simtime_pre': simtime_pre,
                      'geckoport': geckoport + i, 'gdebug': gdebug}
            t = td.Thread(target=self._start_sim_single, kwargs=kwargs, name=str(i))
            t.start()
            threads.append(t)

        # Wait for the threads to complete
        for t in threads:
            t.join()

        # ************ Gecko End **********

        # Progressbar end
        self.pbar.close()

        # Rename the keys according to convention
        # da_sim_results_temp = dict()
        # for k, v in self.da_sim_results.items():
        #     da_sim_results_temp['sim_' + k] = v
        # self.da_sim_results = da_sim_results_temp

        debug(self.da_sim_results)
        return self.da_sim_results

    @timeit
    def start_sim_multi(self, mesh_V1: np.ndarray, mesh_V2: np.ndarray,
                        mod_phi: np.ndarray, mod_tau1: np.ndarray, mod_tau2: np.ndarray,
                        simfilepath: str, timestep: float = None, simtime: float = None,
                        timestep_pre: float = 0, simtime_pre: float = 0, geckoport: int = 43036,
                        gdebug: bool = False) -> dict:
        """Start multiple GeckoCIRCUITS simulations in parallel."""
        # Init arrays to store simulation results
        for k in self.l_means_keys:
            self.da_sim_results[k] = np.full_like(mod_phi, np.nan)
        for k in self.l_rms_keys:
            self.da_sim_results[k] = np.full_like(mod_phi, np.nan)

        # Progressbar init
        # Calc total number of iterations to simulate
        it_total = mod_phi.size
        self.pbar = tqdm(total=it_total)

        # ************ Gecko Start **********

        self.mutex = mp.Lock()
        processes = []
        # Start the worker threads
        for i in range(self.thread_count):
            kwargs = {'mesh_V1': mesh_V1, 'mesh_V2': mesh_V2, 'mod_phi': mod_phi, 'mod_tau1': mod_tau1,
                      'mod_tau2': mod_tau2, 'simfilepath': simfilepath, 'timestep': timestep,
                      'simtime': simtime, 'timestep_pre': timestep_pre, 'simtime_pre': simtime_pre,
                      'geckoport': geckoport + i, 'gdebug': gdebug}
            t = mp.Process(target=self._start_sim_single, kwargs=kwargs)
            t.start()
            processes.append(t)

        # Wait for the threads to complete
        for t in processes:
            t.join()

        # kwargs = {'mesh_V1':   mesh_V1, 'mesh_V2': mesh_V2, 'mod_phi': mod_phi, 'mod_tau1': mod_tau1,
        #           'mod_tau2':  mod_tau2, 'simfilepath': simfilepath, 'timestep': timestep,
        #           'simtime':   simtime, 'timestep_pre': timestep_pre, 'simtime_pre': simtime_pre,
        #           'geckoport': geckoport + 1, 'gdebug': gdebug}
        # self._start_sim_single(**kwargs)

        # self._start_sim_single(mesh_V1, mesh_V2, mod_phi, mod_tau1, mod_tau2, simfilepath, timestep, simtime, timestep_pre, simtime_pre, geckoport)

        # ************ Gecko End **********

        # Progressbar end
        self.pbar.close()

        # Rename the keys according to convention
        # da_sim_results_temp = dict()
        # for k, v in self.da_sim_results.items():
        #     da_sim_results_temp['sim_' + k] = v
        # self.da_sim_results = da_sim_results_temp

        debug(self.da_sim_results)
        return self.da_sim_results

    def _start_sim_single(self, mesh_V1: np.ndarray, mesh_V2: np.ndarray,
                          mod_phi: np.ndarray, mod_tau1: np.ndarray, mod_tau2: np.ndarray,
                          simfilepath: str, timestep: float = None, simtime: float = None,
                          timestep_pre: float = 0, simtime_pre: float = 0, geckoport: int = 43036,
                          gdebug: bool = False):

        info(geckoport, simfilepath)

        # ************ Gecko Start **********

        try:
            # use pyjnius here

            if not __debug__:
                # Gecko Basics
                dab_converter = pgc.GeckoSimulation(simfilepath=simfilepath, geckoport=geckoport, debug=gdebug)

            for vec_vvp in np.ndindex(mod_phi.shape):
                # debug(vec_vvp, mod_phi[vec_vvp], mod_tau1[vec_vvp], mod_tau2[vec_vvp], sep='\n')

                # set simulation parameters and convert tau to inverse-tau for Gecko
                sim_params = {
                    # TODO find a way to do this with sparse arrays
                    'v_dc1': mesh_V1[vec_vvp].item(),
                    'v_dc2': mesh_V2[vec_vvp].item(),
                    'phi': mod_phi[vec_vvp].item() / np.pi * 180,
                    'tau1': mod_tau1[vec_vvp].item() / np.pi * 180,
                    'tau2': mod_tau2[vec_vvp].item() / np.pi * 180
                }
                # debug(sim_params)

                # start simulation for this operation point
                # TODO optimize for multithreading, maybe multiple Gecko instances needed
                if not __debug__:
                    dab_converter.set_global_parameters(sim_params)
                if not __debug__:
                    # TODO time settings should be variable
                    # dab_converter.run_simulation(timestep=100e-12, simtime=15e-6, timestep_pre=50e-9, simtime_pre=10e-3)
                    # TODO Bug in LPT with _pre settings! Does this still run a pre-simulation like in the model?
                    # Start the simulation and get the results
                    dab_converter.run_simulation(timestep=timestep, simtime=simtime)
                    values_mean = dab_converter.get_values(
                        nodes=self.l_means_keys,
                        operations=['mean']
                    )
                    values_rms = dab_converter.get_values(
                        nodes=self.l_rms_keys,
                        operations=['rms']
                    )
                else:
                    # generate some fake data for debugging
                    # values_mean = {'mean': {'p_dc1':      np.random.uniform(0.0, 1000),
                    #                         'S11_p_sw':   np.random.uniform(0.0, 10),
                    #                         'S11_p_cond': np.random.uniform(0.0, 10),
                    #                         'S12_p_sw':   np.random.uniform(0.0, 1000),
                    #                         'S12_p_cond': np.random.uniform(0.0, 100)}}
                    # values_rms = {'rms': {'i_Ls': np.random.uniform(0.0, 10)}}
                    values_mean = defaultdict(dict)
                    values_rms = defaultdict(dict)
                    for k in self.l_means_keys:
                        values_mean['mean'][k] = np.random.uniform(0.0, 1)
                    for k in self.l_rms_keys:
                        values_rms['rms'][k] = np.random.uniform(0.0, 1)

                # ***** LOCK Start *****
                self.mutex.acquire()
                # save simulation results in arrays
                for k in self.l_means_keys:
                    self.da_sim_results[k][vec_vvp] = values_mean['mean'][k]
                for k in self.l_rms_keys:
                    self.da_sim_results[k][vec_vvp] = values_rms['rms'][k]

                # Progressbar update, default increment +1
                self.pbar.update()
                self.mutex.release()
                # ***** LOCK End *****

            # dab_converter.__del__()

        finally:
            # jnius.detach()
            pass
        # ************ Gecko End **********

def start_sim(mesh_V1: np.ndarray, mesh_V2: np.ndarray, mesh_P: np.ndarray,
              mod_phi: np.ndarray, mod_tau1: np.ndarray, mod_tau2: np.ndarray,
              t_dead1: float | np.ndarray, t_dead2: float | np.ndarray, fs: int | np.ndarray,
              Ls: float, Lc1: float, Lc2: float, n: float,
              temp: float,
              simfilepath: str, timestep: float, simtime: float,
              timestep_pre: float = 0, simtime_pre: float = 0, geckoport: int = 43036, gdebug: bool = False,
              C_HB1: float = None, C_HB2: float = None) -> dict:
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
    :param C_HB1:
    :param C_HB2:
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

    # Progressbar init
    # Calc total number of iterations to simulate
    it_total = mod_phi.size
    pbar = tqdm(total=it_total)

    # ************ Gecko Start **********
    # TODO optimize for multithreading, maybe multiple Gecko instances needed
    if not __debug__:
        # Find a free port if zero is given as port
        if geckoport == 0:
            geckoport = get_free_port()
        # Gecko Basics
        print(f"{simfilepath=}")
        dab_converter = pgc.GeckoSimulation(simfilepath=simfilepath, geckoport=geckoport, debug=gdebug,
                                            timestep=timestep, simtime=simtime,
                                            timestep_pre=timestep_pre, simtime_pre=simtime_pre)

    for vec_vvp in np.ndindex(mod_phi.shape):
        # debug(vec_vvp, mod_phi[vec_vvp], mod_tau1[vec_vvp], mod_tau2[vec_vvp], sep='\n')

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
            # to disable set this to very high value not to zero or inf
            'Lc1': float(Lc1),
            # to disable set this to very high value not to zero or inf
            'Lc2_': float(Lc2_),
            'n': float(n),
            'temp': float(temp)
        }
        if C_HB1 is not None:
            sim_params['C_HB11'] = C_HB1
            sim_params['C_HB12'] = C_HB1
        if C_HB2 is not None:
            sim_params['C_HB21'] = C_HB2
            sim_params['C_HB22'] = C_HB2
        # info(sim_params)

        # Only run simulation if all params are valid
        if not np.any(np.isnan(list(sim_params.values()))):
            # start simulation for this operation point
            if not __debug__:
                dab_converter.set_global_parameters(sim_params)
                # Start the simulation and get the results. Do not set times here because it is set while init Gecko.
                dab_converter.run_simulation()
                values_mean = dab_converter.get_values(
                    nodes=l_means_keys,
                    operations=['mean'],
                    # Just use the last part, because the beginning is garbage
                    # simtime must be greater than 2*Ts
                    # Use only second half of simtime and make it multiple of Ts
                    range_start_stop=[math.ceil(simtime * mesh_fs[vec_vvp].item() / 2) * 1 / mesh_fs[vec_vvp].item(),
                                      'end']
                )
                values_rms = dab_converter.get_values(
                    nodes=l_rms_keys,
                    operations=['rms'],
                    # Just use the last part, because the beginning is garbage
                    # simtime must be greater than 2*Ts
                    # Use only second half of simtime and make it multiple of Ts
                    range_start_stop=[math.ceil(simtime * mesh_fs[vec_vvp].item() / 2) * 1 / mesh_fs[vec_vvp].item(),
                                      'end']
                )
                values_min = dab_converter.get_values(
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
        dab_converter.__del__()
    # ************ Gecko End **********

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

    # Rename the keys according to convention
    # da_sim_results_temp = dict()
    # for k, v in da_sim_results.items():
    #     da_sim_results_temp['sim_' + k] = v
    # da_sim_results = da_sim_results_temp

    # info(da_sim_results)
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
        #     print('Port {} is open, used'.format(port))
        # else:
        #     print('Port {} is closed, free to use'.format(port))
    return port


def next_free_port(port=1024, max_port=65535) -> int:
    """Get next free port of GeckoCIRCUITS."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    while port <= max_port:
        try:
            sock.bind(('', port))
            sock.close()
            return port
        except OSError:
            port += 1
    raise IOError('no free ports')

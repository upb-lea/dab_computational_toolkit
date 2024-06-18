"""Validate results using gecko simulations."""
# python libraries
import copy
import os
import datetime

# 3rd party libraries
import numpy as np
import transistordatabase as tdb
import argparse


# own libraries
import dct
import dct.design_check.datasets as ds
from dct.debug_tools import *
import dct.debug_tools as db
import dct.design_check.mod_zvs as mod_zvs
import dct.design_check.geckosimulation as sim_gecko
import dct.design_check.plot_dab as plot_dab


def gecko_simulate_zvs_modulation(design_specification: str,
                                  transistor_1: tdb.Transistor, transistor_2: tdb.Transistor):
    """
    Run GeckoCIRCUITS simulations for the given design_specification for all operating points.

    :param design_specification: Dual-Active Bridge (DAB) design configuration
    :param transistor_1: Transistor bridge 1
    :param transistor_2: Transistor bridge 2
    :return:
    """
    # Load the DAB Specification from the given input
    dab = dct.load_dab_specification(design_specification)

    dab.coss_1 = transistor_1.wp.graph_v_qoss[1]
    dab.coss_2 = transistor_2.wp.graph_v_qoss[1]

    dab.coss_1 = dab.coss_1 + dab.C_HB11
    dab.coss_2 = dab.coss_2 + dab.C_HB22

    # Set sim defaults
    simfilepath = '../circuits/DAB_MOSFET_Modulation_v8.ipes'
    simfilepath = os.path.abspath(simfilepath)
    timestep = 1e-9
    simtime = 50e-6
    timestep_pre = 50e-9
    simtime_pre = 5e-3
    # Automatically select a free port
    geckoport = 0
    # Set file names
    directory = '../dab_modulation_output/'
    name_extra = 'n3_tdead1_100_Coss2'
    name_L = 'Ls_{}uH__Lc1_{}uH__Lc2_{}uH'.format(int(dab.Ls * 1e6),
                                                  int(dab.Lc1 * 1e6),
                                                  int(dab.Lc2 * 1e6))
    name = 'sim_Gv8_' + name_extra + '_' + name_L + '_v{}-v{}-p{}'.format(int(dab.V1_step),
                                                                          int(dab.V2_step),
                                                                          int(dab.P_step))
    name_pre = 'mod_zvs_'
    if __debug__:
        name_pre = 'debug_' + name_pre
    comment = 'Simulation results for mod_zvs with V1 {}, V2 {} and P {} steps.'.format(
        int(dab.V1_step),
        int(dab.V2_step),
        int(dab.P_step))
    comment = comment + '\n' + 'Using simfilepath = ' + simfilepath
    if __debug__:
        comment = 'Debug ' + comment

    # Saving
    # Create new dir for all files
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.mkdir(directory)

    directory = directory + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + name_pre + name
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.mkdir(directory)
    directory = os.path.expanduser(directory)
    directory = os.path.expandvars(directory)
    directory = os.path.abspath(directory)
    if not os.path.exists(directory):
        os.mkdir(directory)
    # Logging
    log = db.log(filename=os.path.join(directory, 'dab_opt.log'))

    # Calculate ZVS modulation parameters and append it to the given dict.
    log.info('Starting ZVS Modulation...')
    da_mod = mod_zvs.calc_modulation(dab.n, dab.Ls, dab.Lc1, dab.Lc2, dab.fs, dab.coss_1, dab.coss_2, dab.mesh_V1, dab.mesh_V2, dab.mesh_P)
    dab.append_result_dict(da_mod, name_pre='mod_zvs_')
    # Perform the GeckoCIRCUITS simulation
    da_sim = sim_gecko.start_sim(dab.mesh_V1,
                                 dab.mesh_V2,
                                 dab.mesh_P,
                                 dab.mod_zvs_phi,
                                 dab.mod_zvs_tau1,
                                 dab.mod_zvs_tau2,
                                 dab.t_dead1,
                                 dab.t_dead2,
                                 dab.fs,
                                 dab.Ls,
                                 dab.Lc1,
                                 dab.Lc2,
                                 dab.n,
                                 dab.temp,
                                 simfilepath, timestep, simtime, timestep_pre, simtime_pre, geckoport=geckoport)
    # Unpack the results from the GeckoCIRCUITS simulation
    dab.append_result_dict(da_sim, name_pre='sim_zvs_')

    # Show ZVS coverage based on calculation
    log.info('Calculation ZVS coverage: {}'.format(dab.mod_zvs_zvs_coverage))
    # Show ZVS coverage based on simulation
    log.info(f'Simulation ZVS coverage (Bridge 1, Bridge 2): {dab.sim_zvs_zvs_coverage} ({dab.sim_zvs_zvs_coverage1}, {dab.sim_zvs_zvs_coverage2})')

    # Only non NaN areas:
    # Show ZVS coverage based on calculation
    log.info('Calculation ZVS coverage (non NaN): {}'.format(dab.mod_zvs_zvs_coverage_notnan))
    # Show ZVS coverage based on simulation
    log.info('Simulation ZVS coverage (Bridge 1, Bridge 2) (non NaN): {} ({}, {})'.format(
        dab.sim_zvs_zvs_coverage_notnan, dab.sim_zvs_zvs_coverage1_notnan,
        dab.sim_zvs_zvs_coverage2_notnan))

    # Save data
    # ds.save_to_file(dab, directory=directory, name=name, comment=comment)
    dab.save_to_file(directory=directory, name='dab_' + name, comment=comment, timestamp=False)
    dab.pprint_to_file(os.path.join(directory, 'dab_' + name + '.txt'))

    # Save to csv for DAB-Controller
    # Meshes to save:
    keys = ['mod_zvs_phi', 'mod_zvs_tau1', 'mod_zvs_tau2']
    # Convert phi, tau1/2 from rad to duty cycle * 10000
    # In DAB-Controller we need duty cycle * 10000 (2pi eq. 10000)
    # Copy dab first
    dab2 = copy.deepcopy(dab)
    for key in keys:
        dab2[key] = dab2[key] / (2 * np.pi) * 10000
        ds.save_to_csv(dab2, key, directory, 'control_' + name, timestamp=False)
    # Convert phi, tau1/2 from rad to degree
    # Copy dab first
    dab2 = copy.deepcopy(dab)
    for key in keys:
        dab2[key] = dab2[key] / (2 * np.pi) * 360
        ds.save_to_csv(dab2, key, directory, 'degree_' + name, timestamp=False)

    # Plotting
    plot_mod_sim(dab, name, comment, directory, ['zvs'])


@timeit
def plot_mod_sim(dab_config, simulation_name: str, comment: str, directory: str, mod_keys: list[str],
                 show_plot: bool = True, logfile=str()):
    """
    Plot the results from the GeckoCIRCUITS simulation.

    :param dab_config: Dual-Active-Bridge (DAB) configuration file
    :param simulation_name: Simulation name
    :param comment: Comment for the simulation, appears in the logs.
    :param directory: Directory to plot
    :param mod_keys: modulation names, e.g. ['zvs', 'mpl']
    :param show_plot: True (default) to show the plots.
    :param logfile: Logfile name
    :return:
    """
    # Logging
    log = db.log(filename=os.path.join(directory, logfile) if logfile else '')
    # Plotting
    log.info("\nStart Plotting\n")
    debug(mod_keys)

    # When dim=1 the v1_middle calc does not work.
    # Therefore, we stretch the array to use the same algo for every data.
    if dab_config.mesh_V1.shape[1] == 1:
        # Broadcast arrays for plotting
        for k, _ in dab_config.items():
            if k.startswith(('mesh_', 'mod_', 'sim_')):
                dab_config[k] = np.broadcast_to(dab_config[k], (dab_config.V2_step, 3, dab_config.P_step))

    # Plot a cross-section through the V1 plane
    v1_middle = int(np.shape(dab_config.mesh_P)[1] / 2)
    log.info('View plane: U_1 = {:.1f}V'.format(dab_config.mesh_V1[0, v1_middle, 0]))
    simulation_name += '_V1_{:.0f}V'.format(dab_config.mesh_V1[0, v1_middle, 0])
    comment += ' View plane: V_1 = {:.1f}V'.format(dab_config.mesh_V1[0, v1_middle, 0])

    plt = plot_dab.Plot_DAB(latex=False, show=show_plot, figsize=(15, 5), fontsize=22)

    for modulation_name in mod_keys:
        log.info('Plotting modulation: ' + modulation_name)
        # Show ZVS coverage based on simulation:
        log.info(modulation_name + ' Simulation ZVS coverage (Bridge 1, Bridge 2): {} ({}, {})'.format(
            round(np.array(dab_config['sim_' + modulation_name + '_zvs_coverage']).item(0), 3),
            round(np.array(dab_config['sim_' + modulation_name + '_zvs_coverage1']).item(0), 3),
            round(np.array(dab_config['sim_' + modulation_name + '_zvs_coverage2']).item(0), 3)))
        # Only non NaN areas:
        log.info(modulation_name + ' Simulation ZVS coverage (Bridge 1, Bridge 2) (non NaN): {} ({}, {})'.format(
            round(np.array(dab_config['sim_' + modulation_name + '_zvs_coverage_notnan']).item(0), 3),
            round(np.array(dab_config['sim_' + modulation_name + '_zvs_coverage1_notnan']).item(0), 3),
            round(np.array(dab_config['sim_' + modulation_name + '_zvs_coverage2_notnan']).item(0), 3)))
        # Mean of I1:
        log.info(modulation_name + ' Simulation I_1-total-mean: {}'.format(
            round(np.array(dab_config['sim_' + modulation_name + '_i_HF1_total_mean']).item(0), 3)))
        log.info(modulation_name + ' Simulation I^2_1-total-mean: {}'.format(
            round(np.array(dab_config['sim_' + modulation_name + '_I1_squared_total_mean']).item(0), 3)))

        # Set masks according to mod for later usage
        match modulation_name:
            case 'sps':
                mask1 = None
                mask2 = None
                mask3 = None
                maskZVS = None
            case 'mcl':
                mask1 = dab_config['mod_' + modulation_name + '_mask_tcm'][:, v1_middle, :]
                mask2 = dab_config['mod_' + modulation_name + '_mask_cpm'][:, v1_middle, :]
                mask3 = None
                maskZVS = None
            case s if s.startswith('zvs'):
                # Hide less useful masks
                mask1 = None
                mask2 = None
                # mask1 = dab['mod_' + m + '_mask_Im2'][:, v1_middle, :]
                # mask2 = dab['mod_' + m + '_mask_IIm2'][:, v1_middle, :]
                mask3 = dab_config['mod_' + modulation_name + '_mask_IIIm1'][:, v1_middle, :]
                maskZVS = dab_config['mod_' + modulation_name + '_mask_zvs'][:, v1_middle, :]
            case _:
                mask1 = None
                mask2 = None
                mask3 = None
                maskZVS = None

        # Plot all modulation angles
        plt.plot_modulation(dab_config.mesh_P[:, v1_middle, :],
                            dab_config.mesh_V2[:, v1_middle, :],
                            dab_config['mod_' + modulation_name + '_phi'][:, v1_middle, :],
                            dab_config['mod_' + modulation_name + '_tau1'][:, v1_middle, :],
                            dab_config['mod_' + modulation_name + '_tau2'][:, v1_middle, :],
                            mask1=mask1,
                            mask2=mask2,
                            mask3=mask3,
                            maskZVS=maskZVS,
                            tab_title=modulation_name + ' Modulation Angles'
                            )
        fname = 'mod_' + modulation_name + '_' + simulation_name + '_' + 'fig1'
        plt.save_fig(plt.figs_axes[-1][0], directory, fname, comment, timestamp=False)

        # v_ds plots
        plt.new_fig(nrows=1, ncols=2, tab_title=modulation_name + ' ZVS')
        plt.subplot_contourf(dab_config.mesh_P[:, v1_middle, :],
                             dab_config.mesh_V2[:, v1_middle, :],
                             dab_config['sim_' + modulation_name + '_v_ds_S11_sw_on'][:, v1_middle, :] / dab_config.mesh_V1[:, v1_middle, :],
                             mask1=mask1,
                             mask2=mask2,
                             mask3=mask3,
                             z_min=0,
                             z_max=1,
                             ax=plt.figs_axes[-1][1][0],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             ylabel=r'$U_\mathrm{DC2} \:/\: \mathrm{V}$',
                             title=r'$u_\mathrm{DS,S11,sw-on} \:/\: U_\mathrm{DC1}$')
        plt.subplot_contourf(dab_config.mesh_P[:, v1_middle, :],
                             dab_config.mesh_V2[:, v1_middle, :],
                             dab_config['sim_' + modulation_name + '_v_ds_S23_sw_on'][:, v1_middle, :] / dab_config.mesh_V2[:, v1_middle, :],
                             mask1=mask1,
                             mask2=mask2,
                             mask3=mask3,
                             z_min=0,
                             z_max=1,
                             ax=plt.figs_axes[-1][1][1],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             title=r'$u_\mathrm{DS,S23,sw-on} \:/\: U_\mathrm{DC2}$')
        fname = 'mod_' + modulation_name + '_' + simulation_name + '_' + 'fig2'
        plt.save_fig(plt.figs_axes[-1][0], directory, fname, comment, timestamp=False)

        # i_l plots 'i_HF1_S11_sw_on', 'i_HF2_S23_sw_on'
        plt.new_fig(nrows=1, ncols=2, tab_title=modulation_name + ' i_L')
        plt.subplot_contourf(dab_config.mesh_P[:, v1_middle, :],
                             dab_config.mesh_V2[:, v1_middle, :],
                             dab_config['sim_' + modulation_name + '_i_HF1_S11_sw_on'][:, v1_middle, :],
                             mask1=mask1,
                             mask2=mask2,
                             mask3=mask3,
                             ax=plt.figs_axes[-1][1][0],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             ylabel=r'$U_\mathrm{DC2} \:/\: \mathrm{V}$',
                             title=r'$i_\mathrm{1,S11,sw-on} \:/\: \mathrm{A}$')
        plt.subplot_contourf(dab_config.mesh_P[:, v1_middle, :],
                             dab_config.mesh_V2[:, v1_middle, :],
                             dab_config['sim_' + modulation_name + '_i_HF2_S23_sw_on'][:, v1_middle, :],
                             mask1=mask1,
                             mask2=mask2,
                             mask3=mask3,
                             ax=plt.figs_axes[-1][1][1],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             title=r'$i_\mathrm{2,S23,sw-on} \:/\: \mathrm{A}$')
        fname = 'mod_' + modulation_name + '_' + simulation_name + '_' + 'fig3'
        plt.save_fig(plt.figs_axes[-1][0], directory, fname, comment, timestamp=False)

        # Total loss
        plt.new_fig(nrows=1, ncols=3, tab_title=modulation_name + ' Total Loss')
        plt.subplot_contourf(dab_config.mesh_P[:, v1_middle, :],
                             dab_config.mesh_V2[:, v1_middle, :],
                             dab_config['sim_' + modulation_name + '_power_deviation'][:, v1_middle, :],
                             mask1=mask1,
                             mask2=mask2,
                             mask3=mask3,
                             ax=plt.figs_axes[-1][1][0],
                             z_min=0.5,
                             z_max=1.5,
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             ylabel=r'$U_\mathrm{DC2} \:/\: \mathrm{V}$',
                             title=r'$P_\mathrm{out,Sim} \:/\: P_\mathrm{out,desired}$')
        plt.subplot_contourf(dab_config.mesh_P[:, v1_middle, :],
                             dab_config.mesh_V2[:, v1_middle, :],
                             dab_config['sim_' + modulation_name + '_p_sw1'][:, v1_middle, :] + dab_config['sim_' + modulation_name + '_p_sw2'][:, v1_middle, :],
                             mask1=mask1,
                             mask2=mask2,
                             mask3=mask3,
                             ax=plt.figs_axes[-1][1][1],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             title=r'$P_\mathrm{sw,total} \:/\: \mathrm{W}$')
        plt.subplot_contourf(dab_config.mesh_P[:, v1_middle, :],
                             dab_config.mesh_V2[:, v1_middle, :],
                             dab_config['sim_' + modulation_name + '_p_cond1'][:, v1_middle, :] + dab_config['sim_' + modulation_name + '_p_cond2'][:, v1_middle, :],
                             mask1=mask1,
                             mask2=mask2,
                             mask3=mask3,
                             ax=plt.figs_axes[-1][1][2],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             title=r'$P_\mathrm{cond,total} \:/\: \mathrm{W}$')
        fname = 'mod_' + modulation_name + '_' + simulation_name + '_' + 'fig4'
        plt.save_fig(plt.figs_axes[-1][0], directory, fname, comment, timestamp=False)

        # Plot power loss
        plt.new_fig(nrows=1, ncols=4, tab_title=modulation_name + ' Power Loss')
        plt.subplot_contourf(dab_config.mesh_P[:, v1_middle, :],
                             dab_config.mesh_V2[:, v1_middle, :],
                             dab_config['sim_' + modulation_name + '_S11_p_sw'][:, v1_middle, :],
                             mask1=mask1,
                             mask2=mask2,
                             mask3=mask3,
                             ax=plt.figs_axes[-1][1][0],
                             xlabel=r'$P \:/\: \mathrm{W}$', ylabel=r'$U_\mathrm{DC2} \:/\: \mathrm{V}$',
                             title=r'$P_\mathrm{S11,sw} \:/\: \mathrm{W}$')
        plt.subplot_contourf(dab_config.mesh_P[:, v1_middle, :],
                             dab_config.mesh_V2[:, v1_middle, :],
                             dab_config['sim_' + modulation_name + '_S11_p_cond'][:, v1_middle, :],
                             mask1=mask1,
                             mask2=mask2,
                             mask3=mask3,
                             ax=plt.figs_axes[-1][1][1],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             title=r'$P_\mathrm{S11,cond} \:/\: \mathrm{W}$')
        plt.subplot_contourf(dab_config.mesh_P[:, v1_middle, :],
                             dab_config.mesh_V2[:, v1_middle, :],
                             dab_config['sim_' + modulation_name + '_S23_p_sw'][:, v1_middle, :],
                             mask1=mask1,
                             mask2=mask2,
                             mask3=mask3,
                             ax=plt.figs_axes[-1][1][2],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             title=r'$P_\mathrm{S23,sw} \:/\: \mathrm{W}$')
        plt.subplot_contourf(dab_config.mesh_P[:, v1_middle, :],
                             dab_config.mesh_V2[:, v1_middle, :],
                             dab_config['sim_' + modulation_name + '_S23_p_cond'][:, v1_middle, :],
                             mask1=mask1,
                             mask2=mask2,
                             mask3=mask3,
                             ax=plt.figs_axes[-1][1][3],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             title=r'$P_\mathrm{S23,cond} \:/\: \mathrm{W}$')
        fname = 'mod_' + modulation_name + '_' + simulation_name + '_' + 'fig5'
        plt.save_fig(plt.figs_axes[-1][0], directory, fname, comment, timestamp=False)

        # Plot inductor currents
        plt.new_fig(nrows=1, ncols=3, tab_title=modulation_name + ' Inductor currents')
        plt.subplot_contourf(dab_config.mesh_P[:, v1_middle, :],
                             dab_config.mesh_V2[:, v1_middle, :],
                             dab_config['sim_' + modulation_name + '_i_HF1'][:, v1_middle, :],
                             mask1=mask1,
                             mask2=mask2,
                             mask3=mask3,
                             ax=plt.figs_axes[-1][1][0],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             ylabel=r'$U_\mathrm{DC2} \:/\: \mathrm{V}$',
                             title=r'$I_\mathrm{1} \:/\: \mathrm{A}$')
        plt.subplot_contourf(dab_config.mesh_P[:, v1_middle, :],
                             dab_config.mesh_V2[:, v1_middle, :],
                             dab_config['sim_' + modulation_name + '_i_Lc1'][:, v1_middle, :],
                             mask1=mask1,
                             mask2=mask2,
                             mask3=mask3,
                             ax=plt.figs_axes[-1][1][1],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             title=r'$I_\mathrm{L1} \:/\: \mathrm{A}$')
        plt.subplot_contourf(dab_config.mesh_P[:, v1_middle, :],
                             dab_config.mesh_V2[:, v1_middle, :],
                             dab_config['sim_' + modulation_name + '_i_Lc2'][:, v1_middle, :] / dab_config.n,
                             mask1=mask1,
                             mask2=mask2,
                             mask3=mask3,
                             ax=plt.figs_axes[-1][1][2],
                             xlabel=r'$P \:/\: \mathrm{W}$',
                             title=r'$I^\prime_\mathrm{L2} \:/\: \mathrm{A}$')
        fname = 'mod_' + modulation_name + '_' + simulation_name + '_' + 'fig6'
        plt.save_fig(plt.figs_axes[-1][0], directory, fname, comment, timestamp=False)
        if not show_plot:
            plt.close()

    info('Plotting is done!')
    # Finally show everything
    if show_plot:
        plt.show()
    else:
        plt.close()


def main_init() -> None:
    """
    Parse arguments for terminal usage.

    :return: None
    """
    parser = argparse.ArgumentParser()
    # parser.add_argument("configfile", help="config file")
    parser.add_argument("-l", help="Log to file: <datetime>_<name>.log with -l <name>")
    parser.add_argument("-d", help="Set log output to debug level", action="store_true")
    args = parser.parse_args()


if __name__ == '__main__':
    """
    Run this file using
    python -O gecko_validation.py
    to run the gecko simulations.
    """
    dbm = tdb.DatabaseManager()
    dbm.set_operation_mode_json()

    mosfet1 = 'CREE_C3M0065100J'
    mosfet2 = 'CREE_C3M0060065J'
    transistor_1 = dbm.load_transistor(mosfet1)
    transistor_2 = dbm.load_transistor(mosfet2)

    transistor_1.quickstart_wp()
    transistor_2.quickstart_wp()

    # design_specification = 'dab_ds_default_Gv8_sim'
    design_specification = 'initial'

    gecko_simulate_zvs_modulation(design_specification, transistor_1, transistor_2)

"""Plot GeckoCIRCUITS simulation results."""
# python libraries
import os

# own libraries
from dct.debug_tools import Log, timeit, info
from dct import DabDTO
from dct import Plot_DAB

# 3rd party libraries
import numpy as np


@timeit
def plot_gecko_simulation_results(dab_config: DabDTO, simulation_name: str, comment: str, directory: str,
                                  show_plot: bool = True, logfile=str()):
    """
    Plot the results from the GeckoCIRCUITS simulation.

    :param dab_config: Dual-Active-Bridge (DAB) configuration file
    :param simulation_name: Simulation name
    :param comment: Comment for the simulation, appears in the logs.
    :param directory: Directory to plot
    :param show_plot: True (default) to show the plots.
    :param logfile: Logfile name
    :return:
    """
    # Logging
    log = Log(filename=os.path.join(directory, logfile) if logfile else '')
    # Plotting
    log.info("\nStart Plotting\n")

    # Plot a cross-section through the V1 plane
    v1_middle = int(np.shape(dab_config.calc_config.mesh_P)[1] / 2)
    log.info('View plane: U_1 = {:.1f}V'.format(dab_config.calc_config.mesh_V1[0, v1_middle, 0]))
    simulation_name += '_V1_{:.0f}V'.format(dab_config.calc_config.mesh_V1[0, v1_middle, 0])
    comment += ' View plane: V_1 = {:.1f}V'.format(dab_config.calc_config.mesh_V1[0, v1_middle, 0])

    plt = Plot_DAB(latex=False, show=show_plot, figsize=(15, 5), fontsize=22)

    modulation_name = 'zvs'

    log.info('Plotting modulation: ' + modulation_name)
    # Show ZVS coverage based on simulation:
    log.info(modulation_name + ' Simulation ZVS coverage (Bridge 1, Bridge 2): {} ({}, {})'.format(
        round(np.array(dab_config.gecko_results.zvs_coverage).item(0), 3),
        round(np.array(dab_config.gecko_results.zvs_coverage1).item(0), 3),
        round(np.array(dab_config.gecko_results.zvs_coverage2).item(0), 3)))
    # Only non NaN areas:
    log.info(modulation_name + ' Simulation ZVS coverage (Bridge 1, Bridge 2) (non NaN): {} ({}, {})'.format(
        round(np.array(dab_config.gecko_results.zvs_coverage_notnan).item(0), 3),
        round(np.array(dab_config.gecko_results.zvs_coverage1_notnan).item(0), 3),
        round(np.array(dab_config.gecko_results.zvs_coverage2_notnan).item(0), 3)))
    # Mean of I1:
    log.info(modulation_name + ' Simulation I_1-total-mean: {}'.format(
        round(np.array(dab_config.gecko_results.i_HF1_total_mean).item(0), 3)))
    log.info(modulation_name + ' Simulation I^2_1-total-mean: {}'.format(
        round(np.array(dab_config.gecko_results.I1_squared_total_mean).item(0), 3)))

    # Set masks according to mod for later usage
    match modulation_name:
        case s if s.startswith('zvs'):
            # Hide less useful masks
            mask1 = None
            mask2 = None
            # mask1 = dab['mod_' + m + '_mask_Im2'][:, v1_middle, :]
            # mask2 = dab['mod_' + m + '_mask_IIm2'][:, v1_middle, :]
            mask3 = dab_config.calc_modulation.mask_IIIm1[:, v1_middle, :]
            maskZVS = dab_config.calc_modulation.mask_zvs[:, v1_middle, :]
        case _:
            mask1 = None
            mask2 = None
            mask3 = None
            maskZVS = None

    # Plot all modulation angles
    plt.plot_modulation(dab_config.calc_config.mesh_P[:, v1_middle, :],
                        dab_config.calc_config.mesh_V2[:, v1_middle, :],
                        dab_config.calc_modulation.phi[:, v1_middle, :],
                        dab_config.calc_modulation.tau1[:, v1_middle, :],
                        dab_config.calc_modulation.tau2[:, v1_middle, :],
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
    plt.subplot_contourf(dab_config.calc_config.mesh_P[:, v1_middle, :],
                         dab_config.calc_config.mesh_V2[:, v1_middle, :],
                         dab_config.gecko_results.v_ds_S11_sw_on[:, v1_middle, :] / dab_config.calc_config.mesh_V1[:, v1_middle, :],
                         mask1=mask1,
                         mask2=mask2,
                         mask3=mask3,
                         z_min=0,
                         z_max=1,
                         ax=plt.figs_axes[-1][1][0],
                         xlabel=r'$P \:/\: \mathrm{W}$',
                         ylabel=r'$U_\mathrm{DC2} \:/\: \mathrm{V}$',
                         title=r'$u_\mathrm{DS,S11,sw-on} \:/\: U_\mathrm{DC1}$')
    plt.subplot_contourf(dab_config.calc_config.mesh_P[:, v1_middle, :],
                         dab_config.calc_config.mesh_V2[:, v1_middle, :],
                         dab_config.gecko_results.v_ds_S23_sw_on[:, v1_middle, :] / dab_config.calc_config.mesh_V2[:, v1_middle, :],
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
    plt.subplot_contourf(dab_config.calc_config.mesh_P[:, v1_middle, :],
                         dab_config.calc_config.mesh_V2[:, v1_middle, :],
                         dab_config.gecko_results.i_HF1_S11_sw_on[:, v1_middle, :],
                         mask1=mask1,
                         mask2=mask2,
                         mask3=mask3,
                         ax=plt.figs_axes[-1][1][0],
                         xlabel=r'$P \:/\: \mathrm{W}$',
                         ylabel=r'$U_\mathrm{DC2} \:/\: \mathrm{V}$',
                         title=r'$i_\mathrm{1,S11,sw-on} \:/\: \mathrm{A}$')
    plt.subplot_contourf(dab_config.calc_config.mesh_P[:, v1_middle, :],
                         dab_config.calc_config.mesh_V2[:, v1_middle, :],
                         dab_config.gecko_results.i_HF2_S23_sw_on[:, v1_middle, :],
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
    plt.subplot_contourf(dab_config.calc_config.mesh_P[:, v1_middle, :],
                         dab_config.calc_config.mesh_V2[:, v1_middle, :],
                         dab_config.gecko_results.power_deviation[:, v1_middle, :],
                         mask1=mask1,
                         mask2=mask2,
                         mask3=mask3,
                         ax=plt.figs_axes[-1][1][0],
                         z_min=0.5,
                         z_max=1.5,
                         xlabel=r'$P \:/\: \mathrm{W}$',
                         ylabel=r'$U_\mathrm{DC2} \:/\: \mathrm{V}$',
                         title=r'$P_\mathrm{out,Sim} \:/\: P_\mathrm{out,desired}$')
    plt.subplot_contourf(dab_config.calc_config.mesh_P[:, v1_middle, :],
                         dab_config.calc_config.mesh_V2[:, v1_middle, :],
                         dab_config.gecko_results.p_sw1[:, v1_middle, :] + dab_config.gecko_results.p_sw2[:, v1_middle, :],
                         mask1=mask1,
                         mask2=mask2,
                         mask3=mask3,
                         ax=plt.figs_axes[-1][1][1],
                         xlabel=r'$P \:/\: \mathrm{W}$',
                         title=r'$P_\mathrm{sw,total} \:/\: \mathrm{W}$')
    plt.subplot_contourf(dab_config.calc_config.mesh_P[:, v1_middle, :],
                         dab_config.calc_config.mesh_V2[:, v1_middle, :],
                         dab_config.gecko_results.p_cond1[:, v1_middle, :] + dab_config.gecko_results.p_cond2[:, v1_middle, :],
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
    plt.subplot_contourf(dab_config.calc_config.mesh_P[:, v1_middle, :],
                         dab_config.calc_config.mesh_V2[:, v1_middle, :],
                         dab_config.gecko_results.S11_p_sw[:, v1_middle, :],
                         mask1=mask1,
                         mask2=mask2,
                         mask3=mask3,
                         ax=plt.figs_axes[-1][1][0],
                         xlabel=r'$P \:/\: \mathrm{W}$', ylabel=r'$U_\mathrm{DC2} \:/\: \mathrm{V}$',
                         title=r'$P_\mathrm{S11,sw} \:/\: \mathrm{W}$')
    plt.subplot_contourf(dab_config.calc_config.mesh_P[:, v1_middle, :],
                         dab_config.calc_config.mesh_V2[:, v1_middle, :],
                         dab_config.gecko_results.S11_p_cond[:, v1_middle, :],
                         mask1=mask1,
                         mask2=mask2,
                         mask3=mask3,
                         ax=plt.figs_axes[-1][1][1],
                         xlabel=r'$P \:/\: \mathrm{W}$',
                         title=r'$P_\mathrm{S11,cond} \:/\: \mathrm{W}$')
    plt.subplot_contourf(dab_config.calc_config.mesh_P[:, v1_middle, :],
                         dab_config.calc_config.mesh_V2[:, v1_middle, :],
                         dab_config.gecko_results.S23_p_sw[:, v1_middle, :],
                         mask1=mask1,
                         mask2=mask2,
                         mask3=mask3,
                         ax=plt.figs_axes[-1][1][2],
                         xlabel=r'$P \:/\: \mathrm{W}$',
                         title=r'$P_\mathrm{S23,sw} \:/\: \mathrm{W}$')
    plt.subplot_contourf(dab_config.calc_config.mesh_P[:, v1_middle, :],
                         dab_config.calc_config.mesh_V2[:, v1_middle, :],
                         dab_config.gecko_results.S23_p_cond[:, v1_middle, :],
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
    plt.subplot_contourf(dab_config.calc_config.mesh_P[:, v1_middle, :],
                         dab_config.calc_config.mesh_V2[:, v1_middle, :],
                         dab_config.gecko_results.i_HF1[:, v1_middle, :],
                         mask1=mask1,
                         mask2=mask2,
                         mask3=mask3,
                         ax=plt.figs_axes[-1][1][0],
                         xlabel=r'$P \:/\: \mathrm{W}$',
                         ylabel=r'$U_\mathrm{DC2} \:/\: \mathrm{V}$',
                         title=r'$I_\mathrm{1} \:/\: \mathrm{A}$')
    plt.subplot_contourf(dab_config.calc_config.mesh_P[:, v1_middle, :],
                         dab_config.calc_config.mesh_V2[:, v1_middle, :],
                         dab_config.gecko_results.i_Lc1[:, v1_middle, :],
                         mask1=mask1,
                         mask2=mask2,
                         mask3=mask3,
                         ax=plt.figs_axes[-1][1][1],
                         xlabel=r'$P \:/\: \mathrm{W}$',
                         title=r'$I_\mathrm{L1} \:/\: \mathrm{A}$')
    plt.subplot_contourf(dab_config.calc_config.mesh_P[:, v1_middle, :],
                         dab_config.calc_config.mesh_V2[:, v1_middle, :],
                         dab_config.gecko_results.i_Lc2[:, v1_middle, :] / dab_config.input_config.n,
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

"""Calculate the DAB operating points and show the results."""

# python libraries
import os

# own libraries
from dct.topology.dab.dab_datasets_dtos import DabCircuitDTO
from dct.topology.dab.dab_plotdab import PlotDAB

# 3rd party libraries
import numpy as np

def plot_calculation_results(dab_config: DabCircuitDTO) -> None:
    """
    Calculate the DAB operating points and show the results.

    :param dab_config: DAB configuration file
    :type dab_config: DabCircuitDTO
    """
    zvs_coverage = np.count_nonzero(dab_config.calc_modulation.mask_zvs) / np.size(dab_config.calc_modulation.mask_zvs)

    # Plotting
    directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    directory = os.path.join(directory, 'results')
    if not os.path.exists(directory):
        os.mkdir(directory)
    directory = os.path.join(directory, 'zvs_mod')
    if not os.path.exists(directory):
        os.mkdir(directory)
    name = 'mod_zvs'

    plt = PlotDAB(is_latex=False)

    # Plot OptZVS mod results
    # Plot a cross-section through the V1 plane
    v1_middle = int(np.shape(dab_config.input_config.mesh_p)[1] / 2)
    # Plot all modulation angles
    plt.plot_modulation(dab_config.input_config.mesh_p[:, v1_middle, :],
                        dab_config.input_config.mesh_v2[:, v1_middle, :],
                        dab_config.calc_modulation.phi[:, v1_middle, :],
                        dab_config.calc_modulation.tau1[:, v1_middle, :],
                        dab_config.calc_modulation.tau2[:, v1_middle, :],
                        mask1=dab_config.calc_modulation.mask_Im2[:, v1_middle, :],
                        mask2=dab_config.calc_modulation.mask_IIm2[:, v1_middle, :],
                        mask3=dab_config.calc_modulation.mask_IIIm1[:, v1_middle, :],
                        maskZVS=dab_config.calc_modulation.mask_zvs[:, v1_middle, :],
                        tab_title='OptZVS Modulation Angles (U_1 = {:.1f}V)'.format(dab_config.input_config.mesh_v1[0, v1_middle, 0])
                        )
    fname = name + '_V1_{:.0f}V'.format(dab_config.input_config.mesh_v1[0, v1_middle, 0])
    fcomment = 'View plane: V_1 = {:.1f}V'.format(dab_config.input_config.mesh_v1[0, v1_middle, 0])
    PlotDAB.save_fig(plt.figs_axes[-1][0], directory, fname, fcomment)

    # Plot all modulation angles but separately with autoscale
    plt.new_fig(nrows=1, ncols=3, tab_title='OptZVS Modulation Angles (autoscale)')
    plt.subplot_contourf(dab_config.input_config.mesh_p[:, v1_middle, :],
                         dab_config.input_config.mesh_v2[:, v1_middle, :],
                         dab_config.calc_modulation.phi[:, v1_middle, :],
                         ax=plt.figs_axes[-1][1][0],
                         xlabel='P / W', ylabel='U2 / V', title='phi in rad')
    plt.subplot_contourf(dab_config.input_config.mesh_p[:, v1_middle, :],
                         dab_config.input_config.mesh_v2[:, v1_middle, :],
                         dab_config.calc_modulation.tau1[:, v1_middle, :],
                         ax=plt.figs_axes[-1][1][1],
                         xlabel='P / W', ylabel='U2 / V', title='tau1 in rad')
    plt.subplot_contourf(dab_config.input_config.mesh_p[:, v1_middle, :],
                         dab_config.input_config.mesh_v2[:, v1_middle, :],
                         dab_config.calc_modulation.tau2[:, v1_middle, :],
                         ax=plt.figs_axes[-1][1][2],
                         xlabel='P / W', ylabel='U2 / V', title='tau2 in rad')
    fname = name + '_V1_{:.0f}V_autoscale'.format(dab_config.input_config.mesh_v1[0, v1_middle, 0])
    fcomment = 'View plane: V_1 = {:.1f}V'.format(dab_config.input_config.mesh_v1[0, v1_middle, 0])
    PlotDAB.save_fig(plt.figs_axes[-1][0], directory, fname, fcomment)

    # Plot a cross-section through the V2 plane
    v2_middle = int(np.shape(dab_config.input_config.mesh_p)[0] / 2)

    plt.plot_modulation(dab_config.input_config.mesh_p[v2_middle, :, :],
                        dab_config.input_config.mesh_v1[v2_middle, :, :],
                        dab_config.calc_modulation.phi[v2_middle, :, :],
                        dab_config.calc_modulation.tau1[v2_middle, :, :],
                        dab_config.calc_modulation.tau2[v2_middle, :, :],
                        mask1=dab_config.calc_modulation.mask_Im2[v2_middle, :, :],
                        mask2=dab_config.calc_modulation.mask_IIm2[v2_middle, :, :],
                        mask3=dab_config.calc_modulation.mask_IIIm1[v2_middle, :, :],
                        maskZVS=dab_config.calc_modulation.mask_zvs[v2_middle, :, :],
                        Vnum=1,
                        tab_title='OptZVS Modulation Angles (U_2 = {:.1f}V)'.format(dab_config.input_config.mesh_v2[v2_middle, 0, 0])
                        )
    fname = name + '_V2_{:.0f}V'.format(dab_config.input_config.mesh_v2[v2_middle, 0, 0])
    fcomment = 'View plane: V_2 = {:.1f}V'.format(dab_config.input_config.mesh_v2[v2_middle, 0, 0])
    PlotDAB.save_fig(plt.figs_axes[-1][0], directory, fname, fcomment)

    # Plot Coss and Qoss of transistor 1
    plt.new_fig(nrows=1, ncols=2, tab_title='Coss ' + str(dab_config.input_config.transistor_dto_1), sharex=False, sharey=False)
    plt.subplot(np.arange(dab_config.calc_config.c_oss_1.shape[0]),
                dab_config.calc_config.c_oss_1,
                ax=plt.figs_axes[-1][1][0],
                xlabel='U_DS / V', ylabel='C_oss / pF', title='Coss ' + str(dab_config.input_config.transistor_dto_1),
                yscale='log')

    plt.subplot(np.arange(dab_config.calc_config.q_oss_1.shape[0]),
                dab_config.calc_config.q_oss_1,
                ax=plt.figs_axes[-1][1][1],
                xlabel='U_DS / V', ylabel='Q_oss / nC', title='Qoss ' + str(dab_config.input_config.transistor_dto_1))

    # Plot Coss and Qoss of transistor 2
    plt.new_fig(nrows=1, ncols=2, tab_title='Coss ' + str(dab_config.input_config.transistor_dto_2), sharex=False, sharey=False)
    plt.subplot(np.arange(dab_config.calc_config.c_oss_2.shape[0]),
                dab_config.calc_config.c_oss_2,
                ax=plt.figs_axes[-1][1][0],
                xlabel='U_DS / V', ylabel='C_oss / pF', title='Coss ' + str(dab_config.input_config.transistor_dto_2),
                yscale='log')
    plt.subplot(np.arange(dab_config.calc_config.q_oss_2.shape[0]),
                dab_config.calc_config.q_oss_2,
                ax=plt.figs_axes[-1][1][1],
                xlabel='U_DS / V', ylabel='Q_oss / nC', title='Qoss ' + str(dab_config.input_config.transistor_dto_2))

    plt.show()

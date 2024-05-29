
import dct
import numpy as np
import os

def calculate_and_plot_dab_results(dab_design_config, mosfet1, mosfet2):
    da_mod = dct.calc_modulation(dab_design_config.n,
                                 dab_design_config.Ls,
                                 dab_design_config.Lc1,
                                 dab_design_config.Lc2,
                                 dab_design_config.fs,
                                 dab_design_config['coss_' + mosfet1],
                                 dab_design_config['coss_' + mosfet2],
                                 dab_design_config.mesh_V1,
                                 dab_design_config.mesh_V2,
                                 dab_design_config.mesh_P)

    # Unpack the results
    dab_design_config.append_result_dict(da_mod, name_pre='mod_zvs_')
    dct.debug(da_mod)
    dct.debug('phi min:', np.nanmin(dab_design_config.mod_zvs_phi), 'phi max:', np.nanmax(dab_design_config.mod_zvs_phi))
    zvs_coverage = np.count_nonzero(dab_design_config.mod_zvs_mask_zvs) / np.size(dab_design_config.mod_zvs_mask_zvs)
    dct.debug('zvs coverage:', zvs_coverage)

    ## Plotting
    dct.info("\nStart Plotting\n")

    directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
    directory = os.path.join(directory, 'results')
    if not os.path.exists(directory):
        os.mkdir(directory)
    directory = os.path.join(directory, 'zvs_mod')
    if not os.path.exists(directory):
        os.mkdir(directory)
    name = 'mod_zvs'
    comment = 'Only modulation calculation results for mod_zvs with V1 {}, V2 {} and P {} steps.'.format(
        int(dab_design_config.V1_step),
        int(dab_design_config.V2_step),
        int(dab_design_config.P_step))

    plt = dct.plot_dab.Plot_DAB(latex=False)

    # Plot OptZVS mod results
    # Plot a cross-section through the V1 plane
    v1_middle = int(np.shape(dab_design_config.mesh_P)[1] / 2)
    dct.debug('View plane: U_1 = {:.1f}V'.format(dab_design_config.mesh_V1[0, v1_middle, 0]))
    # Plot all modulation angles
    # plt.new_fig(nrows=1, ncols=3,
    #             tab_title='OptZVS Modulation Angles (U_1 = {:.1f}V)'.format(dab.mesh_V1[0, v1_middle, 0]))
    plt.plot_modulation(dab_design_config.mesh_P[:, v1_middle, :],
                        dab_design_config.mesh_V2[:, v1_middle, :],
                        dab_design_config.mod_zvs_phi[:, v1_middle, :],
                        dab_design_config.mod_zvs_tau1[:, v1_middle, :],
                        dab_design_config.mod_zvs_tau2[:, v1_middle, :],
                        mask1=dab_design_config.mod_zvs_mask_Im2[:, v1_middle, :],
                        mask2=dab_design_config.mod_zvs_mask_IIm2[:, v1_middle, :],
                        mask3=dab_design_config.mod_zvs_mask_IIIm1[:, v1_middle, :],
                        maskZVS=dab_design_config.mod_zvs_mask_zvs[:, v1_middle, :],
                        tab_title='OptZVS Modulation Angles (U_1 = {:.1f}V)'.format(dab_design_config.mesh_V1[0, v1_middle, 0])
                        )
    fname = name + '_V1_{:.0f}V'.format(dab_design_config.mesh_V1[0, v1_middle, 0])
    fcomment = comment + ' View plane: V_1 = {:.1f}V'.format(dab_design_config.mesh_V1[0, v1_middle, 0])
    plt.save_fig(plt.figs_axes[-1][0], directory, fname, fcomment)

    # Plot all modulation angles but separately with autoscale
    plt.new_fig(nrows=1, ncols=3, tab_title='OptZVS Modulation Angles (autoscale)')
    plt.subplot_contourf(dab_design_config.mesh_P[:, v1_middle, :],
                         dab_design_config.mesh_V2[:, v1_middle, :],
                         dab_design_config.mod_zvs_phi[:, v1_middle, :],
                         ax=plt.figs_axes[-1][1][0],
                         xlabel='P / W', ylabel='U2 / V', title='phi in rad')
    plt.subplot_contourf(dab_design_config.mesh_P[:, v1_middle, :],
                         dab_design_config.mesh_V2[:, v1_middle, :],
                         dab_design_config.mod_zvs_tau1[:, v1_middle, :],
                         ax=plt.figs_axes[-1][1][1],
                         xlabel='P / W', ylabel='U2 / V', title='tau1 in rad')
    plt.subplot_contourf(dab_design_config.mesh_P[:, v1_middle, :],
                         dab_design_config.mesh_V2[:, v1_middle, :],
                         dab_design_config.mod_zvs_tau2[:, v1_middle, :],
                         ax=plt.figs_axes[-1][1][2],
                         xlabel='P / W', ylabel='U2 / V', title='tau2 in rad')
    fname = name + '_V1_{:.0f}V_autoscale'.format(dab_design_config.mesh_V1[0, v1_middle, 0])
    fcomment = comment + ' View plane: V_1 = {:.1f}V'.format(dab_design_config.mesh_V1[0, v1_middle, 0])
    plt.save_fig(plt.figs_axes[-1][0], directory, fname, fcomment)

    # Plot a cross-section through the V2 plane
    v2_middle = int(np.shape(dab_design_config.mesh_P)[0] / 2)
    dct.debug('View plane: U_2 = {:.1f}V'.format(dab_design_config.mesh_V2[v2_middle, 0, 0]))
    # Plot all modulation angles
    # plt.new_fig(nrows=1, ncols=3,
    #             tab_title='OptZVS Modulation Angles (U_2 = {:.1f}V)'.format(dab.mesh_V2[v2_middle, 0, 0]))
    plt.plot_modulation(dab_design_config.mesh_P[v2_middle, :, :],
                        dab_design_config.mesh_V1[v2_middle, :, :],
                        dab_design_config.mod_zvs_phi[v2_middle, :, :],
                        dab_design_config.mod_zvs_tau1[v2_middle, :, :],
                        dab_design_config.mod_zvs_tau2[v2_middle, :, :],
                        mask1=dab_design_config.mod_zvs_mask_Im2[v2_middle, :, :],
                        mask2=dab_design_config.mod_zvs_mask_IIm2[v2_middle, :, :],
                        mask3=dab_design_config.mod_zvs_mask_IIIm1[v2_middle, :, :],
                        maskZVS=dab_design_config.mod_zvs_mask_zvs[v2_middle, :, :],
                        Vnum=1,
                        tab_title='OptZVS Modulation Angles (U_2 = {:.1f}V)'.format(dab_design_config.mesh_V2[v2_middle, 0, 0])
                        )
    fname = name + '_V2_{:.0f}V'.format(dab_design_config.mesh_V2[v2_middle, 0, 0])
    fcomment = comment + ' View plane: V_2 = {:.1f}V'.format(dab_design_config.mesh_V2[v2_middle, 0, 0])
    plt.save_fig(plt.figs_axes[-1][0], directory, fname, fcomment)

    # Plot Coss
    plt.new_fig(nrows=1, ncols=2, tab_title='Coss ' + mosfet1, sharex=False, sharey=False)
    plt.subplot(np.arange(dab_design_config['coss_' + mosfet1].shape[0]),
                dab_design_config['coss_' + mosfet1],
                ax=plt.figs_axes[-1][1][0],
                xlabel='U_DS / V', ylabel='C_oss / pF', title='Coss ' + mosfet1,
                yscale='log')
    plt.subplot(np.arange(dab_design_config['qoss_' + mosfet1].shape[0]),
                dab_design_config['qoss_' + mosfet1],
                ax=plt.figs_axes[-1][1][1],
                xlabel='U_DS / V', ylabel='Q_oss / nC', title='Qoss ' + mosfet1)

    # Plot Coss
    plt.new_fig(nrows=1, ncols=2, tab_title='Coss ' + mosfet2, sharex=False, sharey=False)
    plt.subplot(np.arange(dab_design_config['coss_' + mosfet2].shape[0]),
                dab_design_config['coss_' + mosfet2],
                ax=plt.figs_axes[-1][1][0],
                xlabel='U_DS / V', ylabel='C_oss / pF', title='Coss ' + mosfet2,
                yscale='log')
    plt.subplot(np.arange(dab_design_config['qoss_' + mosfet2].shape[0]),
                dab_design_config['qoss_' + mosfet2],
                ax=plt.figs_axes[-1][1][1],
                xlabel='U_DS / V', ylabel='Q_oss / nC', title='Qoss ' + mosfet2)

    plt.show()

import os
import sys

import math
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.ticker as ticker

from dct.debug_tools import *
from dct.design_check.plot_window import *


class Plot_DAB:
    """
    Class storing and managing the plotwindow, figs and axes.
    """
    pw: PlotWindow
    figs_axes: list

    def __init__(self, latex=False, window_title: str = 'DAB Plots', figsize=(12, 5), fontsize=16, show=True):
        """
        Create the object with default settings for all further plots

        :param latex: Use Latex fonts (if available) for labels
        :param window_title:
        :param figsize: Set default figsize for all plots and savefig (figsize * dpi = px)
        """
        self.show_pw = show
        if show:
            # Create new plotWindow that holds the tabs
            self.pw = PlotWindow(window_title=window_title, figsize=figsize)
        else:
            plt.ioff()
        # Set pyplot figsize for savefig
        # Alternative default figsize=(10, 5) with default dpi = 100 may be used
        self.figsize = figsize
        plt.rcParams.update({'figure.figsize': figsize})
        self.fontsize = fontsize
        # Create empty list to store the fig and axe handlers
        self.figs_axes = []
        # Switch between latex math usage and plain text where possible
        # Note: For latex to work you must have it installed on your system!
        self.latex = latex
        if latex:
            plt.rcParams.update({
                "text.usetex":      True,
                "font.family":      "STIXGeneral",
                "mathtext.fontset": "cm",
                # "font.size":        12
                "font.size":        fontsize
            })
            plt.rcParams['figure.constrained_layout.use'] = False
            plt.rcParams["figure.autolayout"] = False
        else:
            plt.rcParams['figure.constrained_layout.use'] = False
            plt.rcParams["figure.autolayout"] = False

    def apply_spacings(self, fig):

        # Set some default spacings
        if self.fontsize < 14:
            fig.subplots_adjust(left=0.05, right=0.98, bottom=0.08, top=0.91, wspace=0.06, hspace=0.2)
            if self.figsize == (16, 8):
                fig.subplots_adjust(left=0.04, right=0.995, bottom=0.065, top=0.96, wspace=0.07, hspace=0.2)
            if self.figsize == (12.8, 8):
                fig.subplots_adjust(left=0.05, right=0.98, bottom=0.06, top=0.96, wspace=0.12, hspace=0.2)
            if self.figsize == (10, 5):
                fig.subplots_adjust(left=0.062, right=0.975, bottom=0.092, top=0.94, wspace=0.17, hspace=0.2)
            if self.figsize == (12, 5):
                fig.subplots_adjust(left=0.062, right=0.975, bottom=0.092, top=0.94, wspace=0.17, hspace=0.2)
        else:
            if self.figsize == (5, 4):
                fig.subplots_adjust(left=0.18, right=0.99, bottom=0.15, top=0.92, wspace=0.17, hspace=0.2)
            if self.figsize == (5, 5):
                # fig.subplots_adjust(left=0.15, right=0.98, bottom=0.12, top=0.94, wspace=0.17, hspace=0.2)
                fig.subplots_adjust(left=0.17, right=0.98, bottom=0.12, top=0.94, wspace=0.17, hspace=0.2)
            if self.figsize == (10, 5):
                fig.subplots_adjust(left=0.077, right=0.955, bottom=0.127, top=0.935, wspace=0.17, hspace=0.25)
            if self.figsize == (12, 5):
                fig.subplots_adjust(left=0.062, right=0.97, bottom=0.117, top=0.935, wspace=0.17, hspace=0.25)
            if self.figsize == (15, 5):
                fig.subplots_adjust(left=0.065, right=0.975, bottom=0.15, top=0.93, wspace=0.17, hspace=0.25)

    def new_fig(self, nrows: int = 1, ncols: int = 1, sharex: str = True, sharey: str = True,
                tab_title='add Plot title'):
        """
        Create a new fig in a new tab with the amount of subplots specified

        :param nrows:
        :param ncols:
        :param sharex:
        :param sharey:
        :param figsize:
        :param tab_title: Set the title of the tab-selector
        """
        # self.figs_axes.append(plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey,
        #                                    figsize=figsize, num=num))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, figsize=self.figsize)

        self.apply_spacings(fig)

        # Store the handlers in our list with tuples
        # TODO do we have to store axs if we can access them with fig.axes?
        self.figs_axes.append((fig, axs))
        if self.show_pw:
            self.pw.addPlot(title=tab_title, figure=fig)

    def save_fig(self, fig, directory=None, name='', comment='',
                 timestamp=True):
        """
        Saves the given fig as PNG and PDF
        :param fig:
        :param directory:
        :param name:
        :param comment:
        :param timestamp: If the filename should start with a timestamp
        """
        if __debug__:
            name = 'debug_' + name
        if __debug__:
            comment = 'Debug ' + comment

        if directory is None:
            directory = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))
            directory = os.path.join(directory, 'results')
            if not os.path.exists(directory):
                os.mkdir(directory)
            directory = os.path.join(directory, 'zvs_mod')
            if not os.path.exists(directory):
                os.mkdir(directory)

        # Adding a timestamp to the filename if requested
        if timestamp:
            if name:
                filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + name
            else:
                filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        else:
            if name:
                filename = name
            else:
                # set some default non-empty filename
                filename = "dab_plot"

        # Expand dir name to full path
        directory = os.path.expanduser(directory)
        directory = os.path.expandvars(directory)
        directory = os.path.abspath(directory)
        if not os.path.isdir(directory):
            warning("Directory does not exist!")
            sys.exit(1)

        # Save plots
        metadata = {'Title':    filename,
                    'Author':   'Felix Langemeier',
                    'Keywords': 'python, matplotlib, dab'}
        # Save as PNG
        fname = os.path.join(directory, filename + '.png')
        fig.savefig(fname=fname, metadata=metadata)
        # Save as PDF
        if not __debug__:
            fname = os.path.join(directory, filename + '.pdf')
            fig.savefig(fname=fname, metadata=metadata)

    def plot_3by1(self, fig_axes: tuple, x, y, z1, z2, z3, xl: str = 'x', yl: str = 'y', t1: str = 'z1', t2: str = 'z2',
                  t3: str = 'z3'):
        """
        Plots three contourf plots with a shared colorbar.

        :param fig_axes: Provide the tuple (fig, axs)
        :param x:
        :param y:
        :param z1:
        :param z2:
        :param z3:
        """
        # plot
        fig = fig_axes[0]
        axs = fig_axes[1]
        # fig.suptitle("subtitle")
        # fig.tight_layout()
        cf = axs[0].contourf(x, y, z1)
        axs[1].contourf(x, y, z2)
        axs[2].contourf(x, y, z3)
        axs[0].set_title(t1)
        axs[1].set_title(t2)
        axs[2].set_title(t3)
        for ax in axs.flat:
            ax.set(xlabel=xl, ylabel=yl)
            ax.label_outer()
        # Only add colorbar if there was none
        if fig.axes[-1].get_label() == '<colorbar>':
            # TODO update colorbar
            debug("update colorbar")
            cbar = fig.axes[-1]
        else:
            cbar = fig.colorbar(cf, ax=axs)
        # tight_layout and colorbar are tricky
        # fig.tight_layout()
        # Redraw the current figure
        plt.draw()

    def plot_modulation(self, x, y, z1, z2, z3, title: str = '', mask1=None, mask2=None, mask3=None,
                        maskZVS=None, Vnum=2, tab_title='add Plot title'):
        """
        Plots three contourf plots with a shared colorbar.

        :param fig_axes: Provide the tuple (fig, axs)
        :param x: x mesh, e.g. P
        :param y: y mesh, e.g. V2
        :param z1: z for subplot 1, e.g. phi
        :param z2: z for subplot 2, e.g. tau1
        :param z3: z for subplot 3, e.g. tau2
        :param mask1: optional mask contour line
        :param mask2: optional mask contour line
        :param mask3: optional mask contour line
        :param Vnum: Voltage number of y-axis {1, 2}
        """
        # Add a new tab with subplot
        fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=self.figsize,
                                gridspec_kw={'width_ratios': [0.38, 0.31, 0.31]})
        self.apply_spacings(fig)
        # Store the handlers in our list with tuples
        # TODO do we have to store axs if we can access them with fig.axes?
        self.figs_axes.append((fig, axs))
        if self.show_pw:
            self.pw.addPlot(title=tab_title, figure=fig)
        fig_axes = self.figs_axes[-1]

        # Some defaults
        fig = fig_axes[0]
        axs = fig_axes[1]
        num_cont_lines = 20
        cmap = 'viridis'
        pz_min = -np.pi / 4
        pz_max = np.pi / 4
        z_min = -np.pi
        z_max = np.pi
        # Clear only the 3 subplots in case we update the same figure. Colorbar stays.
        axs[0].clear()
        axs[1].clear()
        axs[2].clear()
        # Plot the contourf maps
        axs[0].contourf(x, y, z1, num_cont_lines, alpha=1, antialiased=True, cmap=cmap, vmin=pz_min, vmax=pz_max)
        if not mask1 is None: axs[0].contour(x, y, mask1, levels=[0.5], colors=['red'], alpha=1)
        if not mask2 is None: axs[0].contour(x, y, mask2, levels=[0.5], colors=['blue'], alpha=1)
        if not mask3 is None: axs[0].contour(x, y, mask3, levels=[0.5], colors=['black'], alpha=1)
        if not maskZVS is None: axs[0].contourf(x, y, np.ma.masked_where(maskZVS == 1, maskZVS), 1, alpha=0.5,
                                                antialiased=True, cmap='Greys_r', vmin=0, vmax=1)

        # Apply the limits to the colorbar. That way the colorbar does not depend on one plot.
        mappable = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=pz_min, vmax=pz_max), cmap=cmap)
        # cbar = fig.colorbar(mappable, ax=axs[0], pad=0.001, boundaries=np.linspace(pz_min, pz_max, num_cont_lines),
        #                     ticks=[-np.pi / 4, -np.pi / 8, 0, np.pi / 8, np.pi / 4])
        cbar = fig.colorbar(mappable, ax=axs[0], pad=0.001,
                            ticks=[-np.pi / 4, -np.pi / 8, 0, np.pi / 8, np.pi / 4])
        if self.latex:
            cbar.ax.set_yticklabels(
                [r'$-\frac{1}{4} \pi$', r'$-\frac{1}{8} \pi$', r'$0$',
                 r'$\frac{1}{8} \pi$', r'$\frac{1}{4} \pi$'])
        else:
            cbar.ax.set_yticklabels(['-π/4', '-π/8', '0', 'π/8', 'π/4'])

        axs[1].contourf(x, y, z2, num_cont_lines, alpha=1, antialiased=True, cmap=cmap, vmin=z_min, vmax=z_max)
        if not mask1 is None: axs[1].contour(x, y, mask1, levels=[0.5], colors=['red'], alpha=1)
        if not mask2 is None: axs[1].contour(x, y, mask2, levels=[0.5], colors=['blue'], alpha=1)
        if not mask3 is None: axs[1].contour(x, y, mask3, levels=[0.5], colors=['black'], alpha=1)
        if not maskZVS is None: axs[1].contourf(x, y, np.ma.masked_where(maskZVS == 1, maskZVS), 1, alpha=0.5,
                                                antialiased=True, cmap='Greys_r', vmin=0, vmax=1)
        axs[2].contourf(x, y, z3, num_cont_lines, alpha=1, antialiased=True, cmap=cmap, vmin=z_min, vmax=z_max)
        if not mask1 is None: axs[2].contour(x, y, mask1, levels=[0.5], colors=['red'], alpha=1)
        if not mask2 is None: axs[2].contour(x, y, mask2, levels=[0.5], colors=['blue'], alpha=1)
        if not mask3 is None: axs[2].contour(x, y, mask3, levels=[0.5], colors=['black'], alpha=1)
        if not maskZVS is None: axs[2].contourf(x, y, np.ma.masked_where(maskZVS == 1, maskZVS), 1, alpha=0.5,
                                                antialiased=True, cmap='Greys_r', vmin=0, vmax=1)
        # Set the labels
        if title: fig.suptitle(title)
        axs[0].set_title(r"$\varphi \:/\: \mathrm{rad}$" if self.latex else "phi in rad")
        axs[1].set_title(r"$\tau_1 \:/\: \mathrm{rad}$" if self.latex else "tau1 in rad")
        axs[2].set_title(r"$\tau_2 \:/\: \mathrm{rad}$" if self.latex else "tau2 in rad")
        for ax in axs.flat:
            if self.latex:
                ax.set(xlabel=r'$P \:/\: \mathrm{W}$',
                       ylabel=r'$U_\mathrm{{DC{:.0f}}} \:/\: \mathrm{{V}}$'.format(Vnum))
            else:
                ax.set(xlabel='P / W', ylabel='U{:.0f} / V'.format(Vnum))
            ax.label_outer()
        # Apply the limits to the colorbar. That way the colorbar does not depend on one plot.
        mappable = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=z_min, vmax=z_max), cmap=cmap)
        # Only add colorbar if there was none
        # if fig.axes[-1].get_label() == '<colorbar>':
        if False:
            # TODO update colorbar
            warning("update colorbar not implemented")
            cbar = fig.axes[-1]
        else:
            cbar = fig.colorbar(mappable=mappable, ax=axs, fraction=0.05, pad=0.02,
                                ticks=[-np.pi, -np.pi * 3 / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2,
                                       np.pi * 3 / 4, np.pi])
            if self.latex:
                cbar.ax.set_yticklabels(
                    [r'$-\pi$', r'$-\frac{3}{4} \pi$', r'$-\frac{1}{2} \pi$', r'$-\frac{1}{4} \pi$', r'$0$',
                     r'$\frac{1}{4} \pi$', r'$\frac{1}{2} \pi$', r'$\frac{3}{4} \pi$', r'$\pi$'])
            else:
                cbar.ax.set_yticklabels(['-π', '-π3/4', '-π/2', '-π/4', '0', 'π/4', 'π/2', 'π3/4', 'π'])
            # alternative to this quick fix: https://stackoverflow.com/a/53586826
        # tight_layout and colorbar are tricky
        # fig.tight_layout()
        # Redraw the current figure
        # plt.draw()
        fig.canvas.draw()
        fig.canvas.flush_events()

    def plot_modulation_classic(self, fig_axes: tuple, x, y, z1, z2, z3, title: str = '', mask1=None, mask2=None,
                                mask3=None,
                                maskZVS=None):
        """
        Plots three contourf plots with a shared colorbar.

        :param fig_axes: Provide the tuple (fig, axs)
        :param x: x mesh, e.g. P
        :param y: y mesh, e.g. V2
        :param z1: z for subplot 1, e.g. phi
        :param z2: z for subplot 2, e.g. tau1
        :param z3: z for subplot 3, e.g. tau2
        :param mask1: optional mask contour line
        :param mask2: optional mask contour line
        :param mask3: optional mask contour line
        """
        # Some defaults
        fig = fig_axes[0]
        axs = fig_axes[1]
        num_cont_lines = 20
        cmap = 'viridis'
        z_min = 0
        z_max = np.pi
        # Clear only the 3 subplots in case we update the same figure. Colorbar stays.
        axs[0].clear()
        axs[1].clear()
        axs[2].clear()
        # Plot the contourf maps
        axs[0].contourf(x, y, z1, num_cont_lines, alpha=1, antialiased=True, cmap=cmap, vmin=z_min, vmax=z_max)
        if not mask1 is None: axs[0].contour(x, y, mask1, levels=[0.5], colors=['red'], alpha=1)
        if not mask2 is None: axs[0].contour(x, y, mask2, levels=[0.5], colors=['blue'], alpha=1)
        if not mask3 is None: axs[0].contour(x, y, mask3, levels=[0.5], colors=['black'], alpha=1)
        if not maskZVS is None: axs[0].contourf(x, y, np.ma.masked_where(maskZVS == 1, maskZVS), 1, alpha=0.5,
                                                antialiased=True, cmap='Greys_r', vmin=0, vmax=1)
        axs[1].contourf(x, y, z2, num_cont_lines, alpha=1, antialiased=True, cmap=cmap, vmin=z_min, vmax=z_max)
        if not mask1 is None: axs[1].contour(x, y, mask1, levels=[0.5], colors=['red'], alpha=1)
        if not mask2 is None: axs[1].contour(x, y, mask2, levels=[0.5], colors=['blue'], alpha=1)
        if not mask3 is None: axs[1].contour(x, y, mask3, levels=[0.5], colors=['black'], alpha=1)
        if not maskZVS is None: axs[1].contourf(x, y, np.ma.masked_where(maskZVS == 1, maskZVS), 1, alpha=0.5,
                                                antialiased=True, cmap='Greys_r', vmin=0, vmax=1)
        axs[2].contourf(x, y, z3, num_cont_lines, alpha=1, antialiased=True, cmap=cmap, vmin=z_min, vmax=z_max)
        if not mask1 is None: axs[2].contour(x, y, mask1, levels=[0.5], colors=['red'], alpha=1)
        if not mask2 is None: axs[2].contour(x, y, mask2, levels=[0.5], colors=['blue'], alpha=1)
        if not mask3 is None: axs[2].contour(x, y, mask3, levels=[0.5], colors=['black'], alpha=1)
        if not maskZVS is None: axs[2].contourf(x, y, np.ma.masked_where(maskZVS == 1, maskZVS), 1, alpha=0.5,
                                                antialiased=True, cmap='Greys_r', vmin=0, vmax=1)
        # Set the labels
        if title: fig.suptitle(title)
        axs[0].set_title(r"$\varphi / \mathrm{rad}$" if self.latex else "phi in rad")
        axs[1].set_title(r"$\tau_1 / \mathrm{rad}$" if self.latex else "tau1 in rad")
        axs[2].set_title(r"$\tau_2 / \mathrm{rad}$" if self.latex else "tau2 in rad")
        for ax in axs.flat:
            if self.latex:
                ax.set(xlabel=r'$P / \mathrm{W}$', ylabel=r'$U_2 / \mathrm{V}$')
            else:
                ax.set(xlabel='P / W', ylabel='U2 / V')
            ax.label_outer()
        # Apply the limits to the colorbar. That way the colorbar does not depend on one plot.
        mappable = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=z_min, vmax=z_max), cmap=cmap)
        # Only add colorbar if there was none
        if fig.axes[-1].get_label() == '<colorbar>':
            # TODO update colorbar
            warning("update colorbar not implemented")
            cbar = fig.axes[-1]
        else:
            cbar = fig.colorbar(mappable=mappable, ax=axs, fraction=0.05, pad=0.02,
                                ticks=[0, np.pi / 4, np.pi / 2, np.pi * 3 / 4, np.pi])
            if self.latex:
                cbar.ax.set_yticklabels(
                    [r'$0$', r'$\frac{1}{4} \pi$', r'$\frac{1}{2} \pi$', r'$\frac{3}{4} \pi$', r'$\pi$'])
            else:
                cbar.ax.set_yticklabels(['0', 'π/4', 'π/2', 'π3/4', 'π'])
            # alternative to this quick fix: https://stackoverflow.com/a/53586826
        # tight_layout and colorbar are tricky
        # fig.tight_layout()
        # Redraw the current figure
        # plt.draw()
        fig.canvas.draw()
        fig.canvas.flush_events()

    @timeit
    def plot_rms_current(self, mesh_V2, mesh_P, mvvp_iLs):
        # plot
        fig, axs = plt.subplots(1, 3, sharey=True)
        fig.suptitle("DAB RMS Currents")
        cf = axs[0].contourf(mesh_P[:, 1, :], mesh_V2[:, 1, :], mvvp_iLs[:, 1, :])
        axs[1].contourf(mesh_P[:, 1, :], mesh_V2[:, 1, :], mvvp_iLs[:, 1, :])
        axs[2].contourf(mesh_P[:, 1, :], mesh_V2[:, 1, :], mvvp_iLs[:, 1, :])
        axs[0].set_title("i_Ls")
        axs[1].set_title("i_Ls")
        axs[2].set_title("i_Ls")
        for ax in axs.flat:
            ax.set(xlabel='P / W', ylabel='U2 / V')
            ax.label_outer()
        # fig.colorbar(cf, ax=axs.ravel().tolist())
        fig.colorbar(cf, ax=axs)

        # plt.show()
        return fig

    def subplot_contourf(self, x, y, z, mask1=None, mask2=None, mask3=None,
                         nan_matrix=None, ax: str = None,
                         num_cont_lines: int = 12, alpha: float = 0.75, cmap: str = 'viridis',
                         axlinewidth=0.5, axlinecolor: str = 'r', wp_x: float = None, wp_y: float = None,
                         inlinespacing: int = -10, xlabel='', ylabel: str = '', title: str = "", clabel: bool = False,
                         markerstyle: str = 'star',
                         z_min: float = None, z_max: float = None,
                         square=False, same_xy_ticks=False) -> None:
        """
        Draw a subplot contourf.
        The area of z where a nan can be found in nan_matrix will be shaded.

        :param x: x-coordinate
        :param y: y-coordinate
        :param z: z-coordinate
        :param nan_matrix: [optional] z-values where a nan is in nan_matrix will be plotted shaded
        :param ax: choose the axis to draw this plot
        :param num_cont_lines: [optional] number of contour lines, default to 20
        :param alpha: [optional] shading 0...1. 1 = 100%, default to 0.5
        :param cmap: [optional] cmap type, e.g. inferno [default]
        :param axlinewidth: [optional] line width of axvline and axhline, default to 0.5
        :param axlinecolor: [optional] color of axline and star, default to red
        :param wp_x: [optional] working point in x (for marker line or star marker)
        :param wp_y: [optional] working point in y (for marker line or star marker)
        :param inlinespacing: [optional] default to -10
        :param xlabel: [optional] x-label
        :param ylabel: [optional] y-label
        :param fontsize_axis: [optional] default to 9
        :param fontsize_title: [optional] default to 9
        :param title: [optional] subplot figure title
        :param clabel: [optional] True to write labels inside the plot, default to False
        :param markerstyle: [optional] marker style: 'star' or 'line'
        :param z_min: [optional] clip to minimum z-value
        :param z_max: [optional] clip to maximum z-value
        :return: -
        """
        fontsize_axis = self.fontsize
        fontsize_title = self.fontsize

        # check if z input matrix is out ouf None's only. If Ture, raise exception.
        # Note: the 1-value is a random value, hopefully no one has sum(array) with array_size
        # search_nones = z.copy()
        # search_nones[np.isnan(search_nones)] = 1
        # if np.sum(search_nones) == np.size(search_nones):
        #     raise Exception("in subplot_contourf(), z input out of None's only is not allowed")
        if np.all(np.isnan(z)):
            warning('subplot_contourf(): z input out of NaN only is not allowed!', z)

        if ax is None:
            ax = plt.gca()

        if z_min is None or z_min > np.nanmax(z):
            z_min = np.nanmin(z)
        if z_max is None or z_max < np.nanmin(z):
            z_max = np.nanmax(z)
        # To prevent error in cbar and get at least a plot even it is one color
        if z_min == z_max:
            z_min = z_min - z_min * 0.1
            z_max = z_max + z_max * 0.1
        # Try to round min/max to sane values
        digits_min = 0 if z_min == 0 else -math.floor(math.log10(abs(z_min)))
        digits_max = 0 if z_max == 0 else -math.floor(math.log10(abs(z_max)))
        _z_min = round(z_min, digits_min)
        _z_max = round(z_max, digits_max)
        if _z_min != _z_max:
            z_min = _z_min
            z_max = _z_max
        # z_min = round(z_min, 1)
        # z_max = round(z_max, 1)
        # Set fixed cont_lines
        levels = np.linspace(z_min, z_max, num_cont_lines)
        # in case of nan_matrix is not set
        if nan_matrix is None:
            cs_full = ax.contourf(x, y, np.clip(z, z_min, z_max), levels=levels, alpha=1, antialiased=True, cmap=cmap,
                                  vmin=z_min, vmax=z_max)
            if not mask1 is None: ax.contour(x, y, mask1, levels=[0.5], colors=['red'], alpha=1)
            if not mask2 is None: ax.contour(x, y, mask2, levels=[0.5], colors=['blue'], alpha=1)
            if not mask3 is None: ax.contour(x, y, mask3, levels=[0.5], colors=['black'], alpha=1)
        # in case of nan_matrix is set
        else:
            # plot background 50% visible
            cs_background = ax.contourf(x, y, np.clip(z, z_min, z_max), levels=levels, alpha=alpha, antialiased=True,
                                        cmap=cmap, vmin=z_min, vmax=z_max)

            # generate matrix for foreground, 100% visible
            z_nan = z * nan_matrix

            # plot foreground, 100% visible
            # Note: levels taken from first plot
            # cs_full = ax.contourf(x, y, z_nan.clip(z_min, z_max), num_cont_lines, alpha=1, antialiased=True, cmap=cmap,
            #                       vmin=z_min, vmax=z_max, levels=cs_background.levels)
            cs_full = ax.contourf(x, y, z_nan.clip(z_min, z_max), levels=levels, alpha=1, antialiased=True, cmap=cmap,
                                  vmin=z_min, vmax=z_max)
            if not mask1 is None: ax.contour(x, y, mask1, levels=[0.5], colors=['red'], alpha=1)
            if not mask2 is None: ax.contour(x, y, mask2, levels=[0.5], colors=['blue'], alpha=1)
            if not mask3 is None: ax.contour(x, y, mask3, levels=[0.5], colors=['black'], alpha=1)

        ax.set_xlabel(xlabel, fontsize=fontsize_axis)
        ax.set_ylabel(ylabel, fontsize=fontsize_axis)
        ax.set_title(title, fontsize=fontsize_title)
        # Apply the limits to the colorbar. That way the colorbar does not depend on one plot.
        mappable = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=z_min, vmax=z_max), cmap=cmap)
        # if z_min is not None or z_min is not None:
        # Plot the colorbar with the same discrete steps as contourf plot
        cb = plt.colorbar(mappable=mappable, ax=ax, boundaries=np.linspace(z_min, z_max, num_cont_lines + 1))
        # This limits the number of digits to 2. If value hat no decimals none is shown.
        if digits_max > -2:
            cb.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2g'))
        else:
            cb.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
        # else:
        # These give a non-discrete colorbar
        # cb = plt.colorbar(mappable=mappable, ax=ax)
        # cb = plt.colorbar(cs_full)
        # Should make the colorbar discrete but does it?
        # cb.ax.locator_params(nbins=num_cont_lines)
        ax.grid()
        if clabel == True:
            ax.clabel(cs_full, inline=1, inline_spacing=inlinespacing, fontsize=10, fmt='%1.1f', colors='k')
        if wp_x is not None and markerstyle.lower() == 'line':
            ax.axvline(wp_x, linewidth=axlinewidth, color=axlinecolor)
        if wp_y is not None and markerstyle.lower() == 'line':
            ax.axhline(wp_y, linewidth=axlinewidth, color=axlinecolor)
        if wp_x is not None and wp_y is not None and markerstyle.lower() == 'star':
            ax.plot(wp_x, wp_y, marker="*", color=axlinecolor)
        if square:
            ax.set_aspect('equal', adjustable='box')
        if same_xy_ticks:
            ax.set_yticks(ax.get_xticks())

    def subplot_contourf_fixedz(self, x, y, z, mask1=None, mask2=None, mask3=None,
                                nan_matrix=None, ax: str = None,
                                num_cont_lines: int = 12, alpha: float = 0.75, cmap: str = 'viridis',
                                axlinewidth=0.5, axlinecolor: str = 'r', wp_x: float = None, wp_y: float = None,
                                inlinespacing: int = -10, xlabel='', ylabel: str = '', title: str = "",
                                clabel: bool = False, markerstyle: str = 'star',
                                z_min: float = None, z_max: float = None) -> None:
        """
        Draw a subplot contourf.
        The area of z where a nan can be found in nan_matrix will be shaded.

        :param x: x-coordinate
        :param y: y-coordinate
        :param z: z-coordinate
        :param nan_matrix: [optional] z-values where a nan is in nan_matrix will be plotted shaded
        :param ax: choose the axis to draw this plot
        :param num_cont_lines: [optional] number of contour lines, default to 20
        :param alpha: [optional] shading 0...1. 1 = 100%, default to 0.5
        :param cmap: [optional] cmap type, e.g. inferno [default]
        :param axlinewidth: [optional] line width of axvline and axhline, default to 0.5
        :param axlinecolor: [optional] color of axline and star, default to red
        :param wp_x: [optional] working point in x (for marker line or star marker)
        :param wp_y: [optional] working point in y (for marker line or star marker)
        :param inlinespacing: [optional] default to -10
        :param xlabel: [optional] x-label
        :param ylabel: [optional] y-label
        :param fontsize_axis: [optional] default to 9
        :param fontsize_title: [optional] default to 9
        :param title: [optional] subplot figure title
        :param clabel: [optional] True to write labels inside the plot, default to False
        :param markerstyle: [optional] marker style: 'star' or 'line'
        :param z_min: [optional] clip to minimum z-value
        :param z_max: [optional] clip to maximum z-value
        :return: -
        """
        fontsize_axis = self.fontsize
        fontsize_title = self.fontsize

        # check if z input matrix is out ouf None's only. If Ture, raise exception.
        # Note: the 1-value is a random value, hopefully no one has sum(array) with array_size
        # search_nones = z.copy()
        # search_nones[np.isnan(search_nones)] = 1
        # if np.sum(search_nones) == np.size(search_nones):
        #     raise Exception("in subplot_contourf(), z input out of None's only is not allowed")
        if np.all(np.isnan(z)):
            warning('subplot_contourf(): z input out of NaN only is not allowed!', z)

        if ax is None:
            ax = plt.gca()

        if z_min is None or z_min > np.nanmax(z):
            z_min = np.nanmin(z)
        if z_max is None or z_max < np.nanmin(z):
            z_max = np.nanmax(z)
        # To prevent error in cbar and get at least a plot even it is one color
        if z_min == z_max:
            z_min = z_min - z_min * 0.1
        # Set fixed cont_lines
        levels = np.linspace(z_min, z_max, num_cont_lines)
        # in case of nan_matrix is not set
        if nan_matrix is None:
            cs_full = ax.contourf(x, y, np.clip(z, z_min, z_max), levels=levels, alpha=1, antialiased=True, cmap=cmap,
                                  vmin=z_min, vmax=z_max)
            if not mask1 is None: ax.contour(x, y, mask1, levels=[0.5], colors=['red'], alpha=1)
            if not mask2 is None: ax.contour(x, y, mask2, levels=[0.5], colors=['blue'], alpha=1)
            if not mask3 is None: ax.contour(x, y, mask3, levels=[0.5], colors=['black'], alpha=1)
        # in case of nan_matrix is set
        else:
            # plot background 50% visible
            cs_background = ax.contourf(x, y, np.clip(z, z_min, z_max), levels=levels, alpha=alpha, antialiased=True,
                                        cmap=cmap, vmin=z_min, vmax=z_max)

            # generate matrix for foreground, 100% visible
            z_nan = z * nan_matrix

            # plot foreground, 100% visible
            # Note: levels taken from first plot
            cs_full = ax.contourf(x, y, z_nan.clip(z_min, z_max), levels=levels, alpha=1, antialiased=True, cmap=cmap,
                                  vmin=z_min, vmax=z_max)
            if not mask1 is None: ax.contour(x, y, mask1, levels=[0.5], colors=['red'], alpha=1)
            if not mask2 is None: ax.contour(x, y, mask2, levels=[0.5], colors=['blue'], alpha=1)
            if not mask3 is None: ax.contour(x, y, mask3, levels=[0.5], colors=['black'], alpha=1)

        ax.set_xlabel(xlabel, fontsize=fontsize_axis)
        ax.set_ylabel(ylabel, fontsize=fontsize_axis)
        ax.set_title(title, fontsize=fontsize_title)
        # Apply the limits to the colorbar. That way the colorbar does not depend on one plot.
        mappable = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=z_min, vmax=z_max), cmap=cmap)
        cb = plt.colorbar(mappable=mappable, ax=ax, boundaries=np.linspace(z_min, z_max, num_cont_lines + 1))
        # cb = plt.colorbar(mappable=mappable, ax=ax)
        # cb = plt.colorbar(cs_full)
        cb.ax.locator_params(nbins=num_cont_lines)
        ax.grid()
        if clabel == True:
            ax.clabel(cs_full, inline=1, inline_spacing=inlinespacing, fontsize=10, fmt='%1.1f', colors='k')
        if wp_x is not None and markerstyle.lower() == 'line':
            ax.axvline(wp_x, linewidth=axlinewidth, color=axlinecolor)
        if wp_y is not None and markerstyle.lower() == 'line':
            ax.axhline(wp_y, linewidth=axlinewidth, color=axlinecolor)
        if wp_x is not None and wp_y is not None and markerstyle.lower() == 'star':
            ax.plot(wp_x, wp_y, marker="*", color=axlinecolor)

    @timeit
    def subplot_contourf_nan(self, x, y, z, nan_matrix=None, ax: str = None,
                             num_cont_lines: int = 20, alpha: float = 0.75, cmap: str = 'inferno', axlinewidth=0.5,
                             axlinecolor: str = 'r', wp_x: float = None, wp_y: float = None, inlinespacing: int = -10,
                             xlabel='Lambda = f * L', ylabel: str = 'Turns ratio n', fontsize_axis: int = 9,
                             fontsize_title: int = 9, title: str = "", clabel: bool = False, markerstyle: str = 'star',
                             z_min: float = None, z_max: float = None) -> None:
        """
        Draw a subplot contourf.
        The area of z where a nan can be found in nan_matrix will be shaded.

        :param x: x-coordinate
        :param y: y-coordinate
        :param z: z-coordinate
        :param nan_matrix: [optional] z-values where a nan is in nan_matrix will be plotted shaded
        :param ax: choose the axis to draw this plot
        :param num_cont_lines: [optional] number of contour lines, default to 20
        :param alpha: [optional] shading 0...1. 1 = 100%, default to 0.5
        :param cmap: [optional] cmap type, e.g. inferno [default]
        :param axlinewidth: [optional] line width of axvline and axhline, default to 0.5
        :param axlinecolor: [optional] color of axline and star, default to red
        :param wp_x: [optional] working point in x (for marker line or star marker)
        :param wp_y: [optional] working point in y (for marker line or star marker)
        :param inlinespacing: [optional] default to -10
        :param xlabel: [optional] x-label
        :param ylabel: [optional] y-label
        :param fontsize_axis: [optional] default to 9
        :param fontsize_title: [optional] default to 9
        :param title: [optional] subplot figure title
        :param clabel: [optional] True to write labels inside the plot, default to False
        :param markerstyle: [optional] marker style: 'star' or 'line'
        :param z_min: [optional] clip to minimum z-value
        :param z_max: [optional] clip to maximum z-value
        :return: -
        """
        # check if z input matrix is out ouf None's only. If Ture, raise exception.
        # Note: the 1-value is a random value, hopefully no one has sum(array) with array_size
        search_nones = z.copy()
        search_nones[np.isnan(search_nones)] = 1
        if np.sum(search_nones) == np.size(search_nones):
            raise Exception("in subplot_contourf_nan(), z input out of None's only is not allowed")

        if ax is None:
            ax = plt.gca()

        if z_min is None or z_min > np.nanmax(z):
            z_min = np.nanmin(z)
        if z_max is None or z_max < np.nanmin(z):
            z_max = np.nanmax(z)
        # in case of nan_matrix is not set
        if nan_matrix is None:
            cs_full = ax.contourf(x, y, z.clip(z_min, z_max), num_cont_lines, alpha=1, antialiased=True, cmap=cmap,
                                  vmin=z_min, vmax=z_max)
        # in case of nan_matrix is set
        else:
            # plot background 50% visible
            cs_background = ax.contourf(x, y, z.clip(z_min, z_max), num_cont_lines, alpha=alpha, antialiased=True,
                                        cmap=cmap, vmin=z_min, vmax=z_max)

            # generate matrix for foreground, 100% visible
            z_nan = z * nan_matrix

            # plot foreground, 100% visible
            # Note: levels taken from first plot
            cs_full = ax.contourf(x, y, z_nan.clip(z_min, z_max), num_cont_lines, alpha=1, antialiased=True, cmap=cmap,
                                  vmin=z_min, vmax=z_max, levels=cs_background.levels)

        ax.set_xlabel(xlabel, fontsize=fontsize_axis)
        ax.set_ylabel(ylabel, fontsize=fontsize_axis)
        ax.set_title(title, fontsize=fontsize_title)
        plt.colorbar(cs_full, ax=ax)
        ax.grid()
        if clabel == True:
            ax.clabel(cs_full, inline=1, inline_spacing=inlinespacing, fontsize=10, fmt='%1.1f', colors='k')
        if wp_x is not None and markerstyle.lower() == 'line':
            ax.axvline(wp_x, linewidth=axlinewidth, color=axlinecolor)
        if wp_y is not None and markerstyle.lower() == 'line':
            ax.axhline(wp_y, linewidth=axlinewidth, color=axlinecolor)
        if wp_x is not None and wp_y is not None and markerstyle.lower() == 'star':
            ax.plot(wp_x, wp_y, marker="*", color=axlinecolor)

    def subplot(self, x, y, ax: str = None,
                xlabel='x', ylabel: str = 'y', title: str = '',
                xscale='linear', yscale='linear') -> None:
        """
        Plot a simple line plot in a subplot
        :param x:
        :param y:
        :param ax:
        :param xlabel:
        :param ylabel:
        :param title:
        :param xscale: {"linear", "log", "symlog", "logit", ...}
        :param yscale: {"linear", "log", "symlog", "logit", ...}
        """
        fontsize_axis = self.fontsize
        fontsize_title = self.fontsize

        # If no subplot axis was given find one
        if ax is None:
            ax = plt.gca()
        # Simple line plot
        ax.plot(x, y)

        ax.set_xlabel(xlabel, fontsize=fontsize_axis)
        ax.set_ylabel(ylabel, fontsize=fontsize_axis)
        ax.set_title(title, fontsize=fontsize_title)
        ax.set_xscale(xscale)
        ax.set_yscale(yscale)
        ax.grid()
        # FIXME A quick fix for layout problems, but may affect contourf plots
        plt.rcParams["figure.autolayout"] = True
        # fig.tight_layout()

    def show(self):
        if self.show_pw:
            # just to show the plots all at once
            self.pw.show()
        else:
            plt.close('all')

    def close(self):
        if self.show_pw:
            # Close figures
            # FIXME It seems that this does not close nicely
            self.pw.close()
        else:
            plt.close('all')


@timeit
def plot_modulation(x, y, z1, z2, z3, title: str = '', mask1=None, mask2=None, mask3=None,
                    maskZVS=None, Vnum=2, filename='Plot_title', latex=False):
    """
    Plots three contourf plots with a shared colorbar.

    :param fig_axes: Provide the tuple (fig, axs)
    :param x: x mesh, e.g. P
    :param y: y mesh, e.g. V2
    :param z1: z for subplot 1, e.g. phi
    :param z2: z for subplot 2, e.g. tau1
    :param z3: z for subplot 3, e.g. tau2
    :param mask1: optional mask contour line
    :param mask2: optional mask contour line
    :param mask3: optional mask contour line
    :param Vnum: Voltage number of y-axis {1, 2}
    """
    figsize = (10, 5)
    if latex:
        plt.rcParams.update({
            "text.usetex":      True,
            "font.family":      "STIXGeneral",
            "mathtext.fontset": "cm",
            "font.size":        12
        })
        plt.rcParams['figure.constrained_layout.use'] = False
        plt.rcParams["figure.autolayout"] = False
    else:
        plt.rcParams['figure.constrained_layout.use'] = False
        plt.rcParams["figure.autolayout"] = False

    if np.all(mask1 == mask1[0]): mask1 = None
    if np.all(mask2 == mask2[0]): mask2 = None
    if np.all(mask3 == mask3[0]): mask3 = None
    if np.all(maskZVS == maskZVS[0]): maskZVS = None

    # Add a new tab with subplot
    fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=figsize,
                            gridspec_kw={'width_ratios': [0.38, 0.31, 0.31]})

    num_cont_lines = 20
    cmap = 'viridis'
    pz_min = -np.pi / 4
    pz_max = np.pi / 4
    z_min = -np.pi
    z_max = np.pi
    # Clear only the 3 subplots in case we update the same figure. Colorbar stays.
    axs[0].clear()
    axs[1].clear()
    axs[2].clear()
    # Plot the contourf maps
    axs[0].contourf(x, y, z1, num_cont_lines, alpha=1, antialiased=True, cmap=cmap, vmin=pz_min, vmax=pz_max)
    if not mask1 is None: axs[0].contour(x, y, mask1, levels=[0.5], colors=['red'], alpha=1)
    if not mask2 is None: axs[0].contour(x, y, mask2, levels=[0.5], colors=['blue'], alpha=1)
    if not mask3 is None: axs[0].contour(x, y, mask3, levels=[0.5], colors=['black'], alpha=1)
    if not maskZVS is None: axs[0].contourf(x, y, np.ma.masked_where(maskZVS == 1, maskZVS), 1, alpha=0.5,
                                            antialiased=True, cmap='Greys_r', vmin=0, vmax=1)

    # Apply the limits to the colorbar. That way the colorbar does not depend on one plot.
    mappable = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=pz_min, vmax=pz_max), cmap=cmap)
    cbar = fig.colorbar(mappable, ax=axs[0], pad=0.001, boundaries=np.linspace(pz_min, pz_max, num_cont_lines),
                        ticks=[-np.pi / 4, -np.pi / 8, 0, np.pi / 8, np.pi / 4])
    if latex:
        cbar.ax.set_yticklabels(
            [r'$-\frac{1}{4} \pi$', r'$-\frac{1}{8} \pi$', r'$0$',
             r'$\frac{1}{8} \pi$', r'$\frac{1}{4} \pi$'])
    else:
        cbar.ax.set_yticklabels(['-π/4', '-π/8', '0', 'π/8', 'π/4'])

    axs[1].contourf(x, y, z2, num_cont_lines, alpha=1, antialiased=True, cmap=cmap, vmin=z_min, vmax=z_max)
    if not mask1 is None: axs[1].contour(x, y, mask1, levels=[0.5], colors=['red'], alpha=1)
    if not mask2 is None: axs[1].contour(x, y, mask2, levels=[0.5], colors=['blue'], alpha=1)
    if not mask3 is None: axs[1].contour(x, y, mask3, levels=[0.5], colors=['black'], alpha=1)
    if not maskZVS is None: axs[1].contourf(x, y, np.ma.masked_where(maskZVS == 1, maskZVS), 1, alpha=0.5,
                                            antialiased=True, cmap='Greys_r', vmin=0, vmax=1)
    axs[2].contourf(x, y, z3, num_cont_lines, alpha=1, antialiased=True, cmap=cmap, vmin=z_min, vmax=z_max)
    if not mask1 is None: axs[2].contour(x, y, mask1, levels=[0.5], colors=['red'], alpha=1)
    if not mask2 is None: axs[2].contour(x, y, mask2, levels=[0.5], colors=['blue'], alpha=1)
    if not mask3 is None: axs[2].contour(x, y, mask3, levels=[0.5], colors=['black'], alpha=1)
    if not maskZVS is None: axs[2].contourf(x, y, np.ma.masked_where(maskZVS == 1, maskZVS), 1, alpha=0.5,
                                            antialiased=True, cmap='Greys_r', vmin=0, vmax=1)
    # Set the labels
    if title: fig.suptitle(title)
    axs[0].set_title(r"$\varphi \:/\: \mathrm{rad}$" if latex else "phi in rad")
    axs[1].set_title(r"$\tau_1 \:/\: \mathrm{rad}$" if latex else "tau1 in rad")
    axs[2].set_title(r"$\tau_2 \:/\: \mathrm{rad}$" if latex else "tau2 in rad")
    for ax in axs.flat:
        if latex:
            ax.set(xlabel=r'$P \:/\: \mathrm{W}$', ylabel=r'$U_{{{:.0f}}} \:/\: \mathrm{{V}}$'.format(Vnum))
        else:
            ax.set(xlabel='P / W', ylabel='U{:.0f} / V'.format(Vnum))
        ax.label_outer()
    # Apply the limits to the colorbar. That way the colorbar does not depend on one plot.
    mappable = plt.cm.ScalarMappable(norm=plt.Normalize(vmin=z_min, vmax=z_max), cmap=cmap)
    cbar = fig.colorbar(mappable=mappable, ax=axs, fraction=0.05, pad=0.02,
                        ticks=[-np.pi, -np.pi * 3 / 4, -np.pi / 2, -np.pi / 4, 0, np.pi / 4, np.pi / 2,
                               np.pi * 3 / 4, np.pi])
    if latex:
        cbar.ax.set_yticklabels(
            [r'$-\pi$', r'$-\frac{3}{4} \pi$', r'$-\frac{1}{2} \pi$', r'$-\frac{1}{4} \pi$', r'$0$',
             r'$\frac{1}{4} \pi$', r'$\frac{1}{2} \pi$', r'$\frac{3}{4} \pi$', r'$\pi$'])
    else:
        cbar.ax.set_yticklabels(['-π', '-π3/4', '-π/2', '-π/4', '0', 'π/4', 'π/2', 'π3/4', 'π'])

    # Save plots
    metadata = {'Title':    filename,
                'Author':   'Felix Langemeier',
                'Keywords': 'python, matplotlib, dab'}
    # Save as PNG
    filename = os.path.expanduser(filename)
    filename = os.path.expandvars(filename)
    filename = os.path.abspath(filename)
    fname = filename + '.png'
    fig.savefig(fname=fname, metadata=metadata)
    # Save as PDF
    # fname = filename + '.pdf'
    # fig.savefig(fname=fname, metadata=metadata)


@timeit
def plot_rms_current(mesh_V2, mesh_P, mvvp_iLs):
    # plot
    fig, axs = plt.subplots(1, 3, sharey=True)
    fig.suptitle("DAB RMS Currents")
    cf = axs[0].contourf(mesh_P[:, 1, :], mesh_V2[:, 1, :], mvvp_iLs[:, 1, :])
    axs[1].contourf(mesh_P[:, 1, :], mesh_V2[:, 1, :], mvvp_iLs[:, 1, :])
    axs[2].contourf(mesh_P[:, 1, :], mesh_V2[:, 1, :], mvvp_iLs[:, 1, :])
    axs[0].set_title("i_Ls")
    axs[1].set_title("i_Ls")
    axs[2].set_title("i_Ls")
    for ax in axs.flat:
        ax.set(xlabel='P / W', ylabel='U2 / V')
        ax.label_outer()
    # fig.colorbar(cf, ax=axs.ravel().tolist())
    fig.colorbar(cf, ax=axs)

    # plt.show()
    return fig


def show_plot():
    # just to show the plots all at once
    plt.show()

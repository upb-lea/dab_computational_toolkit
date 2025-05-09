"""Plot the DAB calculations."""
# python libraries
import os
import datetime
import warnings
import math

# 3rd party libraries
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.axes

# own libraries
from dct.debug_tools import *
from dct.plot_window import *

class PlotDAB:
    """Class storing and managing the plot window, figs and axes."""

    pw: PlotWindow
    figs_axes: list

    def __init__(self, is_latex: bool = False, window_title: str = 'DAB Plots', figsize: tuple = (12, 5), fontsize: int = 16, show: bool = True) -> None:
        """
        Create the object with default settings for all further plots.

        :param is_latex: Use Latex fonts (if available) for labels
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
        self.latex = is_latex
        if is_latex:
            plt.rcParams.update({
                "text.usetex": True,
                "font.family": "STIXGeneral",
                "mathtext.fontset": "cm",
                # "font.size":        12
                "font.size": fontsize
            })
            plt.rcParams['figure.constrained_layout.use'] = False
            plt.rcParams["figure.autolayout"] = False
        else:
            plt.rcParams['figure.constrained_layout.use'] = False
            plt.rcParams["figure.autolayout"] = False

    def apply_spacings(self, fig):
        """
        Set default spacings for the plots.

        :param fig: matplotlib figure
        :type fig: matplotlib.plot.figure
        """
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

    def new_fig(self, nrows: int = 1, ncols: int = 1, sharex: bool = True, sharey: bool = True,
                tab_title: str = 'add Plot title') -> None:
        """
        Create a new fig in a new tab with the amount of subplots specified.

        :param nrows: number of figures in a row
        :type nrows: int
        :param ncols: number of figures in a column
        :type ncols: int
        :param sharex: 1 to share the x-axis between the figures
        :type sharex: int
        :param sharey: 1 to share the y-axis between the figures
        :type sharey: int
        :param tab_title: Set the title of the tab-selector
        :type tab_title: str
        """
        # self.figs_axes.append(plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey,
        #                                    figsize=figsize, num=num))
        fig, axs = plt.subplots(nrows=nrows, ncols=ncols, sharex=sharex, sharey=sharey, figsize=self.figsize)

        self.apply_spacings(fig)

        # Store the handlers in our list with tuples
        # TODO do we have to store axs if we can access them with fig.axes?
        self.figs_axes.append((fig, axs))
        if self.show_pw:
            self.pw.add_plot(title=tab_title, figure=fig)

    def save_fig(self, fig: plt.Figure, directory: str | None = None, name: str = '', comment: str = '', timestamp: bool = True) -> None:
        """
        Save the given fig as PNG and PDF.

        :param fig: matplotlib figure
        :type fig: matplotlib.pyplot.figure
        :param directory: Directory to store the figure
        :type directory: str
        :param name: Name of the figure
        :type name: str
        :param comment: Comment
        :type comment: str
        :param timestamp: If the filename should start with a timestamp
        :type timestamp: bool
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
                filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + name
            else:
                filename = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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
            warnings.warn("Directory does not exist!", stacklevel=2)
            sys.exit(1)

        # Save plots
        metadata = {'Title': filename,
                    'Author': 'Felix Langemeier',
                    'Keywords': 'python, matplotlib, dab'}
        # Save as PNG
        fname = os.path.join(directory, filename + '.png')
        fig.savefig(fname=fname, metadata=metadata)
        # Save as PDF
        if not __debug__:
            fname = os.path.join(directory, filename + '.pdf')
            fig.savefig(fname=fname, metadata=metadata)

    def plot_3by1(self, fig_axes: tuple, x: np.ndarray, y: np.ndarray, z1: np.ndarray, z2: np.ndarray, z3: np.ndarray, xl: str = 'x', yl: str = 'y',
                  t1: str = 'z1', t2: str = 'z2', t3: str = 'z3') -> None:
        """
        Plot three contourf plots with a shared colorbar.

        :param fig_axes: Provide the tuple (fig, axs)
        :type fig_axes: tuple
        :param x: x mesh, e.g. P
        :type x: np.ndarray
        :param y: y mesh, e.g. V2
        :type y: np.ndarray
        :param z1: z for subplot 1, e.g. phi
        :type z1: np.ndarray
        :param z2: z for subplot 2, e.g. tau1
        :type z2: np.ndarray
        :param z3: z for subplot 3, e.g. tau2
        :type z3: np.ndarray
        :param t1: title for plot 1
        :type t1: str
        :param t2: title for plot 2
        :type t2: str
        :param t3: title for plot 3
        :type t3: str
        :param xl: x label
        :type xl: str
        :param yl: y label
        :type yl: str
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
            print("update colorbar")
            cbar = fig.axes[-1]
        else:
            cbar = fig.colorbar(cf, ax=axs)
        # tight_layout and colorbar are tricky
        # fig.tight_layout()
        # Redraw the current figure
        plt.draw()

    def plot_modulation(self, x: np.ndarray, y: np.ndarray, z1: np.ndarray, z2: np.ndarray, z3: np.ndarray, title: str = '', mask1: np.ndarray | None = None,
                        mask2: np.ndarray | None = None, mask3: np.ndarray | None = None, maskZVS: np.ndarray | None = None,
                        Vnum: float = 2, tab_title: str = 'add Plot title') -> None:
        """
        Plot three contourf plots with a shared colorbar.

        :param x: x mesh, e.g. P
        :type x: np.ndarray
        :param y: y mesh, e.g. V2
        :type y: np.ndarray
        :param z1: z for subplot 1, e.g. phi
        :type z1: np.ndarray
        :param z2: z for subplot 2, e.g. tau1
        :type z2: np.ndarray
        :param z3: z for subplot 3, e.g. tau2
        :type z3: np.ndarray
        :param mask1: optional mask contour line
        :type mask1: np.ndarray
        :param mask2: optional mask contour line
        :type mask2: np.ndarray
        :param mask3: optional mask contour line
        :type mask3: np.ndarray
        :param Vnum: Voltage number of y-axis {1, 2}
        :type Vnum: int
        :param tab_title: Set the title of the tab-selector
        :type tab_title: str
        :param maskZVS: mask for ZVS
        :type maskZVS: np.ndarray
        :param title: title of plot
        :type title: str
        """
        # Add a new tab with subplot
        fig, axs = plt.subplots(nrows=1, ncols=3, sharex=True, sharey=True, figsize=self.figsize,
                                gridspec_kw={'width_ratios': [0.38, 0.31, 0.31]})
        self.apply_spacings(fig)
        # Store the handlers in our list with tuples
        # TODO do we have to store axs if we can access them with fig.axes?
        self.figs_axes.append((fig, axs))
        if self.show_pw:
            self.pw.add_plot(title=tab_title, figure=fig)
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
        if mask1 is not None:
            axs[0].contour(x, y, mask1, levels=[0.5], colors=['red'], alpha=1)
        if mask2 is not None:
            axs[0].contour(x, y, mask2, levels=[0.5], colors=['blue'], alpha=1)
        if mask3 is not None:
            axs[0].contour(x, y, mask3, levels=[0.5], colors=['black'], alpha=1)
        if maskZVS is not None:
            axs[0].contourf(x, y, np.ma.masked_where(maskZVS == 1, maskZVS), 1, alpha=0.5, antialiased=True, cmap='Greys_r', vmin=0, vmax=1)

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
        if mask1 is not None:
            axs[1].contour(x, y, mask1, levels=[0.5], colors=['red'], alpha=1)
        if mask2 is not None:
            axs[1].contour(x, y, mask2, levels=[0.5], colors=['blue'], alpha=1)
        if mask3 is not None:
            axs[1].contour(x, y, mask3, levels=[0.5], colors=['black'], alpha=1)
        if maskZVS is not None:
            axs[1].contourf(x, y, np.ma.masked_where(maskZVS == 1, maskZVS), 1, alpha=0.5, antialiased=True, cmap='Greys_r', vmin=0, vmax=1)
        axs[2].contourf(x, y, z3, num_cont_lines, alpha=1, antialiased=True, cmap=cmap, vmin=z_min, vmax=z_max)
        if mask1 is not None:
            axs[2].contour(x, y, mask1, levels=[0.5], colors=['red'], alpha=1)
        if mask2 is not None:
            axs[2].contour(x, y, mask2, levels=[0.5], colors=['blue'], alpha=1)
        if mask3 is not None:
            axs[2].contour(x, y, mask3, levels=[0.5], colors=['black'], alpha=1)
        if maskZVS is not None:
            axs[2].contourf(x, y, np.ma.masked_where(maskZVS == 1, maskZVS), 1, alpha=0.5, antialiased=True, cmap='Greys_r', vmin=0, vmax=1)
        # Set the labels
        if title:
            fig.suptitle(title)
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
            print("update colorbar not implemented")
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

    def plot_modulation_classic(self, fig_axes: tuple, x: np.ndarray, y: np.ndarray, z1: np.ndarray, z2: np.ndarray, z3: np.ndarray, title: str = '',
                                mask1: np.ndarray | None = None, mask2: np.ndarray | None = None, mask3: np.ndarray | None = None,
                                maskZVS: np.ndarray | None = None) -> None:
        """
        Plot three contourf plots with a shared colorbar.

        :param fig_axes: Provide the tuple (fig, axs)
        :type fig_axes: tuple
        :param x: x mesh, e.g. P
        :type x: np.ndarray
        :param y: y mesh, e.g. V2
        :type y: np.ndarray
        :param z1: z for subplot 1, e.g. phi
        :type z1: np.ndarray
        :param z2: z for subplot 2, e.g. tau1
        :type z2: np.ndarray
        :param z3: z for subplot 3, e.g. tau2
        :type z3: np.ndarray
        :param mask1: optional mask contour line
        :type mask1: np.ndarray
        :param mask2: optional mask contour line
        :type mask2: np.ndarray
        :param mask3: optional mask contour line
        :type mask3: np.ndarray
        :param title: title of plot
        :type title: str
        :param maskZVS: mask for ZVS
        :type maskZVS: np.ndarray
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
        if mask1 is not None:
            axs[0].contour(x, y, mask1, levels=[0.5], colors=['red'], alpha=1)
        if mask2 is not None:
            axs[0].contour(x, y, mask2, levels=[0.5], colors=['blue'], alpha=1)
        if mask3 is not None:
            axs[0].contour(x, y, mask3, levels=[0.5], colors=['black'], alpha=1)
        if maskZVS is not None:
            axs[0].contourf(x, y, np.ma.masked_where(maskZVS == 1, maskZVS), 1, alpha=0.5, antialiased=True, cmap='Greys_r', vmin=0, vmax=1)
        axs[1].contourf(x, y, z2, num_cont_lines, alpha=1, antialiased=True, cmap=cmap, vmin=z_min, vmax=z_max)
        if mask1 is not None:
            axs[1].contour(x, y, mask1, levels=[0.5], colors=['red'], alpha=1)
        if mask2 is not None:
            axs[1].contour(x, y, mask2, levels=[0.5], colors=['blue'], alpha=1)
        if mask3 is not None:
            axs[1].contour(x, y, mask3, levels=[0.5], colors=['black'], alpha=1)
        if maskZVS is not None:
            axs[1].contourf(x, y, np.ma.masked_where(maskZVS == 1, maskZVS), 1, alpha=0.5, antialiased=True, cmap='Greys_r', vmin=0, vmax=1)
        axs[2].contourf(x, y, z3, num_cont_lines, alpha=1, antialiased=True, cmap=cmap, vmin=z_min, vmax=z_max)
        if mask1 is not None:
            axs[2].contour(x, y, mask1, levels=[0.5], colors=['red'], alpha=1)
        if mask2 is not None:
            axs[2].contour(x, y, mask2, levels=[0.5], colors=['blue'], alpha=1)
        if mask3 is not None:
            axs[2].contour(x, y, mask3, levels=[0.5], colors=['black'], alpha=1)
        if maskZVS is not None:
            axs[2].contourf(x, y, np.ma.masked_where(maskZVS == 1, maskZVS), 1, alpha=0.5, antialiased=True, cmap='Greys_r', vmin=0, vmax=1)
        # Set the labels
        if title:
            fig.suptitle(title)
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
            warnings.warn("update colorbar not implemented", stacklevel=2)
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
    def plot_rms_current(self, mesh_V2: np.ndarray, mesh_P: np.ndarray, mvvp_iLs: np.ndarray) -> plt.Figure:
        """
        Plot RMS currents.

        :param mesh_V2: mesh of voltage v2 in V
        :type mesh_V2: np.ndarray
        :param mesh_P: mesh of the power P in W
        :type mesh_P: np.ndarray
        :param mvvp_iLs: current i_Ls in A
        :type mvvp_iLs: np.ndarray
        """
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

    def subplot_contourf(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, mask1: np.ndarray | None = None, mask2: np.ndarray | None = None,
                         mask3: np.ndarray | None = None,
                         nan_matrix: np.ndarray | None = None, ax: matplotlib.axes.Axes | None = None,
                         num_cont_lines: int = 12, alpha: float = 0.75, cmap: str = 'viridis',
                         axlinewidth: float = 0.5, axlinecolor: str = 'r', wp_x: float | None = None, wp_y: float | None = None,
                         inlinespacing: int = -10, xlabel: str = '', ylabel: str = '', title: str = "", clabel: bool = False,
                         markerstyle: str = 'star',
                         z_min: float | None = None, z_max: float | None = None,
                         square: bool = False, same_xy_ticks: bool = False) -> None:
        """
        Draw a subplot contourf.

        The area of z where a nan can be found in nan_matrix will be shaded.

        :param x: x-coordinate
        :type x: np.ndarray
        :param y: y-coordinate
        :type y: np.ndarray
        :param z: z-coordinate
        :type z: np.ndarray
        :param nan_matrix: [optional] z-values where a nan is in nan_matrix will be plotted shaded
        :type nan_matrix: np.ndarray
        :param ax: choose the axis to draw this plot
        :type ax: str
        :param num_cont_lines: [optional] number of contour lines, default to 20
        :type num_cont_lines: int
        :param alpha: [optional] shading 0...1. 1 = 100%, default to 0.5
        :type alpha: float
        :param cmap: [optional] cmap type, e.g. inferno
        :type cmap: str
        :param axlinewidth: [optional] line width of axvline and axhline, default to 0.5
        :type axlinewidth: float
        :param axlinecolor: [optional] color of axline and star, default to red
        :type axlinecolor: str
        :param wp_x: [optional] working point in x (for marker line or star marker)
        :type wp_x: float
        :param wp_y: [optional] working point in y (for marker line or star marker)
        :type wp_y: float
        :param inlinespacing: [optional] default to -10
        :type inlinespacing: float
        :param xlabel: [optional] x-label
        :type xlabel: str
        :param ylabel: [optional] y-label
        :type ylabel: str
        :param title: [optional] subplot figure title
        :type title: str
        :param clabel: [optional] True to write labels inside the plot, default to False
        :type clabel: str
        :param markerstyle: [optional] marker style: 'star' or 'line'
        :type markerstyle: str
        :param z_min: [optional] clip to minimum z-value
        :type z_min: float
        :param z_max: [optional] clip to maximum z-value
        :type z_max: float
        :param mask1: optional mask contour line
        :type mask1: np.ndarray
        :param mask2: optional mask contour line
        :type mask2: np.ndarray
        :param mask3: optional mask contour line
        :type mask3: np.ndarray
        :param square:
        :type square: bool
        :param same_xy_ticks:
        :type same_xy_ticks: bool
        """
        fontsize_axis = self.fontsize
        fontsize_title = self.fontsize

        # check if z input matrix is out of None's only. If True, raise exception.
        # Note: the 1-value is a random value, hopefully no one has sum(array) with array_size
        # search_nones = z.copy()
        # search_nones[np.isnan(search_nones)] = 1
        # if np.sum(search_nones) == np.size(search_nones):
        #     raise Exception("in subplot_contourf(), z input out of None's only is not allowed")
        if np.all(np.isnan(z)):
            warnings.warn(f'subplot_contourf(): z input {z} out of NaN only is not allowed!', stacklevel=2)

        if ax is None:
            ax = plt.gca()

        if z_min is None or z_min > np.nanmax(z):
            z_min = np.nanmin(z)
        if z_max is None or z_max < np.nanmin(z):
            z_max = np.nanmax(z)
        if z_min is None:  # mypy issue handling
            raise TypeError("Failure in assignment.")
        if z_max is None:  # mypy issue handling
            raise TypeError("Failure in assignment.")
        # To prevent error in cbar and get at least a plot even it is one color
        if z_min == z_max and z_min is not None and z_max is not None:
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
            if mask1 is not None:
                ax.contour(x, y, mask1, levels=[0.5], colors=['red'], alpha=1)
            if mask2 is not None:
                ax.contour(x, y, mask2, levels=[0.5], colors=['blue'], alpha=1)
            if mask3 is not None:
                ax.contour(x, y, mask3, levels=[0.5], colors=['black'], alpha=1)
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
            if mask1 is not None:
                ax.contour(x, y, mask1, levels=[0.5], colors=['red'], alpha=1)
            if mask2 is not None:
                ax.contour(x, y, mask2, levels=[0.5], colors=['blue'], alpha=1)
            if mask3 is not None:
                ax.contour(x, y, mask3, levels=[0.5], colors=['black'], alpha=1)

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
        if clabel:
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

    def subplot_contourf_fixedz(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, mask1: np.ndarray | None = None, mask2: np.ndarray | None = None,
                                mask3: np.ndarray | None = None,
                                nan_matrix: np.ndarray | None = None, ax: matplotlib.axes.Axes | None = None,
                                num_cont_lines: int = 12, alpha: float = 0.75, cmap: str = 'viridis',
                                axlinewidth: float = 0.5, axlinecolor: str = 'r', wp_x: float | None = None, wp_y: float | None = None,
                                inlinespacing: int = -10, xlabel: str = '', ylabel: str = '', title: str = "",
                                clabel: bool = False, markerstyle: str = 'star',
                                z_min: float | None = None, z_max: float | None = None) -> None:
        """
        Draw a subplot contourf.

        The area of z where a nan can be found in nan_matrix will be shaded.

        :param x: x-coordinate
        :type x: np.ndarray
        :param y: y-coordinate
        :type y: np.ndarray
        :param z: z-coordinate
        :type z: np.ndarray
        :param nan_matrix: [optional] z-values where a nan is in nan_matrix will be plotted shaded
        :type nan_matrix: np.ndarray
        :param ax: choose the axis to draw this plot
        :type ax: str
        :param num_cont_lines: [optional] number of contour lines, default to 20
        :type num_cont_lines: int
        :param alpha: [optional] shading 0...1. 1 = 100%, default to 0.5
        :type alpha: float
        :param cmap: [optional] cmap type, e.g. inferno
        :type cmap: str
        :param axlinewidth: [optional] line width of axvline and axhline, default to 0.5
        :type axlinewidth: float
        :param axlinecolor: [optional] color of axline and star, default to red
        :type axlinecolor: str
        :param wp_x: [optional] working point in x (for marker line or star marker)
        :type wp_x: float
        :param wp_y: [optional] working point in y (for marker line or star marker)
        :type wp_y: float
        :param inlinespacing: [optional] default to -10
        :type inlinespacing: float
        :param xlabel: [optional] x-label
        :type xlabel: str
        :param ylabel: [optional] y-label
        :type ylabel: str
        :param title: [optional] subplot figure title
        :type title: str
        :param clabel: [optional] True to write labels inside the plot, default to False
        :type clabel: str
        :param markerstyle: [optional] marker style: 'star' or 'line'
        :type markerstyle: str
        :param z_min: [optional] clip to minimum z-value
        :type z_min: float
        :param z_max: [optional] clip to maximum z-value
        :type z_max: float
        :param mask1: optional mask contour line
        :type mask1: np.ndarray
        :param mask2: optional mask contour line
        :type mask2: np.ndarray
        :param mask3: optional mask contour line
        :type mask3: np.ndarray
        """
        fontsize_axis = self.fontsize
        fontsize_title = self.fontsize

        # check if z input matrix is out of None's only. If True, raise exception.
        # Note: the 1-value is a random value, hopefully no one has sum(array) with array_size
        # search_nones = z.copy()
        # search_nones[np.isnan(search_nones)] = 1
        # if np.sum(search_nones) == np.size(search_nones):
        #     raise Exception("in subplot_contourf(), z input out of None's only is not allowed")
        if np.all(np.isnan(z)):
            warnings.warn(f'subplot_contourf(): z input {z} out of NaN only is not allowed!', stacklevel=2)

        if ax is None:
            ax = plt.gca()

        if z_min is None or z_min > np.nanmax(z):
            z_min = np.nanmin(z)
        if z_max is None or z_max < np.nanmin(z):
            z_max = np.nanmax(z)
        if z_min is None:  # mypy issue handling
            raise TypeError("Failure in assignment.")
        if z_max is None:  # mypy issue handling
            raise TypeError("Failure in assignment.")
        # To prevent error in cbar and get at least a plot even it is one color
        if z_min == z_max:
            z_min = z_min - z_min * 0.1
        # Set fixed cont_lines
        levels = np.linspace(z_min, z_max, num_cont_lines)
        # in case of nan_matrix is not set
        if nan_matrix is None:
            cs_full = ax.contourf(x, y, np.clip(z, z_min, z_max), levels=levels, alpha=1, antialiased=True, cmap=cmap,
                                  vmin=z_min, vmax=z_max)
            if mask1 is not None:
                ax.contour(x, y, mask1, levels=[0.5], colors=['red'], alpha=1)
            if mask2 is not None:
                ax.contour(x, y, mask2, levels=[0.5], colors=['blue'], alpha=1)
            if mask3 is not None:
                ax.contour(x, y, mask3, levels=[0.5], colors=['black'], alpha=1)
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
            if mask1 is not None:
                ax.contour(x, y, mask1, levels=[0.5], colors=['red'], alpha=1)
            if mask2 is not None:
                ax.contour(x, y, mask2, levels=[0.5], colors=['blue'], alpha=1)
            if mask3 is not None:
                ax.contour(x, y, mask3, levels=[0.5], colors=['black'], alpha=1)

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
        if clabel:
            ax.clabel(cs_full, inline=1, inline_spacing=inlinespacing, fontsize=10, fmt='%1.1f', colors='k')
        if wp_x is not None and markerstyle.lower() == 'line':
            ax.axvline(wp_x, linewidth=axlinewidth, color=axlinecolor)
        if wp_y is not None and markerstyle.lower() == 'line':
            ax.axhline(wp_y, linewidth=axlinewidth, color=axlinecolor)
        if wp_x is not None and wp_y is not None and markerstyle.lower() == 'star':
            ax.plot(wp_x, wp_y, marker="*", color=axlinecolor)

    @timeit
    def subplot_contourf_nan(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, nan_matrix: np.ndarray | None = None, ax: matplotlib.axes.Axes | None = None,
                             num_cont_lines: int = 20, alpha: float = 0.75, cmap: str = 'inferno', axlinewidth: float = 0.5,
                             axlinecolor: str = 'r', wp_x: float | None = None, wp_y: float | None = None, inlinespacing: int = -10,
                             xlabel: str = 'Lambda = f * L', ylabel: str = 'Turns ratio n', fontsize_axis: int = 9,
                             fontsize_title: int = 9, title: str = "", clabel: bool = False, markerstyle: str = 'star',
                             z_min: float | None = None, z_max: float | None = None) -> None:
        """
        Draw a subplot contourf.

        The area of z where a nan can be found in nan_matrix will be shaded.

        :param x: x-coordinate
        :type x: np.ndarray
        :param y: y-coordinate
        :type y: np.ndarray
        :param z: z-coordinate
        :type z: np.ndarray
        :param nan_matrix: [optional] z-values where a nan is in nan_matrix will be plotted shaded
        :type nan_matrix: np.ndarray
        :param ax: choose the axis to draw this plot
        :type ax: str
        :param num_cont_lines: [optional] number of contour lines, default to 20
        :type num_cont_lines: int
        :param alpha: [optional] shading 0...1. 1 = 100%, default to 0.5
        :type alpha: float
        :param cmap: [optional] cmap type, e.g. inferno
        :type cmap: str
        :param axlinewidth: [optional] line width of axvline and axhline, default to 0.5
        :type axlinewidth: float
        :param axlinecolor: [optional] color of axline and star, default to red
        :type axlinecolor: str
        :param wp_x: [optional] working point in x (for marker line or star marker)
        :type wp_x: float
        :param wp_y: [optional] working point in y (for marker line or star marker)
        :type wp_y: float
        :param inlinespacing: [optional] default to -10
        :type inlinespacing: float
        :param xlabel: [optional] x-label
        :type xlabel: str
        :param ylabel: [optional] y-label
        :type ylabel: str
        :param fontsize_axis: [optional] default to 9
        :type fontsize_axis: float
        :param fontsize_title: [optional] default to 9
        :type fontsize_title: float
        :param title: [optional] subplot figure title
        :type title: str
        :param clabel: [optional] True to write labels inside the plot, default to False
        :type clabel: str
        :param markerstyle: [optional] marker style: 'star' or 'line'
        :type markerstyle: str
        :param z_min: [optional] clip to minimum z-value
        :type z_min: float
        :param z_max: [optional] clip to maximum z-value
        :type z_max: float
        """
        # check if z input matrix is out of None's only. If True, raise exception.
        # Note: the 1-value is a random value, hopefully no one has sum(array) with array_size
        search_nones = z.copy()
        search_nones[np.isnan(search_nones)] = 1
        if np.sum(search_nones) == np.size(search_nones):
            raise Exception("in subplot_contourf_nan(), z input out of None values only is not allowed")

        if ax is None:
            ax = plt.gca()

        if z_min is None or z_min > np.nanmax(z):
            z_min = np.nanmin(z)
        if z_max is None or z_max < np.nanmin(z):
            z_max = np.nanmax(z)
        if z_min is None or z_max is None:
            raise ValueError("Issue with setting z_min or z_max")  # mypy workaround
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
        if clabel:
            ax.clabel(cs_full, inline=1, inline_spacing=inlinespacing, fontsize=10, fmt='%1.1f', colors='k')
        if wp_x is not None and markerstyle.lower() == 'line':
            ax.axvline(wp_x, linewidth=axlinewidth, color=axlinecolor)
        if wp_y is not None and markerstyle.lower() == 'line':
            ax.axhline(wp_y, linewidth=axlinewidth, color=axlinecolor)
        if wp_x is not None and wp_y is not None and markerstyle.lower() == 'star':
            ax.plot(wp_x, wp_y, marker="*", color=axlinecolor)

    def subplot(self, x: np.ndarray, y: np.ndarray, ax: matplotlib.axes.Axes | None = None,
                xlabel: str = 'x', ylabel: str = 'y', title: str = '',
                xscale: str = 'linear', yscale: str = 'linear') -> None:
        """
        Plot a simple line plot in a subplot.

        :param x:
        :type x: np.ndarray
        :param y:
        :type y: np.ndarray
        :param ax: axis
        :type ax: str
        :param xlabel: x label
        :type xlabel: str
        :param ylabel: y label
        :type ylabel: str
        :param title: title
        :type title: str
        :param xscale: {"linear", "log", "symlog", "logit", ...}
        :type xscale: str
        :param yscale: {"linear", "log", "symlog", "logit", ...}
        :type yscale: str
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
        """Show the plots all at once."""
        if self.show_pw:

            self.pw.show()
        else:
            plt.close('all')

    def close(self):
        """Close plot window."""
        if self.show_pw:
            # Close figures
            # FIXME: It seems that this does not close nicely
            self.pw.close()
        else:
            plt.close('all')


@timeit
def plot_modulation(x: np.ndarray, y: np.ndarray, z1: np.ndarray, z2: np.ndarray, z3: np.ndarray, title: str = '', mask1: np.ndarray | None = None,
                    mask2: np.ndarray | None = None, mask3: np.ndarray | None = None,
                    maskZVS: np.ndarray | None = None, Vnum: int = 2, filename: str = 'Plot_title', latex: bool = False) -> None:
    """
    Plot three contourf plots with a shared colorbar.

    :param x: x mesh, e.g. P
    :type x: np.ndarray
    :param y: y mesh, e.g. V2
    :type y: np.ndarray
    :param z1: z for subplot 1, e.g. phi
    :type z1: np.ndarray
    :param z2: z for subplot 2, e.g. tau1
    :type z2: np.ndarray
    :param z3: z for subplot 3, e.g. tau2
    :type z3: np.ndarray
    :param mask1: optional mask contour line
    :type mask1: np.ndarray
    :param mask2: optional mask contour line
    :type mask2: np.ndarray
    :param mask3: optional mask contour line
    :type mask3: np.ndarray
    :param Vnum: Voltage number of y-axis {1, 2}
    :type Vnum: int
    :param maskZVS: ZVS mask
    :type maskZVS: np.ndarray
    :param Vnum:
    :type Vnum: int
    :param filename: name of the file
    :type filename: str
    :param latex: True to set font to LaTeX font
    :type latex: bool
    :param title: title of the plot
    :type title: str
    """
    figsize = (10, 5)
    if latex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "cm",
            "font.size": 12
        })
        plt.rcParams['figure.constrained_layout.use'] = False
        plt.rcParams["figure.autolayout"] = False
    else:
        plt.rcParams['figure.constrained_layout.use'] = False
        plt.rcParams["figure.autolayout"] = False

    if mask1 is not None:
        if np.all(mask1 == mask1[0]):
            mask1 = None
    if mask2 is not None:
        if np.all(mask2 == mask2[0]):
            mask2 = None
    if mask3 is not None:
        if np.all(mask3 == mask3[0]):
            mask3 = None
    if maskZVS is not None:
        if np.all(maskZVS == maskZVS[0]):  # type: ignore
            maskZVS = None

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
    if mask1 is not None:
        axs[0].contour(x, y, mask1, levels=[0.5], colors=['red'], alpha=1)
    if mask2 is not None:
        axs[0].contour(x, y, mask2, levels=[0.5], colors=['blue'], alpha=1)
    if mask3 is not None:
        axs[0].contour(x, y, mask3, levels=[0.5], colors=['black'], alpha=1)
    if maskZVS is not None:
        axs[0].contourf(x, y, np.ma.masked_where(maskZVS == 1, maskZVS), 1, alpha=0.5, antialiased=True, cmap='Greys_r', vmin=0, vmax=1)

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
    if mask1 is not None:
        axs[1].contour(x, y, mask1, levels=[0.5], colors=['red'], alpha=1)
    if mask2 is not None:
        axs[1].contour(x, y, mask2, levels=[0.5], colors=['blue'], alpha=1)
    if mask3 is not None:
        axs[1].contour(x, y, mask3, levels=[0.5], colors=['black'], alpha=1)
    if maskZVS is not None:
        axs[1].contourf(x, y, np.ma.masked_where(maskZVS == 1, maskZVS), 1, alpha=0.5, antialiased=True, cmap='Greys_r', vmin=0, vmax=1)
    axs[2].contourf(x, y, z3, num_cont_lines, alpha=1, antialiased=True, cmap=cmap, vmin=z_min, vmax=z_max)
    if mask1 is not None:
        axs[2].contour(x, y, mask1, levels=[0.5], colors=['red'], alpha=1)
    if mask2 is not None:
        axs[2].contour(x, y, mask2, levels=[0.5], colors=['blue'], alpha=1)
    if mask3 is not None:
        axs[2].contour(x, y, mask3, levels=[0.5], colors=['black'], alpha=1)
    if maskZVS is not None:
        axs[2].contourf(x, y, np.ma.masked_where(maskZVS == 1, maskZVS), 1, alpha=0.5, antialiased=True, cmap='Greys_r', vmin=0, vmax=1)
    # Set the labels
    if title:
        fig.suptitle(title)
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
    metadata = {'Title': filename,
                'Author': 'Felix Langemeier',
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
def plot_rms_current(mesh_v2: np.ndarray, mesh_p: np.ndarray, mvvp_i_ls: np.ndarray) -> plt.Figure:
    """
    Plot the RMS currents.

    :param mesh_v2: mesh of voltage v2 in V
    :type mesh_v2: np.ndarray
    :param mesh_p: mesh of the power P in W
    :type mesh_p: np.ndarray
    :param mvvp_i_ls: current i_Ls in A
    :type mvvp_i_ls: np.ndarray
    """
    # plot
    fig, axs = plt.subplots(1, 3, sharey=True)
    fig.suptitle("DAB RMS Currents")
    cf = axs[0].contourf(mesh_p[:, 1, :], mesh_v2[:, 1, :], mvvp_i_ls[:, 1, :])
    axs[1].contourf(mesh_p[:, 1, :], mesh_v2[:, 1, :], mvvp_i_ls[:, 1, :])
    axs[2].contourf(mesh_p[:, 1, :], mesh_v2[:, 1, :], mvvp_i_ls[:, 1, :])
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
    """Show the plots all at once."""
    plt.show()

"""GUI application."""
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QTabWidget, QVBoxLayout
import sys


# prevent NoneType error for versions of matplotlib 3.1.0rc1+ by calling matplotlib.use()
# For more on why it's necessary, see
# https://stackoverflow.com/questions/59656632/using-qt5agg-backend-with-matplotlib-3-1-2-get-backend-changes-behavior
# matplotlib.use('qt5agg')


class PlotWindow:
    """Class to initialize the GUI."""

    def __init__(self, parent: None = None, window_title: str = 'plot window', figsize: tuple = (12.8, 8)) -> None:
        """Initialize PlotWindow instance."""
        self.app = QApplication(sys.argv)
        self.MainWindow = QMainWindow()
        self.MainWindow.__init__()  # type: ignore
        self.MainWindow.setWindowTitle(window_title)
        self.canvases: list[FigureCanvas] = []
        self.figure_handles: list[plt.Figure] = []
        self.toolbar_handles: list[NavigationToolbar] = []
        self.tab_handles: list[QWidget] = []
        self.current_window = -1
        self.tabs = QTabWidget()
        self.MainWindow.setCentralWidget(self.tabs)
        self.MainWindow.resize(int(figsize[0] * 100) + 24, int(figsize[1] * 100) + 109)
        self.figsize = figsize
        self.MainWindow.show()

    def add_plot(self, title: str, figure: plt.Figure) -> None:
        """Add plot to the GUI instance.

        :param title: title for the plot
        :type title: str
        :param figure: Figure to add the plot
        :type figure: matplotlib.Figure
        """
        new_tab = QWidget()
        layout = QVBoxLayout()
        new_tab.setLayout(layout)

        if self.figsize == (5, 4):
            figure.subplots_adjust(left=0.18, right=0.99, bottom=0.15, top=0.92, wspace=0.17, hspace=0.2)
        if self.figsize == (5, 5):
            figure.subplots_adjust(left=0.17, right=0.98, bottom=0.12, top=0.94, wspace=0.17, hspace=0.2)
        if self.figsize == (10, 5):
            figure.subplots_adjust(left=0.077, right=0.955, bottom=0.127, top=0.935, wspace=0.17, hspace=0.25)
        if self.figsize == (12, 5):
            figure.subplots_adjust(left=0.062, right=0.97, bottom=0.117, top=0.935, wspace=0.17, hspace=0.25)
        if self.figsize == (15, 5):
            figure.subplots_adjust(left=0.065, right=0.975, bottom=0.15, top=0.93, wspace=0.17, hspace=0.25)

        new_canvas = FigureCanvas(figure)
        new_toolbar = NavigationToolbar(new_canvas, new_tab)

        layout.addWidget(new_canvas)
        layout.addWidget(new_toolbar)
        self.tabs.addTab(new_tab, title)

        self.toolbar_handles.append(new_toolbar)
        self.canvases.append(new_canvas)
        self.figure_handles.append(figure)
        self.tab_handles.append(new_tab)

    def show(self):
        """Show the GUI application."""
        self.app.exec_()

    def close(self):
        """Close the GUI application."""
        self.app.exit()


if __name__ == '__main__':
    import numpy as np

    pw = PlotWindow()

    x = np.arange(0, 10, 0.001)

    f = plt.figure()
    ysin = np.sin(x)
    plt.plot(x, ysin, '--')
    pw.add_plot("sin", f)

    f = plt.figure()
    ycos = np.cos(x)
    plt.plot(x, ycos, '--')
    pw.add_plot("cos", f)
    pw.show()

    # sys.exit(app.exec_())

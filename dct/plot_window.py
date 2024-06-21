"""GUI application."""
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QMainWindow, QApplication, QWidget, QTabWidget, QVBoxLayout
import sys

import matplotlib

# prevent NoneType error for versions of matplotlib 3.1.0rc1+ by calling matplotlib.use()
# For more on why it's necessary, see
# https://stackoverflow.com/questions/59656632/using-qt5agg-backend-with-matplotlib-3-1-2-get-backend-changes-behavior
matplotlib.use('qt5agg')


class PlotWindow:
    """Class to initialize the GUI."""

    def __init__(self, parent=None, window_title: str = 'plot window', figsize=(12.8, 8)):
        """Initialize PlotWindow instance."""
        self.app = QApplication(sys.argv)
        self.MainWindow = QMainWindow()
        self.MainWindow.__init__()
        self.MainWindow.setWindowTitle(window_title)
        self.canvases = []
        self.figure_handles = []
        self.toolbar_handles = []
        self.tab_handles = []
        self.current_window = -1
        self.tabs = QTabWidget()
        self.MainWindow.setCentralWidget(self.tabs)
        self.MainWindow.resize(int(figsize[0] * 100) + 24, int(figsize[1] * 100) + 109)
        self.figsize = figsize
        self.MainWindow.show()

    def addPlot(self, title, figure):
        """Add plot to the GUI instance."""
        new_tab = QWidget()
        layout = QVBoxLayout()
        new_tab.setLayout(layout)

        # # Set some default spacings
        # figure.subplots_adjust(left=0.05, right=0.98, bottom=0.08, top=0.91, wspace=0.06, hspace=0.2)
        # if self.figsize == (16, 8):
        #     figure.subplots_adjust(left=0.04, right=0.995, bottom=0.065, top=0.96, wspace=0.07, hspace=0.2)
        # if self.figsize == (12.8, 8):
        #     figure.subplots_adjust(left=0.05, right=0.98, bottom=0.06, top=0.96, wspace=0.12, hspace=0.2)
        # if self.figsize == (10, 5):
        #     figure.subplots_adjust(left=0.062, right=0.975, bottom=0.092, top=0.94, wspace=0.17, hspace=0.2)
        # if self.figsize == (12, 5):
        #     figure.subplots_adjust(left=0.062, right=0.97, bottom=0.117, top=0.935, wspace=0.17, hspace=0.25)
        # for fontsize 16
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
    pw.addPlot("sin", f)

    f = plt.figure()
    ycos = np.cos(x)
    plt.plot(x, ycos, '--')
    pw.addPlot("cos", f)
    pw.show()

    # sys.exit(app.exec_())

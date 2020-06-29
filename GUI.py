import wx

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import (
    NavigationToolbar2WxAgg as NavigationToolbar,
)
from matplotlib.figure import Figure
from matplotlib.widgets import Cursor

from typing import Tuple
from typing import Optional

import pkg_resources.py2_warn

from scipy.signal import find_peaks

import pandas as pd

import numpy as np
import os


class MyApp(wx.App):
    """Main application class."""

    def __init__(self):
        super(MyApp, self).__init__(clearSigInt=True)

        self.mainFrame = MyFrame(None)
        self.mainFrame.Show()


class MyFrame(wx.Frame):
    """Class which generates a frame.
    
    All of the visual positioning of the blocks are described here.
    """

    def __init__(
        self, parent, title="Thickness Interferometry", pos=(100, 100)
    ) -> None:

        super(MyFrame, self).__init__(parent, title=title, pos=pos)
        self.init_frame()
        self.SetBackgroundColour("white")

    def init_frame(self) -> None:
        # Sizers
        self.mainSizer = wx.BoxSizer(wx.HORIZONTAL)
        self.graphSizer = wx.BoxSizer()
        self.controlSizer = wx.BoxSizer(wx.VERTICAL)

        # Panels
        self.graph = MyPanel(self)
        self.graph.draw_graph()
        self.graphs = MyPanel(self)
        self.graphs.draw_graphs()
        self.graphs.Hide()
        self.control = MyPanel(self)
        self.control.show_buttons()
        self.control.graph = self.graph
        self.control.graphs = self.graphs

        #
        self.graphSizer.Add(self.graph, 0, wx.ALL, 5)
        self.graphSizer.Add(self.graphs, 0, wx.ALL, 5)
        self.controlSizer.Add(self.control, 0, wx.ALL, 5)
        self.mainSizer.Add(self.graphSizer, 0, wx.LEFT, 5)
        self.mainSizer.Add(self.controlSizer, 0, wx.RIGHT, 5)
        self.SetSizer(self.mainSizer)
        self.Fit()


class MyPanel(wx.Panel):
    """Custom panel class."""

    def __init__(self, parent: MyFrame) -> None:
        super(MyPanel, self).__init__(parent=parent)

    def show_buttons(self) -> None:
        hbox = wx.BoxSizer(wx.VERTICAL)
        self.btn = wx.Button(self, label="Open Text File")
        self.btn.Bind(wx.EVT_BUTTON, self.on_open)

        # -----------------
        self.cwd = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "Refraction_index"
        )

        refl_files = get_files_list(self.cwd)
        self.wvlngh, self.n_coef = refection_coef_read(
            os.path.join(self.cwd, refl_files[1])
        )
        cb = wx.ComboBox(
            self, choices=refl_files, style=wx.CB_READONLY, value=refl_files[1]
        )
        cb.Bind(wx.EVT_COMBOBOX, self.on_select)
        self.st = wx.StaticText(self, label="")
        self.st.SetLabel("Refractive index file:")
        # -----------------

        self.auto = wx.Button(self, label="Auto GaAs depth")
        self.auto.Enable(False)
        self.auto.Bind(wx.EVT_BUTTON, self.calc_auto)
        # ------------------

        self.calc_btn = wx.Button(self, label="Calculate depth")
        self.calc_btn.Enable(False)
        self.calc_btn.Bind(wx.EVT_BUTTON, self.calc_choice)
        # ------------------

        self.chk_box = wx.CheckBox(self, label="Full Report")

        hbox.Add(self.btn, 0, wx.CENTER, 5)
        hbox.Add(self.auto, 0, wx.CENTER, 5)
        hbox.Add(self.calc_btn, 0, wx.CENTER | wx.BOTTOM, 10)
        hbox.Add(self.st, 0, wx.EXPAND | wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, 5)
        hbox.Add(cb, 0, wx.EXPAND | wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, 1)
        hbox.Add(self.chk_box, 0, wx.EXPAND | wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, 5)

        self.SetSizer(hbox)

    def on_select(self, event: wx.Event) -> None:
        i = event.GetString()
        self.wvlngh, self.n_coef = refection_coef_read(os.path.join(self.cwd, i))

    def on_open(self, event: wx.Event) -> None:

        wildcard = "TXT files (*.txt)|*.txt"
        dialog = wx.FileDialog(
            self,
            "Open Text Files",
            wildcard=wildcard,
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        )

        if dialog.ShowModal() == wx.ID_CANCEL:
            return

        path = dialog.GetPath()

        if os.path.exists(path):
            self.data = np.loadtxt(path)

        self.calc_btn.Enable(False)

        draw_data(self.graph, self.data[:, 0], self.data[:, 1], name="Spectrum")

        self.auto.Enable(True)

        if self.graphs.IsShown():
            self.graphs.Hide()
            self.graph.Show()

        self.Parent.Layout()
        self.Parent.Fit()

    def calc_choice(self, event: wx.Event) -> None:
        """Executes calculation based on checked/unchecked box."""

        if self.chk_box.GetValue():
            self.make_report(*self.data_prep())
        else:
            self.ShowMessage(*self.data_prep())
            while self.graph.toolbar.lx:
                self.graph.toolbar.lx.pop(0).remove()
            self.graph.canvas.draw()

    def calc_auto(self, event: wx.Event) -> None:
        """Returns thickness of the sample in a message box."""

        self.graph.toolbar.x = [0, 9050.0, 9200.0]
        self.ShowMessage(*self.data_prep(), auto_ga_as=True)
        while self.graph.toolbar.lx:
            self.graph.toolbar.lx.pop(0).remove()
        self.graph.canvas.draw()

    def ShowMessage(
        self,
        dat_X: Optional[np.ndarray] = None,
        dat_Y: Optional[np.ndarray] = None,
        wavelength: float = 9000.0,
        n_wv_idx: Optional[float] = None,
        n_true: float = 3.54,
        auto_ga_as: bool = False,
    ) -> None:
        """Draws the message box with info about sample thickness."""

        if auto_ga_as:
            h = rolling_dist(dat_X=dat_X, dat_Y=dat_Y)
        else:
            self.psd, self.freq, true_freq, true_ind = fourier_analysis(dat_X, dat_Y)
            h = thickness(wavelength, 1 / true_freq, n_true)

        msg = "Sample thickness is {h} um".format(h=h)

        dial = wx.MessageDialog(None, msg, "Info", wx.OK)
        dial.ShowModal()
        self.calc_btn.Enable(False)

    def data_prep(self) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        """Prepares data to be analysed."""
        self.x = self.graph.toolbar.x[1:]
        self.graph.toolbar.x[:] = []
        self.x.sort()

        try:
            self.x != []
        except:
            print("Processing range not specified")

        x = np.searchsorted(self.data[:, 0], [self.x[0]])[0]
        y = np.searchsorted(self.data[:, 0], [self.x[1]])[0]
        dat_X = self.data[x:y, 0]
        dat_Y = self.data[x:y, 1]

        wavelength = dat_X[int(dat_X.shape[-1] / 2)]
        n_wv_idx = find_nearest(self.wvlngh, wavelength / 10000.0)
        n_true = self.n_coef[n_wv_idx]

        return dat_X, dat_Y, wavelength, n_wv_idx, n_true

    def make_report(
        self,
        dat_X: Optional[np.ndarray] = None,
        dat_Y: Optional[np.ndarray] = None,
        wavelength: float = 9000.0,
        n_wv_idx: Optional[float] = None,
        n_true=3.54,
    ) -> None:
        """Draws another panel covered with plots filled with additional info."""

        self.calc_btn.Enable(False)

        # dial = wx.ProgressDialog('Progress', 'Computation could take some time', maximum=100, parent=self)
        # dial.Update(5)

        # Moving average method
        idx, peaks = find_peaks(dat_Y)
        draw_data(
            self.graphs.graphFour,
            dat_X,
            dat_Y,
            scatter_x=dat_X[idx],
            scatter_y=dat_Y[idx],
            name="Peaks detection",
        )
        periodicity = pd.Series(dat_X[idx], index=dat_X[idx]).diff().dropna()

        periodicity_mean = periodicity.rolling(window=10, center=True).mean().values

        period = periodicity_mean[int(periodicity_mean.shape[-1] / 2)]

        draw_data(
            self.graphs.graphOne,
            periodicity.index.values,
            periodicity_mean,
            name="Average periodicity",
            text=textstr(wavelength, period, n_true),
        )
        # dial.Update(20)

        # Fourier methods
        self.psd, self.freq, true_freq, true_ind = fourier_analysis(dat_X, dat_Y)
        reverse = np.zeros_like(dat_Y)
        reverse[self.psd == true_ind] = true_freq
        reverse = np.real(np.fft.ifft(reverse))

        ## Flattening Process
        coef = np.polyfit(dat_X, dat_Y, 1)
        poly1d_fn = np.poly1d(coef)
        new_dat = dat_Y - poly1d_fn(dat_X)

        self.psd, self.freq, true_freq, true_ind = fourier_analysis(
            dat_X, new_dat - 200 * reverse
        )
        new_dat = new_dat / new_dat.max()

        draw_data(
            self.graphs.graphTwo,
            self.freq,
            self.psd,
            # style="o",
            text=textstr(wavelength, 1 / true_freq, n_true),
            name="FFT after signal flattening",
            x_label=r"Frequency $[\frac{1}{\AA}]$",
        )
        # dial.Update(40)
        draw_data(self.graphs.graphFive, dat_X, reverse * new_dat.max() / reverse.max())
        draw_data(
            self.graphs.graphFive,
            dat_X,
            new_dat,
            name="IFFT and flattened signal",
            clear=False,
        )

        # Refractive index
        draw_data(
            self.graphs.graphThree,
            self.wvlngh,
            self.n_coef,
            scatter_x=[self.wvlngh[n_wv_idx]],
            scatter_y=[n_true],
            name="Refractive index",
            label_l="n = " + str(n_true),
            x_label=r"Wavelength [um]",
            y_label=r"$n(\lambda)$",
        )
        # dial.Update(60)

        draw_data(
            self.graphs.graphSix,
            0,
            0,
            text=r"$h=\frac{\lambda^2}{2n \Delta\lambda}$",
            name="Execution Formula",
            size=16,
        )
        # dial.Update(100)

        if self.graph.IsShown():
            self.graph.Hide()
            self.graphs.Show()

        self.Parent.Layout()
        self.Parent.Fit()

    def draw_graphs(self) -> None:
        """Prepares canvas for 6 additional plots with supportive info."""

        self.column = wx.BoxSizer(wx.VERTICAL)
        self.firstLine = wx.BoxSizer()
        self.secondLine = wx.BoxSizer()

        # Add first line graphs
        self.graphOne = MyPanel(self)
        self.firstLine.Add(self.graphOne, 0, wx.ALL | wx.EXPAND, 3)
        self.graphTwo = MyPanel(self)
        self.firstLine.Add(self.graphTwo, 0, wx.ALL | wx.EXPAND, 3)
        self.graphThree = MyPanel(self)
        self.firstLine.Add(self.graphThree, 0, wx.ALL | wx.EXPAND, 3)

        # Add second line graphs
        self.graphFour = MyPanel(self)
        self.secondLine.Add(self.graphFour, 0, wx.ALL | wx.EXPAND, 3)
        self.graphFive = MyPanel(self)
        self.secondLine.Add(self.graphFive, 0, wx.ALL | wx.EXPAND, 3)
        self.graphSix = MyPanel(self)
        self.secondLine.Add(self.graphSix, 0, wx.ALL | wx.EXPAND, 3)

        self.column.Add(self.firstLine, 0, wx.ALL | wx.EXPAND, 0)
        self.column.Add(self.secondLine, 0, wx.ALL | wx.EXPAND, 0)

        self.SetSizer(self.column)

        width, _ = wx.GetDisplaySize()
        dpi = 100 / 1920.0 * width

        self.graphOne.draw_graph(fig_size=(5, 4), is_special=False, dpi=dpi)
        self.graphTwo.draw_graph(fig_size=(5, 4), is_special=False, dpi=dpi)
        self.graphThree.draw_graph(fig_size=(5, 4), is_special=False, dpi=dpi)
        self.graphFour.draw_graph(fig_size=(5, 4), is_special=False, dpi=dpi)
        self.graphFive.draw_graph(fig_size=(5, 4), is_special=False, dpi=dpi)
        self.graphSix.draw_graph(fig_size=(5, 4), is_special=False, dpi=dpi)

        self.Fit()

    def draw_graph(
        self,
        fig_size: Tuple[float, float] = (5.0, 5.0),
        is_special: bool = True,
        dpi: int = 100,
    ) -> None:
        """Creates plot."""

        self.figure = Figure(figsize=fig_size, dpi=dpi)
        self.axes = self.figure.add_subplot()
        self.cursor = Cursor(self.axes, useblit=True, color="red")
        self.canvas = FigureCanvas(self, -1, self.figure)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 0, wx.LEFT | wx.TOP)

        self.toolbar = MyNavigationToolbar(self.canvas, is_special=is_special)
        self.toolbar.Realize()

        self.sizer.Add(self.toolbar, 0, wx.LEFT)

        # update the axes menu on the toolbar
        self.toolbar.update()

        self.SetSizer(self.sizer)
        self.Fit()


class MyNavigationToolbar(NavigationToolbar):
    """Extend the default wx toolbar with your own event handlers."""

    def __init__(self, canvas: FigureCanvas, is_special: bool = True) -> None:
        NavigationToolbar.__init__(self, canvas)
        POSITION_OF_CONFIGURE_SUBPLOTS_BTN = 6
        self.DeleteToolByPos(POSITION_OF_CONFIGURE_SUBPLOTS_BTN)

        if is_special:
            bmp = wx.ArtProvider.GetBitmap(wx.ART_CROSS_MARK, wx.ART_TOOLBAR)
            tool = self.AddTool(wx.ID_ANY, "Click me", bmp, "Activate custom contol")
            self.Bind(wx.EVT_TOOL, self._on_custom, id=tool.GetId())

        self.counter = 0
        self.clicks = True
        self.lines = []
        self.x = []

    def _on_custom(self, evt: wx.Event) -> None:
        self.ax = self.canvas.figure.axes[0]
        self.cid = self.canvas.figure.canvas.mpl_connect(
            "button_press_event", self.on_press
        )
        self.move = self.canvas.figure.canvas.mpl_connect(
            "motion_notify_event", self.on_curser
        )
        self.lx = []

    def on_press(self, event: wx.Event) -> None:
        self.background = self.canvas.figure.canvas.copy_from_bbox(
            self.canvas.figure.bbox
        )
        self.counter += 1

        if self.clicks:
            self.x.append(event.xdata)
            self.lx.append(self.ax.axvline(event.xdata, color="k"))
            # print(type(self.lx))
            self.clicks = False
            self.canvas.draw()

        else:
            self.x.append(event.xdata)
            self.lx.append(self.ax.axvline(event.xdata, color="k"))
            self.clicks = True
            self.canvas.draw()

        if self.counter > 2:
            self.canvas.figure.canvas.mpl_disconnect(self.move)
            self.canvas.figure.canvas.mpl_disconnect(self.cid)
            self.Parent.Parent.control.calc_btn.Enable(True)
            self.counter = 0

    def on_curser(self, event: wx.Event) -> None:
        if not event.inaxes:
            return

        if self.lx != []:
            line = self.lx[-1]
            line.set_xdata(event.xdata)
            self.canvas.restore_region(self.background)
            self.Parent.axes.draw_artist(line)
            self.canvas.figure.canvas.blit(line.axes.bbox)


def draw_data(
    graphNum: MyPanel,
    x: np.ndarray,
    y: np.ndarray,
    style="-",
    text: Optional[str] = None,
    scatter_x: Optional[np.ndarray] = None,
    scatter_y: Optional[np.ndarray] = None,
    name: Optional[str] = None,
    label_l: str = "Peaks",
    size: int = 9,
    clear: bool = True,
    x_label: str = r"Wavelength $[\AA]$",
    y_label: str = "Intensity",
) -> None:
    """Draws data on plot."""

    scatter_x = [] if scatter_x is None else scatter_x
    scatter_y = [] if scatter_y is None else scatter_y

    if clear:
        graphNum.axes.clear()
        graphNum.toolbar.update()
    graphNum.axes.grid(True)
    graphNum.axes.plot(x, y, style)
    graphNum.axes.set_xlabel(x_label)
    graphNum.axes.set_ylabel(y_label)
    graphNum.axes.ticklabel_format(style="sci", useMathText=True, scilimits=(0, 0))

    if text != None:

        # these are matplotlib.patch.Patch properties
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)

        # place a text box in upper left in axes coords
        graphNum.axes.text(
            0.05,
            0.95,
            text,
            transform=graphNum.axes.transAxes,
            fontsize=size,
            verticalalignment="top",
            bbox=props,
        )

    if name != None:
        graphNum.axes.set_title(name, fontsize=11)

    if (scatter_x != []) & (scatter_y != []):
        graphNum.axes.scatter(scatter_x, scatter_y, s=30, c="red", label=label_l)
        graphNum.axes.legend(fontsize=9)
    graphNum.canvas.draw()
    graphNum.toolbar.update()


def find_nearest(array: np.ndarray, value: float) -> float:
    """Finds closest value in array for given one."""

    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def adv_dist_calc(waveln_1: float, waveln_2: float) -> float:
    return np.abs(
        waveln_1
        * waveln_2
        / (2 * (waveln_1 * sellmeyer_eq(waveln_2) - waveln_2 * sellmeyer_eq(waveln_1)))
    )


def rolling_window(a: np.ndarray, window: int) -> np.ndarray:
    """Returns rolling windows."""

    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_dist(
    dat_X: Optional[np.ndarray] = None, dat_Y: Optional[np.ndarray] = None,
) -> float:
    """Returns mean of distances obtained using rolling windows."""

    idx, _ = find_peaks(dat_Y)
    fragments = rolling_window(dat_X[idx] / 10000.0, 2)
    dist = np.array(
        list(map(lambda x: adv_dist_calc(waveln_1=x[0], waveln_2=x[1],), fragments,))
    )

    # wv_add = wv_add[:-1]
    # n_add = sellmeyer_eq(wv_add)
    # d_mean = np.round(dist.mean())
    # m = np.round(2 * n_add * d_mean / wv_add)
    # d_new = m * wv_add / (2 * n_add)
    return np.round(dist.mean())


def get_files_list(path_to_dir: str) -> np.ndarray:
    """Returns list of files in given directory."""

    f = []

    if os.path.exists(path_to_dir):
        for (dirpath, dirnames, filenames) in os.walk(path_to_dir):
            f.extend(filenames)
            break
    return np.array(f)


def refection_coef_read(path_to_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Reads the data for reflection coefficient from file."""

    if os.path.exists(path_to_file):
        n_lam = pd.read_csv(
            path_to_file, header=0, delimiter=",", encoding="utf-8", names=["x", "y"]
        )
    return n_lam["x"].values, n_lam["y"].values


def thickness(wavelength: float, period: float, n: float) -> float:
    """Calculates the thickness of the sample using interference method."""

    return np.round(wavelength ** 2 / (2.0 * n * period * 10000.0), 2)


def textstr(wvln: float, period: float, n_true: float) -> str:

    return "\n".join(
        (
            r"$h=%.2f$ um" % (thickness(wvln, period, n_true),),
            r"$f =%.2f \frac{1}{\AA}$" % (1 / period,),
            r"$\rho=%.2f\AA$" % (period,),
        )
    )


def sellmeyer_eq(
    wavelength: np.ndarray, a: float = 8.950, b: float = 2.054, c2: float = 0.390
) -> np.ndarray:
    """Returns refractive index for given wavelength.
    
    The Sellmeier equation is an empirical relationship between refractive 
    index and wavelength for a particular transparent medium. 
    The equation is used to determine the dispersion of light
    in the medium. Default values corresponds to GaAs at room
    temperature.
    
    Args:
        wavelength: List of wavelengths.
        a: empirical coefficient, default value is given for GaAs
        b: empirical coefficient, default value is given for GaAs
        c2: empirical coefficient, default value is given for GaAs
        keys: A sequence of strings representing the key of each table row
            to fetch.
        other_silly_variable: Another optional variable, that has a much
            longer name than the other args, and which does nothing.

    Returns:
        List of refractive indexes.
    """

    return np.sqrt(a + b / (1 - c2 / wavelength ** 2))


def fourier_analysis(
    x: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Perfoms FFT and searches for main frequencies."""

    sp = np.fft.fft(y)
    freq = np.fft.fftfreq(y.shape[-1], np.diff(x).mean())
    psd = np.abs(sp)
    new_freq = freq[freq > 0.12]
    new_psd = psd[freq > 0.12]
    maxInd, _ = find_peaks(new_psd)

    true_freq = new_freq[maxInd]
    true_psd = new_psd[maxInd]

    return psd, freq, true_freq[true_psd == true_psd.max()], true_psd.max()


if __name__ == "__main__":
    app = MyApp()
    app.MainLoop()
    wx.Exit()

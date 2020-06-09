import wx
import wx.lib.agw.aui as aui
import wx.lib.mixins.inspection as wit

import matplotlib as mpl
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import (
    NavigationToolbar2WxAgg as NavigationToolbar,
)
import matplotlib.backends.backend_wxagg as wxagg
from matplotlib.figure import Figure
from matplotlib.widgets import Cursor

# mpl.use('WXAgg')
import pkg_resources.py2_warn

from scipy.signal import find_peaks
import pandas as pd

import numpy as np
import os

# from win32api import GetSystemMetrics


class MyApp(wx.App):
    def __init__(self):
        super(MyApp, self).__init__(clearSigInt=True)

        self.mainFrame = MyFrame(None)
        self.mainFrame.Show()


class MyFrame(wx.Frame):
    def __init__(self, parent, title="Thickness Interferometry", pos=(100, 100)):
        # super(MyFrame, self).__init__(parent, title=title, size=(800, 600), pos=pos)
        super(MyFrame, self).__init__(parent, title=title, pos=pos)
        self.init_frame()
        self.SetBackgroundColour("white")

    def init_frame(self):
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

        # self.Bind(wx.EVT_PAINT, self.OnPaint)


class MyPanel(wx.Panel):
    def __init__(self, parent):
        super(MyPanel, self).__init__(parent=parent)
        # self.SetAutoLayout(True)
        # self.InitPanel()
        # self.graph = wx.Panel()
        # self.graphs = wx.Panel()

    def show_buttons(self):
        hbox = wx.BoxSizer(wx.VERTICAL)
        # self.my_text = wx.TextCtrl(self, style=wx.TE_MULTILINE)
        self.btn = wx.Button(self, label="Open Text File")
        self.btn.Bind(wx.EVT_BUTTON, self.on_open)

        # -----------------
        # self.cwd = os.path.join(os.getcwd(), 'Refraction_index')
        self.cwd = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "Refraction_index"
        )

        refl_files = get_files_list(self.cwd)
        self.wvlngh, self.n_coef = refection_coef_read(
            os.path.join(self.cwd, refl_files[0])
        )
        # print('Real Path: ', self.cwd+refl_files[0])
        cb = wx.ComboBox(
            self, choices=refl_files, style=wx.CB_READONLY, value=refl_files[0]
        )
        cb.Bind(wx.EVT_COMBOBOX, self.on_select)
        self.st = wx.StaticText(self, label="")
        self.st.SetLabel("Refractive index file:")
        # -----------------

        self.calc_btn = wx.Button(self, label="Calculate depth")
        self.calc_btn.Enable(False)
        self.calc_btn.Bind(wx.EVT_BUTTON, self.calc_choice)
        # ------------------

        self.chk_box = wx.CheckBox(self, label="Full Report")

        # self.reset_btn = wx.Button(self, label='Reset')
        # self.reset_btn.Enable(False)
        # self.reset_btn.Bind(wx.EVT_BUTTON, self.onReset)

        # hbox2.Add(self.my_text, 1, wx.ALL|wx.EXPAND)
        hbox.Add(self.btn, 0, wx.CENTER, 5)
        # hbox.Add(self.reset_btn, 0, wx.CENTER, 5)
        hbox.Add(self.calc_btn, 0, wx.CENTER | wx.BOTTOM, 10)
        hbox.Add(self.st, 0, wx.EXPAND | wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, 5)
        hbox.Add(cb, 0, wx.EXPAND | wx.ALIGN_CENTER_HORIZONTAL | wx.ALL, 1)
        hbox.Add(self.chk_box, 0, wx.EXPAND | wx.ALIGN_CENTER_HORIZONTAL | wx.TOP, 5)

        self.SetSizer(hbox)

    def on_select(self, event):
        i = event.GetString()
        self.wvlngh, self.n_coef = refection_coef_read(os.path.join(self.cwd, i))

    def onReset(self, event):
        self.Parent.Restore()
        # GetApp().OnInit()

    def on_open(self, event):

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

        # self.graph.axes.clear()
        # self.graph.axes.grid(True)
        # self.graph.axes.plot(self.data[:,0], self.data[:, 1])
        # # self.Layout()
        # self.graph.canvas.draw()

        self.calc_btn.Enable(False)

        # self.graph.axes.clear()
        # self.graph.toolbar.update()

        draw_data(self.graph, self.data[:, 0], self.data[:, 1], name="Spectrum")

        if self.graphs.IsShown():
            self.graphs.Hide()
            self.graph.Show()

        self.Parent.Layout()
        # self.Parent.graphSizer.Layout()
        self.Parent.Fit()

    def calc_choice(self, event):

        if self.chk_box.GetValue():
            self.make_report(*self.data_prep())
        else:
            self.ShowMessage1(*self.data_prep())
            while self.graph.toolbar.lx:
                self.graph.toolbar.lx.pop(0).remove()
            self.graph.canvas.draw()

    def ShowMessage1(
        self, dat_X=None, dat_Y=None, wavelength=9000, n_wv_idx=None, n_true=3.54
    ):

        self.psd, self.freq, true_freq, true_ind = fourier_analysis(dat_X, dat_Y)
        msg = "Sample thickness is {h} um".format(
            h=thickness(wavelength, 1 / true_freq, n_true)
        )

        dial = wx.MessageDialog(None, msg, "Info", wx.OK)
        dial.ShowModal()
        self.calc_btn.Enable(False)

    def data_prep(self):
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
        self, dat_X=None, dat_Y=None, wavelength=9000, n_wv_idx=None, n_true=3.54
    ):
        self.calc_btn.Enable(False)

        # dial.ShowModal()

        # if self.graph.IsShown():
        #     self.graph.Hide()
        #     self.graphs.Show()

        # # self.Parent.graphSizer.SendSizeEvent()
        # self.Parent.graphSizer.Layout()

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
        period = (
            periodicity.rolling(window=10, center=True)
            .mean()
            .values[
                int(
                    periodicity.rolling(window=10, center=True).mean().values.shape[-1]
                    / 2
                )
            ]
        )
        thickness(wavelength, period, n_true)

        textstr = "\n".join(
            (
                r"$h=%.2f$ um" % (thickness(wavelength, period, n_true),),
                r"$f =%.2f \frac{1}{\AA}$" % (1 / period,),
                r"$\rho=%.2f\AA$" % (period,),
            )
        )

        draw_data(
            self.graphs.graphOne,
            periodicity.index.values,
            periodicity.rolling(window=10, center=True).mean().values,
            name="Average periodicity",
            text=textstr,
        )
        # dial.Update(20)

        # Fourier methods
        self.psd, self.freq, true_freq, true_ind = fourier_analysis(dat_X, dat_Y)
        reverse = np.zeros_like(dat_Y)
        reverse[self.psd == true_ind] = true_freq
        # print('Reverse:', reverse)
        reverse = np.fft.ifft(reverse)

        ## Flattening Process
        coef = np.polyfit(dat_X, dat_Y, 1)
        poly1d_fn = np.poly1d(coef)
        new_dat = dat_Y - poly1d_fn(dat_X)

        textstr = "\n".join(
            (
                r"$h=%.2f$ um" % (thickness(wavelength, 1 / true_freq, n_true),),
                r"$f =%.2f \frac{1}{\AA}$" % (true_freq,),
                r"$\rho=%.2f\AA$" % (1 / true_freq,),
            )
        )

        self.psd, self.freq, true_freq, true_ind = fourier_analysis(
            dat_X, new_dat - 200 * reverse
        )
        new_dat = new_dat / new_dat.max()
        draw_data(
            self.graphs.graphTwo,
            self.freq,
            self.psd,
            style="o",
            text=textstr,
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

        textstr = r"$h=\frac{\lambda^2}{2n \Delta\lambda}$"
        draw_data(
            self.graphs.graphSix, 0, 0, text=textstr, name="Execution Formula", size=16
        )
        # dial.Update(100)

        # self.Parent.Layout()
        if self.graph.IsShown():
            self.graph.Hide()
            self.graphs.Show()

        # self.Parent.graphSizer.SendSizeEvent()
        # self.Parent.graphSizer.Layout()
        self.Parent.Layout()
        self.Parent.Fit()

    def draw_graphs(self):

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

    # def draw_new_graph(self, fig_size=(5,5)):

    #     draw_graph(self, fig_size=(5,5)):

    def draw_graph(self, fig_size=(5, 5), is_special=True, dpi=100):
        self.figure = Figure(figsize=fig_size, dpi=dpi)
        self.axes = self.figure.add_subplot()
        self.cursor = Cursor(self.axes, useblit=True, color="red")
        self.canvas = FigureCanvas(self, -1, self.figure)
        # self.figure.canvas.mpl_connect('key_press_event', self.on_key)

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.sizer.Add(self.canvas, 0, wx.LEFT | wx.TOP)

        self.toolbar = MyNavigationToolbar(self.canvas, is_special=is_special)
        self.toolbar.Realize()

        self.sizer.Add(self.toolbar, 0, wx.LEFT)
        # update the axes menu on the toolbar
        self.toolbar.update()

        self.SetSizer(self.sizer)
        self.Fit()

    # def on_key(self, event):
    #     """If the user presses the Escape key then stop picking points and
    #     reset the list of picked points."""
    #     # if 'escape' == event.key:
    #     #     self._is_pick_started = False
    #     #     self._picked_indices = None
    #     # return
    #     ax = self.canvas.figure.axes[0]
    #     x, y = np.random.rand(2)  # generate a random location
    #     rgb = np.random.rand(3)  # generate a random color
    #     ax.text(x, y, 'You clicked me', transform=ax.transAxes, color=rgb)
    #     self.canvas.draw()
    #     # evt.Skip()

    # def add_toolbar(self):


class MyNavigationToolbar(NavigationToolbar):
    """Extend the default wx toolbar with your own event handlers."""

    def __init__(self, canvas, is_special=True):
        # self.Parent.SetToolBitmapSize(wx.Size(3, 2))
        NavigationToolbar.__init__(self, canvas)
        # We use a stock wx bitmap, but you could also use your own image file.
        POSITION_OF_CONFIGURE_SUBPLOTS_BTN = 6
        self.DeleteToolByPos(POSITION_OF_CONFIGURE_SUBPLOTS_BTN)
        # self.DCsize = tuple(self.GetClientSize()/10)
        # print(self.DCsize)
        # self.SetToolBitmapSize(wx.Size(3, 2))
        # print(self.GetToolSize())
        # self.SetToolBitmapSize(self.DCsize)

        if is_special:
            bmp = wx.ArtProvider.GetBitmap(wx.ART_CROSS_MARK, wx.ART_TOOLBAR)
            tool = self.AddTool(wx.ID_ANY, "Click me", bmp, "Activate custom contol")
            self.Bind(wx.EVT_TOOL, self._on_custom, id=tool.GetId())

        self.counter = 0
        self.clicks = True
        self.lines = []
        self.x = []

    def _on_custom(self, evt):
        self.ax = self.canvas.figure.axes[0]
        self.cid = self.canvas.figure.canvas.mpl_connect(
            "button_press_event", self.on_press
        )
        self.move = self.canvas.figure.canvas.mpl_connect(
            "motion_notify_event", self.on_curser
        )
        # self.pick = self.canvas.figure.canvas.mpl_connect('pick_event', self.on_pick)
        self.lx = []
        # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        # self.text = self.ax.text(0.05, 0.95, transform=self.ax.transAxes, fontsize=11, verticalalignment='top', bbox=props)

    # def pick(self, event):

    #     if text != None:
    #         # str_ = 'h = ' + str(text) + ' um'
    #         # graphNum.axes.text(1.0, y.max(), str_)
    #         # these are matplotlib.patch.Patch properties
    #         props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

    #         # place a text box in upper left in axes coords
    #         self.ax.text(0.05, 0.95, text, transform=graphNum.axes.transAxes, fontsize=size, verticalalignment='top', bbox=props)

    def on_press(self, event):
        self.background = self.canvas.figure.canvas.copy_from_bbox(
            self.canvas.figure.bbox
        )
        self.counter += 1

        if self.clicks:
            self.x.append(event.xdata)
            self.lx.append(self.ax.axvline(event.xdata, color="k"))
            self.clicks = False
            self.canvas.draw()

        else:
            # self.background = self.canvas.figure.canvas.copy_from_bbox(self.canvas.figure.bbox)
            self.x.append(event.xdata)
            self.lx.append(self.ax.axvline(event.xdata, color="k"))
            self.clicks = True
            self.canvas.draw()

        if self.counter > 2:
            self.canvas.figure.canvas.mpl_disconnect(self.move)
            self.canvas.figure.canvas.mpl_disconnect(self.cid)
            self.Parent.Parent.control.calc_btn.Enable(True)
            self.counter = 0

    def on_curser(self, event):
        if not event.inaxes:
            return

        if self.lx != []:
            line = self.lx[-1]
            line.set_xdata(event.xdata)
            # self.text.set_text('X: {x}'.format(x= event.xdata))
            # self.canvas.restore_region(self.canvas.copy_from_bbox(self.Parent.figure.bbox))
            self.canvas.restore_region(self.background)
            self.Parent.axes.draw_artist(line)
            self.canvas.figure.canvas.blit(line.axes.bbox)
            # self.canvas.blit(line.get_window_extent())
        # self.canvas.draw()


def draw_data(
    graphNum,
    x,
    y,
    style="-",
    text=None,
    scatter_x=None,
    scatter_y=None,
    name=None,
    label_l="Peaks",
    size=9,
    clear=True,
    x_label=r"Wavelength $[\AA]$",
    y_label="Intensity",
):
    scatter_x = [] if scatter_x is None else scatter_x
    scatter_y = [] if scatter_y is None else scatter_y

    if clear:
        graphNum.axes.clear()
        graphNum.toolbar.update()
    # graphNum.axes.clear()
    graphNum.axes.grid(True)
    # graphNum.axes.set_xlabel(r'$\Delta_i^j$')
    # graphNum.axes.set_ylabel(r'$\Delta_{i+1}^j$')
    graphNum.axes.plot(x, y, style)
    graphNum.axes.set_xlabel(x_label)
    graphNum.axes.set_ylabel(y_label)
    # graphNum.axes.autoscale(tight=True)
    graphNum.axes.ticklabel_format(style="sci", useMathText=True, scilimits=(0, 0))
    # graphNum.axes.set_scientific(True)

    if text != None:
        # str_ = 'h = ' + str(text) + ' um'
        # graphNum.axes.text(1.0, y.max(), str_)
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


def find_nearest(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def get_files_list(path_to_dir):
    f = []

    if os.path.exists(path_to_dir):
        for (dirpath, dirnames, filenames) in os.walk(path_to_dir):
            f.extend(filenames)
            break
    return np.array(f)


def refection_coef_read(path_to_file):
    if os.path.exists(path_to_file):
        n_lam = pd.read_csv(
            path_to_file, header=None, delimiter=",", encoding="utf-8"
        ).rename(columns={0: "x", 1: "y"})
    wvlngh = n_lam["x"].values
    n_coef = n_lam["y"].values

    return wvlngh, n_coef


def thickness(wavelength, period, n):
    return np.round(wavelength * wavelength / (2.0 * n * period * 10000.0), 2)


def fourier_analysis(x, y):
    # sig_fft = np.fft.fft(y)
    # # power = np.abs(sig_fft)
    # sample_freq = np.fft.fftfreq(y.shape[-1], (x[0]-x[-1])/(len(x)-1))
    # # freq = np.fft.fftfreq(y.shape[-1], )
    # mask = sample_freq > 0
    # ind = np.arange(1, int(y.shape[-1]/2) + 1)
    # # print(ind)
    # power = np.abs(sig_fft[ind])**2 + np.abs(sig_fft[-ind])**2
    # return power, sample_freq
    ########################
    sp = np.fft.fft(y)
    # sp = np.fft.fftshift(sp)
    freq = np.fft.fftfreq(y.shape[-1], np.diff(x).mean())
    # freq = np.fft.fftshift(freq)
    # mask = freq > 0
    # ind = np.arange(1, int(y.shape[-1]/2))

    # print(ind)
    # psd = np.abs(sp[ind])**2 + np.abs(sp[-ind])**2
    psd = np.abs(sp)
    # peaks, _ = find_peaks(psd, height=0.25, width=2)
    # peaks.sort()
    # print("Freq: ", freq[-peaks])
    # plt.plot(freq[-ind], psd, 'o')
    # masl = freq > 0.1
    # true_ind = (freq > 0.2)
    true_freq = freq
    true_freq = true_freq[true_freq > 0.2]
    true_psd = psd[freq > 0.2]

    return psd, freq, true_freq[true_psd.argmax()], true_psd.max()


if __name__ == "__main__":
    app = MyApp()
    app.MainLoop()
    wx.Exit()

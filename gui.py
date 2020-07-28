import wx  # type: ignore

from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas  # type: ignore
from matplotlib.backends.backend_wxagg import (
    NavigationToolbar2WxAgg as NavigationToolbar,
)  # type: ignore
from matplotlib.figure import Figure  # type: ignore
from matplotlib.widgets import Cursor  # type: ignore

from typing import Tuple, Optional, List, Union

# import pkg_resources.py2_warn  # type: ignore

from scipy.signal import find_peaks  # type: ignore

from utils import *
from analysis import *

import numpy as np  # type: ignore
import os


class MyApp(wx.App):
    """Main application class."""

    def __init__(self):
        super().__init__(clearSigInt=True)

        self.mainFrame = MyFrame(None)
        self.mainFrame.Show()


class MyFrame(wx.Frame):
    """Class which generates a frame.
    
    All of the visual positioning of the blocks are described here.
    """

    def __init__(
        self, parent, title="Thickness Interferometry", pos=(100, 100)
    ) -> None:

        super().__init__(parent, title=title, pos=pos)
        self.init_frame()
        self.SetBackgroundColour("white")

    def init_frame(self) -> None:
        """Initializes all the panels and set sizers.
    
        There are 3 panels: control panel for control buttons, graph panel
        for raw data visualization and preparation and graphs panel which
        switches from graph panel and shows stages of data process.

        Args:
            None

        Returns:
            None
        """
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
        super().__init__(parent=parent)

    def show_buttons(self) -> None:
        """Initializes all of the buttons.
    
        Buttons are placed at control panel. All the buttons use event-handlers
        to react on user activity and make an appropriate functions call.

        Args:
            None

        Returns:
            None
        """

        hbox = wx.BoxSizer(wx.VERTICAL)
        self.btn = wx.Button(self, label="Open Text File")
        self.btn.Bind(wx.EVT_BUTTON, self.on_open)

        # -----------------
        self.cwd = os.path.join(
            os.path.abspath(os.path.dirname(__file__)), "Refraction_index"
        )

        refl_files = get_files_list(self.cwd)
        self.wavelength, self.n_coef = refection_coef_read(
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
        """Updates refractive index data due to chosen source.
    
        When the user selects an item in the drop-down menu, the new refractive index data is read from the selected file.

        Args:
            event: wx.EVT_BUTTON. Checks that user have picked a member of the list.

        Returns:
            None
        """

        i = event.GetString()
        self.wavelength, self.n_coef = refection_coef_read(os.path.join(self.cwd, i))

    def on_open(self, event: wx.Event) -> None:
        """Open a raw data to process.
    
        When user clicks on "Open Text File" and find an appropriate file, 
        this function imports the data to `self.data` and plots it on graph
        panel.

        Args:
            event: wx.EVT_COMBOBOX. Checks that user have clicked on button.

        Returns:
            None
        """

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
        """Executes calculation based on checked/unchecked box.
    
        If the Checkbox is marked then calculations are launched with detailed 
        information about each step. Otherwise a dialog box appears with thickness 
        information.

        Args:
            event: wx.EVT_BUTTON

        Returns:
            None
        """

        if self.chk_box.GetValue():
            make_report(self, *data_prep(self))
        else:
            self.ShowMessage(*data_prep(self))
            while self.graph.toolbar.lx:
                self.graph.toolbar.lx.pop(0).remove()
            self.graph.canvas.draw()

    def calc_auto(self, event: wx.Event) -> None:
        """Returns thickness of the sample in a message box.
    
        One can run thickness calculation without marking data boundaries
        and choosing reflective coefficient values (valid for GaAs samples). 
        In this case one may press "Auto GaAs depth" and this function will 
        automate the process.

        Args:
            event: wx.EVT_BUTTON

        Returns:
            None
        """

        self.graph.toolbar.x = [0, 9050.0, 9200.0]
        self.ShowMessage(*data_prep(self), auto_ga_as=True)
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
        """Draws the message box with info about sample thickness.
    
        Starts a quick thickness calculation and displays a dialog box with the correct answer.

        Args:
            dat_X: raw data X (wavelength)

            dat_Y: raw data Y (intensity)

            wavelength: the wavelength for which the thickness will be calculated

            n_wv_idx: deprecated positional argument

            n_true: value of appropriate refractive index for given wavelength

            auto_ga_as: boolean flag for auto calculation

        Returns:
            None
        """

        if auto_ga_as:
            h = rolling_dist(dat_X=dat_X, dat_Y=dat_Y)
        else:
            self.psd, self.freq, true_freq, true_ind = fourier_analysis(dat_X, dat_Y)
            h = thickness(wavelength, 1 / true_freq, n_true)

        msg = "Sample thickness is {h} um".format(h=h)

        dial = wx.MessageDialog(None, msg, "Info", wx.OK)
        dial.ShowModal()
        self.calc_btn.Enable(False)    

    def draw_graphs(self) -> None:
        """Prepares canvas for 6 additional plots with supportive info.
    
        Defines 6 figures on the panel graphs. Sets sizers.

        Args:
            None

        Returns:
            None
        """

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
        """Creates plot.
    
        Prepare canvas and define figures.

        Args:
            fig_size: sets figure size

            is_special: flag for different kind of navigation tools on figure

            dpi: dpi

        Returns:
            None
        """

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
        # self.lines = []
        self.x: List[np.float64] = []

    def _on_custom(self, evt: wx.Event) -> None:
        self.ax = self.canvas.figure.axes[0]
        self.cid = self.canvas.figure.canvas.mpl_connect(
            "button_press_event", self.on_press
        )
        self.move = self.canvas.figure.canvas.mpl_connect(
            "motion_notify_event", self.on_curser
        )
        self.lx: List[Figure.Line2D] = []

    def on_press(self, event: wx.Event) -> None:
        """Draw the line on click.
    
        Draws vertical lines when the user sets the boundaries of the range of data analysis.

        Args:
            event: button_press_event

        Returns:
            None
        """
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
        """Draw the line that follows the cursor.
    
        Data may be limited on two sides. After drawing the first border, the second follows the cursor.

        Args:
            event: button_press_event

        Returns:
            None
        """
        if not event.inaxes:
            return

        if len(self.lx) != 0:
            line = self.lx[-1]
            line.set_xdata(event.xdata)
            self.canvas.restore_region(self.background)
            self.Parent.axes.draw_artist(line)
            self.canvas.figure.canvas.blit(line.axes.bbox)





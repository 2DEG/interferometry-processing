import numpy as np  # type: ignore
import pandas as pd

from typing import Tuple, Optional, List, Union


import os

def draw_data(
    graphNum,
    x: np.ndarray,
    y: np.ndarray,
    style="-",
    text: Optional[str] = None,
    scatter_x: Union[np.ndarray, List[float]] = None,
    scatter_y: Union[np.ndarray, List[float]] = None,
    name: Optional[str] = None,
    label_l: str = "Peaks",
    size: int = 9,
    clear: bool = True,
    x_label: str = r"Wavelength $[\AA]$",
    y_label: str = "Intensity",
) -> None:
    """Draws data.
    
    Draws data that it takes on the figure that is takes.

    Args:
        graphNum: shows on which particular canvas to draw

        x: data for axis-X

        y: data for axis-Y

        style: sets a style of plot

        text: text that would be written on plot

        scatter_x: data for axis-X (scatter style)

        scatter_y: data for axis-Y (scatter style)

        name: name of the plot

        label_l: labels if any

        size: fontsize

        clear: one may want to clear the plot

        x_label: str = Label to the axis-X

        y_label: str = Label to the axis-X

    Returns:
        None
    """

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

    if text is not None:

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

    if name is not None:
        graphNum.axes.set_title(name, fontsize=11)

    if (len(scatter_x) != 0) & (len(scatter_y) != 0):
        graphNum.axes.scatter(scatter_x, scatter_y, s=30, c="red", label=label_l)
        graphNum.axes.legend(fontsize=9)
    graphNum.canvas.draw()
    graphNum.toolbar.update()


def rolling_window(a: np.ndarray, window: int) -> np.ndarray:
    """Returns rolling windows.

    Args:
        a: array

        window: number of element on the window

    Returns:
        window
    """

    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)

def get_files_list(path_to_dir: str) -> np.ndarray:
    """Returns list of files in given directory.

    Args:
        path_to_dir: path to directory

    Returns:
        list of files names
    """

    f = []

    if os.path.exists(path_to_dir):
        for (dirpath, dirnames, filenames) in os.walk(path_to_dir):
            f.extend(filenames)
            break
    return np.array(f)


def refection_coef_read(path_to_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """Reads the data for reflection coefficient from file.
    
    Args:
        path_to_file: path to file

    Returns:
        lists of wavelength and intensities
    """

    if os.path.exists(path_to_file):
        n_lam = pd.read_csv(
            path_to_file, header=0, delimiter=",", encoding="utf-8", names=["x", "y"]
        )
    return n_lam["x"].values, n_lam["y"].values



def textstr(wavelength: float, period: float, n_true: float) -> str:
    """Returns the standard caption for a graph.

    Args:
        wavelength: wavelength

        period: period between intensities maximums

        n_true: reflection coefficient for given wavelength

    Returns:
        caption
    """
    from analysis import thickness
    return "\n".join(
        (
            r"$h=%.2f$ um" % (thickness(wavelength, period, n_true),),
            r"$f =%.2f \frac{1}{\AA}$" % (1 / period,),
            r"$\rho=%.2f\AA$" % (period,),
        )
    )
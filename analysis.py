
import numpy as np  # type: ignore
from typing import Tuple, Optional, List, Union
from scipy.signal import find_peaks  # type: ignore
import pandas as pd
from utils import rolling_window, draw_data, textstr

def find_nearest(array: np.ndarray, value: float) -> float:
    """Finds closest value in array for given one.
    
    Searches for the argument of the array element that is closest to some given value.

    Args:
        array: the array in which the element will be searched

        value: value for search

    Returns:
        index of the array element of closest value
    """

    array = np.asarray(array)
    return (np.abs(array - value)).argmin()


def adv_dist_calc(waveln_1: float, waveln_2: float) -> float:
    """Thickness calculation method.
    
    Calculation of thickness from exact wavelengths and refractive index. This
    method is used with rolling windows approach which result with good accuracy. 

    Args:
        waveln_1: first wavelength

        waveln_2: first wavelength

    Returns:
        sample thickness
    """
    return np.abs(
        waveln_1
        * waveln_2
        / (2 * (waveln_1 * sellmeyer_eq(waveln_2) - waveln_2 * sellmeyer_eq(waveln_1)))
    )





def rolling_dist(dat_X: np.ndarray, dat_Y: np.ndarray,) -> float:
    """Returns mean of distances obtained using rolling windows.
    
    Finds the distance between the intensity maxima. For each pair of highs,
    it considers the depth. Then it returns the average of all the resulting 
    thicknesses.

    Args:
        dat_X: array of wavelength

        dat_Y: array of intenseties

    Returns:
        mean value of thickness
    """

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


def thickness(wavelength: float, period: float, n: float) -> float:
    """Calculates the thickness of the sample using interference method.
    
    Simplified thickness calculation method 

    Args:
        wavelength: wavelength

        period: period between intensities maximums

        n: reflection coefficient for given wavelength

    Returns:
        sample thickness
    """

    return np.round(wavelength ** 2 / (2.0 * n * period * 10000.0), 2)



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
    """Performs FFT and searches for main frequencies.

    Args:
        x: array of wavelengths

        y: array of intensities

    Returns:
        psd: amplitudes

        freq: frequencies

        true_freq: frequency of interest
        
        true_psd_max: amplitude of interest
    """

    sp = np.fft.fft(y)
    freq = np.fft.fftfreq(y.shape[-1], np.diff(x).mean())
    psd = np.abs(sp)
    new_freq = freq[freq > 0.12]
    new_psd = psd[freq > 0.12]
    maxInd, _ = find_peaks(new_psd)

    true_freq = new_freq[maxInd]
    true_psd = new_psd[maxInd]

    return psd, freq, true_freq[true_psd == true_psd.max()], true_psd.max()

def make_report(
        panel,
        dat_X: np.ndarray,
        dat_Y: np.ndarray,
        wavelength: float = 9000.0,
        n_wv_idx: Optional[float] = None,
        n_true=3.54,
    ) -> None:
        """Draws another panel covered with plots filled with additional info.
    
        Performs full calculations of sample thickness. Draws graphs of all the 
        intermediate steps on the graphs panel.

        Args:
            dat_X: raw data X (wavelength)

            dat_Y: raw data Y (intensity)

            wavelength: the wavelength for which the thickness will be calculated

            n_wv_idx: deprecated positional argument

            n_true: value of appropriate refractive index for given wavelength

        Returns:
            None
        """

        panel.calc_btn.Enable(False)

        # dial = wx.ProgressDialog('Progress', 'Computation could take some time', maximum=100, parent=panel)
        # dial.Update(5)

        # Moving average method
        idx, peaks = find_peaks(dat_Y)
        draw_data(
            panel.graphs.graphFour,
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
            panel.graphs.graphOne,
            periodicity.index.values,
            periodicity_mean,
            name="Average periodicity",
            text=textstr(wavelength, period, n_true),
        )
        # dial.Update(20)

        # Fourier methods
        panel.psd, panel.freq, true_freq, true_ind = fourier_analysis(dat_X, dat_Y)
        reverse = np.zeros_like(dat_Y)
        reverse[panel.psd == true_ind] = true_freq
        reverse = np.real(np.fft.ifft(reverse))

        ## Flattening Process
        coef = np.polyfit(dat_X, dat_Y, 1)
        poly1d_fn = np.poly1d(coef)
        new_dat = dat_Y - poly1d_fn(dat_X)

        panel.psd, panel.freq, true_freq, true_ind = fourier_analysis(
            dat_X, new_dat - 200 * reverse
        )
        new_dat = new_dat / new_dat.max()

        draw_data(
            panel.graphs.graphTwo,
            panel.freq,
            panel.psd,
            # style="o",
            text=textstr(wavelength, 1 / true_freq, n_true),
            name="FFT after signal flattening",
            x_label=r"Frequency $[\frac{1}{\AA}]$",
        )
        # dial.Update(40)
        draw_data(panel.graphs.graphFive, dat_X, reverse * new_dat.max() / reverse.max())
        draw_data(
            panel.graphs.graphFive,
            dat_X,
            new_dat,
            name="IFFT and flattened signal",
            clear=False,
        )

        # Refractive index
        draw_data(
            panel.graphs.graphThree,
            panel.wvlngh,
            panel.n_coef,
            scatter_x=[panel.wvlngh[n_wv_idx]],
            scatter_y=[n_true],
            name="Refractive index",
            label_l="n = " + str(n_true),
            x_label=r"Wavelength [um]",
            y_label=r"$n(\lambda)$",
        )
        # dial.Update(60)

        draw_data(
            panel.graphs.graphSix,
            0,
            0,
            text=r"$h=\frac{\lambda^2}{2n \Delta\lambda}$",
            name="Execution Formula",
            size=16,
        )
        # dial.Update(100)

        if panel.graph.IsShown():
            panel.graph.Hide()
            panel.graphs.Show()

        panel.Parent.Layout()
        panel.Parent.Fit()   

def data_prep(panel) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
        """Prepares data to be analysed.

        Cuts data at user-specified boundaries and prepares them for depth calculations.
        Selects the average wavelength of the range and finds the corresponding refractive
        index.

        Args:
            event: wx.EVT_BUTTON

        Returns:
            dat_X: raw data X (wavelength)

            dat_Y: raw data Y (intensity)

            wavelength: the wavelength for which the thickness will be calculated

            n_wv_idx: index of n_true

            n_true: value of appropriate refractive index for given wavelength
        """
        panel.x = panel.graph.toolbar.x[1:]
        panel.graph.toolbar.x[:] = []
        panel.x.sort()

        try:
            len(panel.x) != 0
        except:
            print("Processing range not specified")

        x = np.searchsorted(panel.data[:, 0], [panel.x[0]])[0]
        y = np.searchsorted(panel.data[:, 0], [panel.x[1]])[0]
        dat_X = panel.data[x:y, 0]
        dat_Y = panel.data[x:y, 1]

        wavelength = dat_X[int(dat_X.shape[-1] / 2)]
        n_wv_idx = find_nearest(panel.wvlngh, wavelength / 10000.0)
        n_true = panel.n_coef[n_wv_idx]

        return dat_X, dat_Y, wavelength, n_wv_idx, n_true
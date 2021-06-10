import pandas as pd
import numpy as np
# from scipy.signal import find_peaks, savgol_filter
from scipy.signal import savgol_filter, find_peaks, find_peaks_cwt
import time
# from lmfit.models import LorentzianModel
from scipy.optimize import curve_fit
from utilities import *


def import_data(filename):
    df = pd.read_csv(filename, header=0, delimiter='\t', usecols=[0, 1], names=['MW', 'ODMR'])
    df.sort_values('MW', inplace=True)
    df_crop = pd.DataFrame(df[(df['MW'] > 2300) & (df['MW'] < 3490)])
    df_crop.index = np.arange(0, len(df_crop))
    scale = (max(df_crop['ODMR']) - min(df_crop['ODMR'])) / max(df_crop['ODMR']) / 1.
    df_crop['ODMR_norm'] = (1 - df_crop['ODMR'] / max(df_crop['ODMR'])) / scale
    return df_crop

def detect_peaks_weighted(x_data, y_data, weight_order = 2, min_contrast = 0.001):
    """
    Find peak position using weighted average
    Parameters
    ----------
    x_data : array like
    y_data : array like
    weight_order : int = 2
    min_contrast : float = 0.001 - minimum contrast (max-min)/max of the peak,
    if contrast smaller than min_contrast => no peak found, return 0
    """
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    contrast = (max(y_data) - min(y_data))/max(y_data)
    if contrast >= min_contrast:
        y_weights = 1.0 - (y_data - min(y_data)) / (max(y_data) - min(y_data))
        x_weighted_sum = sum(x_data * y_weights ** weight_order)
        sum_weights = sum(y_weights ** weight_order)
        x_peak = x_weighted_sum/sum_weights
    else:
        x_peak = 0
    return x_peak

def detect_peaks(x_data, y_data, debug=False):
    """
    Detect peak position with Lorentz fit.

    Parameters
    ----------
    x_data
    y_data
    debug

    Returns
    -------
    peak_positions, peak_amplitudes
    If by fitting optimal parameters not found returns 0,0

    """
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    #x, x0, amplitude, gamma, y0
    time0 = time.time()
    x0_init = np.mean(x_data)
    y0_init = max(y_data)
    amplitude_init = min(y_data) - max(y_data)
    gamma_init = 5
    try:
        popt, pconv = curve_fit(lorentz, x_data, y_data, p0=[x0_init, amplitude_init, gamma_init, y0_init])
        time1 = time.time()

        peak_positions = popt[0]
        peak_amplitudes = lorentz(peak_positions, popt[0], popt[1], popt[2], popt[3])


        if debug:
            import matplotlib.pyplot as plt
            print('fitted peak Lorenz:')
            print(popt)
            print('Time:', time1 - time0)
            y_fitted = lorentz(x_data, popt[0], popt[1], popt[2], popt[3])
            plt.plot(x_data, y_data, color='k', markersize=5, marker='o', linewidth=1)
            p = plt.plot(x_data, y_fitted)
            col = p[0].get_color()
            plt.plot(peak_positions, peak_amplitudes, "x", color=col, label='exp peaks')
            # plt.show()



    except Exception as e:
        peak_positions = 0
        peak_amplitudes = 0
        print(e)

    return peak_positions, peak_amplitudes


def detect_peaks_simple(x_data, y_data, height=None, debug=False):
    # df = pd.read_csv(filename, header=0, delimiter='\t', usecols=[0, 1], names=['MW', 'ODMR'])
    # df.sort_values('MW', inplace=True)
    # df_crop = pd.DataFrame(df[(df['MW'] > 2300) & (df['MW'] < 3490)])
    # df_crop.index = np.arange(0, len(df_crop))

    # d = {'MW': x_data, 'ODMR': y_data}
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    df = pd.DataFrame(data={'MW': x_data, 'ODMR': y_data})
    # df['ODMR'] = df['ODMR_raw'] #savgol_filter(df['ODMR_raw'], 71, 4)

    min_distance = len(df) / (max(df['MW']) - min(df['MW'])) * 50
    # print(min_distance)
    if height:
        height_norm = (max(-df['ODMR']) - min(-df['ODMR'])) * height
    time0 = time.time()
    if height:
        peaks, properties = find_peaks(-df['ODMR'], distance=min_distance, height=height_norm)
    else:
        peaks, properties = find_peaks(-df['ODMR'], distance=min_distance)
    time1 = time.time()
    peak_positions = np.array(df['MW'][peaks])
    peak_amplitudes = np.array(df['ODMR'][peaks])

    if debug:
        import matplotlib.pyplot as plt
        print('find_peaks')
        print('Time:', time1 - time0)
        print(peaks)
        print(len(peak_positions), peak_positions)
        # plt.plot(df['MW'], df['ODMR_raw'], color='b', markersize=5, marker='o', linewidth=1)
        plt.plot(df['MW'], df['ODMR'], color='k', markersize=5, marker='o', linewidth=1)
        plt.plot(df['MW'][peaks], df['ODMR'][peaks], "x", label='exp peaks')
        # plt.show()

    return peak_positions, peak_amplitudes

def detect_peaks_cwt(x_data, y_data, widthMHz = np.array([5]), min_snr=1, debug=False):
    # df = pd.read_csv(filename, header=0, delimiter='\t', usecols=[0, 1], names=['MW', 'ODMR'])
    # df.sort_values('MW', inplace=True)
    # df_crop = pd.DataFrame(df[(df['MW'] > 2300) & (df['MW'] < 3490)])
    # df_crop.index = np.arange(0, len(df_crop))

    # d = {'MW': x_data, 'ODMR': y_data}
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    df = pd.DataFrame(data={'MW': x_data, 'ODMR': y_data})
    # df['ODMR'] = df['ODMR_raw'] #savgol_filter(df['ODMR_raw'], 71, 4)

    min_distance = len(df) / (max(df['MW']) - min(df['MW'])) * 50
    width = len(df) / (max(df['MW']) - min(df['MW'])) * np.array(widthMHz)
    print(width)
    # print(min_distance)

    time0 = time.time()

    peaks = find_peaks_cwt(-df['ODMR'], widths = width, min_snr=min_snr)

    time1 = time.time()
    peak_positions = np.array(df['MW'][peaks])
    peak_amplitudes = np.array(df['ODMR'][peaks])

    if debug:
        import matplotlib.pyplot as plt
        print('find_peaks')
        print('Time:', time1 - time0)
        print(peaks)
        print(len(peak_positions), peak_positions)
        # plt.plot(df['MW'], df['ODMR_raw'], color='b', markersize=5, marker='o', linewidth=1)
        plt.plot(df['MW'], df['ODMR'], color='k', markersize=5, marker='o', linewidth=1)
        plt.plot(df['MW'][peaks], df['ODMR'][peaks], "x", label='exp peaks')
        # plt.show()

    return peak_positions, peak_amplitudes


if __name__ == '__main__':

    filename = 'test_data/test_dev20.0_peak2611.0.dat'
    dataframe = import_data(filename)
    print(dataframe.head())
    peaks, amplitudes = detect_peaks(dataframe['MW'], dataframe['ODMR'], debug=True)
    print(peaks)
    print(amplitudes)

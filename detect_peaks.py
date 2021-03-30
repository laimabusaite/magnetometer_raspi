import pandas as pd
import numpy as np
from scipy.signal import find_peaks
# from scipy.signal import savgol_filter, find_peaks, find_peaks_cwt
import time

def import_data(filename):
    df = pd.read_csv(filename, header=0, delimiter='\t', usecols=[0, 1], names=['MW', 'ODMR'])
    df.sort_values('MW', inplace=True)
    df_crop = pd.DataFrame(df[(df['MW'] > 2300) & (df['MW'] < 3490)])
    df_crop.index = np.arange(0, len(df_crop))
    scale = (max(df_crop['ODMR']) - min(df_crop['ODMR'])) / max(df_crop['ODMR']) / 1.
    df_crop['ODMR_norm'] = (1 - df_crop['ODMR'] / max(df_crop['ODMR'])) / scale
    return df_crop

def detect_peaks(x_data, y_data, debug=False):
    # df = pd.read_csv(filename, header=0, delimiter='\t', usecols=[0, 1], names=['MW', 'ODMR'])
    # df.sort_values('MW', inplace=True)
    # df_crop = pd.DataFrame(df[(df['MW'] > 2300) & (df['MW'] < 3490)])
    # df_crop.index = np.arange(0, len(df_crop))

    # d = {'MW': x_data, 'ODMR': y_data}
    x_data = np.array(x_data)
    y_data = np.array(y_data)
    df = pd.DataFrame(data={'MW': x_data, 'ODMR': y_data})

    min_distance = len(df) / (max(df['MW']) - min(df['MW'])) * 50
    height = (max(-df['ODMR']) - min(-df['ODMR'])) * 0.1
    time0 = time.time()
    # peaks, properties = find_peaks(-df['ODMR'], distance=min_distance, height=height)
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
        plt.plot(df['MW'], df['ODMR'], color='k', markersize=5, marker='o', linewidth=1)
        plt.plot(df['MW'][peaks], df['ODMR'][peaks], "x", label='exp peaks')
        plt.show()

    return peak_positions, peak_amplitudes


if __name__ == '__main__':
    filename = 'data/test_2920_650_16dBm_1024_ODMR.dat'
    dataframe = import_data(filename)
    print(dataframe.head())
    peaks, amplitudes = detect_peaks(dataframe['MW'], dataframe['ODMR'], debug=True)
    print(peaks)
    print(amplitudes)


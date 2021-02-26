import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
# from scipy.signal import savgol_filter, find_peaks, find_peaks_cwt
import time

def detect_peaks(filename, debug=False):
    df = pd.read_csv(filename, header=0, delimiter='\t', usecols=[0, 1], names=['MW', 'ODMR'])
    df.sort_values('MW', inplace=True)
    df_crop = pd.DataFrame(df[(df['MW'] > 2300) & (df['MW'] < 3490)])
    df_crop.index = np.arange(0, len(df_crop))
    min_distance = len(df_crop) / (max(df['MW']) - min(df['MW'])) * 50
    time0 = time.time()
    peaks, properties = find_peaks(-df_crop['ODMR'], distance=min_distance, height=0.00035)
    time1 = time.time()
    peak_positions = np.array(df_crop['MW'][peaks])

    if debug:
        print('find_peaks')
        print('Time:', time1 - time0)
        print(peaks)
        print(len(peak_positions), peak_positions)
        plt.plot(df_crop['MW'], df_crop['ODMR'], color='k', markersize=5, marker='o', linewidth=1)
        plt.plot(df_crop['MW'][peaks], df_crop['ODMR'][peaks], "x", label='exp peaks')
        plt.show()

    return peak_positions


if __name__ == '__main__':
    filename = 'data/test_2920_650_16dBm_1024_ODMR.dat'
    peaks = detect_peaks(filename)
    print(peaks)
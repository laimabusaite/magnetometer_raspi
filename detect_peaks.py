import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
# from scipy.signal import savgol_filter, find_peaks, find_peaks_cwt
import time


filename = 'data/test_2920_650_16dBm_1024_ODMR.dat'
df = pd.read_csv(filename, header=0, delimiter='\t', usecols=[0, 1], names=['MW', 'ODMR'])
df.sort_values('MW', inplace=True)

df_crop = pd.DataFrame(df[(df['MW'] > 2300) & (df['MW'] < 3490)])
df_crop.index = np.arange(0, len(df_crop))
# plt.plot(df_crop['MW'], df_crop['ODMR'], color='k', markersize=5, marker='o', linewidth=1)


print('find_peaks')
min_distance = len(df_crop) / (max(df['MW']) - min(df['MW'])) * 50
print('min_distance', min_distance)
time0 = time.time()
peaks, properties = find_peaks(-df_crop['ODMR'], distance=min_distance, height=0.00035)
print('Time:', time.time() - time0)
print(peaks)
# plt.plot(df_crop['MW'][peaks], df_crop['ODMR'][peaks], "x", label='exp peaks')
peak_positions = np.array(df_crop['MW'][peaks])
print(len(peak_positions), peak_positions)
#
#
# print('find_peaks_cwt')
# width_idx = len(df_crop) / (max(df['MW']) - min(df['MW'])) * 6.8
# time1 = time.time()
# peaks_cwt = find_peaks_cwt(-df_crop['ODMR'], widths=np.full(8, width_idx), min_snr=1.2)
# print("Time cwt:", time.time() - time1)
# print(peaks_cwt)
# peak_positions_cwt = np.array(df_crop['MW'][peaks_cwt])
# print(len(peak_positions_cwt), peak_positions_cwt)
# plt.plot(df_crop['MW'][peaks_cwt], df_crop['ODMR'][peaks_cwt], "x", label='exp peaks')

# plt.show()
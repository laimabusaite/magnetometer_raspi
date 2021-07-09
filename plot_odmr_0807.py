import glob
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import fit_odmr
from detect_peaks import detect_peaks

import NVcenter as nv
from utilities import *

if __name__ == '__main__':

    dB = 0.001
    foldername = 'RQnc/arb/0G'
    num = 12
    if num == 13:
        num_str = f'{num}_real'
    else:
        num_str = f'{num}'

    filename_list = sorted(glob.glob(f'{foldername}/a{num_str}*.dat'))
    # filename_list = ['RQnc/arb/2G/a12_dev30_avg16_dev30.0_peak2616.8_2021-07-07_02-38-57.360786.dat',
    #                  'RQnc/arb/2G/a12_dev30_avg16_dev30.0_peak2891.7_2021-07-07_02-39-01.311106.dat',
    #                  'RQnc/arb/2G/a12_dev30_avg16_dev30.0_peak3223.8_2021-07-07_02-39-02.089214.dat',
    #                  'RQnc/arb/2G/a12_dev30_avg16_dev30.0_peak3374.1_2021-07-07_02-39-02.866583.dat']
    print(filename_list)
    filename_array = np.reshape(filename_list, (4, -1))
    print(filename_array)

    filename_list = filename_array[:, 0]
    print(filename_list)

    full_filename_list = sorted(glob.glob(f'{foldername}/full_scan{num}*.dat'))
    print(full_filename_list)

    # filename_list.append(full_filename_list[0])

    # print(filename_list)

    dataframe_full = pd.read_csv(full_filename_list[0], names=['MW', 'ODMR'], sep='\t')
    print(dataframe_full.head())
    # plt.plot(dataframe_full['MW'], dataframe_full['ODMR'], c='k')
    B = 200
    theta = 80
    phi = 20
    Blab = CartesianVector(B, theta, phi)
    print(Blab)
    init_params = {'B_labx': Blab[0], 'B_laby': Blab[1], 'B_labz': Blab[2], 'glor': 10., 'D': 2870.00, 'Mz1': 1.67,
                   'Mz2': 1.77, 'Mz3': 1.83, 'Mz4': 2.04}

    parameters = fit_odmr.fit_full_odmr(dataframe_full['MW'], dataframe_full['ODMR'], init_params=init_params,
                                        debug=False)

    D = parameters["D"]  # 2867.61273
    Mz_array = np.array([parameters["Mz1"], parameters["Mz2"], parameters["Mz3"], parameters[
        "Mz4"]])  # np.array([1.85907453, 2.16259814, 1.66604227, 2.04334145]) #np.array([7.32168327, 6.66104172, 9.68158138, 5.64605102])
    B_lab = np.array([parameters["B_labx"], parameters["B_laby"], parameters[
        "B_labz"]])  # np.array([169.121512, 87.7180839, 40.3986877]) #np.array([191.945068, 100.386360, 45.6577322])

    print('B_lab = ', B_lab)
    # NV center orientation in laboratory frame
    # (100)
    nv_center_set = nv.NVcenterSet(D=D, Mz_array=Mz_array)
    nv_center_set.setMagnetic(B_lab=B_lab)
    # print(nv_center_set.B_lab)
    frequencies0 = nv_center_set.four_frequencies(np.array([2000, 3500]), nv_center_set.B_lab)
    print('frequencies0')
    print(frequencies0)

    A_inv = nv_center_set.calculateAinv(nv_center_set.B_lab, dB=dB)

    plt.plot(dataframe_full['MW'], dataframe_full['ODMR'], c='k')

    peak_list = []
    peak_full_list = []
    for filename in filename_list:
        filename_split = re.split('peak|_', filename)
        print(filename_split)
        # print(filename_split[6])
        if 'real' in filename_split:
            fr_centr = float(filename_split[6])
        else:
            fr_centr = float(filename_split[5])
        fr_dev = 15
        x_full = \
        dataframe_full[(dataframe_full['MW'] >= fr_centr - fr_dev) & (dataframe_full['MW'] <= fr_centr + fr_dev)]['MW']
        y_full = \
        dataframe_full[(dataframe_full['MW'] >= fr_centr - fr_dev) & (dataframe_full['MW'] <= fr_centr + fr_dev)][
            'ODMR']

        peak_full = detect_peaks(x_full, y_full, debug=True)
        peak_full_list.append(peak_full)

        dataframe = pd.read_csv(filename, names=['MW', 'ODMR'], sep='\t')
        print(dataframe.head())

        peak = detect_peaks(dataframe['MW'], dataframe['ODMR'], debug=True)
        peak_list.append(peak)

        plt.plot(dataframe['MW'], dataframe['ODMR'])

    peaks_list = np.array(peak_list).flatten()
    print(peaks_list)
    peaks_full_list = np.array(peak_full_list).flatten()
    print(peaks_full_list)

    # delta_frequencies = frequencies0 - peaks_list
    delta_frequencies = peaks_full_list - peaks_list
    print('delta frequencies:', delta_frequencies)
    # delta_frequencies = np.zeros(4) + 0.3

    Bsens = deltaB_from_deltaFrequencies(A_inv, delta_frequencies)

    print("\nB =", Bsens, np.linalg.norm(Bsens))

    plt.show()

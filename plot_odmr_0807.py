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
    # foldername = 'RQnc/arb1/0G'
    # foldername = 'RQnc/reverse_scan'
    foldername = 'RQnc/bidirectional_scan'
    # foldername = 'example_data'
    num = 1
    if num == 13:
        num_str = f'{num}_real'
    else:
        num_str = f'{num}'

    filename_list = sorted(glob.glob(f'{foldername}/a{num_str}_*.dat'))
    # filename_list = ['RQnc/arb/2G/a12_dev30_avg16_dev30.0_peak2616.8_2021-07-07_02-38-57.360786.dat',
    #                  'RQnc/arb/2G/a12_dev30_avg16_dev30.0_peak2891.7_2021-07-07_02-39-01.311106.dat',
    #                  'RQnc/arb/2G/a12_dev30_avg16_dev30.0_peak3223.8_2021-07-07_02-39-02.089214.dat',
    #                  'RQnc/arb/2G/a12_dev30_avg16_dev30.0_peak3374.1_2021-07-07_02-39-02.866583.dat']

    filename_list.sort(key=lambda x: f"{x.split('_')[-2]}_{x.split('_')[-1]}")
    print('filename_list')
    print(filename_list)
    # filename_array = np.reshape(filename_list, (4, -1))
    filename_array = np.reshape(filename_list, (-1, 4)).transpose()
    print('filename_array')
    print(filename_array)

    # filename_list = filename_array[:, 0]
    print('filename_array[:, 0]')
    print(filename_list)

    full_filename_list = sorted(glob.glob(f'{foldername}/full_scan{num}*.dat'))
    print(full_filename_list)

    # filename_list.append(full_filename_list[0])

    # print(filename_list)

    dataframe_full = pd.read_csv(full_filename_list[0], names=['MW', 'ODMR'], sep='\t')
    print(dataframe_full.head())
    plt.plot(dataframe_full['MW'], dataframe_full['ODMR'], c='k')
    B = 200
    theta = 80
    phi = 20
    Blab = CartesianVector(B, theta, phi)
    print(Blab)
    init_params = {'B_labx': Blab[0], 'B_laby': Blab[1], 'B_labz': Blab[2], 'glor': 10., 'D': 2870.00, 'Mz1': 1.67,
                   'Mz2': 1.77, 'Mz3': 1.83, 'Mz4': 2.04}

    parameters = fit_odmr.fit_full_odmr(dataframe_full['MW'], dataframe_full['ODMR'], init_params=init_params,
                                        debug=False, save=False)

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

    # plt.figure('ODMR', figsize=(5,4))
    # plt.plot(dataframe_full['MW'], dataframe_full['ODMR'], c='k')

    print(filename_array.shape)
    for idx in range(filename_array.shape[1]):
        # if idx > 0:
        #     break
    # for idx in range(1):
        # print(idx)
        filename_list = filename_array[:, idx]
        peak_list = []
        peak_full_list = []
        for idx2, filename in enumerate(filename_list):
            print(idx, idx2)
            # print(filename, type(filename))
            filename_split = re.split('peak|_', filename)
            print(filename_split)
            peak_idx = filename.find('peak')
            date_idx = filename.find('_2021-07')
            fr_centr = float(filename[peak_idx + 4:date_idx])
            # print(filename_split[6])
            # if 'real' in filename_split:
            #     fr_centr = float(filename_split[6])
            # else:
            #     fr_centr = float(filename_split[5])

            #fit full scan

            fr_dev = 15
            x_full = \
            dataframe_full[(dataframe_full['MW'] >= fr_centr - fr_dev) & (dataframe_full['MW'] <= fr_centr + fr_dev)]['MW']
            y_full = \
            dataframe_full[(dataframe_full['MW'] >= fr_centr - fr_dev) & (dataframe_full['MW'] <= fr_centr + fr_dev)][
                'ODMR']

            if idx == 0:
                debug=True
            else:
                debug=False
            peak_full = detect_peaks(x_full, y_full, debug=False)
            peak_full_list.append(peak_full)

            dataframe = pd.read_csv(filename, names=['MW', 'ODMR'], sep='\t')
            print(dataframe.head())

            peak = detect_peaks(dataframe['MW'], dataframe['ODMR'], debug=False)
            peak_list.append(peak)

            plt.plot(dataframe['MW'], dataframe['ODMR'], marker='.', ls='-')

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

        plt.xlabel('Microwave frequency (MHz)')
        plt.ylabel('Normalized fluorescence intensity (arb. units)')
        fig = plt.gcf()
        fig.set_size_inches(5, 4)
        proc = 0.15

        # for fr in peaks_full_list:
        #     plt.xlim(fr - 30, fr + 30)
        #     ymin = min(dataframe_full[(dataframe_full['MW'] > fr - 30) & (dataframe_full['MW'] < fr + 30)]['ODMR'])
        #     ymax = max(
        #         dataframe_full[(dataframe_full['MW'] > fr - 30) & (dataframe_full['MW'] < fr + 30)]['ODMR'])
        #
        #     plt.ylim(ymin=ymin - (ymax - ymin)*proc, ymax=ymax + (ymax - ymin)*proc)
        #     plt.tight_layout()
            # plt.savefig(f'/home/laima/Dropbox/Apps/Overleaf/ESA D10 - Software  Design Document/full_plus_meas/v3_fr{fr:.0f}.pdf', bbox_inches='tight')
            # plt.savefig(f'/home/laima/Dropbox/Apps/Overleaf/ESA D09 - Electronics Design Document/v3_fr{fr:.0f}.pdf', bbox_inches='tight')
            # plt.savefig(f'/home/laima/Dropbox/Apps/Overleaf/ESA D09 - Electronics Design Document/v3_fr{fr:.0f}.png',
            #             bbox_inches='tight')

        fig = plt.gcf()
        fig.set_size_inches(9, 4)
        plt.xlim(min(dataframe_full['MW']), max(dataframe_full['MW']))
        ymin = min(dataframe_full['ODMR'])
        ymax = max(dataframe_full['ODMR'])
        plt.ylim(ymin=ymin - (ymax - ymin) * proc, ymax=ymax + (ymax - ymin) * proc)
        plt.tight_layout()
        # plt.savefig(f'/home/laima/Dropbox/Apps/Overleaf/ESA D10 - Software  Design Document/full_plus_meas/v3_full.pdf', bbox_inches='tight')

    plt.show()

import NVcenter as nv
import numpy as np
from detect_peaks import *
from utilities import *
import glob
import json

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import os
    ### parameters 06.04.2021
    # B_labx: 169.121512 + / - 0.01092155(0.01 %)(init=169.1182)
    # B_laby: 87.7180839 + / - 0.08322728(0.09 %)(init=87.70597)
    # B_labz: 40.3986877 + / - 0.11793138(0.29 %)(init=40.4266)
    # glor: 4.44135742 + / - 0.01707828(0.38 %)(init=5)
    # D: 2867.61273 + / - 32865.3284(1146.09 %)(init=2870)
    # Mz1: 1.85907453 + / - 32864.3339(1767779.26 %)(init=0)
    # Mz2: 2.16259814 + / - 32865.4396(1519720.14 %)(init=0)
    # Mz3: 1.66604227 + / - 32864.9733(1972637.42 %)(init=0)
    # Mz4: 2.04334145 + / - 32864.9365(1608391.81 %)(init=0)

    # read parameters
    filename = "ODMR_fit_parameters.json"
    a_file = open(filename, "r")
    parameters = json.load(a_file)
    # parameters = dict(parameters)
    print(parameters)


    D = parameters["D"] #2867.61273
    Mz_array = np.array([parameters["Mz1"], parameters["Mz2"], parameters["Mz3"], parameters["Mz4"]]) #np.array([1.85907453, 2.16259814, 1.66604227, 2.04334145]) #np.array([7.32168327, 6.66104172, 9.68158138, 5.64605102])

    B_lab = np.array([parameters["B_labx"], parameters["B_laby"], parameters["B_labz"]]) #np.array([169.121512, 87.7180839, 40.3986877]) #np.array([191.945068, 100.386360, 45.6577322])

    print('D = ', D)
    print('Mz_array =', Mz_array)
    print('B_lab =', B_lab)

    # NV center orientation in laboratory frame
    # (100)
    nv_center_set = nv.NVcenterSet(D=D, Mz_array=Mz_array)
    nv_center_set.setMagnetic(B_lab=B_lab)
    # print(nv_center_set.B_lab)

    frequencies0 = nv_center_set.four_frequencies(np.array([2000, 3500]), nv_center_set.B_lab)

    # print(frequencies0)

    A_inv = nv_center_set.calculateAinv(nv_center_set.B_lab)

    # print(A_inv)

    # filename = 'data/test_2920_650_16dBm_1024_ODMR.dat'
    # filenames = sorted(glob.glob("test_data/*.dat"))
    filenames_all = sorted(glob.glob("data/*.dat"))
    filenames = [filename for filename in filenames_all if '0_magnet' in filename]
    print(filenames)
    peaks_list = []
    #print()
    # plt.figure()
    plt.figure(1)
    for idx, filename in enumerate(filenames):
        # plt.figure(idx+1)
        print(filename)
        dataframe = import_data(filename)
        # print(dataframe)
        # dataframe.to_csv(f'test_pandas_save/{os.path.basename(filename)}', header=None, index=None, sep=' ', mode='a')
        peaks, amplitudes = detect_peaks(dataframe['MW'], dataframe['ODMR'], debug=True)
        #print(peaks)
        peaks_list.append(peaks)
    # plt.show()

    peaks_list = np.array(peaks_list).flatten()
    print(peaks_list)

    delta_frequencies = frequencies0 - peaks_list #peaks[1::2]
    #print("\ndelta F =",delta_frequencies)

    Bsens = deltaB_from_deltaFrequencies(A_inv, delta_frequencies)

    print("\nB =",Bsens)

    plt.figure(2)
    print()
    print('cwt')
    peaks_list = []
    for idx, filename in enumerate(filenames):
        # plt.figure(idx+1)
        print(filename)
        dataframe = import_data(filename)
        # print(dataframe)
        # dataframe.to_csv(f'test_pandas_save/{os.path.basename(filename)}', header=None, index=None, sep=' ', mode='a')
        peaks, amplitudes = detect_peaks_cwt(dataframe['MW'], dataframe['ODMR'], debug=True)
        #print(peaks)
        peaks_list.append(peaks)
    # plt.show()

    peaks_list = np.array(peaks_list).flatten()
    print(peaks_list)

    delta_frequencies = frequencies0 - peaks_list #peaks[1::2]
    #print("\ndelta F =",delta_frequencies)

    Bsens = deltaB_from_deltaFrequencies(A_inv, delta_frequencies)

    print("\nB =",Bsens)

    peaks_list = []
    plt.figure(3)
    for idx, filename in enumerate(filenames):
        # plt.figure(idx+1)
        print(filename)
        dataframe = import_data(filename)
        # print(dataframe)
        # dataframe.to_csv(f'test_pandas_save/{os.path.basename(filename)}', header=None, index=None, sep=' ', mode='a')
        peaks, _ = detect_peaks_simple(dataframe['MW'], dataframe['ODMR'], debug=True)
        print(peaks)
        peaks_list.append(peaks)

    plt.show()

    peaks_list = np.array(peaks_list).flatten()
    print(peaks_list)

    delta_frequencies = frequencies0 - peaks_list  # peaks[1::2]
    # print("\ndelta F =",delta_frequencies)

    Bsens = deltaB_from_deltaFrequencies(A_inv, delta_frequencies)

    print("\nB =", Bsens)



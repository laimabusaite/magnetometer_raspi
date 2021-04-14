import numpy as np
import NVcenter as nv
import matplotlib.pyplot as plt
from detect_peaks import *
from utilities import *
import json


if __name__ == '__main__':

    # D = 2851.26115
    # Mz_array = np.array([7.32168327, 6.66104172, 9.68158138, 5.64605102])
    # B_lab = np.array([191.945068, 100.386360, 45.6577322])
    # read parameters
    filename = "ODMR_fit_parameters.json"
    a_file = open(filename, "r")
    parameters = json.load(a_file)
    # parameters = dict(parameters)
    print(parameters)

    D = parameters["D"]
    Mz_array = np.array([parameters["Mz1"], parameters["Mz2"], parameters["Mz3"], parameters[
        "Mz4"]])
    B_lab = np.array([parameters["B_labx"], parameters["B_laby"], parameters[
        "B_labz"]])

    nv_center_set = nv.NVcenterSet(D=D, Mz_array=Mz_array)
    nv_center_set.setMagnetic(B_lab=B_lab)
    print(nv_center_set.B_lab)

    # omega = np.linspace(2000, 3800, 5000)
    omega = np.arange(2000, 3800, 0.4)
    g_lor = 10
    odmr_signal = - nv_center_set.sum_odmr(omega, g_lor)
    peaks_clean, amplitudes_clean = detect_peaks(omega, odmr_signal)
    # plt.plot(omega, odmr_signal)

    # omega = np.linspace(2000, 3800, 1000)
    # omega = np.arange(2000, 3800, 0.1)
    g_lor = 10
    odmr_signal = - nv_center_set.sum_odmr(omega, g_lor)

    noise_std = 0.1
    noisy_odmr_signal = add_noise(odmr_signal, noise_std)
    peaks, amplitudes = detect_peaks(omega, noisy_odmr_signal, height=0.5)
    peaks_cwt, amplitudes_cwt = detect_peaks_cwt(omega, noisy_odmr_signal, widthMHz=[5]*8, min_snr=2.5)
    print(peaks_cwt)

    odmr_smooth_signal = savgol_filter(noisy_odmr_signal, 61, 2)
    peaks_smooth, amplitudes_smooth = detect_peaks(omega, odmr_smooth_signal, height=0.2)


    plt.plot(omega, noisy_odmr_signal, color='C0')
    plt.plot(omega, odmr_smooth_signal, color='C2')
    plt.plot(omega, odmr_signal, color='C1')

    plt.vlines(peaks, min(noisy_odmr_signal), max(noisy_odmr_signal), colors='C0')
    plt.vlines(peaks_clean, min(noisy_odmr_signal), max(noisy_odmr_signal), colors='C1')
    plt.vlines(peaks_smooth, min(noisy_odmr_signal), max(noisy_odmr_signal), colors='C2')
    plt.vlines(peaks_cwt, min(noisy_odmr_signal), max(noisy_odmr_signal), colors='C3')
    plt.show()

    print('peaks_clean\n', peaks_clean)
    print('peaks\n', peaks)
    print('peaks_smooth\n', peaks_smooth)
    print('peaks_cwt\n', peaks_cwt)

    A_inv = nv_center_set.calculateAinv(nv_center_set.B_lab)
    print('A_inv\n', A_inv)

    print('From noisy signal')
    delta_frequencies = (peaks_clean - peaks)[1::2]
    print('delta_frequencies\n', delta_frequencies)
    Bsens = deltaB_from_deltaFrequencies(A_inv, delta_frequencies)
    print('Bsens', Bsens)
    print('From noisy signal cwt')
    delta_frequencies = (peaks_clean - peaks_cwt)[1::2]
    print('delta_frequencies\n', delta_frequencies)
    Bsens = deltaB_from_deltaFrequencies(A_inv, delta_frequencies)
    print('Bsens', Bsens)

    print('From smooth signal')
    delta_frequencies_smooth = (peaks_clean - peaks_smooth)[1::2]
    print('delta_frequencies\n', delta_frequencies)
    Bsens = deltaB_from_deltaFrequencies(A_inv, delta_frequencies_smooth)
    print('Bsens', Bsens)

    print('From clean signal')
    delta_frequencies_clean = (peaks_clean - peaks_clean)[1::2]
    print('delta_frequencies\n', delta_frequencies_clean)
    Bsens = deltaB_from_deltaFrequencies(A_inv, delta_frequencies_clean)
    print('Bsens', Bsens)

    print('\nnoisy peaks')
    print(peaks_clean)
    for i in range(10):
        peaks_noisy = add_noise(peaks_clean, 0.5)
        # print(peaks_noisy)
        delta_frequencies = (peaks_clean - peaks_noisy)[1::2]
        # print('delta_frequencies\n', delta_frequencies)
        Bsens = deltaB_from_deltaFrequencies(A_inv, delta_frequencies)
        print('Bsens', Bsens, 'delta_frequencies', delta_frequencies)









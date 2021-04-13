import numpy as np
import NVcenter as nv
import matplotlib.pyplot as plt
from detect_peaks import *
from utilities import *


if __name__ == '__main__':

    D = 2851.26115
    Mz_array = np.array([7.32168327, 6.66104172, 9.68158138, 5.64605102])
    B_lab = np.array([191.945068, 100.386360, 45.6577322])

    nv_center_set = nv.NVcenterSet(D=D, Mz_array=Mz_array)
    nv_center_set.setMagnetic(B_lab=B_lab)
    print(nv_center_set.B_lab)

    # omega = np.linspace(2000, 3800, 5000)
    omega = np.arange(2000, 3800, 0.2)
    g_lor = 10
    odmr_signal = - nv_center_set.sum_odmr(omega, g_lor)
    peaks_clean, amplitudes_clean = detect_peaks(omega, odmr_signal)
    plt.plot(omega, odmr_signal)

    # omega = np.linspace(2000, 3800, 1000)
    # omega = np.arange(2000, 3800, 0.1)
    g_lor = 10
    odmr_signal = - nv_center_set.sum_odmr(omega, g_lor)

    noise_std = 0.1
    noisy_odmr_signal = add_noise(odmr_signal, noise_std)
    # print(odmr_signal)
    plt.plot(omega, odmr_signal)
    plt.plot(omega, noisy_odmr_signal)
    peaks, amplitudes = detect_peaks(omega, noisy_odmr_signal, height=0.5)
    odmr_smooth_signal = savgol_filter(noisy_odmr_signal, 61, 2)
    plt.plot(omega, odmr_smooth_signal)
    peaks_smooth, amplitudes_smooth = detect_peaks(omega, odmr_smooth_signal, height=0.2)
    plt.vlines(peaks, min(noisy_odmr_signal), max(noisy_odmr_signal))
    plt.vlines(peaks_clean, min(noisy_odmr_signal), max(noisy_odmr_signal), colors='k')
    plt.vlines(peaks_smooth, min(noisy_odmr_signal), max(noisy_odmr_signal), colors='r')
    plt.show()

    print(peaks_clean)
    print(peaks)
    print(peaks_smooth)

    A_inv = nv_center_set.calculateAinv(nv_center_set.B_lab)
    print(A_inv)

    delta_frequencies = (peaks_clean - peaks)[1::2]
    print(delta_frequencies)
    Bsens = deltaB_from_deltaFrequencies(A_inv, delta_frequencies)
    print(Bsens)

    delta_frequencies_smooth = (peaks_clean - peaks_smooth)[1::2]
    print(delta_frequencies_smooth)
    Bsens = deltaB_from_deltaFrequencies(A_inv, delta_frequencies_smooth)
    print(Bsens)

    delta_frequencies_clean = (peaks_clean - peaks_clean)[1::2]
    print(delta_frequencies_clean)
    Bsens = deltaB_from_deltaFrequencies(A_inv, delta_frequencies_clean)
    print(Bsens)









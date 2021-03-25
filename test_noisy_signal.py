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

    omega = np.linspace(2000, 3800, 50000)
    g_lor = 10
    odmr_signal = - nv_center_set.sum_odmr(omega, g_lor)
    peaks_clean, amplitudes_clean = detect_peaks(omega, odmr_signal)
    plt.plot(omega, odmr_signal)

    # omega = np.linspace(2000, 3800, 1000)
    # omega = np.arange(2000, 3800, 0.1)
    g_lor = 10
    odmr_signal = - nv_center_set.sum_odmr(omega, g_lor)

    noise_std = 0.005
    noisy_odmr_signal = nv.add_noise(odmr_signal, noise_std)
    # print(odmr_signal)
    plt.plot(omega, odmr_signal)
    plt.plot(omega, noisy_odmr_signal)
    peaks, amplitudes = detect_peaks(omega, noisy_odmr_signal)
    # plt.plot(peaks, noisy_odmr_signal[peaks], "x", label='exp peaks')
    plt.vlines(peaks, min(noisy_odmr_signal), max(noisy_odmr_signal))
    plt.vlines(peaks_clean, min(noisy_odmr_signal), max(noisy_odmr_signal))

    delta_frequencies = (peaks_clean - peaks)[1::2]
    print(delta_frequencies)

    A_inv = nv_center_set.calculateAinv(nv_center_set.B_lab)
    print(A_inv)

    Bsens = deltaB_from_deltaFrequencies(A_inv, delta_frequencies)
    print(Bsens)




    nv_center_set_test = nv.NVcenterSet(D=D, Mz_array=Mz_array)
    nv_center_set_test.setMagnetic(B_lab=B_lab)

    B_sens = np.array([0, 0, 0])






    plt.show()


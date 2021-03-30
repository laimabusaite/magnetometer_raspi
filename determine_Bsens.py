import NVcenter as nv
import numpy as np
from detect_peaks import *
from utilities import *
import glob

if __name__ == '__main__':
    # B_labx: 191.922866 + / - 0.03154444(0.02 %)(init=191.9451)
    # B_laby: 100.320155 + / - 0.01157694(0.01 %)(init=100.3864)
    # B_labz: 45.4196647 + / - 0.01582443(0.03 %)(init=45.65773)
    # glor: 7.09188935 + / - 0.00750257(0.11 %)(init=10)
    # D: 2866.66847 + / - 16128.0752(562.61 %)(init=2851.261)
    # Mz1: -8.23252697 + / - 16128.0808(195906.81 %)(init=0)
    # Mz2: -8.59166628 + / - 16128.0432(187717.29 %)(init=0)
    # Mz3: -6.41586921 + / - 16127.9931(251376.59 %)(init=0)
    # Mz4: -9.94692311 + / - 16128.3296(162143.91 %)(init=0)

    D = 2866.66847 #2851.26115
    Mz_array = np.array([-8.23252697, -8.59166628, -6.41586921, -9.94692311]) #np.array([7.32168327, 6.66104172, 9.68158138, 5.64605102])


    B_lab = np.array([191.922866, 100.320155, 45.4196647]) #np.array([191.945068, 100.386360, 45.6577322])

    # NV center orientation in laboratory frame
    # (100)
    nv_center_set = nv.NVcenterSet(D=D, Mz_array=Mz_array)
    nv_center_set.setMagnetic(B_lab=B_lab)
    print(nv_center_set.B_lab)

    frequencies0 = nv_center_set.four_frequencies(np.array([2000, 3500]), nv_center_set.B_lab)

    print(frequencies0)

    A_inv = nv_center_set.calculateAinv(nv_center_set.B_lab)

    print(A_inv)

    # filename = 'data/test_2920_650_16dBm_1024_ODMR.dat'
    filenames = sorted(glob.glob("test_data/*.dat"))
    peaks_list = []
    for filename in filenames:
        dataframe = import_data(filename)
        # print(dataframe)
        peaks, amplitudes = detect_peaks(dataframe['MW'], dataframe['ODMR'], debug=False)
        print(peaks)
        peaks_list.append(peaks)

    peaks_list = np.array(peaks_list).flatten()
    print(peaks_list)

    delta_frequencies = frequencies0 - peaks_list #peaks[1::2]
    print(delta_frequencies)

    Bsens = deltaB_from_deltaFrequencies(A_inv, delta_frequencies)

    print(Bsens)

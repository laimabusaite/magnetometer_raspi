import glob

import NVcenter
import detect_peaks as dp
import matplotlib.pyplot as plt
import numpy as np

import fit_odmr
import utilities

if __name__ == '__main__':

    filedir = 'test_data_rp'
    filenames = sorted(glob.glob(f'{filedir}/test_full_scan*.dat'))
    fullscan_filename = filenames[-1]
    print(fullscan_filename)
    #import full scan
    dataframe_full = dp.import_data(fullscan_filename)
    print(dataframe_full.head())

    x_data_full = dataframe_full['MW']
    y_data_full = dataframe_full['ODMR']
    # peak_positions, peak_amplitudes = dp.detect_peaks_simple(x_data, y_data, height=0.1, debug=False)
    peak_positions, peak_amplitudes = dp.detect_peaks_cwt(x_data_full, y_data_full, widthMHz=np.array([10]), min_snr=1,
                                                          debug=False)
    plt.plot(x_data_full, y_data_full, lw=2, c='k', label='fullscan')
    # plt.scatter(peak_positions, peak_amplitudes, c='k')

    B = 200
    theta = 80
    phi = 20
    Blab = utilities.CartesianVector(B, theta, phi)
    # Blab = get_initial_magnetic_field(x_data, y_data) #CartesianVector(B, theta, phi)
    print(Blab)
    B = np.linalg.norm(Blab)
    print(B)
    init_params = {'B_labx': Blab[0], 'B_laby': Blab[1], 'B_labz': Blab[2],
                   'glor': 5, 'D': 2870, 'Mz1': 0,
                   'Mz2': 0, 'Mz3': 0, 'Mz4': 0}
    parameters = fit_odmr.fit_full_odmr(x_data_full, y_data_full, init_params=init_params, save=False, debug=False)
    D = parameters["D"]
    Mz_array = np.array([parameters["Mz1"], parameters["Mz2"], parameters["Mz3"], parameters[
        "Mz4"]])
    B_lab = np.array([parameters["B_labx"], parameters["B_laby"], parameters[
        "B_labz"]])

    # NV center orientation in laboratory frame
    # (100)
    nv_center_set = NVcenter.NVcenterSet(D=D, Mz_array=Mz_array)
    nv_center_set.setMagnetic(B_lab=B_lab)

    frequencies0 = nv_center_set.four_frequencies(np.array([2000, 3500]), nv_center_set.B_lab)

    A_inv = nv_center_set.calculateAinv(nv_center_set.B_lab)

    fr2_range = (dataframe_full['MW'] >= 2600) & (dataframe_full['MW'] <= 2640)
    fr4_range = (dataframe_full['MW'] >= 2870) & (dataframe_full['MW'] <= 2920)
    fr6_range = (dataframe_full['MW'] >= 3210) & (dataframe_full['MW'] <= 3250)
    fr8_range = (dataframe_full['MW'] >= 3350) & (dataframe_full['MW'] <= 3400)
    # print(fr2_range)

    fr_range_list = [fr2_range, fr4_range, fr6_range, fr8_range]

    for fr_range in fr_range_list:
        x_data = dataframe_full[fr_range]['MW']
        y_data = dataframe_full[fr_range]['ODMR']
        # peak_positions, peak_amplitudes = dp.detect_peaks_simple(x_data, y_data, height=0.1, debug=False)
        # peak_positions, peak_amplitudes = dp.detect_peaks_cwt(x_data, y_data, widthMHz=np.array([10]), min_snr=1, debug=False)
        peak, peak_amplitudes = dp.detect_peaks(x_data, y_data, debug=False)
        print(peak)
        # plt.plot(x_data, y_data, lw=2, c='k', label='fullscan')
        plt.scatter(peak, peak_amplitudes, c='k')

        # for peak in peak_positions:
        if 2610 <= peak <= 2630:
            fr2 = peak
        elif 2880 <= peak <=2900:
            fr4 = peak
        elif 3220 <= peak <= 3240:
            fr6 = peak
        elif 3370 <= peak <= 3390:
            fr8 = peak

    fr2_list = []
    fr4_list = []
    fr6_list = []
    fr8_list = []
    filedir = 'test_width_contrast_data'
    filenames = sorted(glob.glob(f'{filedir}/*.dat'))
    print(filenames)
    for filename in filenames:
        dataframe = dp.import_data(filename)
        print(dataframe.head())

        x_data = dataframe['MW']
        y_data = dataframe['ODMR']
        plt.plot(x_data, y_data)

        # peak_positions, peak_amplitudes = dp.detect_peaks_cwt(x_data, y_data, widthMHz=np.array([5]), min_snr=1,
        #                                                       debug=False)
        peak, peak_amplitudes = dp.detect_peaks(x_data, y_data, debug=False)
        plt.scatter(peak, peak_amplitudes)
        # for peak in peak_positions:
        if 2610 <= peak <= 2630:
            fr2_list.append(peak)
        elif 2880 <= peak <= 2900:
            fr4_list.append(peak)
        elif 3220 <= peak <= 3240:
            fr6_list.append(peak)
        elif 3370 <= peak <= 3390:
            fr8_list.append(peak)


    print('fr2:')
    print(fr2)
    print(fr2_list)
    fr2_diff = [fr - fr2 for fr in fr2_list]
    print(fr2_diff)
    fr2_diff_mean = np.mean(fr2_diff)
    print(fr2_diff_mean)

    print('fr4:')
    print(fr4)
    print(fr4_list)
    fr4_diff = [fr - fr4 for fr in fr4_list]
    print(fr4_diff)
    fr4_diff_mean = np.mean(fr4_diff)
    print(fr4_diff_mean)

    print('fr6:')
    print(fr6)
    print(fr6_list)
    fr6_diff = [fr - fr6 for fr in fr6_list]
    print(fr6_diff)
    fr6_diff_mean = np.mean(fr6_diff)
    print(fr6_diff_mean)

    print('fr8:')
    print(fr8)
    print(fr8_list)
    fr8_diff = [fr - fr8 for fr in fr8_list]
    print(fr8_diff)
    fr8_diff_mean = np.mean(fr8_diff)
    print(fr8_diff_mean)

    fr_full_list = np.array([fr2, fr4, fr6, fr8])

    fr_diff_list = np.array([fr2_diff_mean, fr4_diff_mean, fr6_diff_mean, fr8_diff_mean])
    Brot = utilities.deltaB_from_deltaFrequencies(A_inv, fr_diff_list)
    print(Brot)
    print(np.linalg.norm(Brot))

    rotation_angles = {"alpha": 1.9626607183487732, "phi": 20.789077311199208, "theta": 179.4794019370279}
    B_tot = B_lab + utilities.rotate(Brot, alpha=-rotation_angles['alpha'], phi=-rotation_angles['phi'], theta=-rotation_angles['theta'])

    # nv_center_set.setMagnetic(B_lab=B_tot)
    nv_fit = fit_odmr.NVsetForFitting()
    B_labx = B_tot[0]
    B_laby = B_tot[1]
    B_labz = B_tot[2]
    glor = parameters['glor']
    fraction = parameters['fraction']
    y_data2 = nv_fit.sum_odmr_voigt(x_data_full,
                                    B_labx=B_labx, B_laby=B_laby, B_labz=B_labz,
                                    glor=glor, D=parameters['D'],
                                    Mz1=parameters['Mz1'], Mz2=parameters['Mz2'], Mz3=parameters['Mz3'],
                                    Mz4=parameters['Mz4'], fraction=fraction)

    y_data2 = max(y_data_full) - (max(y_data_full) - min(y_data_full)) * y_data2
    dataframe_full['ODMR_calc2'] = y_data2

    plt.plot(x_data_full, y_data2, c='r')

    for fr_range in fr_range_list:
        x_data = dataframe_full[fr_range]['MW']
        y_data = dataframe_full[fr_range]['ODMR_calc2']
        # peak_positions, peak_amplitudes = dp.detect_peaks_simple(x_data, y_data, height=0.1, debug=False)
        # peak_positions, peak_amplitudes = dp.detect_peaks_cwt(x_data, y_data, widthMHz=np.array([10]), min_snr=1, debug=False)
        peak, peak_amplitudes = dp.detect_peaks(x_data, y_data, debug=False)
        # print(peak)
        # plt.plot(x_data, y_data, lw=2, c='k', label='fullscan')
        plt.scatter(peak, peak_amplitudes, c='k')

        # for peak in peak_positions:
        if 2610 <= peak <= 2630:
            fr2_2 = peak
        elif 2880 <= peak <=2900:
            fr4_2 = peak
        elif 3220 <= peak <= 3240:
            fr6_2 = peak
        elif 3370 <= peak <= 3390:
            fr8_2 = peak

    fr_list_2  = np.array([fr2_2, fr4_2, fr6_2, fr8_2])

    fr_list_2_diff = fr_list_2 - fr_full_list
    print(fr_diff_list)
    print(fr_list_2_diff)

    plt.legend()
    plt.show()
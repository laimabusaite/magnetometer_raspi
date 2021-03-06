
from fit_odmr import *


if __name__ == '__main__':

    import datetime

    D = 2851.26115
    Mz_array = np.array([7.32168327, 6.66104172, 9.68158138, 5.64605102])
    B_lab = np.array([191.945068, 100.386360, 45.6577322])
    # nv_center_set = nv.NVcenterSet(D=D, Mz_array=Mz_array)
    # nv_center_set.setMagnetic(B_lab=B_lab)

    nv_for_fit = NVsetForFitting()

    # nv_for_fit.setMagnetic()
    print(nv_for_fit.B_lab)

    # filename = 'data/test_2920_650_16dBm_1024_ODMR.dat'
    # filename = 'full_scan1.dat'
    # filename = 'full_scan2.dat'
    filename = 'full_scan_200.dat'
    dataframe = import_data(filename)
    print(dataframe)

    plt.plot(dataframe['MW'], dataframe['ODMR'])
    plt.show()

    # dataframe['ODMR_smooth'] = savgol_filter(dataframe['ODMR_norm'], 41, 2)
    dataframe['ODMR_base'] = savgol_filter(dataframe['ODMR_norm'], 95, 2)

    min_distance_min = len(dataframe) / (max(dataframe['MW']) - min(dataframe['MW'])) * 40
    # peaks_min, properties_min = find_peaks(dataframe['ODMR'], distance=min_distance)
    peaks_min, properties_min = find_peaks(-dataframe['ODMR_base'], distance=min_distance_min)
    # time1 = time.time()
    peak_positions_min = np.array(dataframe['MW'][peaks_min])
    # peak_amplitudes_min = np.array(dataframe['ODMR_norm'][peaks_min])
    peak_amplitudes_min = np.array(dataframe['ODMR_base'][peaks_min])

    interpolate_peaks_min = interpolate.interp1d(peak_positions_min, peak_amplitudes_min, kind='linear',
                                                 fill_value="extrapolate")  # (peak_amplitudes_min[0], peak_amplitudes_min[-1]))#"#extrapolate")
    wavelet_min = interpolate_peaks_min(dataframe['MW'])

    plt.plot(dataframe['MW'], -dataframe['ODMR_base'])
    plt.xlim(min(dataframe['MW']), max(dataframe['MW']))
    plt.xlabel('MW frequency, MHz')
    plt.ylabel('Fluorescence intensity, arb. units')
    plt.show()

    plt.plot(dataframe['MW'], dataframe['ODMR_norm'])
    plt.plot(dataframe['MW'], dataframe['ODMR_base'])
    plt.plot(dataframe['MW'], wavelet_min)
    plt.show()

    dataframe['ODMR_minus_base'] = (dataframe['ODMR_norm'] - wavelet_min)
    # plt.plot(dataframe['MW'], dataframe['ODMR_minus_base'])
    # plt.show()

    min_distance = len(dataframe) / (max(dataframe['MW']) - min(dataframe['MW'])) * 50
    height = 0.05  # (max(-dataframe['ODMR']) - min(-dataframe['ODMR'])) * 0.1
    min_width = len(dataframe) / (max(dataframe['MW']) - min(dataframe['MW'])) * 5
    max_width = len(dataframe) / (max(dataframe['MW']) - min(dataframe['MW'])) * 15
    # time0 = time.time()
    # peaks, properties = find_peaks(dataframe['ODMR_norm'], distance=min_distance, height=height)

    dataframe['ODMR_smooth'] = savgol_filter(dataframe['ODMR_minus_base'], 41, 2)

    peaks, properties = find_peaks(dataframe['ODMR_smooth'], distance=min_distance,
                                   height=height)  # , width=[min_width,max_width])
    # time1 = time.time()
    peak_positions = np.array(dataframe['MW'][peaks])
    # peak_amplitudes = np.array(dataframe['ODMR_norm'][peaks])
    peak_amplitudes = np.array(dataframe['ODMR_smooth'][peaks])
    print(peak_positions)

    interpolate_peaks = interpolate.interp1d(peak_positions, peak_amplitudes, kind='linear', fill_value="extrapolate")
    wavelet = interpolate_peaks(dataframe['MW'])

    plt.plot(dataframe['MW'], dataframe['ODMR_minus_base'])
    plt.plot(dataframe['MW'], dataframe['ODMR_smooth'])
    plt.plot(dataframe['MW'], wavelet)
    plt.show()

    # min_distance_min = 0 #len(dataframe) / (max(dataframe['MW']) - min(dataframe['MW'])) * 2#25
    # # peaks_min, properties_min = find_peaks(dataframe['ODMR'], distance=min_distance)
    # peaks_min, properties_min = find_peaks(-dataframe['ODMR_base'], distance=min_distance)
    # # time1 = time.time()
    # peak_positions_min = np.array(dataframe['MW'][peaks_min])
    # # peak_amplitudes_min = np.array(dataframe['ODMR_norm'][peaks_min])
    # peak_amplitudes_min = np.array(dataframe['ODMR_base'][peaks_min])
    #
    # interpolate_peaks_min = interpolate.interp1d(peak_positions_min, peak_amplitudes_min, kind='linear', fill_value="extrapolate")#(peak_amplitudes_min[0], peak_amplitudes_min[-1]))#"#extrapolate")
    # wavelet_min = interpolate_peaks_min(dataframe['MW'])

    # min_distance_min = len(dataframe) / (max(dataframe['MW']) - min(dataframe['MW'])) * 20
    # # peaks_min, properties_min = find_peaks(dataframe['ODMR'], distance=min_distance)
    # peaks_min, properties_min = find_peaks(-dataframe['ODMR_base'], distance=min_distance_min)
    # # time1 = time.time()
    # peak_positions_min = np.array(dataframe['MW'][peaks_min])
    # # peak_amplitudes_min = np.array(dataframe['ODMR_norm'][peaks_min])
    # peak_amplitudes_min = np.array(dataframe['ODMR_base'][peaks_min])
    #
    # interpolate_peaks_min = interpolate.interp1d(peak_positions_min, peak_amplitudes_min, kind='quadratic', fill_value="extrapolate")#(peak_amplitudes_min[0], peak_amplitudes_min[-1]))#"#extrapolate")
    # wavelet_min = interpolate_peaks_min(dataframe['MW'])
    #
    # dataframe['ODMR_minus_base'] = (dataframe['ODMR_norm'] - wavelet_min)
    # plt.plot(dataframe['MW'], dataframe['ODMR_minus_base'])

    # dataframe['ODMR_unitary'] = (dataframe['ODMR_norm']-wavelet_min)/(wavelet-wavelet_min)
    dataframe['ODMR_unitary'] = dataframe['ODMR_minus_base'] / wavelet
    plt.plot(dataframe['MW'], dataframe['ODMR_unitary'])
    dataframe['ODMR_unitary'] = savgol_filter(dataframe['ODMR_unitary'], 41, 2)
    plt.plot(dataframe['MW'], dataframe['ODMR_unitary'])
    # plt.show()

    # plt.plot(dataframe['MW'], dataframe['ODMR_norm'])
    #
    # plt.plot(dataframe['MW'], dataframe['ODMR_base'])
    # plt.plot(dataframe['MW'], wavelet)
    # plt.plot(dataframe['MW'], wavelet_min)
    # plt.show()

    D = 2870  # 2851.26115
    Mz_array = np.array([7.32168327, 6.66104172, 9.68158138, 5.64605102])

    # B_labx: 191.945068 + / - 0.06477634(0.03 %)(init=194.7307)
    # B_laby: 100.386360 + / - 0.04340693(0.04 %)(init=94.97649)
    # B_labz: 45.6577322 + / - 0.02524832(0.06 %)(init=38.2026)
    B_lab = np.array([191.945068, 100.386360, 45.6577322])  # + np.array([20, 0, 0])
    # B_labx: 169.118217 + / - 0.01067529(0.01 %)(init=177.0279)
    # B_laby: 87.7059717 + / - 0.01101454(0.01 %)(init=86.34226)
    # B_labz: 40.4265965 + / - 0.01059013(0.03 %)(init=34.72964)
    # glor: 4.44378131 + / - 0.01645117(0.37 %)(init=5)

    B_lab = np.array([169.118217, 87.7059717, 40.4265965])

    init_params = {'B_labx': 169.12151185371846, 'B_laby': 87.71808391787972, 'B_labz': 40.39868769723933,
                   'glor': 4.441357423298525, 'D': 2867.612730518484, 'Mz1': 0,
                   'Mz2': 0, 'Mz3': 0, 'Mz4': 0}

    print(init_params['B_labx'])

    omega = np.linspace(2000, 3800, 1800)
    # glor = 10
    # odmr_signal = nv_for_fit.sum_odmr(omega, B_lab[0], B_lab[1], B_lab[2], glor, D, Mz_array[0], Mz_array[1], Mz_array[2], Mz_array[3])
    # plt.plot(omega, odmr_signal)
    # odmr_signal_voigt = nv_for_fit.sum_odmr_voigt(omega, B_lab[0], B_lab[1], B_lab[2], glor, D, Mz_array[0], Mz_array[1],
    #                                   Mz_array[2], Mz_array[3], asym_coef=0.4, fraction=0.5)
    # plt.plot(omega, odmr_signal_voigt, c='k')
    #
    # omega = np.linspace(2000, 3800, 1800)
    init_B = 200  # 220
    init_theta = 80.0
    init_phi = 26  # 80 #26.9
    init_glor = 5  # 10.
    print(nv.CartesianVector(init_B, init_theta, init_phi), B_lab)
    init_Blab = B_lab  # nv.CartesianVector(init_B, init_theta, init_phi)
    summodel = Model(nv_for_fit.sum_odmr)
    y = np.array(dataframe['ODMR_unitary'])
    x = np.array(dataframe['MW'])
    params = summodel.make_params(B_labx=init_Blab[0], B_laby=init_Blab[1], B_labz=init_Blab[2], glor=10, D=D, Mz1=0,
                                  Mz2=0, Mz3=0, Mz4=0)
    params.update(summodel.make_params())
    params['B_labx'].set(init_Blab[0], min=0, max=400, vary=True)
    params['B_laby'].set(init_Blab[1], min=0, max=400, vary=True)
    params['B_labz'].set(init_Blab[2], min=0, max=400, vary=True)
    params['glor'].set(init_glor, min=0, max=50, vary=True)
    params['D'].set(D, min=2750, max=2900, vary=True)
    params['Mz1'].set(0, min=-10, max=10, vary=True)
    params['Mz2'].set(0, min=-10, max=10, vary=True)
    params['Mz3'].set(0, min=-10, max=10, vary=True)
    params['Mz4'].set(0, min=-10, max=10, vary=True)

    fitResult = summodel.fit(y, params, x=x)

    print(fitResult.best_values)

    # plt.plot(df_crop['MW'], df_crop['ODMR_norm'], color='k', markersize=5, marker='o', linewidth=1)
    # plt.plot(df_crop['MW'][peaks], df_crop['ODMR_norm'][peaks], "x", label='exp peaks')
    plt.plot(x, summodel.eval(params, x=x), 'g--')
    plt.plot(x, fitResult.best_fit, 'g-')
    save_filename = f"ODMR_fit_parameters{time.time()}.json" #f"ODMR_fit_parameters{datetime.datetime.now().isoformat()}.json" #'ODMR_fit_parameters.json' #
    print(save_filename)
    nv_for_fit.fit_odmr_lorentz(x, y, init_params, varyB=True, varyGlor=False, varyD=False,
                     varyMz=False, save_filename=save_filename)
    print(nv_for_fit.fitResultLorentz.fit_report())
    print(nv_for_fit.fitResultLorentz.best_values)
    plt.plot(x, nv_for_fit.summodel.eval(nv_for_fit.params, x=x), 'r--')
    plt.plot(x, nv_for_fit.fitResultLorentz.best_fit, 'r-')

    # read parameters
    filename = save_filename #"ODMR_fit_parameters.json"
    a_file = open(filename, "r")
    init_params = json.load(a_file)
    # parameters = dict(parameters)
    print('Second iteration: vary all \n init:')
    print(init_params)

    nv_for_fit.fit_odmr_lorentz(x, y, init_params, varyB=True, varyGlor=True, varyD=True,
                                varyMz=True, save_filename = save_filename)
    print(nv_for_fit.fitResultLorentz.fit_report())
    print(nv_for_fit.fitResultLorentz.best_values)
    plt.plot(x, nv_for_fit.summodel.eval(nv_for_fit.params, x=x), 'k--')
    plt.plot(x, nv_for_fit.fitResultLorentz.best_fit, 'k-')



    ### fit voigt
    # omega = np.linspace(2000, 3800, 1800)
    #
    # summodel_voigt = Model(nv_for_fit.sum_odmr_voigt)
    # y = np.array(dataframe['ODMR_unitary'])
    # x = np.array(dataframe['MW'])
    # params = summodel_voigt.make_params(B_labx=init_Blab[0], B_laby=init_Blab[1], B_labz=init_Blab[2], glor=10, D=D, Mz1=0,
    #                               Mz2=0, Mz3=0, Mz4=0, asym_coef=0.4, fraction=0.7)
    # params.update(summodel_voigt.make_params())
    # params['B_labx'].set(init_Blab[0], min=0, max=400, vary=True)
    # params['B_laby'].set(init_Blab[1], min=0, max=400, vary=True)
    # params['B_labz'].set(init_Blab[2], min=0, max=400, vary=True)
    # params['glor'].set(init_glor, min=0, max=50, vary=False)
    # params['D'].set(D, min=2750, max=2900, vary=False)
    # params['Mz1'].set(0, min=-10, max=10, vary=False)
    # params['Mz2'].set(0, min=-10, max=10, vary=False)
    # params['Mz3'].set(0, min=-10, max=10, vary=False)
    # params['Mz4'].set(0, min=-10, max=10, vary=False)
    # params['asym_coef'].set(0.0, min=0.0, max=1.0, vary=False)
    # params['fraction'].set(0.5, min=0.0, max=1.0, vary=False)
    #
    # fitResult_voigt = summodel_voigt.fit(y, params, x=x)
    #
    # print(fitResult_voigt.fit_report())
    #
    # # plt.plot(df_crop['MW'], df_crop['ODMR_norm'], color='k', markersize=5, marker='o', linewidth=1)
    # # plt.plot(df_crop['MW'][peaks], df_crop['ODMR_norm'][peaks], "x", label='exp peaks')
    # plt.plot(x, summodel_voigt.eval(params, x=x), 'b--')
    # plt.plot(x, fitResult_voigt.best_fit, 'b-')

    plt.show()

    # x = np.linspace(-100,100,1000)
    # # asym = asymetrical_voigt(x, 10, 1, 5, -0.4, 0.5)
    # # plt.plot(x, asym)
    # #
    # asym_curve = asymetrical_voigt_curve(x, [0, 15], [1, 1], 1, 0.1, 0.5)
    # plt.plot(x, asym_curve)
    #
    # # for x0 in [-5, 0, 5, 20]:
    # #
    # #     asym = asymetrical_voigt(x, x0, 1, 5, 0.4, 0.5)
    # #     plt.plot(x, asym)
    #
    #
    # plt.show()

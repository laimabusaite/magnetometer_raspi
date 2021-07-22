import numpy as np
import NVcenter as nv
import matplotlib.pyplot as plt
from detect_peaks import *
from lmfit.models import Model
from scipy import interpolate
from scipy.signal import find_peaks, savgol_filter
from utilities import *
import json
from scipy.optimize import curve_fit

from scipy.special import wofz


class NVsetForFitting(nv.NVcenterSet):
    # def __init__(self, D=2870, Mz_array=np.array([0, 0, 0, 0])):
    #     super().__init__(self, D=2870, Mz_array=np.array([0, 0, 0, 0]))

    def sum_odmr_voigt(self, x, B_labx, B_laby, B_labz, glor, D, Mz1, Mz2, Mz3, Mz4, fraction):
        '''
        Calculate Voigt profile
        Parameters
        ----------
        x
        B_labx
        B_laby
        B_labz
        glor
        D
        Mz1
        Mz2
        Mz3
        Mz4
        fraction

        Returns
        -------
        sum_odmr
        '''
        asym_coef = 0
        # fraction = 0.9
        B_lab = np.array([B_labx, B_laby, B_labz])
        Mz_array = np.array([Mz1, Mz2, Mz3, Mz4])
        B = np.linalg.norm(B_lab)
        ODMR = np.empty(4, dtype='object')
        for m in range(4):
            cos = np.dot(B_lab, self.rNV[m]) / (np.linalg.norm(B_lab) * np.linalg.norm(self.rNV[m]))
            if cos >= 1.0:
                cos = 1.0
            thetaBnv = np.arccos(cos) * 180 / np.pi
            phiBnv = 0
            Bcarnv = nv.CartesianVector(B, thetaBnv, phiBnv)
            self.nvlist[m].setNVparameters(D=D, Mz=Mz_array[m])
            self.nvlist[m].setMagnetic(Bx=Bcarnv[0], By=Bcarnv[1], Bz=Bcarnv[2])
            # ODMR[m] = self.nvlist[m].nv_lorentz(x, glor)
            ODMR[m] = self.nvlist[m].nv_pseudo_voigt(x, glor, asym_coef=asym_coef, fraction=fraction)

        sum_odmr = ODMR[0] + ODMR[1] + ODMR[2] + ODMR[3]
        sum_odmr /= max(sum_odmr)

        return sum_odmr

    def sum_odmr(self, x, B_labx, B_laby, B_labz, glor, D, Mz1, Mz2, Mz3, Mz4):
        '''
        Clculate Lorenzian profile
        Parameters
        ----------
        x
        B_labx
        B_laby
        B_labz
        glor
        D
        Mz1
        Mz2
        Mz3
        Mz4

        Returns
        -------
        sum_odmr
        '''
        B_lab = np.array([B_labx, B_laby, B_labz])
        Mz_array = np.array([Mz1, Mz2, Mz3, Mz4])
        B = np.linalg.norm(B_lab)
        ODMR = np.empty(4, dtype='object')
        for m in range(4):
            cos = np.dot(B_lab, self.rNV[m]) / (np.linalg.norm(B_lab) * np.linalg.norm(self.rNV[m]))
            if cos >= 1.0:
                cos = 1.0
            thetaBnv = np.arccos(cos) * 180 / np.pi
            phiBnv = 0
            Bcarnv = nv.CartesianVector(B, thetaBnv, phiBnv)
            self.nvlist[m].setNVparameters(D=D, Mz=Mz_array[m])
            self.nvlist[m].setMagnetic(Bx=Bcarnv[0], By=Bcarnv[1], Bz=Bcarnv[2])
            ODMR[m] = self.nvlist[m].nv_lorentz(x, glor)

        sum_odmr = ODMR[0] + ODMR[1] + ODMR[2] + ODMR[3]
        sum_odmr /= max(sum_odmr)

        return sum_odmr

    def fit_odmr_lorentz(self, x, y, init_params, varyB=True, varyGlor=True, varyD=True,
                         varyMz=False, save=True, save_filename='ODMR_fit_parameters.json'):
        '''
        Parameters
        ----------
        x
        y
        init_params
        varyB
        varyGlor
        varyD
        varyMz
        save
        save_filename

        '''
        self.summodel = Model(self.sum_odmr)
        # self.summodel = Model(self.sum_odmr_voigt)
        params = self.summodel.make_params(B_labx=init_params['B_labx'], B_laby=init_params['B_laby'],
                                           B_labz=init_params['B_labz'], glor=init_params['glor'],
                                           D=init_params['D'],
                                           Mz1=init_params['Mz1'], Mz2=init_params['Mz2'], Mz3=init_params['Mz3'],
                                           Mz4=init_params['Mz4'])
        params.update(self.summodel.make_params())
        params['B_labx'].set(init_params['B_labx'], min=0, max=400, vary=varyB)
        params['B_laby'].set(init_params['B_laby'], min=0, max=400, vary=varyB)
        params['B_labz'].set(init_params['B_labz'], min=0, max=400, vary=varyB)
        params['glor'].set(init_params['glor'], min=0, max=50, vary=varyGlor)
        params['D'].set(init_params['D'], min=2750, max=2900, vary=varyD)
        params['Mz1'].set(init_params['Mz1'], min=-10, max=10, vary=varyMz)
        params['Mz2'].set(init_params['Mz2'], min=-10, max=10, vary=varyMz)
        params['Mz3'].set(init_params['Mz3'], min=-10, max=10, vary=varyMz)
        params['Mz4'].set(init_params['Mz4'], min=-10, max=10, vary=varyMz)

        self.params = params

        self.fitResultLorentz = self.summodel.fit(y, self.params, x=x)

        if save:
            self.save_parameters(self.fitResultLorentz.best_values, save_filename)

    def fit_odmr_voigt(self, x, y, init_params, varyB=True, varyGlor=True, varyD=True,
                         varyMz=False, varyFraction=False, save=True, save_filename='ODMR_fit_parameters.json'):
        '''

        Parameters
        ----------
        x
        y
        init_params
        varyB
        varyGlor
        varyD
        varyMz
        varyFraction
        save
        save_filename

        Returns
        -------

        '''
        # self.summodel = Model(self.sum_odmr)
        self.summodel = Model(self.sum_odmr_voigt)
        params = self.summodel.make_params(B_labx=init_params['B_labx'], B_laby=init_params['B_laby'],
                                           B_labz=init_params['B_labz'], glor=init_params['glor'],
                                           D=init_params['D'],
                                           Mz1=init_params['Mz1'], Mz2=init_params['Mz2'], Mz3=init_params['Mz3'],
                                           Mz4=init_params['Mz4'],
                                           fraction = 0.9)
        params.update(self.summodel.make_params())
        params['B_labx'].set(init_params['B_labx'], min=0, max=400, vary=varyB)
        params['B_laby'].set(init_params['B_laby'], min=0, max=400, vary=varyB)
        params['B_labz'].set(init_params['B_labz'], min=0, max=400, vary=varyB)
        params['glor'].set(init_params['glor'], min=0, max=50, vary=varyGlor)
        params['D'].set(init_params['D'], min=2750, max=2900, vary=varyD)
        params['Mz1'].set(init_params['Mz1'], min=-10, max=10, vary=varyMz)
        params['Mz2'].set(init_params['Mz2'], min=-10, max=10, vary=varyMz)
        params['Mz3'].set(init_params['Mz3'], min=-10, max=10, vary=varyMz)
        params['Mz4'].set(init_params['Mz4'], min=-10, max=10, vary=varyMz)
        params['fraction'].set(0.9, min=0, max=1, vary=varyFraction)

        self.params = params

        self.fitResultLorentz = self.summodel.fit(y, self.params, x=x)

        if save:
            self.save_parameters(self.fitResultLorentz.best_values, save_filename)

    def save_parameters(self, dictionary_data, filename):
        '''
        Save dictionary data to json
        Parameters
        ----------
        dictionary_data
        filename
        '''
        a_file = open(filename, "w")
        json.dump(dictionary_data, a_file)
        a_file.close()



def normalize_data(x_data, y_data, debug=False):
    '''
    Normalize ODMR data in range 0 to 1 for all peaks
    Parameters
    ----------
    x_data : array like, microwave frequency data
    y_data : array like, fluoescence data
    debug : boolean, plots data for debuging

    Returns
    -------
    y_unitary : normalized fluorescence
    '''
    # normalized odmr
    y_norm = 1. - (y_data - min(y_data)) / (max(y_data) - min(y_data))
    # y_base = y_norm
    step_size = (max(x_data) - min(x_data)) / len(x_data)
    steps_per_mhz = int(len(x_data) / (max(x_data) - min(x_data)))
    print('step size', step_size, steps_per_mhz)
    y_base = savgol_filter(y_norm, 12*steps_per_mhz+1, 2)

    min_distance = steps_per_mhz * 50
    height = 0.1  # 0.055  # (max(-dataframe['ODMR']) - min(-dataframe['ODMR'])) * 0.1
    peaks00, properties00 = find_peaks(y_base, distance=min_distance,
                                   height=height)  # , width=[min_width,max_width])
    peak_positions00 = np.array(x_data[peaks00])
    peak_amplitudes00 = np.array(y_base[peaks00])
    # print(peaks00)
    # print(int(peaks00[0] - steps_per_mhz * 10))
    # print(y_base[:int(peaks00[0] - steps_per_mhz * 20)])
    # print(y_base[int(peaks00[0] + steps_per_mhz * 20): int(peaks00[1] - steps_per_mhz * 20)])
    base_array_selected = np.concatenate((
        y_base[:int(peaks00[0] - steps_per_mhz * 20)],
        y_base[int(peaks00[0] + steps_per_mhz * 20): int(peaks00[1] - steps_per_mhz * 20)],
        y_base[int(peaks00[1] + steps_per_mhz * 20): int(peaks00[2] - steps_per_mhz * 20)],
        y_base[int(peaks00[2] + steps_per_mhz * 20): int(peaks00[3] - steps_per_mhz * 20)],
        y_base[int(peaks00[3] + steps_per_mhz * 20): int(peaks00[4] - steps_per_mhz * 20)],
        y_base[int(peaks00[4] + steps_per_mhz * 20): int(peaks00[5] - steps_per_mhz * 20)],
        y_base[int(peaks00[5] + steps_per_mhz * 20): int(peaks00[6] - steps_per_mhz * 20)],
        y_base[int(peaks00[6] + steps_per_mhz * 20): int(peaks00[7] - steps_per_mhz * 20)],
        y_base[int(peaks00[7] + steps_per_mhz * 20):],
                        ))
    base_mean = np.mean(base_array_selected)

    min_distance_min = steps_per_mhz * 40
    # print('mean base',base_mean, np.mean((1-y_base)**5*y_base)/np.mean((1-y_base)**5))
    peaks_min, properties_min = find_peaks(-y_base, distance=min_distance_min)
    peak_positions_min = np.array(x_data[peaks_min])
    # peak_amplitudes_min = np.array(y_base[peaks_min])
    peak_amplitudes_min = np.full_like(peak_positions_min, base_mean)

    interpolate_peaks_min = interpolate.interp1d(peak_positions_min, peak_amplitudes_min, kind='linear',
                                                 fill_value="extrapolate")  # (peak_amplitudes_min[0], peak_amplitudes_min[-1]))#"#extrapolate")
    wavelet_min = interpolate_peaks_min(x_data)

    # y_smooth = y_norm - wavelet_min
    y_smooth = savgol_filter(y_norm - wavelet_min, 10*steps_per_mhz+1, 2)

    min_distance = steps_per_mhz * 50
    height = 0.1 # 0.055  # (max(-dataframe['ODMR']) - min(-dataframe['ODMR'])) * 0.1
    min_width = steps_per_mhz * 5
    max_width = steps_per_mhz * 15
    peaks, properties = find_peaks(y_smooth, distance=min_distance,
                                   height=height)  # , width=[min_width,max_width])
    # peakso, prperties = find_peaks(y_smooth, prominence=0.004, width=[5, 25])

    peak_positions = np.array(x_data[peaks])
    peak_amplitudes = np.array(y_smooth[peaks])

    interpolate_peaks = interpolate.interp1d(peak_positions, peak_amplitudes, kind='linear', fill_value="extrapolate")
    wavelet = interpolate_peaks(x_data)

    y_unitary = y_smooth / wavelet

    if debug:
        print('normalize_data debug')
        print(peaks)
        print(peak_positions)
        print(peak_amplitudes)

        plt.figure(1, figsize=(5,4))
        plt.plot(x_data, y_norm, c='k', label='measured signal')
        plt.plot(x_data, y_base, label='smoothed data')
        plt.plot(x_data, wavelet_min, label='base')
        plt.plot(peak_positions00, peak_amplitudes00, label='peaks')
        plt.xlim(x_data[0], x_data[-1])
        plt.xlabel('Microwave frequency (MHz)')
        plt.ylabel('Normalized fluorescence intensity (arb. units)')
        plt.legend()
        # plt.savefig('/home/laima/Dropbox/Apps/Overleaf/ESA D10 - Software  Design Document/fit_odmr/smooth1.pdf')
        # plt.savefig('/home/laima/Dropbox/Apps/Overleaf/ESA D10 - Software  Design Document/fit_odmr/smooth1.png')

        plt.figure(2, figsize=(5,4))
        plt.plot(x_data, y_norm - wavelet_min)
        plt.plot(x_data, y_smooth)
        plt.plot(x_data, wavelet)
        plt.xlim(x_data[0], x_data[-1])
        plt.xlabel('Microwave frequency (MHz)')
        plt.ylabel('Normalized fluorescence intensity (arb. units)')
        # plt.savefig('/home/laima/Dropbox/Apps/Overleaf/ESA D10 - Software  Design Document/fit_odmr/smooth2.pdf')
        # plt.savefig('/home/laima/Dropbox/Apps/Overleaf/ESA D10 - Software  Design Document/fit_odmr/smooth2.png')

        plt.figure('unitary', figsize=(5,4))
        plt.plot(x_data, (y_norm - wavelet_min) / wavelet , c='k', label='normalized data')
        plt.plot(x_data, y_unitary, c='gray', label='smoothed data')
        plt.xlim(x_data[0], x_data[-1])

        # plt.show()

    return y_unitary


def fit_full_odmr(x_data, y_data,
                  init_params={'B_labx': 169.12, 'B_laby': 87.71, 'B_labz': 40.39, 'glor': 4.44, 'D': 2870, 'Mz1': 0,
                               'Mz2': 0, 'Mz3': 0, 'Mz4': 0}, save=True,
                  save_filename="ODMR_fit_parameters.json", debug=False):
    '''

    Parameters
    ----------
    x_data : array like, microwave frequency data
    y_data : array like, fluoescence data
    init_params : dictionary of inital parameters
    save : boolean, whether to save parametrs to json
    save_filename : string, json filename to store fitting parameters
    debug : boolean, plots data for debuging

    Returns
    -------
    fit_parameters : dictionary of fitted paramters

    '''
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # normalize odmr
    y_unitary = normalize_data(x_data, y_data, debug=debug)
    min_distance = len(x_data) / (max(x_data) - min(x_data)) * 50
    height = 0.5
    peaks_exp, properties_exp = find_peaks(y_unitary, distance=min_distance,
                                   height=height)

    # define NV center:
    nv_for_fit = NVsetForFitting()

    # create fitting model
    print('Fitting')
    # print('First iteration: vary B \n init:')
    init_params0 = init_params
    print(init_params0)


    # TODO fit voigt
    nv_for_fit.fit_odmr_voigt(x_data, y_unitary, init_params0, varyB=True, varyGlor=False, varyD=False,
                                varyMz=False, varyFraction=False, save=False)
    if debug:
        # print('nv_for_fit.fitResultLorentz.fit_report()')
        # print(nv_for_fit.fitResultLorentz.fit_report())
        # print('nv_for_fit.fitResultLorentz.best_values')
        # print(nv_for_fit.fitResultLorentz.best_values)
        plt.figure('unitary', figsize=(5,4))
        # plt.plot(x_data, y_unitary, c='k')
        plt.plot(x_data, nv_for_fit.summodel.eval(nv_for_fit.params, x=x_data), 'g--', label='initial conditions')
        plt.plot(x_data, nv_for_fit.fitResultLorentz.best_fit, 'g-')
        plt.xlim(x_data[0], x_data[-1])
        # plt.xlabel('Microwave frequency (MHz)')
        # plt.ylabel('Normalized fluorescence intensity (arb. units)')
        # plt.savefig('/home/laima/Dropbox/Apps/Overleaf/ESA D10 - Software  Design Document/fit_odmr/fitted.pdf')
        # plt.savefig('/home/laima/Dropbox/Apps/Overleaf/ESA D10 - Software  Design Document/fit_odmr/fitted.png')
        # plt.show()

    init_params = nv_for_fit.fitResultLorentz.best_values
    print('Second iteration: vary all \n init:')
    print(init_params)
    # nv_for_fit.fit_odmr_voigt(x_data, y_unitary, init_params, varyB=True, varyGlor=True, varyD=False,
    #                           varyMz=True, varyFraction=True, save_filename=save_filename, save=save)
    nv_for_fit.fit_odmr_voigt(x_data, y_unitary, init_params, varyB=True, varyGlor=True, varyD=True,
                              varyMz=False, varyFraction=True, save_filename=save_filename, save=save)
    print('Result voigt')
    print(nv_for_fit.fitResultLorentz.best_values)
    steps_per_mhz = int(len(x_data) / (max(x_data) - min(x_data)))
    min_distance = steps_per_mhz * 50
    height = 0.5
    peaks_fit, properties_fit = find_peaks(nv_for_fit.fitResultLorentz.best_fit, distance=min_distance,
                                           height=height)
    print(peaks_exp, peaks_fit)
    peak_diff = peaks_exp - peaks_fit
    print('peak diff idx', peak_diff)
    peak_diff_mhz = peak_diff / steps_per_mhz
    print('peak diff mhz', peak_diff_mhz)
    if (np.abs(peak_diff_mhz) > 2).any():
        print('Optimum values not found. Adjust initial conditions.')

    if debug:
        # print('nv_for_fit.fitResultLorentz.fit_report()')
        # print(nv_for_fit.fitResultLorentz.fit_report())
        # print('nv_for_fit.fitResultLorentz.best_values')
        # print(nv_for_fit.fitResultLorentz.best_values)
        plt.plot(x_data, nv_for_fit.summodel.eval(nv_for_fit.params, x=x_data), 'b--')
        plt.figure('unitary', figsize=(5, 4))
        plt.plot(x_data, nv_for_fit.fitResultLorentz.best_fit, 'b-', label='fit')
        plt.xlim(x_data[0], x_data[-1])
        plt.xlabel('Microwave frequency (MHz)')
        plt.ylabel('Normalized fluorescence intensity (arb. units)')
        plt.legend()
        # plt.savefig('/home/laima/Dropbox/Apps/Overleaf/ESA D10 - Software  Design Document/fit_odmr/fitted2.pdf')
        # plt.savefig('/home/laima/Dropbox/Apps/Overleaf/ESA D10 - Software  Design Document/fit_odmr/fitted2.png')
        plt.show()

    fit_parameters = nv_for_fit.fitResultLorentz.best_values

    return fit_parameters


def get_initial_magnetic_field(x_data, y_data):
    B_lab = np.array([179., 72., 36.])

    peaks0 = np.array([2458., 2620., 2766., 2893., 3141., 3231., 3310., 3379.])
    nv_center_set = nv.NVcenterSet()
    nv_center_set.setMagnetic(B_lab=B_lab)
    A_inv = nv_center_set.calculateAinv(B_lab)

    peaks, _ = detect_peaks_simple(x_data, y_data, height=0.1, debug=False)

    delta_freqs = (peaks0 - peaks)[1::2]

    Binit = B_lab + deltaB_from_deltaFrequencies(A_inv, delta_freqs)

    return Binit


if __name__ == '__main__':
    import glob

    filedir = 'RQnc/steps/0G'
    filenames = glob.glob(f'{filedir}/full_scan*.dat')

    print(filenames)
    filename = filenames[0]
    for filename in filenames[:]:
        # filename = 'full_scan_rp_1.dat'

        print(filename)
        dataframe = import_data(filename)
        # print(dataframe)

        x_data = dataframe['MW']
        y_data = dataframe['ODMR']

        B = 200
        theta = 80
        phi = 20
        Blab = CartesianVector(B, theta, phi)
        # Blab = get_initial_magnetic_field(x_data, y_data) #CartesianVector(B, theta, phi)
        print(Blab)

        B = np.linalg.norm(Blab)
        print(B)

        init_params = {'B_labx': Blab[0], 'B_laby': Blab[1], 'B_labz': Blab[2],
                       'glor': 5, 'D': 2870, 'Mz1': 0,
                       'Mz2': 0, 'Mz3': 0, 'Mz4': 0}
        save_filename = "ODMR_fit_parameters.json"
        parameters = fit_full_odmr(x_data, y_data, init_params=init_params, save_filename=save_filename, debug=True,
                                   save=False)
        print(parameters)

    # plt.show()
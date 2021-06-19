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

    def sum_odmr_voigt(self, x, B_labx, B_laby, B_labz, glor, D, Mz1, Mz2, Mz3, Mz4, asym_coef, fraction):
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
            ODMR[m] = self.nvlist[m].nv_pseudo_voigt(x, glor, asym_coef, fraction)

        sum_odmr = ODMR[0] + ODMR[1] + ODMR[2] + ODMR[3]
        sum_odmr /= max(sum_odmr)

        return sum_odmr

    def sum_odmr(self, x, B_labx, B_laby, B_labz, glor, D, Mz1, Mz2, Mz3, Mz4):
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

        self.summodel = Model(self.sum_odmr)
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

    def save_parameters(self, dictionary_data, filename):

        a_file = open(filename, "w")
        json.dump(dictionary_data, a_file)
        a_file.close()
        #
        # a_file = open(filename, "r")
        # output = a_file.read()
        # print(output)
        # # OUTPUT
        # # {"a": 1, "b": 2}
        # a_file.close()


def normalize_data(x_data, y_data, debug=False):
    # normalized odmr
    y_norm = 1. - (y_data - min(y_data)) / (max(y_data) - min(y_data))
    # y_base = y_norm
    y_base = savgol_filter(y_norm, 11, 2)

    min_distance_min = len(x_data) / (max(x_data) - min(x_data)) * 40
    peaks_min, properties_min = find_peaks(-y_base, distance=min_distance_min)
    peak_positions_min = np.array(x_data[peaks_min])
    peak_amplitudes_min = np.array(y_base[peaks_min])

    interpolate_peaks_min = interpolate.interp1d(peak_positions_min, peak_amplitudes_min, kind='linear',
                                                 fill_value="extrapolate")  # (peak_amplitudes_min[0], peak_amplitudes_min[-1]))#"#extrapolate")
    wavelet_min = interpolate_peaks_min(x_data)

    # y_smooth = y_norm - wavelet_min
    y_smooth = savgol_filter(y_norm - wavelet_min, 11, 2)

    min_distance = len(x_data) / (max(x_data) - min(x_data)) * 50
    height = 0.05  # (max(-dataframe['ODMR']) - min(-dataframe['ODMR'])) * 0.1
    min_width = len(x_data) / (max(x_data) - min(x_data)) * 5
    max_width = len(x_data) / (max(x_data) - min(x_data)) * 15
    peaks, properties = find_peaks(y_smooth, distance=min_distance,
                                   height=height)  # , width=[min_width,max_width])

    peak_positions = np.array(x_data[peaks])
    peak_amplitudes = np.array(y_smooth[peaks])

    interpolate_peaks = interpolate.interp1d(peak_positions, peak_amplitudes, kind='linear', fill_value="extrapolate")
    wavelet = interpolate_peaks(x_data)

    y_unitary = y_smooth / wavelet

    if debug:
        print(peaks)
        print(peak_positions)
        print(peak_amplitudes)

        plt.figure(1)
        plt.plot(x_data, y_norm)
        plt.plot(x_data, y_base)
        plt.plot(x_data, wavelet_min)

        plt.figure(2)
        plt.plot(x_data, y_norm-wavelet_min)
        plt.plot(x_data, y_smooth)
        plt.plot(x_data, wavelet)

        plt.figure(3)
        plt.plot(x_data, y_unitary)

        # plt.show()

    return y_unitary


def fit_full_odmr(x_data, y_data,
                  init_params={'B_labx': 169.12, 'B_laby': 87.71, 'B_labz': 40.39, 'glor': 4.44, 'D': 2867.61, 'Mz1': 0,
                               'Mz2': 0, 'Mz3': 0, 'Mz4': 0}, save=True,
                  save_filename="ODMR_fit_parameters.json", debug=False):
    '''
    x_data and y_data:  array like
    init_params: dictionary of inital parameters default:
    init_params = {'B_labx': 169.12, 'B_laby': 87.71, 'B_labz': 40.39,
                    'glor': 4.44, 'D': 2867.61,
                    'Mz1': 0, 'Mz2': 0, 'Mz3': 0, 'Mz4': 0}
    save_filename: string, json filename to store fitting parameters default:
    save_filename="ODMR_fit_parameters.json"
    debug: Boolen prints fit results and plots data default:
    debug=False
    '''
    x_data = np.array(x_data)
    y_data = np.array(y_data)

    # normalize odmr
    y_unitary = normalize_data(x_data, y_data, debug=debug)

    # define NV center:
    nv_for_fit = NVsetForFitting()

    # create fitting model
    print('Fitting')
    print('First iteration: vary B \n init:')
    print(init_params)
    nv_for_fit.fit_odmr_lorentz(x_data, y_unitary, init_params, varyB=True, varyGlor=False, varyD=False,
                                varyMz=False, save = False)
    if debug:
        print(nv_for_fit.fitResultLorentz.fit_report())
        print(nv_for_fit.fitResultLorentz.best_values)
        plt.figure()
        plt.plot(x_data, y_unitary)
        plt.plot(x_data, nv_for_fit.summodel.eval(nv_for_fit.params, x=x_data), 'r--')
        plt.plot(x_data, nv_for_fit.fitResultLorentz.best_fit, 'r-')
        # plt.show()

    # read parameters
    # filename = save_filename  # "ODMR_fit_parameters.json"
    # a_file = open(filename, "r")
    init_params = nv_for_fit.fitResultLorentz.best_values #json.load(a_file)
    # parameters = dict(parameters)
    print('Second iteration: vary all \n init:')
    print(init_params)

    nv_for_fit.fit_odmr_lorentz(x_data, y_unitary, init_params, varyB=True, varyGlor=True, varyD=False,
                                varyMz=True, save_filename=save_filename, save=save)
    print('Result')
    print(nv_for_fit.fitResultLorentz.best_values)
    if debug:
        print(nv_for_fit.fitResultLorentz.fit_report())
        print(nv_for_fit.fitResultLorentz.best_values)
        plt.plot(x_data, nv_for_fit.summodel.eval(nv_for_fit.params, x=x_data), 'k--')
        plt.plot(x_data, nv_for_fit.fitResultLorentz.best_fit, 'k-')
        plt.show()

    return nv_for_fit.fitResultLorentz.best_values


if __name__ == '__main__':
    filename = 'full_scan_rp_1.dat'
    dataframe = import_data(filename)
    print(dataframe)

    x_data = dataframe['MW']
    y_data = dataframe['ODMR']

    B = 210
    theta = 80
    phi = 20
    Blab = CartesianVector(B, theta, phi)
    print(Blab)
    init_params = {'B_labx': Blab[0], 'B_laby': Blab[1], 'B_labz': Blab[2],
                   'glor': 4.44, 'D': 2867.61, 'Mz1': 0,
                   'Mz2': 0, 'Mz3': 0, 'Mz4': 0}
    save_filename = "ODMR_fit_parameters.json"
    fit_full_odmr(x_data, y_data, init_params=init_params, save_filename=save_filename, debug=True)

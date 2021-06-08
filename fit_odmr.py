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
                         varyMz=False, save_filename='ODMR_fit_parameters.json'):

        self.summodel = Model(self.sum_odmr)
        params = self.summodel.make_params(B_labx=init_params['B_labx'], B_laby=init_params['B_laby'], B_labz=init_params['B_labz'], glor=init_params['glor'],
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


def fit_full_odmr(x_data, y_data, debug=False):

    #normalized odmr
    y_norm = 1. - (y_data-min(y_data))/(max(y_data)-min(y_data))

    #define NV center:
    nv_for_fit = NVsetForFitting()

    if debug:
        plt.plot(x_data, y_data)
        plt.show()
        plt.plot(x_data, y_norm)
        plt.show()


if __name__ == '__main__':
    filename = 'full_scan_200.dat'
    dataframe = import_data(filename)
    print(dataframe)

    x_data = dataframe['MW']
    y_data = dataframe['ODMR']

    fit_full_odmr(x_data, y_data, debug=True)
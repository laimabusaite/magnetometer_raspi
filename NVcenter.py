import numpy as np
import math
from utilities import *


class NVcenter(object):
    def __init__(self,D=2870, Mz=0):
        super(NVcenter, self).__init__()
        self.muB = 1.3996245042  # 1.401 #MHz/G
        self.g_el = 2.00231930436182
        self.Sx = 1 / np.sqrt(2) * np.array([[0, 1, 0],
                                             [1, 0, 1],
                                             [0, 1, 0]], dtype=complex)
        self.Sy = 1.j / np.sqrt(2) * np.array([[0., -1., 0.],
                                               [1., 0., -1.],
                                               [0., 1., 0.]], dtype=complex)
        self.Sz = np.array([[1, 0, 0],
                            [0, 0, 0],
                            [0, 0, -1]], dtype=complex)
        self.setNVparameters(D=D, Mz=Mz)
        self.setMagnetic()

    def setNVparameters(self, D=2870, Mz=0):
        self.D = D
        self.Mz = Mz
        self.dim = 3  # number of states
        self.DSmat = (self.D + self.Mz) * np.dot(self.Sz, self.Sz)

    def setMagnetic(self, Bx=0, By=0, Bz=0):
        """Set magnetic field in Cartesian coordinated Bx, By, Bz"""
        self.Bx = Bx  # MHz (magnetic field interaction along x)
        self.By = By  # MHz (magnetic field interaction along y)
        self.Bz = Bz  # MHz (magnetic field interaction along z)
        self.BSmat = self.g_el * self.muB * (self.Bx * self.Sx + self.By * self.Sy + self.Bz * self.Sz)

    def calculateHamiltonian(self):  # Compute new eigenstates and respective energies
        self.H = self.DSmat + self.BSmat
        self.eVals, self.eVecs = np.linalg.eigh(self.H, UPLO='U')

    def calculateInteraction(self):
        self.MWham = np.dot(np.abs(self.eVecs.transpose().conjugate()), np.dot(self.Sx, np.abs(self.eVecs)))

    def calculatePeaks(self, omega):
        self.calculateHamiltonian()
        self.calculateInteraction()
        frequencies = np.tile(self.eVals, self.dim).reshape(self.dim, -1) - np.tile(self.eVals, self.dim).reshape(
            self.dim, -1).transpose()
        frequencies = frequencies.flatten()
        amplitudes = self.MWham.flatten()

        condlist = (frequencies > np.array(omega)[0]) & (frequencies <= np.array(omega)[-1]) & (amplitudes > 0)
        self.frequencies = np.extract(condlist, frequencies)
        self.amplitudes = np.extract(condlist, amplitudes)

    def nv_lorentz(self, omega, g_lor):
        self.calculatePeaks(omega)
        res = lor_curve(omega, self.frequencies, self.amplitudes, g_lor)
        try:
            self.res = res / max(res)
            return self.res
        except Exception as e:
            print(res)
            print(e)

    def nv_pseudo_voigt(self, omega, g_lor, asym_coef, fraction):
        self.calculatePeaks(omega)
        res = asymetrical_voigt_curve(omega, self.frequencies, self.amplitudes, g_lor, asym_coef, fraction)
        try:
            self.res = res / max(res)
            return self.res
        except Exception as e:
            print(res)
            print(e)



class NVcenterSet(object):
    def __init__(self, D=2870, Mz_array=np.array([0, 0, 0, 0])):
        # super(NVcenterSet, self).__init__()
        self.rNV = 1. / math.sqrt(3.) * np.array([
            [-1, -1, -1],
            [1, 1, -1],
            [1, -1, 1],
            [-1, 1, 1]])
        self.D = D
        self.Mz_array = Mz_array
        self.nvlist = np.array([NVcenter(D, Mz_array[0]),
                                NVcenter(D, Mz_array[1]),
                                NVcenter(D, Mz_array[2]),
                                NVcenter(D, Mz_array[3])])
        self.setMagnetic()

    def setNVparameters(self, D=2870, Mz_array=np.array([0,0,0,0])):
        for m in range(4):
            self.nvlist[m].setNVparameters(self, D=D, Mz=Mz_array[m])

    def setMagnetic(self, B_lab=np.array([0.0, 0.0, 0.0])):
        """Set magnetic field in Cartesian coordinated Bx, By, Bz"""
        self.Bx = B_lab[0]  # MHz (magnetic field interaction along x)
        self.By = B_lab[1]  # MHz (magnetic field interaction along y)
        self.Bz = B_lab[2]  # MHz (magnetic field interaction along z)
        self.B_lab = B_lab

    def all_frequencies(self, frequencyRange):
        B = np.linalg.norm(self.B_lab)
        self.frequencies = np.empty((4, 2), dtype='float')
        for m in range(4):
            cos = np.dot(self.B_lab, self.rNV[m]) / (np.linalg.norm(self.B_lab) * np.linalg.norm(self.rNV[m]))
            if cos >= 1.0:
                cos = 1.0
            thetaBnv = np.arccos(cos) * 180 / np.pi
            phiBnv = 0
            Bcarnv = CartesianVector(B, thetaBnv, phiBnv)
            # nvlist[m].setNVparameters(D=D)
            self.nvlist[m].setMagnetic(Bx=Bcarnv[0], By=Bcarnv[1], Bz=Bcarnv[2])
            self.nvlist[m].calculatePeaks(frequencyRange)
            self.frequencies[m] = self.nvlist[m].frequencies
        return np.array(self.frequencies)

    def four_frequencies(self, frequencyRange, B_total):
        B = np.linalg.norm(B_total)
        frequencies = np.empty((4, 2), dtype='object')
        for m in range(4):
            cos = np.dot(B_total, self.rNV[m]) / (np.linalg.norm(B_total) * np.linalg.norm(self.rNV[m]))
            if cos >= 1.0:
                cos = 1.0
            thetaBnv = np.arccos(cos) * 180 / np.pi
            phiBnv = 0
            Bcarnv = CartesianVector(B, thetaBnv, phiBnv)
            self.nvlist[m].setMagnetic(Bx=Bcarnv[0], By=Bcarnv[1], Bz=Bcarnv[2])
            self.nvlist[m].calculatePeaks(frequencyRange)
            # nv_fr_idx = (m+1) % 2
            # frequencies[m] = nvlist[m].frequencies[nv_fr_idx]
            frequencies[m] = self.nvlist[m].frequencies

        frequencies1 = np.sort(np.array(frequencies).flatten())
        self.four_frequencies_list = frequencies1[1::2]
        return self.four_frequencies_list

    def sum_odmr(self, x, glor):
        # self.setMagnetic(B_lab)
        B = np.linalg.norm(self.B_lab)
        self.ODMR = np.empty(4, dtype='object')
        for m in range(4):
            cos = np.dot(self.B_lab, self.rNV[m]) / (np.linalg.norm(self.B_lab) * np.linalg.norm(self.rNV[m]))
            if cos >= 1.0:
                cos = 1.0
            thetaBnv = np.arccos(cos) * 180 / np.pi
            phiBnv = 0
            Bcarnv = CartesianVector(B, thetaBnv, phiBnv)
            # nvlist[m].setNVparameters(D=D)
            self.nvlist[m].setMagnetic(Bx=Bcarnv[0], By=Bcarnv[1], Bz=Bcarnv[2])
            self.ODMR[m] = self.nvlist[m].nv_lorentz(x, glor)

        sum_odmr = self.ODMR[0] + self.ODMR[1] + self.ODMR[2] + self.ODMR[3]
        sum_odmr /= max(sum_odmr)

        return sum_odmr

    def calculateAinv(self, B0, Bsens=np.array([0, 0, 0]), omega_limits=np.array([2000, 3800]), dB=0.001):
        '''
        B0 - constant bias magnetic field
        Bsens - small additional magnetic field (the one being measured)
        dB - step size of derivatives
        '''
        B_total = B0 + Bsens
        four_freqs = self.four_frequencies(omega_limits, B_total)
        # print(four_freqs)
        dfdB_array = np.empty((4, 3))
        for row_idx, row in enumerate(np.eye(3)):
            four_freqs_plusdB = self.four_frequencies(omega_limits, B_total + row * dB)
            # print(four_freqs_plusdB)
            dfdB = (four_freqs_plusdB - four_freqs) / dB
            dfdB_array[:, row_idx] = dfdB
        self.dfdB_array_inv = np.linalg.pinv(dfdB_array)
        return self.dfdB_array_inv

    # def deltaB_from_deltaFrequencies(self, A_inv, deltaFrequencies):
    #     return np.dot(A_inv, deltaFrequencies.T)

    def noisy_odmr(self, x, glor, NVparameters, noise_std = 0.005):
        '''
        parameters = {"B_labx": 169.12, "B_laby": 87.71, "B_labz": 40.39, "glor": 4.44, "D": 2867.61, "Mz1": 1.85, "Mz2": 2.16, "Mz3": 1.66, "Mz4": 2.04}
        '''
        D = NVparameters['D']
        Mz_array = np.array([NVparameters['Mz1'], NVparameters['Mz2'], NVparameters['Mz3'], NVparameters['Mz4']])
        B_lab = np.array([NVparameters['B_labx'], NVparameters['B_labx'], NVparameters['B_labx']])
        self.setNVparameters(D, Mz_array)
        self.setMagnetic(B_lab=B_lab)
        odmr_signal = nv_center_set.sum_odmr(x, glor)
        noisy_odmr_signal = add_noise(odmr_signal, noise_std)
        return noisy_odmr_signal


if __name__ == '__main__':
    D = 2851.26115
    Mz_array = np.array([7.32168327, 6.66104172, 9.68158138, 5.64605102])

    # B_labx: 191.945068 + / - 0.06477634(0.03 %)(init=194.7307)
    # B_laby: 100.386360 + / - 0.04340693(0.04 %)(init=94.97649)
    # B_labz: 45.6577322 + / - 0.02524832(0.06 %)(init=38.2026)
    B_lab = np.array([191.945068, 100.386360, 45.6577322])

    # NV center orientation in laboratory frame
    # (100)
    nv_center_set = NVcenterSet(D=D, Mz_array=Mz_array)
    nv_center_set.setMagnetic(B_lab=B_lab)
    print(nv_center_set.B_lab)

    A_inv = nv_center_set.calculateAinv(B_lab)

    print(A_inv)

    omega = np.linspace(2000, 3800, 1000)
    g_lor = 10
    odmr_signal = nv_center_set.sum_odmr(omega, g_lor)
    noise_std = 0.05
    noisy_odmr_signal = add_noise(odmr_signal, noise_std)
    # print(odmr_signal)
    import matplotlib.pyplot as plt
    plt.plot(omega, odmr_signal)
    plt.plot(omega, noisy_odmr_signal)
    plt.show()

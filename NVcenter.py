import numpy as np
import math

def lor(x, x0, a, g):
    return a * g**2 / ((x-x0)**2 + g**2)

def lor_curve(omega, omega_tr, ampl_tr, g):
    res = np.zeros(len(omega))
    for i in range(len(omega_tr)):
        if abs(ampl_tr[i]) > 0:
            res += abs(lor(omega, omega_tr[i], ampl_tr[i], g))
    return res

def CartesianVector(r, theta, phi=0):
    """ Return vector in Cartesian by 'deg' degrees """
    x = r * math.sin(theta * np.pi / 180.0) * math.cos(phi * np.pi / 180.0)
    y = r * math.sin(theta * np.pi / 180.0) * math.sin(phi * np.pi / 180.0)
    z = r * math.cos(theta * np.pi / 180.0)
    return np.array([x, y, z])

class NVcenter(object):
    def __init__(self):
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
        self.setNVparameters()
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
        # print(frequencies)
        amplitudes = self.MWham.flatten()

        condlist = (frequencies > np.array(omega)[0]) & (frequencies <= np.array(omega)[-1]) & (amplitudes > 0)
        self.frequencies = np.extract(condlist, frequencies)
        self.amplitudes = np.extract(condlist, amplitudes)

    def nv_lorentz(self, omega, g_lor):
        # self.calculateHamiltonian()
        # self.calculateInteraction()
        self.calculatePeaks(omega)

        res = lor_curve(omega, self.frequencies, self.amplitudes, g_lor)
        try:
            self.res = res / max(res)
            return self.res
        except Exception as e:
            # print(len(omega), len(res))
            print(res)
            print(e)


class NVcenterSet(object):
    def __init__(self, D=2870, Mz_array=np.array([0, 0, 0, 0])):
        super(NVcenterSet, self).__init__()
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
            self.ODMR[m] = nvlist[m].nv_lorentz(x, glor)

        sum_odmr = self.ODMR[0] + self.ODMR[1] + self.ODMR[2] + self.ODMR[3]
        sum_odmr /= max(sum_odmr)

        return sum_odmr


if __name__ == '__main__':
    D = 2851.26115
    Mz_array = np.array([7.32168327, 6.66104172, 9.68158138, 5.64605102])
    # NV center orientation in laboratory frame
    # (100)
    rNV = 1. / math.sqrt(3.) * np.array([
        [-1, -1, -1],
        [1, 1, -1],
        [1, -1, 1],
        [-1, 1, 1]])
    nvlist = np.array([NVcenter(), NVcenter(), NVcenter(), NVcenter()])
    for i in range(4):
        # nvlist[i].setNVparameters()
        nvlist[i].setNVparameters(D=D, Mz=Mz_array[i])
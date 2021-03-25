import numpy as np
import math


def deltaB_from_deltaFrequencies(A_inv, deltaFrequencies):
    return np.dot(A_inv, deltaFrequencies.T)

def add_noise(signal, noise_std=1):
    np_signal = np.array(signal)
    # print(np_signal)
    noise = np.random.normal(0, noise_std, len(np_signal))
    # print(noise)
    noisy_signal = signal + noise

    return noisy_signal

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

def gauss(x, x0, amplitude, gamma):
    """ Return Gaussian line shape at x with HWHM alpha """
    return amplitude * np.sqrt(4*np.log(2) / np.pi) / gamma * np.exp(-((x-x0) / gamma)**2 * 4*np.log(2))

def lorentz(x, x0, amplitude, gamma):
    """ Return Lorentzian line shape at x with HWHM gamma """
    return amplitude * gamma / (2*np.pi) / ((x-x0)**2 + (gamma/2.)**2)


def asymetrical_voigt(x, x0, amplitude, gamma, asym_coef, fraction):
    # x = x - x0
    # x0 = 0
    perturbation = 1 - asym_coef * (x - x0) / gamma * np.exp(-(x-x0)**2/(2*(2*gamma)**2))
    # perturbation = 1 - asym_coef / gamma * np.exp(-(x - x0) ** 2 / (2 * (2 * gamma) ** 2))

    pseudo_voigt = fraction * gauss((x - x0)*perturbation,0,amplitude,gamma) + (1-fraction)*lorentz((x - x0)*perturbation,0,amplitude,gamma)
    return np.array(pseudo_voigt)

def asymetrical_voigt_curve(omega, omega_tr, ampl_tr, g, asym_coef=0, fraction=0.5):
    res = np.zeros(len(omega))
    for i in range(len(omega_tr)):
        if np.real(ampl_tr[i]) > 0:
            res += abs(asymetrical_voigt(omega, np.real(omega_tr[i]), np.real(ampl_tr[i]), g, asym_coef, fraction))
            # print(omega_tr[i], np.real(ampl_tr[i]), max(res))
    return res
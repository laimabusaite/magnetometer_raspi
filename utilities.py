import matplotlib.pyplot as plt
import numpy as np
import math


def round_to_decimal(a, round_to=0):
    if abs(round_to) > 0:
        return np.round(a / round_to) * round_to
    else:
        return np.round(a)


def deltaB_from_deltaFrequencies(A_inv, deltaFrequencies):
    rotation_angles = {"alpha": 1.9626607183487732, "phi": 20.789077311199208, "theta": 179.4794019370279}

    deltaB = np.dot(A_inv, deltaFrequencies.T)

    rotatedB = rotate(deltaB, alpha=rotation_angles['alpha'], phi=rotation_angles['phi'], theta=rotation_angles['theta'])

    return  rotatedB


def add_noise(signal, noise_std=1):
    np_signal = np.array(signal)
    # print(np_signal)
    noise = np.random.normal(0, noise_std, len(np_signal))
    # print(noise)
    noisy_signal = signal + noise

    return noisy_signal


def lor(x, x0, a, g):
    return a * g ** 2 / ((x - x0) ** 2 + g ** 2)


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
    return amplitude * np.sqrt(4 * np.log(2) / np.pi) / gamma * np.exp(-((x - x0) / gamma) ** 2 * 4 * np.log(2))

def gauss_curve(omega, omega_tr, ampl_tr, g):
    res = np.zeros(len(omega))
    for i in range(len(omega_tr)):
        if abs(ampl_tr[i]) > 0:
            res += abs(gauss(omega, omega_tr[i], ampl_tr[i], g))
    return res


def lorentz(x, x0, amplitude, gamma, y0=0):
    """ Return Lorentzian line shape at x with HWHM gamma """
    return y0 + amplitude * gamma / (2 * np.pi) / ((x - x0) ** 2 + (gamma / 2.) ** 2)


def asymetrical_voigt(x, x0, amplitude, gamma, asym_coef, fraction):
    # x = x - x0
    # x0 = 0
    perturbation = 1 - asym_coef * (x - x0) / gamma * np.exp(-(x - x0) ** 2 / (2 * (2 * gamma) ** 2))
    # perturbation = 1 - asym_coef / gamma * np.exp(-(x - x0) ** 2 / (2 * (2 * gamma) ** 2))

    pseudo_voigt = fraction * gauss((x - x0) * perturbation, 0, amplitude, gamma) + (1 - fraction) * lorentz(
        (x - x0) * perturbation, 0, amplitude, gamma)
    return np.array(pseudo_voigt)


def asymetrical_voigt_curve(omega, omega_tr, ampl_tr, g, asym_coef=0, fraction=0.5):
    res = np.zeros(len(omega))
    for i in range(len(omega_tr)):
        if np.real(ampl_tr[i]) > 0:
            res += abs(asymetrical_voigt(omega, np.real(omega_tr[i]), np.real(ampl_tr[i]), g, asym_coef, fraction))
            # pri
    return res

def amper2gauss(current, axis):
    x_list = [0, 'x', 'X']
    y_list = [1, 'y', 'Y']
    z_list = [2, 'z', 'Z']
    # print(current, axis)

    if axis in x_list:
        m = 1.086561069
        b = 0.024495759
    elif axis in y_list:
        m = 0.892602592
        b = 0.11899291
    elif axis in z_list:
        m = 0.810370347
        b = 0.003419481
    else:
        print('invalid axis name')
        m = 0
        b = 0

    # print(m, b)
    return (m * current + b) * 10.

def rotate(vec, theta=0.0, phi=0.0, alpha=0.0):
    theta = np.radians(theta)
    phi = np.radians(phi)
    alpha = np.radians(alpha)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(alpha), -np.sin(alpha)],
                   [0, np.sin(alpha), np.cos(alpha)]])
    Ry = np.array([[np.cos(theta), 0, np.sin(theta)],
                   [0, 1, 0],
                   [-np.sin(theta), 0, np.cos(theta)]])
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0],
                   [np.sin(phi), np.cos(phi), 0],
                   [0, 0, 1]])
    rotated_vec = np.dot(Rz, np.dot(Ry, np.dot(Rx, vec)))
    return rotated_vec

if __name__ == '__main__':

    x = np.linspace(2800, 3000, 1801)
    y = asymetrical_voigt_curve(x, [2870], [1], 5, asym_coef=0, fraction=1)

    plt.plot(x,y)
    plt.show()
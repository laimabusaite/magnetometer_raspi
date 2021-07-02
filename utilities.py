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
    # rotation_angles = {"alpha": 87.34271435510534, "phi": 3.2800280875516754, "theta": 179.51519807708098}
    alpha = rotation_angles['alpha']  # rotate around x
    phi = rotation_angles['phi']  # rotate around z
    theta = rotation_angles['theta']  # rotate around y

    deltaB = np.dot(A_inv, deltaFrequencies.T)

    rotatedB = rotate(deltaB, alpha=alpha, phi=phi, theta=theta)

    return rotatedB


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

def amper2gauss_array(current_x, current_y, current_z):

    #florian
    # m_list = np.array([1.086561069, 0.892602592, 0.810370347]) * 10.
    # b_list = np.array([0.024495759, 0.11899291, 0.003419481]) * 10.

    #reinis
    # m_list = np.array([8.4895, 10.177, 7.6793])
    # b_list = np.array([0.0, 0.0, 0.0])

    #laima
    linear_params = {"slope": [0.0007826239771925043, 0.0009039493509247832, 0.0010428497650549662],
     "intercept": [-0.003527443175143303, -0.008846942705030204, -0.009250214551616835]}
    m_list = np.array(linear_params['slope']) * 10.
    b_list = np.array(linear_params['intercept']) * 10.


    # if np.isscalar(current_x):
    #

    # if isinstance(current_x, list):
    #     current_x = np.array(current_x)

    B_x = current_x * m_list[0] + b_list[0]
    B_y = current_y * m_list[1] + b_list[1]
    B_z = current_z * m_list[2] + b_list[2]


    return B_x, B_y, B_z

def amper2gauss(current_list):

    m_list = np.array([1.086561069, 0.892602592, 0.810370347])
    b_list = np.array([0.024495759, 0.11899291, 0.003419481])
    current_list = np.array(current_list)

    B_coil_list = (m_list * current_list + b_list) * 10.

    return B_coil_list


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

    plt.plot(x, y)
    plt.show()

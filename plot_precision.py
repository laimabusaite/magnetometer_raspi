import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def exponential_fit(x, a, b, c):
    return a*np.exp(-b*x) + c

def parabola_fit(x, a, b, c):
    return a * x**2 + b * x + c

def sqrt_fit(x, a, b):
    return a / np.sqrt(b*x)

def pow_fit(x, a, b, c):
    return a * (b*x)**c

def fit_func(x, a, b, c):
    # return parabola_fit(x, a, b, c)
    # return exponential_fit(x, a, b, c)
    return sqrt_fit(x, a, b)

if __name__ == '__main__':

    filename = 'tables/table_sensitivity.csv'

    dataframe = pd.read_csv(filename, header=0, index_col=0)

    print(dataframe.head())

    dev_set = sorted(list(set(dataframe['dev (MHz)'])))
    print(dev_set)

    B_index_set = sorted(list(set(dataframe['B index'])))
    print(B_index_set)
    print(dataframe.columns)

    column_list = ['B index',
                   'Bmod coil (mT)', 'Bmod coil stderr (mT)',
                   'Bmod measured (mT)', 'Bmod measured stderr (mT)',
                   'Bx coil (mT)', 'Bx coil stderr (mT)',
                   'Bx measured (mT)', 'Bx measured stderr (mT)',
                   'By coil (mT)', 'By coil stderr (mT)',
                   'By measured (mT)', 'By measured stderr (mT)',
                   'Bz coil (mT)', 'Bz coil stderr (mT)',
                   'Bz measured (mT)', 'Bz measured stderr (mT)',
                   'avg', 'dev (MHz)', 'point count']


    for dev in dev_set[:]:
        for B_index in B_index_set[:]:
            dataframe_dev = dataframe[(dataframe['dev (MHz)'] == dev) & (dataframe['B index'] == B_index)][column_list]
            dataframe_dev.sort_values(by='avg', inplace=True)
            print(dataframe_dev)

            x = dataframe_dev['avg']
            y = dataframe_dev['Bmod measured stderr (mT)']
            plt.figure(f'B {B_index}, dev {dev} MHz', figsize=(6, 4))
            plt.title(f'B = {B_index * 0.1} mT, dev = {dev} MHz')
            try:
                x_fit = np.linspace(0, 200, 1000)
                fitting_parameters, covariance = curve_fit(exponential_fit, x, y)
                a, b, c = fitting_parameters
                print('exp', fitting_parameters)
                y_fit = exponential_fit(x_fit, a, b, c)
                plt.plot(x_fit, y_fit, 'r', label=fr'${a:.4f} \cdot \exp{{(-{b:.4f}\cdot x)}} + {c:.4f}$')

                fitting_parameters, covariance = curve_fit(sqrt_fit, x, y)
                a, b = fitting_parameters
                print('sqrt', fitting_parameters)
                y_fit = sqrt_fit(x_fit, a, b)
                plt.plot(x_fit, y_fit, 'g', label=fr'$\dfrac{{{a:.2f}}}{{\sqrt{{{b:.2f} x}}}}$')

                fitting_parameters, covariance = curve_fit(pow_fit, x, y)
                a, b, c = fitting_parameters
                print('pow', fitting_parameters)
                y_fit = pow_fit(x_fit, a, b, c)
                plt.plot(x_fit, y_fit, 'k', label=fr'${a:.3f} \cdot ({b:.3} x)^{{{c:.3f}}}$')
            except Exception as e:
                print(e)
            plt.plot(x, y, marker='o', ms=7)
            plt.xlabel('Number of averages')
            plt.ylabel(r'Bmod measured stderr (mT)')
            plt.xlim(0, 200)
            plt.ylim(0.0,0.002)
            plt.tight_layout()
            plt.legend()
            plt.savefig(f'fig_std_avg/fit_B{B_index}_dev{dev:.0f}MHz.png')
            plt.savefig(f'fig_std_avg/fit_B{B_index}_dev{dev:.0f}MHz.pdf')

            x = dataframe_dev['avg']
            y = dataframe_dev['Bmod measured stderr (mT)']/dataframe_dev['Bmod measured (mT)'] * 100
            plt.figure(f'rel B {B_index}, dev {dev} MHz', figsize=(6, 4))
            plt.title(f'B = {B_index * 0.1:.1f} mT, dev = {dev} MHz')

            try:
                fitting_parameters, covariance = curve_fit(exponential_fit, x, y)
                a, b, c = fitting_parameters
                x_fit = np.linspace(0, 150, 1000)
                y_fit = exponential_fit(x_fit, a, b, c)
                plt.plot(x_fit, y_fit, 'r')

                fitting_parameters, covariance = curve_fit(sqrt_fit, x, y)
                a, b = fitting_parameters
                x_fit = np.linspace(0, 150, 1000)
                y_fit = sqrt_fit(x_fit, a, b)
                plt.plot(x_fit, y_fit, 'g')

                fitting_parameters, covariance = curve_fit(pow_fit, x, y)
                a, b, c = fitting_parameters
                x_fit = np.linspace(0, 150, 1000)
                y_fit = pow_fit(x_fit, a, b, c)
                plt.plot(x_fit, y_fit, 'k')
            except Exception as e:
                print(e)
            plt.plot(x, y, marker='o', ms=10)
            plt.xlabel('Number of averages')
            plt.ylabel(r'$\Delta B / B$ (%)')
            plt.xlim(0, 150)
            plt.tight_layout()
            # plt.savefig(f'fig_std_avg/fit_rel_B{B_index}_dev{dev:.0f}MHz.png')
            # plt.savefig(f'fig_std_avg/fit_rel_B{B_index}_dev{dev:.0f}MHz.pdf')

    plt.show()
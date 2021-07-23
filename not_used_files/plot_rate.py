import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

import utilities


def func_rate(x, tau):

    return tau / x

if __name__ == '__main__':

    filename_rate = 'tables/table_rate.csv'

    dataframe_rate = pd.read_csv(filename_rate, header=[0,1], index_col=0)

    print(dataframe_rate)

    # x_data = dataframe_rate.iloc['dev (MHz)',:] * dataframe_rate.iloc['avg',:]

    print(dataframe_rate.columns[1])

    x_data = np.empty(len(dataframe_rate.columns))
    for idx_col, col in enumerate(dataframe_rate.columns):
        x_data[idx_col] = float(dataframe_rate.columns[idx_col][0]) * float(dataframe_rate.columns[idx_col][1])

    print(x_data)

    y_data = np.array((1. / dataframe_rate).mean())
    print(y_data)

    dataframe_plot = pd.DataFrame(data = {'X': x_data, 'Y': y_data})
    dataframe_plot.sort_values(by='X', inplace=True)
    print(dataframe_plot)
    x = np.array(dataframe_plot['X'])
    y = np.array(dataframe_plot['Y'])
    print(x, y)


    params = np.polyfit(x, y, deg=1)
    print(params)
    x_fit = np.linspace(0, 2500, 100)
    y_fit = params[0] * x_fit + params[1]

    plt.figure(figsize=(5, 4))
    plt.scatter(x, y)
    plt.plot(x_fit, y_fit, label=f'{params[0]:.4f} x + {params[1]:.4f}')
    # plt.xlabel(r'Number of averages $\times$ frequency range (MHz)')
    plt.xlabel(r'Number of averages $\times$ number of measured points')
    # plt.ylabel('Measurement rate (Hz)')
    plt.ylabel('Measurement time (s)')
    plt.xlim(0, 2200)
    plt.ylim(0,15)
    plt.tight_layout()
    plt.legend()
    plt.savefig('tables/time_vs_devavg.png')
    plt.savefig('tables/time_vs_devavg.pdf')

    y_rate = 1. / y_data
    # params = curve_fit(func_rate, x, y_rate)
    # [tau] = params[0]
    # x_fit = np.linspace(x[0], x[-1], 100)
    # y_fit = func_rate(x_fit, tau)
    plt.figure(figsize=(5,4))
    plt.scatter(x_data, y_rate)
    plt.plot(x_fit, 1./y_fit)
    # plt.xlabel(r'Number of averages $\times$ frequency range (MHz)')
    plt.xlabel(r'Number of averages $\times$ number of measured points')
    plt.ylabel('Measurement rate (Hz)')
    plt.xlim(0, 2200)
    plt.ylim(0, 2)
    plt.tight_layout()
    plt.savefig('tables/rate_vs_devavg.png')
    plt.savefig('tables/rate_vs_devavg.pdf')


    plt.show()




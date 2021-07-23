import os
import re
import time

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter

import fit_odmr
import utilities
from detect_peaks import detect_peaks_weighted, detect_peaks

if __name__ == '__main__':

    filename = 'tables/temp_peaks.csv'

    dataframe = pd.read_csv(filename, index_col=0, header=0, names=[f'peak {idx}' for idx in [2,4,6,8]])

    print(dataframe.head())

    # print(savgol_filter(dataframe, 101, 2).head())

    data_smooth = np.array([savgol_filter(dataframe[col], 501, 2) for col in dataframe.columns]).transpose()
    dataframe_smooth = pd.DataFrame(data=data_smooth, columns=[f'peak {idx}' for idx in [2,4,6,8]])
    print(dataframe_smooth.head())

    xlabel = 'Measurement N#'
    ylabel1 = 'Peak frequency (MHz)'
    ylabel2 = 'Peak frequency change (MHz)'
    title = 'Microwave step 1 MHz, room cooling, 23C, with small fan'

    ax1 = dataframe.plot(title=title)
    # dataframe_smooth.plot(ax=ax1, title=title)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel(ylabel1)
    ax1.set_xlim(0, 10000)
    plt.savefig('tables/temp_peaks/peaks.pdf', bbox_inches='tight')
    plt.savefig('tables/temp_peaks/peaks.png', bbox_inches='tight')

    ax2 = (dataframe-dataframe.loc[0, :]).plot()
    (dataframe_smooth - dataframe_smooth.loc[0, :]).plot(ax=ax2, title=title)
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel(ylabel2)
    ax2.set_xlim(0, 10000)
    plt.savefig('tables/temp_peaks/peak_change.pdf', bbox_inches='tight')
    plt.savefig('tables/temp_peaks/peak_change.png', bbox_inches='tight')

    # ax3 = dataframe.plot(subplots=True, title=title)
    # dataframe_smooth.plot(ax=ax3, subplots=True, title=title, color='k')
    ax3 = dataframe_smooth.plot(subplots=True, title=title)
    ax3[3].set_xlabel(xlabel)
    # ax3[1].set_ylabel(ylabel1)
    plt.savefig('tables/temp_peaks/peaks_smooth_subplots1.pdf', bbox_inches='tight')
    plt.savefig('tables/temp_peaks/peaks_smooth_subplots1.png', bbox_inches='tight')

    for ax in ax3:
        # ax.set_ylabel(ylabel1)
        ax.set_xlim(0, 10000)

    ax4 = (dataframe_smooth - dataframe_smooth.loc[0, :]).plot(title=title, figsize=(5,4))
    ax4.set_xlabel(xlabel)
    ax4.set_ylabel(ylabel2)
    ax4.set_xlim(0, 10000)
    # plt.tight_layout()
    plt.savefig('tables/temp_peaks/peak_change_smooth.pdf', bbox_inches='tight')
    plt.savefig('tables/temp_peaks/peak_change_smooth.png', bbox_inches='tight')


    plt.show()

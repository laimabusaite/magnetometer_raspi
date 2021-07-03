import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':
    filename = 'new_calibration/coil_axis_calibration.dat'

    df = pd.read_csv(filename, index_col=0)

    Mz1_mean = df['Mz1'].mean()
    Mz2_mean = df['Mz2'].mean()
    Mz3_mean = df['Mz3'].mean()
    Mz4_mean = df['Mz4'].mean()
    Mz1_std = df['Mz1'].std()
    Mz2_std = df['Mz2'].std()
    Mz3_std = df['Mz3'].std()
    Mz4_std = df['Mz4'].std()
    print(f'Mz1 = {Mz1_mean:.2f} +- {Mz1_std:.2f}')
    print(f'Mz2 = {Mz2_mean:.2f} +- {Mz2_std:.2f}')
    print(f'Mz3 = {Mz3_mean:.2f} +- {Mz3_std:.2f}')
    print(f'Mz4 = {Mz4_mean:.2f} +- {Mz4_std:.2f}')
    # print(Mz1_mean, Mz2_mean, Mz3_mean, Mz4_mean)

    plt.plot(df['Mz1'], marker='o', ls='-')
    plt.plot(df['Mz2'], marker='o', ls='-')
    plt.plot(df['Mz3'], marker='o', ls='-')
    plt.plot(df['Mz4'], marker='o', ls='-')

    plt.axhline(Mz1_mean, c='C0')
    plt.axhline(Mz2_mean, c='C1')
    plt.axhline(Mz3_mean, c='C2')
    plt.axhline(Mz4_mean, c='C3')

    # df['Mz1'].plot()
    # df['Mz2'].plot()
    # df['Mz3'].plot()
    # df['Mz4'].plot()
    # print(df['Mz1'])
    plt.figure()
    plt.plot(df['glor'], marker='o', ls='-')
    plt.legend()
    plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
            plt.figure(f'B {B_index}, dev {dev} MHz', figsize=(6,4))
            plt.title(f'B = {B_index*0.1} mT, dev = {dev} MHz')
            plt.plot(dataframe_dev['avg'], dataframe_dev['Bmod measured stderr (mT)'], marker='o')
            plt.xlabel('Number of averages')
            plt.ylabel(r'Bmod measured stderr (mT)')
            plt.xlim(0,100)
            plt.tight_layout()
            plt.savefig(f'fig_std_avg/B{B_index}_dev{dev:.0f}MHz.png')
            plt.savefig(f'fig_std_avg/B{B_index}_dev{dev:.0f}MHz.pdf')

            plt.figure(f'rel B {B_index}, dev {dev} MHz', figsize=(6, 4))
            plt.title(f'B = {B_index * 0.1:.1f} mT, dev = {dev} MHz')
            plt.plot(dataframe_dev['avg'],
                     dataframe_dev['Bmod measured stderr (mT)']/dataframe_dev['Bmod measured (mT)'] * 100,
                     marker='o')
            plt.xlabel('Number of averages')
            plt.ylabel(r'$\Delta B / B$ (%)')
            plt.xlim(0, 100)
            plt.tight_layout()
            plt.savefig(f'fig_std_avg/rel_B{B_index}_dev{dev:.0f}MHz.png')
            plt.savefig(f'fig_std_avg/rel_B{B_index}_dev{dev:.0f}MHz.pdf')

    plt.show()
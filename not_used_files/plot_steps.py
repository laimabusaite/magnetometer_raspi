import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == '__main__':

    filename = 'tables/table_steps.csv'

    dataframe = pd.read_csv(filename, header=0)

    print(dataframe.columns)

    for b in ['x', 'y', 'z', 'mod']:
        dataframe[f'B{b} diff (mT)'] = np.abs(dataframe[f'B{b} coil (mT)'] - dataframe[f'B{b} measured (mT)'])

    print(dataframe)

    B_set_list = sorted(list(set(dataframe['B index'])))
    print(B_set_list)

    for B_set in B_set_list:
        plt.figure(f'B {B_set}', figsize=(5, 4))
        plt.title(f'B = {B_set * 0.1:.1f} mT')
        x = dataframe[dataframe['B index'] == B_set]['step (MHz)']
        for b in ['x', 'y', 'z', 'mod']:
            y = dataframe[dataframe['B index'] == B_set][f'B{b} diff (mT)']
            plt.plot(x, y, marker='o')
        plt.xlabel('Measurement step size (MHz)')
        plt.ylabel('| B coil - B measurement | (mT)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'tables/Bdiff_vs_step_B{B_set:.0f}.png')
        plt.savefig(f'tables/Bdiff_vs_step_B{B_set:.0f}.pdf')

    plt.show()

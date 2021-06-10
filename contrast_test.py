from detect_peaks import *
import matplotlib.pyplot as plt
import glob

if __name__ == '__main__':
    filenames = sorted(glob.glob("data/*.dat"))[2:]

    contrast_list = []
    for filename in filenames[:]:
        # print(filename)
        dataframe = import_data(filename)
        dataframe['ODMR_norm'] = dataframe['ODMR'] / max(dataframe['ODMR'])
        # print(dataframe.head())
        scale = max(dataframe['ODMR']) - min(dataframe['ODMR'])
        contrast = scale / max(dataframe['ODMR'])
        # print(f"Scale: {scale}")
        # print(f"Contrast: {contrast}")
        contrast_list.append(contrast)

        peak_weighted = detect_peaks_weighted(dataframe['MW'], dataframe['ODMR'],
                                              weight_order=3, min_contrast=0.002)

        print(peak_weighted)
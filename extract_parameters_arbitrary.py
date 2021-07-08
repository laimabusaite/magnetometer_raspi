import re

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

import utilities

if __name__ == '__main__':

    dir_name = 'RQnc/arb'

    folder_list = glob.glob(f'{dir_name}/*/')

    print(folder_list)

    folder = folder_list[0]

    log_file_list = glob.glob(f'{folder}/*.log')
    print(log_file_list)


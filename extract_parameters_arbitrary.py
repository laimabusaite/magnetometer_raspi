import re

import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

import utilities

if __name__ == '__main__':

    dir_name = 'RQnc/arb'

    log_file_list = glob.glob(f'{dir_name}/*.log')
    print(log_file_list)
from terminalplot import plot

import glob
import numpy as np
import time

filenames = sorted(glob.glob("test_data/*.dat"))
filenames = filenames[::1]

#f = filenames[0]

#print(f)

for f in filenames:
    print(f)
    x, y = np.loadtxt(fname = f, delimiter='\t', usecols=(0, 1), unpack = True)
    #x = x[0:10000:1]
    #y = y[0:10000:1]

    y=y/max(y)
    x = x.tolist()
    y = y.tolist()

    plot(x, y)
    time.sleep(1)

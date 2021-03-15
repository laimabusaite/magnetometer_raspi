import NVcenter as nv
import numpy as np
from detect_peaks import detect_peaks

if __name__ == '__main__':
  D = 2851.26115
  Mz_array = np.array([7.32168327, 6.66104172, 9.68158138, 5.64605102])

  # B_labx: 191.945068 + / - 0.06477634(0.03 %)(init=194.7307)
  # B_laby: 100.386360 + / - 0.04340693(0.04 %)(init=94.97649)
  # B_labz: 45.6577322 + / - 0.02524832(0.06 %)(init=38.2026)
  B_lab = np.array([191.945068, 100.386360, 45.6577322])

  # NV center orientation in laboratory frame
  # (100)
  nv_center_set = nv.NVcenterSet(D=D, Mz_array=Mz_array)
  nv_center_set.setMagnetic(B_lab=B_lab)
  print(nv_center_set.B_lab)

  frequencies0 = nv_center_set.four_frequencies(np.array([2000,3500]), nv_center_set.B_lab)

  print(frequencies0)

  A_inv = nv_center_set.calculateAinv(nv_center_set.B_lab)

  print(A_inv)

  filename = 'data/test_2920_650_16dBm_1024_ODMR.dat'
  peaks = detect_peaks(filename)
  print(peaks)

  delta_frequencies = frequencies0 - peaks[1::2]
  print(delta_frequencies)

  Bsens = nv_center_set.deltaB_from_deltaFrequencies(A_inv, delta_frequencies)

  print(Bsens)




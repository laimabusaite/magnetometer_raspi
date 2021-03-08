import NVcenter as nv
import numpy as np

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

  A_inv = nv_center_set.calculateAinv(B_lab)

  print(A_inv)



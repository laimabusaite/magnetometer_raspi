# Coil calibration

Folders x, y, x : raw data from hall probe

Folder "extracted_data": calibrated data for each coil

current_x_coil (mA) 

current_y_coil (mA) 

current_z_coil (mA) 

Bx_probe (mT) 

By_probe (mT) 

Bz_probe (mT) 

Bmod_probe (mT) = 	$\sqrt{B_x^2 + B_y^2 + B_z^2}$

Bx_std_probe (mT)  - pandas method used to calculate std

By_std_probe (mT)  - pandas method used to calculate std

Bz_std_probe (mT)  - pandas method used to calculate std

Bmod_std_probe (mT) - pandas method used to calculate std

Bmod_np_std_probe (mT) - numpy method used to calculate std

slope (mT  / mA)

intercept (mT) 

Bmod_linear (mT) = slope * current + intercept


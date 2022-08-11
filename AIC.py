#compare Akaike Information Criterion (AIC) for different halo models

import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

def read_data_by_column(name, lines):
    return np.loadtxt(name, skiprows=lines, unpack=True)

def read_data(fname):
    f1 = open(fname, 'r')
    info = f1.readlines()
    f1.seek(0)
    f1.close()
    return info

chi_mine_1 = 9562.46
chi_takahashi_1 = 13182.081

chi_mine_3 = 42352.744
chi_takahashi_3 = 38601.1

chi_mine_7 = 8542.0
chi_takahashi_7 = 14682.0

chi_mine_5 = 7250.0
chi_takahashi_5 = 10603.0

chi_mine_1avg = 177636.2
chi_mine_3avg = 56470.3
chi_mine_5avg = 23276.6
chi_mine_7avg = 19713.2

aic_chi_mine_1 = 2*4.0 + chi_mine_1
aic_chi_mine_3 = 2*4.0 + chi_mine_3
aic_chi_mine_5 = 2*4.0 + chi_mine_5
aic_chi_mine_7 = 2*4.0 + chi_mine_7

aic_chiavg_mine_1 = 2.0 + chi_mine_1avg
aic_chiavg_mine_3 = 2.0 + chi_mine_3avg
aic_chiavg_mine_5 = 2.0 + chi_mine_5avg
aic_chiavg_mine_7 = 2.0 + chi_mine_7avg

aic_chi_takahashi_1 = 2*(35.0/16.0) + chi_takahashi_1
aic_chi_takahashi_3 = 2*(35.0/16.0) + chi_takahashi_3
aic_chi_takahashi_5 = 2*(35.0/16.0) + chi_takahashi_5
aic_chi_takahashi_7 = 2*(35.0/16.0) + chi_takahashi_7

aic_rel_mine_1 = 2.0*4.0 + 120.0*(0.033**2)
aic_rel_mine_3 = 2.0*4.0 + 120.0*(0.03**2)
aic_rel_mine_5 = 2.0*4.0 + 120.0*(0.028**2)
aic_rel_mine_7 = 2.0*4.0 + 120.0*(0.028**2)

aic_relavg_mine_1 = 2.0 + 120.0*(0.091**2)
aic_relavg_mine_3 = 2.0 + 120.0*(0.051**2)
aic_relavg_mine_5 = 2.0 + 120.0*(0.037**2)
aic_relavg_mine_7 = 2.0 + 120.0*(0.032**2)

aic_rel_takahashi_1 = 2*(35.0/16.0) + 120.0*(0.053**2)
aic_rel_takahashi_3 = 2*(35.0/16.0) + 120.0*(0.042**2)
aic_rel_takahashi_5 = 2*(35.0/16.0) + 120.0*(0.034**2)
aic_rel_takahashi_7 = 2*(35.0/16.0) + 120.0*(0.037**2)


print(aic_chi_mine_1, aic_chi_mine_3, aic_chi_mine_5, aic_chi_mine_7)
print(aic_chiavg_mine_1, aic_chiavg_mine_3, aic_chiavg_mine_5, aic_chiavg_mine_7)
print(aic_chi_takahashi_1, aic_chi_takahashi_3, aic_chi_takahashi_5, aic_chi_takahashi_7)
print('\n')

print(aic_rel_mine_1, aic_rel_mine_3, aic_rel_mine_5, aic_rel_mine_7)
print(aic_relavg_mine_1, aic_relavg_mine_3, aic_relavg_mine_5, aic_relavg_mine_7)
print(aic_rel_takahashi_1, aic_rel_takahashi_3, aic_rel_takahashi_5, aic_rel_takahashi_7)


import sys
import numpy as np
from matplotlib import pyplot as plt
import eval_data
from scipy.interpolate import make_smoothing_spline

num_files = 1
window_moving_mean = 5

foldername = "./sweep/Save_s21/Batch_04"

# freq, mean_db, meas_db = eval_data.eval_data(foldername, num_files, window_moving_mean)
# eval_data.plot_data(freq, mean_db, meas_db)

## If needs to plot only one file

# foldername = "./sweep/Save_s21/Batch_04/s21_0.s2p"
# freq, mean_db_thrg, meas_db_thrg = eval_data.eval_data(foldername, num_files, window_moving_mean, True)
# eval_data.plot_data(freq, mean_db_thrg, meas_db_thrg)

foldername = "./sweep/Save_s21/Batch_04/s21_t0.s2p"
freq, mean_db_thrg, meas_db_thrg = eval_data.eval_data(foldername, num_files, window_moving_mean, True)

foldername = "./sweep/Save_s21/Batch_04/s21_ress.s2p"
freq, mean_db, meas_db = eval_data.eval_data(foldername, num_files, window_moving_mean, True)


mean_db_calib = mean_db - mean_db_thrg
meas_db_calib = meas_db - meas_db_thrg

eval_data.plot_data(freq, mean_db_calib, meas_db_calib)
# plt.show()

# spl = make_smoothing_spline(freq, mean_db_calib, lam=0.000007)

# print(len(spl(freq)))

# eval_data.plot_data(freq, spl(freq), mean_db_calib)
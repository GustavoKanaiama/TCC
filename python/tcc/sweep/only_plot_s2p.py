import sys
import numpy as np
from matplotlib import pyplot as plt
import eval_data

num_files = 1
window_moving_mean = 5

foldername = "./sweep/Save_s21/Batch_04"

freq, mean_db, meas_db = eval_data.eval_data(foldername, num_files, window_moving_mean)

eval_data.plot_data(freq, mean_db, meas_db)
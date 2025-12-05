# import relevant packages
import os
import numpy as np
import scipy.io as sio # This will be used to load an MATLAB file
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as bpdf # This will be used to create a PDF to store multiple plots in the same file

# Import the functions you'll use from HW8Fun.py (defined in HW7):
from self_py_fun import HW8Fun

# Define variables used in this script:
bp_low = 0.5
bp_upp = 6
electrode_num = 16
electrode_name_ls = ['F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8', 'CP3', 'CP4', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']

parent_dir = '/Users/rmcsweeney/Documents/GitHub/BIOS-584'
parent_data_dir = '{}/data'.format(parent_dir)
time_index = np.linspace(0, 800, 25) # This is a hypothetic time range up to 800 ms after each stimulus.

subject_name = 'K114'
session_name = '001_BCI_TRN' 

# Generate a directory called the subject_name in your working directory:
if os.path.exists("{}/{}".format(parent_dir, subject_name)):
    print("The '{}' folder has already been created.".format(subject_name))
else:
    print("We need to create the subject_name folder.")
    os.mkdir("{}/{}".format(parent_dir, subject_name)) 
    print("The most recent command just created the {} folder.".format(subject_name))

# Load the dataset
eeg_trunc_obj = sio.loadmat("data/{}_001_BCI_TRN_Truncated_Data_0.5_6.mat".format(subject_name)) 
print(type(eeg_trunc_obj))

# Extract information for the Signal and Type keys and assign them to variables
eeg_trunc_signal = eeg_trunc_obj['Signal']
eeg_trunc_type = (np.squeeze(eeg_trunc_obj['Type'], axis = 1)) 

print(eeg_trunc_signal.shape) # This is a two dimensional array
print(eeg_trunc_type.shape) # This is a one dimensional array
print()

print(eeg_trunc_type[:10]) 
print(eeg_trunc_signal[0][:10])
# eeg_trunc_type contains a binary variable, with values of either -1 or 1
# This value notes whether the signal in this row is associated with a target (1) or non-target (-1) stimuli 

### Call the functions to generate the mean and covariance plots: ###

# Calculate the means and covariances using produce_trun_mean_cov():
'''
your output = produce_trun_mean_cov(your input)
'''
K114_mean_cov = HW8Fun.produce_trun_mean_cov(eeg_trunc_signal, eeg_trunc_type, electrode_num) 

# Plot the target and non-target means for each electrode:
HW8Fun.plot_trunc_mean(K114_mean_cov[0], K114_mean_cov[1], subject_name, time_index, 16, electrode_name_ls) 
# Defined variables by position in this code chunk

# Plot the covariance for target signals only:
HW8Fun.plot_trunc_cov(K114_mean_cov[2], "Target", time_index, subject_name, electrode_num, electrode_name_ls, (16,16)) 

# Plot the covariance for non-target signals only:
HW8Fun.plot_trunc_cov(K114_mean_cov[3], "Non-Target", time_index, subject_name, electrode_num, electrode_name_ls, (16,16)) 

# Plot the covariance for all signals:
HW8Fun.plot_trunc_cov(K114_mean_cov[4], "All", time_index, subject_name, electrode_num, electrode_name_ls, (16,16)) 
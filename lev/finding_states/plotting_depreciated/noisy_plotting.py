'''
Plotting file for csv created by testing_witnesses.py
'''
import csv
import matplotlib.pyplot as plt
import ast
import numpy as np

# Load the CSV file into arrays

# Opening three separate csv files each with a specific witness vs chi curve for various etas.
# First file
wm_arr_15 = []
wpl_15 = []
chi_arr = []
with open('phi_chisweep_noisy15eta.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        wm_arr_15.append(ast.literal_eval(row['W'])[0])
        wpl_15.append(ast.literal_eval(row['min_W_prime'])[0])
        chi_arr.append(ast.literal_eval(row['chi'])[0])

# Convert lists to numpy arrays for plotting
wm_arr_30 = np.array(wm_arr_15)
wpl_30 = np.array(wpl_15)
chi_arr_30 = np.array(chi_arr)

# Second file, same chi so not tracked
wm_arr_30 = []
wpl_30 = []
with open('phi_chisweep_noisy30eta.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        wm_arr_30.append(ast.literal_eval(row['W'])[0])
        wpl_30.append(ast.literal_eval(row['min_W_prime'])[0])

# Convert lists to numpy arrays for plotting
wm_arr_30 = np.array(wm_arr_30)
wpl_30 = np.array(wpl_30)

# Third file, same chi so not tracked
wm_arr_45 = []
wpl_45 = []
with open('phi_chisweep_noisy45eta.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        wm_arr_45.append(ast.literal_eval(row['W'])[0])
        wpl_45.append(ast.literal_eval(row['min_W_prime'])[0])

# Convert lists to numpy arrays for plotting
wm_arr_45 = np.array(wm_arr_45)
wpl_45 = np.array(wpl_45)

# To plot witnesses versus chi
fig, ax = plt.subplots()
ax.plot(chi_arr, wm_arr_15, color = 'navy', label = '$W, 15$ ')
ax.plot(chi_arr, wpl_15, color = 'red', label = '$W\prime, 15$')
ax.plot(chi_arr, wm_arr_30, color = 'gray', label = '$W, 30$')
ax.plot(chi_arr, wpl_30, color = 'blue', label = '$W\prime, 30$')
ax.plot(chi_arr, wm_arr_45, color = 'purple', label = '$W, 45$')
ax.plot(chi_arr, wpl_45, color = 'green', label = '$W\prime, 45$')

ax.axhline(0, color='black', linewidth=0.5) 
ax.set_title('$\eta = 15^\degree, 30^\degree, 45^\degree$', fontsize=18)
ax.set_ylabel('Witness value', fontsize=18)
ax.tick_params(axis='both', which='major', labelsize=10)
ax.legend(ncol=2, fontsize=12, loc = 'center right')
ax.set_xlabel('$\chi$', fontsize=18)
plt.tight_layout()
plt.savefig('phi_etasweep_15_30_45_noisy.pdf')

'''
Extra code below, replace the plotting code above with it in order to plot witnesses vs. eta!
'''
# To plot witnesses versus eta
# fig, ax = plt.subplots(figsize = (10, 10))
# ax.plot(eta_arr, wm_arr, color = 'navy', label = '$W$ Witness')
# ax.plot(eta_arr, wpl, color = 'red', label = 'Min $W\prime$ Witness')
# ax.axhline(0, color='black', linewidth=0.5) 
# ax.set_title('$\chi = 45\degree$', fontsize=33)
# ax.set_ylabel('Witness value', fontsize=31)
# ax.tick_params(axis='both', which='major', labelsize=25)
# ax.legend(ncol=2, fontsize=25, loc = 'upper right')
# ax.set_xlabel('$\eta$', fontsize=31)
# plt.show()
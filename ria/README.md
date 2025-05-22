# Overview
This folder contains a variety of files and documentation used in Summer 2025 to wrap up the Summer 2024 work with the W_3 and W_5 data collection.

# Important Files
basic_qp_sweep.py: used to sweep the quartz plate to find a minimum. Intially starts by making phi_plus and measuring in the VV basis, but then moves Alice's and Bob's waveplates according to user defined inputs such that the measurement basis minimizes counts of the desired state.

basic_ratio_tuning.py: used to find the UVHWP position that creates an equal balance of the states the user inputs. Initially starts by making phi_plus and measuring in the VV basis, then moves Alice's and Bob's waveplates as well as the quartz plate to make the state. 

# Collecting Data on An Experimental state
1. Compute the offsets for Bob's waveplates to produce the desired phase shift.
2. Compute the bases for the measurement waveplates to measure in the basis minimizing counts
3. Use basic_qp_sweep.py to minimize qp and basic_ratio_tuning.py to tune the UVHWP
4. If you don't have purity data for phi_plus use pcc_sweep in calibration/phi_plus to collect data
5. Use full_tomo in framework to collect the data (I think?) MAKE SURE TO CHANGE THE SAVED FILE'S NAME!
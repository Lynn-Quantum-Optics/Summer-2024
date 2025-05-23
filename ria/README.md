# Overview
This folder contains a variety of files and documentation used in Summer 2025 to wrap up the Summer 2024 work with the W_3 and W_5 data collection.

# Important Files
basic_qp_sweep.py: used to sweep the quartz plate to find a minimum. Intially starts by making phi_plus and measuring in the VV basis, but then moves Alice's and Bob's waveplates according to user defined inputs such that the measurement basis minimizes counts of the desired state.

basic_ratio_tuning.py: used to find the UVHWP position that creates an equal balance of the states the user inputs. Initially starts by making phi_plus and measuring in the VV basis, then moves Alice's and Bob's waveplates as well as the quartz plate to make the state. 

# Collecting & Processing Data on An Experimental state
1. Compute the offsets for Bob's waveplates to produce the desired phase shift.
2. Compute the bases for the measurement waveplates to measure in the basis minimizing counts
3. Use basic_qp_sweep.py to minimize qp and basic_ratio_tuning.py to tune the UVHWP. These are located in calibration/state_calibration_code
4. If you don't have purity data for phi_plus use pcc_sweep in calibration/phi_plus to collect data
5. Located the full_tomo_and_eta_sweep_TEMPLATE in framework and create a copy for your use. Change the TODOs to match the experiment, and run this file to collect all the necessary data. Note the full_tomo_one_eta_TEMPLATE will run without the eta sweep (using the UVHWP) if you want to verify your state really quick
6. Create a copy of the process_expt_TEMPLATE and run it to process your data. All the todos in the main function should be updated match the experiment.
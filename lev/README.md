# Overview
Much of my work this summer centered around gathering and analyzing experimental data; this includes figuring out what experimental data to take in the first place. As such, my folder in the repo is divided into three sub-folders: finding_states, testing_experiment, and calibration. Below, I give more info on specific files and uses of each:

## finding_states
This folder contains code to determine what states we should gather experimental data on. 
### Important Files:
- any_state_gen.ipynb; this Jupyter notebook contains blocks to generate theoretical density matrices of any state. You can specify the density matrix (or state in the H/V/A/D/R/L bases) and return a plot. This file merges the code in testing_witnesses.py and random_gen.py into one all-encompassing Jupyter notebook. The most important and useful block is the third, 'Test a single pure or mixed state', which does as its name gives. There are also blocks allowing the testing of the W'' witnesses and for larger sweeps of states.  
- testing_mixed_states_cluster.py; this generates a csv of many creatable mixed states in which some criteria is satisfied. It is currently set to only save states for which W_min > 0 and W'_min < 0 by some margin. It is not well optimized and was created to be run on the HMC Physics Dept computer cluster. This file allowed us to determine which states will ultimately be taken experimentally and end up in poster/paper.
- rho_methods.py; this file contains all of the backend code to minimize the witnesses; it has been heavily adapted from the master rho_methods in the main repo folder to mesh well with any_state_gen.ipynb and files in the testing_experiment folder. Note it is identical to the rho_methods in the testing_experiment folder.

## testing_experiment
This folder contains code to analyze experimental data and neural net/population model performance.
### Important Files:
- process_expt_lev.py; this file processes experimental data, calculates its W/W' witness values using rho_methods.py, calculates adjusted theory, and plots all. It also saves these values to a csv file, which can be plotted separately using gen_paper_plots.py (which was used to generate plots for both the paper and poster). You can specify what theoretical state you'd like to compare against and their relative probabilites if mixing.
- process_expt_lev_nn; this file returns whether the W/W' triplet was guessed correctly for the population model and 5 layer neural network.
- gen_paper_plots.py; using the csv output of process_expt_lev.py, this nicely plots our W/W' witness values. As of writingg (7/19/2024) this is set up to generate plots for our Summer 2024 poster.
- mix_expt_data.py; here you can mix two experimentally generated states at different probabilities, outputting an .npy in the same format as the two inputs.
### Important Folders:
- all_summer2024_data; as the name implies, this contains all data we took and analyzed this summer. Important folders within include mixed_phi_psi_45 (Wp1), hrvl_havd_wp2_testmixwithbadhavddata (Wp2), and hrivl_hdiva_mix_flippedminus (Wp3). See the poster for what states these include specifically, with the naming convention of it is in the folder's name first, its file corresponds to _1, second is _2, and the mix of them is the two numbers added together. Notably the Wp1 states don't follow this, with phi being _1, psi being _26, and their mix being _27.
- old-models; contains NN5 and other models trained in Summer 2023. #note, this may not show in github as it is too large, download the folder from Oscar's README in the Summer 2023 repo and place it here!
- richard; contains the psi plus state obtained by Richard in summer 2023.
- 

## calibration
This folder contains the Tex file for the calibration/troubleshooting document I created. This pdf is identical to the one in the actual calibration folder.

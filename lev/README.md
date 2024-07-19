# Overview
Much of my work this summer centered around gathering and analyzing experimental data; this includes figuring out what experimental data to take in the first place. As such, my folder in the repo is divided into three sub-folders: finding_states, testing_experiment (analyzing experimental data for W/W' and neural net/population model performance), and calibration (tex file for calibration/troubleshooting manual I wrote). Below, I give more info on specific files and uses of each:

## finding_states
This folder contains code to determine what states we should gather experimental data on. 
### Important Files:
- any_state_gen.ipynb; this Jupyter notebook contains blocks to generate theoretical density matrices of any state. You can specify the density matrix (or state in the H/V/A/D/R/L bases) and return a plot. This file merges the code in testing_witnesses.py and random_gen.py into one all-encompassing Jupyter notebook. The most important and useful block is the third, 'Test a single pure or mixed state', which does as its name gives. There are also blocks allowing the testing of the W'' witnesses and for larger sweeps of states.  
- testing_mixed_states_cluster.py; this generates a csv of many creatable mixed states in which some criteria is satisfied. It is currently set to only save states for which W_min > 0 and W'_min < 0 by some margin. It is not well optimized and was created to be run on the HMC Physics Dept computer cluster. This file allowed us to determine which states will ultimately be taken experimentally and end up in poster/paper.
- rho_methods.py; this file contains all of the backend code to minimize the witnesses; it has been heavily adapted from the master rho_methods in the main repo folder to mesh well with any_state_gen.ipynb and files in the testing_experiment folder. Note it is identical to the rho_methods in the testing_experiment folder.

## testing_experiment
This folder contains code to analyze experimental data and neural net/population model performance.
### Important Files:
- process_expt_lev.py; this file processes experimental data, calculates its W/W' witness values using rho_methods.py, calculates adjusted theory, and plots all. It also saves these values to a csv file, which can be plotted separately using gen_paper_plots.py (which was used to generate plots for both the paper and poster).

## calibration
This folder contains the Tex file for the calibration/troubleshooting document I created. 

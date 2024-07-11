# Summer-2024
This repository is for the work done over Summer of 2024 in Prof. Lynn's quantum optics lab.

This file contains contact information for the research group members, as well as executive summaries of each project, and only contains brief summaries, contact information for  and instructions for finding more information.

## Contact Information

- Paco Navarro : pnavarro@g.hmc.edu
- Lev Gruber   : lgruber@g.hmc.edu, 510-227-4602
- Stuart Kerr  : stkerr@g.hmc.edu

## Experimental Data Analysis

Contact: Lev

The folder 'lev/testing_experiment' contains code to analyze experimental data for witness values, to test both the population model and the group's neural net, and to mix experimentally derived pure states into mixed states. More info in my folder.

## Calibration / Guide

Contact: Lev

The folder 'calibration' contains a guide to calibrating the experimental apparatus, troubleshooting bugs, and using the Manager. The folder also includes code to calibrate each apparatus element, much of this written by Alec Roberson and Richard Zheng, and adapted by us this summer.

## Finding Theoretical States 

Contact: Lev

The folder 'lev/finding_states' is dedicated to finding states for which a certain witness or witness group (i.e. the Ws or W's) and are creatable by our experimental apparatus. More info in my folder.

## Obtaining Experimental Data

Contact:  Stuart Kerr

The folder 'framework' contains files to generate all of the data we obtained this summer. Please do not edit this folder, and if you'd like to use the files ensure they do not overwrite currently saved data!

## Machine Learning

Contact: Paco Navarro

The folder 'oscar/machine_learning' contains code for the generation of both the five-layer neural net and large set of random states. Specifically, ml_hypersweep.py creates the neural net and was used for hyperparameter optimization and master_datagen.py created the set of 2400000 random states we use for neural net creation, validation, and testing. This is largely unchanged from Oscar Scholin's original code in summer 2023.


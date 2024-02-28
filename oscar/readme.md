Hi there! :) This summer's work can be split into 2 main parts: the entanglement witnessing theory and experimental.

For entanglement witnessing theory, in the `./machine_learning/` directory you'll find most of the important things. At the heart of everything is `rho_methods.py`, which contains a whole load of functions that have to do with density matrices: e.g., if a matrix is a density matrix, how to calculate its purity, concurrence, Stokes parameters, witness values, etc. 'adjust_E0_rho_general' allows for general correction to $E_0$ ,states, but the $x$ parameter must be determined using the gradient descent method 'det_noise()' in 'process_expt.py'.
- To generate random density matrices, you can see the methods in `random_gen.py` or `roik_datagen.py`. The file 'master_datagen.py' integrates all choices of random generation as well as the target quantities to be calculated, so is highly recommended for generating dataframes. Note that I used the Roik style of generation for all mjor datasets, but not the Roik-style of probabilities, which are conditional; unless you are doing something specific with the Roik problem, I recommend using the standard projective probabilities. Note that my simulated data, along with all trained machine learning models, is stored in Google Drive in 'random_gen.zip': https://drive.google.com/drive/folders/13O0Ghobwz5DALPYB19CUrPLoxY5wxI7P?usp=sharing. The sub-folder 'models/saved_models' has all of the best model files I generated this summer and used to make the plots of model performance, in conjunction with the file 'ml_comp.py', which supercedes 'ml_evaluate.py'.* For last year's models, see 'old_models'. Last semester's data can be found on the same Google drive link as 'S22_data'. In `ml_comp.py', use ``plot_comp_acc_simple''' to generate plots of performance for paper/poster. The processed data that was used in the plot is available in the same google drive link, under plot_data/. All calculations for accuracy are done in the same py file with ````eval_perf```.
- Roik et al source code and models can be found in 'roik_etal'.
- To access pre-defined density matrices, you can see `sample_rho.py`: this has the 4 Bell states and Eritas's suggested states.
- For training and testing the NN and XGBoost, see `ml_hypersweep.py`. This allows you to perform a hyperparameter sweep over models using the WandB API. The file 'train_prep' automatically prepares the formatting of calculated data for training, and is integrated into 'ml_hypersweep.py'.`bl_nn_original.py' is from Becca and Laney's work last semester, and 'bl_nn_o.py' is from my work at the beginning of the summer, in particular on XGBoost.
- For visualizing distributions of min eigenvals of the partial transpose and witness values, see `data_analyze.py`.
- In 'oscar/machine_learning/', I also have some Matlab files adapted directly from Eritas Yang: 'Concurrence.m', 'Purity.m', 'MinEig.m', 'findW.m', 'vars.m', 'Random.m', and 'random_load.m' which synthesizes all of these files to generate dataframes matching those in Python.

For reference, some stats on model performance. 1 indicates argmax only for choosing which W', 2 allows for 2nd best.
    Accuracy at 0 concurrence:
    NN5, choice 1: 0.8485454215056251
    NN5, choice 1+2: 0.968075264731634
    Population: 0.7917416946128558
    NN2, choice 1: 0.8093436637866949
    NN2, choice 1+2: 0.9388597954834553

    Fraction of undetected states at 0 concurrence:
    NN5, choice 1: 0.2771518118475924
    NN5, choice 1+2: 0.22360489770544945
    Population: 0.3025987145188326
    NN2, choice 1: 0.29471339394410345
    NN2, choice 1+2: 0.236692827725542
    W: 0.6572827281507054
    W': 0.20930327216751954

On the experimental side,
- In 'oscar/machine_learning', the file `jones.py` has all the code to compute the rotated/translated polarization state as well as compute the gradient descent optimization to determine the settings for the states (InstaQ). The file 'fits_for_jones.py' was used to determine fits of experimental parameters used in the Jones decomposition. 'jones_expt.py' has a gradient descent for an a posteriori determination of phi as a function of the UVHWP and QP angles
- 'process_expt.py' has all the code to process the experimental data, including the gradient descent to determine the noise parameters based on the different models as described in the writeup. This method is called 'det_noise'. Another key method is 'analyze_rhos()', which creates a dataframe of the witness values for the experimental, theoretical, and adjusted theoretical states, in addition to purities and fidelities, which the method 'make_plots_E0()' turns into plots. 'comp_rho_rhoadj()' allows one to compare the magnitude and phase as well as Stokes parameters for an individual experimetnal state. 'comp_w_adj()' allows for the graphs for all unique eta values for E0 states what the witness values are as a function of chi.
- 'process_expt_richard.py' does the same as the above, except adapted for Richard's file formatting and was used to produce the plots on the SQuInT poster.
- The folder 'framework/decomp_test' from the main directory has all the experimental states I measured as 'rho_statename_trialnum.npy'. The file 'decomp_exp.py' in this folder was used to perform these measurements; moreover, I made use of 'sweep_fits.py' in the same folder written by Alec when sweeping QP vs phi.

The writeup for the summer as well as a report on the InstaQ algorithm can be found in the folder 'writing'. The SQInT poster is in the main directory.

As a a brief aside, for maximal LELM distinguishability, see the `./nogo/` directory. The most up-to-date file is 'nogo4.py', which attempts to generate the systems of equations following the criteria in the Lock and Lutkenhaus paper to aid in the effort to solve the systems, ultimately for d=6. Currently this file (and thus my math) is still broken :( but I'm working on it! Note: my thesis code has migrated to the Fall 2023 -- Spring 2024 directory.
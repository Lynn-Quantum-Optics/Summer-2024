This summer's work can be split into 2 main parts: the entanglement witnessing and maximal LELM distinguishability.

For entanglement witnessing, in the `./machine_learning/` directory you'll find most of the important stuff. At the heart of everything is `rho_methods.py`, which contains a whole load of functions that have to do with density matrices: e.g., if a matrix is a density matrix, how to calculate its purity, how to calculate concurrence, etc. 
- To generate random density matrices, you can see the methods in `jones_simplex_datagen.py` or `roik_datagen.py`. The file `jones.py` has all the code to compute the rotated/translated polarization state.
- For visualizing distributions of min eigenvals of the partial transpose and witness values, see `data_analyze.py`.
- To access pre-defined density matrices, you can see `sample_rho.py`: this has the 4 Bell states and I plan to add more.
- For training and testing the NN and XGBoost, see `ml_hypersweep.py`. This allows you to perform a hyperparameter sweep over models using the WandB API.

For maximal LELM distinguishability, see the `./nogo/` directory. The most up-to-date file is 'nogo4.py', which attempts to generate the systems of equations following the criteria in the Lock and Lutkenhaus paper to aid in the effort to solve the systems, ultimately for d=6. Currently this file (and thus my math) is still broken :( but I'm working on it!
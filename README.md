# Noise estimation GenUT
 Journal paper about noise estimation for nonlinear state estimators.

 The files are located in "scripts". A description of each file:
 - plot_batch_gasreactor.py: prints rmse and simulation time statistics based on the loaded data.
 - toy_example_splitting_ut_in_two.py: runs the toy example in Section 5.1 in the paper.
 - main_batch_gasreactor.py: Runs the simulations for the batch gasreactor. The number of simulations (100 were used in the paper), types of tuning methods for the filters (GenUT, linearization and MC) and the case can be chosen. This file also plots state trajectories when finished. If N_sim < 5, you will get an error since statistics for rmse and simulation time are not calculated then (but you will still see the plot of the state trajectories)
 - utils_batch_gasreactor.py: No need to run this, everything is called from main_batch_gasrector.py. A lot of utilities/functions are stored here (e.g. the model).
 - state_estimator: A folder containing different files related to the state estimator. No need to run this, everything is called from main_batch_gasrector.py or toy_example_splitting_ut_in_two.py

The data_ folders are:
- data_gasreactor: the latest simulation from main_batch_gasreactor.py is saved here.
- data_gasreactor_article_values: data used in the article for the gasreactor case study is saved here.
- data_toy_example: data for the toy example/case study of is saved here.

Everardo Villasenor (evillasenor3)
Assignment 4
Code found at: https://github.com/everardoiv/Markov_Decision_Processes
----------------------------------------

This code was built on Python 3.6
The following packages are required to run the experiments:
- Numpy
- Pandas
- csv
- matplotlib
- sklearn
- scipy
- time
- mdptoolbox-hiive
- gym
----------------------------------------

In order to run this code you will need to be using a Python 3.6 environment with the list of dependencies having been installed (pip install X).

The file structure used was the following:
- frozen-lake-experiment.py
- basic_rl.py
- policy_iteration.py
- qlearner.py
- frozen-lake-alternate.py
- forest_experiment.py
- run_q_learner.sh
- plots/
-- forest_experiment
-- frozen_lakes

In order to run the code you must change directories to the parent folder and run (example):
 - VI and PI the following:
  python frozen-lake-experiment.py
  python forest_experiment.py

 - Q Learning for Frozen Lake:
 ./run_q_learner.sh
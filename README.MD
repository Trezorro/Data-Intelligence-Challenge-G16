# Data Intelligence Challenge solution

## Setup
We used pipenv for setting up our virtual environment. It fully specifies dependencies and subdependencies 
in the Pipfile and Pipfile.lock.

For documentation see https://pipenv.pypa.io/en/latest/ 

### Steps:
1. Activate or install a Python 3.8 environment and install pipenv in it.
   - For instance using conda: `conda create -n py38 python=3.8.12 pipenv -c conda-forge`
   - Don't forget to activate it or add it to your path variable.
2. Replicate the project environment by running `pipenv install` in the project folder.
3. Activate the project environment with `pipenv shell` or configure your IDE to use the new pipenv environment (with the project folder name) as the python environment.

Note, that the C++ build tools on your system are required for using this project.

## Headless

The `/Discrete-Simulations/headless.py` script allows for running many simulations without the use of the UI.
This script is not suited for testing the greedy-bot, but is intended for simulating the policy-iteration and value-iteration algorithms.
One can change between policy-iteration and value-iteration by changing the import in line 2. In addition, parameter settings can be specified such as which grids to simulate on and which gamma, theta and p_move values to test on.
In the end, an overview excel file will be generated with semi aggregated results, as well as an excel file with the results of all individual runs.

## Notes

Cel material | Label
-------------|-------
Obstacle (gray)    | -2
Wall (red)         | -1
Clean (green)      |  0
Dirty (white)      |  1
Goal (orange)      |  2
Death (red cross)  |  3
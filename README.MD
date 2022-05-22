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
This script intends to simulate the Monte Carlo, Sarsa, QLearning algorithms and it includes two grid options to apply the algorithms, the 'stay-off-my-grass' grid and the 'experiment-house' grid. One can run the simulations by following the steps shown below.  After running each simulation, a folder named 'output' is generated in the same path as the `headless.py` file. The folder includes 1 excel file for every simulation that is named after the time and date the simulation started running and it consist of the results of the simulation. This folder also includes another folder named 'history' that consist of the path that the robot followed in every simulation and helps in generating the heatmap graph described in the report.

### Steps for running simulations:
1. Open the `headless.py` file
2. Go to line 34 and change the value of `ROBOT_MODULE_NAME` with one of the values sarsa_robot, monte_carlo_robot, q_learning_robot based on which robot you want to use.
3. Go to lines 35 - 38 and uncomment only the grid that you want to include in your simulations, either the 'experiment_house.grid' or the 'stay_off_my_grass.grid'.
4. Note that in line 26, the number of workers is indicated. You can increase or decrease it based on your computer's capacity. 
5. Save the changes and run the code to see the results (approximately 2 hours for every simulation for a '4 cores and 8 threads' computer).

## Notes

Cel material | Label
-------------|-------
Obstacle (gray)    | -2
Wall (red)         | -1
Clean (green)      |  0
Dirty (white)      |  1
Goal (orange)      |  2
Death (red cross)  |  3

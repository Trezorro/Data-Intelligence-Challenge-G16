# Data Intelligence Challenge solution

## Continuous Simulation Setup
We recommend using (mini)conda for installing the dependencies in a separate new environment. This makes installing Pytorch on CUDA and other libraries a little easier. Installing dependencies may be done using `pip install -r requirements.txt`, but we provide no guarantees for this method.


### Steps:
1. Install MPI
   - For windows: Get the .exe from [Microsoft MPI](https://www.microsoft.com/en-us/download/details.aspx?id=100593). Run it, and check whether MSI was added to your system path variable.
   - For Ubuntu: get OpenMPI `sudo apt-get update && sudo apt-get install libopenmpi-dev`
   - For Max OSX: get OpenMPI `brew install openmpi`
   - On some system configurations, you can install OpenMPI using conda `conda install openmpi`. Not guaranteed to work.
2. Replicate the conda environment using:
   1. `conda env create -f conda_env.yaml`
   2. `conda activate dic-env`
3. Install the spinup module ant it's dependencies using: `pip install -e spinningup`
4. Check the installation by running:
   - To check the vacuum robot environment: `python .\Continuous-Simulations\runner.py`

**Note** that the C++ build tools on your system are required for using this project.

**Note (2)** that your path variable will now have the `<path>\swig-<version>` and `<path>\Microsoft MPI\bin`. 
On windows, your path variable can be set by searching for `edit the system environment variables` -> `environment variables` -> Find `Path` variable -> `Edit` -> Add the paths.

## Headless

The `/Discrete-Simulations/headless.py` script allows for running many simulations without the use of the UI.
This script intends to simulate the Monte Carlo, Sarsa, QLearning algorithms and it includes two grid options to apply the algorithms, the 'stay-off-my-grass' grid and the 'experiment-house' grid. One can run the simulations by following the steps shown below.  After running each simulation, a folder named 'output' is generated in the same path as the `headless.py` file. The folder includes 1 excel file for every simulation that is named after the time and date the simulation started running and it consist of the results of the simulation. This folder also includes another folder named 'history' that consist of the path that the robot followed in every simulation and helps in generating the heatmap graph described in the report.

### Steps for running simulations:
1. Open the `headless.py` file
2. Go to line 34 and change the value of `ROBOT_MODULE_NAME` with one of the values sarsa_robot, monte_carlo_robot, q_learning_robot based on which robot you want to use.
3. Go to lines 35 - 38 and uncomment only the grid that you want to include in your simulations, either the 'experiment_house.grid' or the 'stay_off_my_grass.grid'.
4. Note that in line 26, the number of workers is indicated. You can increase or decrease it based on your computer's capacity. 
5. Save the changes and run the code to see the results (approximately 2 hours for every simulation for a '4 cores and 8 threads' computer).

## Environment

The environment and UI were changed with respect to the previous version. To start, the grid object was transposed. 
Therefore, all calls to the numpy array inside the grid objects are now indexed first on rows (y) and then on column (x). 
The visualization was also changed to match the stored numpy array as to have an identical visualization to what is stored in memory (and can be seen in debug mode).
As the grid (and visualization) we're changed, the directions corresponding to the robot orientations were also changed. 
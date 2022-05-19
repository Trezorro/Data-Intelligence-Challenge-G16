from datetime import datetime
import importlib
import itertools
import logging
import pickle
import time
from typing import Type, Union
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from environment import RobotBase
from helpers.td_robot import TDRobotBase

logging.basicConfig(level=logging.WARNING, force=True)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('werkzeug').setLevel('WARNING')

# TODO: specify n-cores in command for parallelization
## ---------- Experiment Settings ---------- ##
OUTPUT_FOLDER = 'output'
RUN_NAME = 'experiment_run'

ROBOT_MODULE_NAME = 'a_sarsa_robot'
GRID_FILES = [
    'experiment_house.grid',
    'stay_off_my_grass.grid',
]
P_MOVES = [0, 0.2]
GAMMAS = [0.2, 0.5, 0.9]
THETAS = [0.1, 0.01]
LEARNING_RATES = [0.99, 0.5]
REPEATS = range(10)
INCLUDED_PARAMETERS = dict(
    # parameter name: [iterable values]
    # comment out what you don't need for the current robot type!
    grid=GRID_FILES,
    p_move=P_MOVES,
    gamma=GAMMAS,
    # theta=THETAS,
    lr=LEARNING_RATES, 
    repeat=REPEATS
)
STOPPING_CRITERIA = 100  # tile percentage at which the room is considered 'clean'

OUTPUT_VALUE_NAMES = ['efficiency', 'cleaned', 'battery', 'dead', 'n_moves', 'time', 'error']

# Initialize csv:
run_filename=f'{datetime.now():%b-%d_%H-%M (%Ss)} - {RUN_NAME}.csv'
run_out_path = Path(OUTPUT_FOLDER, run_filename)
run_history_out_path = Path(OUTPUT_FOLDER, 'histories', run_filename)
if run_out_path.exists():
    raise FileExistsError(f'{run_out_path} already exists!')
else:
    run_out_path.parent.mkdir(parents=True, exist_ok=True)
    run_history_out_path.parent.mkdir(parents=True, exist_ok=True)
    run_out_path.touch()
    run_history_out_path.touch()

with open(run_out_path, 'a') as f:
    f.write(','.join(list(INCLUDED_PARAMETERS.keys()) + OUTPUT_VALUE_NAMES ) + '\n')
with open(run_history_out_path, 'a') as f:
    f.write(','.join(list(INCLUDED_PARAMETERS.keys()))+ ",moves_list\n")

# Dynamically load correct bot class:
robot_module = importlib.import_module('robot_configs.'+ROBOT_MODULE_NAME.split('.py')[0])
if not (hasattr(robot_module, 'Robot') or hasattr(robot_module, 'robot_epoch')):
            raise ImportError(f"No Robot class or robot_epoch function found in {ROBOT_MODULE_NAME}!")
RobotClass: Union[Type[RobotBase], Type[TDRobotBase]] = getattr(robot_module, 'Robot', RobotBase)

try:
    for parameter_tuple in itertools.product(*INCLUDED_PARAMETERS.values()):
        # TODO: skip at random (random search)
        # Initialize statistics:
        parameters = dict(zip(INCLUDED_PARAMETERS.keys(), parameter_tuple))
        print("Starting experiment with parameters:\n", str(parameters).replace(',',',\n'))
        recorded_values = {}
        efficiency = None
        clean_percent = None
        moves = []
        dead = 0

        # Open the grid file.
        with open(f"grid_configs/{parameters['grid']}", 'rb') as f:
            grid = pickle.load(f)

        # Calculate the total visitable tiles:
        n_total_tiles = (grid.cells >= 0).sum()

        # Spawn the robot at (1,1) facing north with battery drainage enabled:
        robot = RobotClass(grid, (1, 1),
                            orientation='e',
                            battery_drain_p=1,
                            battery_drain_lam=2,
                            p_move=parameters['p_move'],
                            gamma=parameters['gamma'],
                            lr=parameters['lr'],
                            number_of_episodes=10
                            )
        if not getattr(robot, 'is_trained', True):
            # If this robot can, and should, be trained:
            robot.train()

        # Keep track of the number of robot decision epochs:
        n_epochs = 0
        while True:
            if not robot.alive:
                if robot.battery_lvl <= 0:
                    dead = 1 # record empty battery death
                break
            start_time = time.time()
            n_epochs += 1
            # Advance a time step and try to make the robot move:
            if hasattr(robot_module, 'robot_epoch'):
                # Use legacy robot_epoch function if available:
                getattr(robot_module, 'robot_epoch')(robot)
            else:
                robot.robot_epoch()  # Use class method otherwise
                # TODO phase this out, pass all parameters to Robot initalization

            # Calculate some statistics:
            clean = (grid.cells == 0).sum()
            dirty = (grid.cells == 1).sum()
            goal = (grid.cells == 2).sum()
            clean_percent = (clean / (dirty + clean)) * 100
            # if int(clean_percent) % 10 == 0 and int(clean_percent) != 0:
            #     cnt += 1
            #     time_tracker.loc[cnt, "Simulation"] = i + 1
            #     time_tracker.loc[cnt, "Percent"] = int(clean_percent)
            #     time_tracker.loc[cnt, "Time"] = round(time.time() - start_time, 2)
            # See if the room can be considered clean, if so, stop the simulaiton instance:
            if clean_percent >= STOPPING_CRITERIA and goal == 0:
                break
            # Calculate the effiency score:
            moves = [(x, y) for (x, y) in zip(robot.history[0], robot.history[1])]
            u_moves = set(moves)
            n_revisted_tiles = len(moves) - len(u_moves)
            efficiency = (100 * n_total_tiles) / (n_total_tiles + n_revisted_tiles)
        # Keep track of the last statistics for each simulation instance:


        recorded_values['efficiency'] = efficiency
        recorded_values['cleaned'] = clean_percent
        recorded_values['battery'] = robot.battery_lvl
        recorded_values['dead'] = dead
        recorded_values['n_moves'] = len(moves)
        recorded_values['time'] = round(time.time() - start_time, 2)
        recorded_values['error'] = 0 # experiment finished successfully

        # TODO: retrieve history of robot and write to csv
        
        # grouped_time = time_tracker[["Percent", "Time"]].groupby("Percent")
        # mean_time = grouped_time.mean()
        # mean_time = mean_time.reset_index()

        output_values= list(parameter_tuple) + [recorded_values[key] for key in OUTPUT_VALUE_NAMES]
        with open(run_out_path, 'a') as f:
            f.write(','.join(repr(v) for v in output_values) + '\n')
        with open(run_history_out_path, 'a') as f:
            f.write(','.join(repr(v) for v in parameter_tuple) + ",'" + repr(moves) + "'\n")
except (KeyboardInterrupt, SystemExit):
    print("Process interrupted!")
except Exception as e:
    print('Exception:', e)
    raise
else:
    print('Successfully completed all experiments.')
finally:
    print("Writting results to file...")



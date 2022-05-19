import concurrent.futures as cf
import importlib
import itertools
import logging
import pickle
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Type, Union
from functools import partialmethod

import pandas as pd
import tqdm

from environment import RobotBase
from helpers.td_robot import TDRobotBase

logging.basicConfig(level=logging.WARNING, force=True)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('werkzeug').setLevel('WARNING')

tqdm.tqdm.__init__ = partialmethod(tqdm.tqdm.__init__, disable=True)

## ---------- Experiment Settings ---------- ##
OUTPUT_FOLDER = 'output'
RUN_NAME = 'experiment_run'
N_WORKERS = 8 # None, or int in [1,63]

ROBOT_MODULE_NAME = 'q_learning_robot'
GRID_FILES = [
    # 'example-random-house-0.grid',
    'stay_off_my_grass.grid',
    # 'snake.grid',
    # 'snake.grid'
        ]
P_MOVES = [0, 0.2]
GAMMAS = [0.2, 0.5, 0.9]
THETAS = [0.1, 0.01]
LEARNING_RATES = [0.99, 0.5]
REPEATS = range(5)
INCLUDED_PARAMETERS = dict(
    # parameter name: [iterable values]
    # comment out what you dont need for the current robot type!
    grid=GRID_FILES,
    p_move=P_MOVES,
    gamma=GAMMAS,
    # theta=THETAS,
    lr=LEARNING_RATES, 
    repeat=REPEATS
)
STOPPING_CRITERIA = 100  # tile percentage at which the room is considered 'clean'

OUTPUT_VALUE_NAMES = ['efficiency', 'cleaned', 'battery', 'dead', 'n_moves', 'time', 'error']


# Dynamically load correct bot class:
robot_module = importlib.import_module('robot_configs.'+ROBOT_MODULE_NAME.split('.py')[0])
if not (hasattr(robot_module, 'Robot') or hasattr(robot_module, 'robot_epoch')):
        raise ImportError(f"No Robot class or robot_epoch function found in {ROBOT_MODULE_NAME}!")
RobotClass: Union[Type[RobotBase], Type[TDRobotBase]]  = getattr(robot_module, 'Robot', RobotBase)

def run_experiment(parameter_tuple: tuple, experiment_number: Optional[int] = None) -> Tuple[Optional[int],str,str]:
    global robot_module, RobotClass, INCLUDED_PARAMETERS, STOPPING_CRITERIA, OUTPUT_VALUE_NAMES
    parameters = dict(zip(INCLUDED_PARAMETERS.keys(), parameter_tuple))
    # Initialize statistics:
    recorded_values = {}
    efficiency = None
    clean_percent = None
    moves = []
    dead = 0

    # Open the grid file.
    with open(f"grid_configs/{parameters['grid']}", 'rb') as f:
        grid = pickle.load(f)
    if not hasattr(grid, 'transposed_version'): # adapt to new grid format
        grid.cells = grid.cells.T

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
                            number_of_episodes=1000
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
        # See if the room can be considered clean, if so, stop the simulaiton instance:
        if clean_percent >= STOPPING_CRITERIA and goal == 0:
            break
        # Calculate the effiency score:
        moves = [(x, y) for (x, y) in zip(robot.history[0], robot.history[1])]
        u_moves = set(moves)
        n_revisted_tiles = len(moves) - len(u_moves)
        efficiency = (100 * n_total_tiles) / (n_total_tiles + n_revisted_tiles)

    recorded_values['efficiency'] = efficiency
    recorded_values['cleaned'] = clean_percent
    recorded_values['battery'] = robot.battery_lvl
    recorded_values['dead'] = dead
    recorded_values['n_moves'] = len(moves)
    recorded_values['time'] = round(time.time() - start_time, 2)
    recorded_values['error'] = 0 # experiment finished successfully

    output_values= list(parameter_tuple) + [recorded_values[key] for key in OUTPUT_VALUE_NAMES]
    measurement_line = ','.join(repr(v) for v in output_values) + '\n'
    moves_line = ','.join(repr(v) for v in parameter_tuple) + ",'" + repr(moves) + "'\n"
    return experiment_number, measurement_line, moves_line

if __name__ == '__main__':
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

    print("Firing up the pool of workers...")
    try:
        with cf.ProcessPoolExecutor(N_WORKERS) as executor:
            # TODO: skip some tuples at random (random search)
            futures = []
            for exp_idx, parameter_tuple in enumerate(itertools.product(*INCLUDED_PARAMETERS.values())):
                print(f"Starting experiment {exp_idx:#>4} with parameters:  ", 
                      str(dict(zip(INCLUDED_PARAMETERS.keys(), parameter_tuple))))
                futures.append(executor.submit(run_experiment, parameter_tuple, exp_idx))
            for result in cf.as_completed(futures):
                exp_idx, measurement_line, moves_line = result.result()
                with open(run_out_path, 'a') as f:
                    f.write(measurement_line)
                with open(run_history_out_path, 'a') as f:
                    f.write(moves_line)
                print(f"Finished experiment {exp_idx:#>4}!")
                
    except (KeyboardInterrupt, SystemExit):
        print("Process interrupted!")
    except Exception as e:
        print('Exception:', e)
        raise
    else:
        print('Successfully completed all experiments.')
    finally:
        print("Writting results to file...")



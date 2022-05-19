# Import our robot algorithm to use in this simulation:
import importlib
from typing import Type
import pickle
from environment import RobotBase
import pandas as pd
import time
import numpy as np
from tqdm import tqdm


import logging
logging.basicConfig(level=logging.WARNING, force=True)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('werkzeug').setLevel('WARNING')

# TODO: use product for grid search
# TODO: specify n-cores in command

# Settings
STOPPING_CRITERIA = 100  # tile percentage at which the room is considered 'clean'
ROBOT_MODULE_NAME = 'a_sarsa_robot'
GRID_FILES = [
    # 'example-random-house-0.grid',
    # 'stay_off_my_grass.grid',
    # 'snake.grid',
    'snake.grid'
        ]
GAMMAS = [0.2, 0.5, 0.9]
THETAS = [0.1, 0.01]
P_MOVES = [0, 0.2]

cleaned_means = []
cleaned_variances = []
efficiencies_means = []
efficiencies_variances = []
experiments = []

# Dynamically load correct bot class:

robot_module = importlib.import_module('robot_configs.'+ROBOT_MODULE_NAME.split('.py')[0])
if not (hasattr(robot_module, 'Robot') or hasattr(robot_module, 'robot_epoch')):
        raise ImportError(f"No Robot class or robot_epoch function found in {ROBOT_MODULE_NAME}!")
RobotClass: Type[RobotBase] = getattr(robot_module, 'Robot', RobotBase)

# Run 100 times:
big_df = pd.DataFrame(columns=['Clean', 'Efficiency', 'Experiment'])

try:
    for grid_file in GRID_FILES:
        for p_move_param in P_MOVES:
            for theta in THETAS:
                for gamma in GAMMAS:
                    # Keep track of some statistics:
                    efficiencies = []
                    n_moves = []
                    deaths = 0
                    cleaned = []
                    experiment_str = 'THETA ' + str(theta) + ' | GAMMA ' + str(gamma) + ' | p_move ' + str(p_move_param) \
                                    + ' | GRID_FILE ' + grid_file

                    time_tracker = pd.DataFrame(columns=["Simulation", "Percent", "Time"])
                    cnt = 0
                    experiment = []

                    print(grid_file, theta, gamma)
                    for i in tqdm(range(10)):
                        # Open the grid file.
                        # (You can create one yourself using the provided editor).
                        with open(f'grid_configs/{grid_file}', 'rb') as f:
                            grid = pickle.load(f)

                        # Calculate the total visitable tiles:
                        n_total_tiles = (grid.cells >= 0).sum()

                        # Spawn the robot at (1,1) facing north with battery drainage enabled:
                        robot = RobotClass(grid, (1, 1),
                                           orientation='e',
                                           battery_drain_p=1,
                                           battery_drain_lam=2,
                                           p_move=p_move_param)
                        if not getattr(robot, 'is_trained', True):
                            # If this robot can, and should, be trained:
                            robot.train()

                        # Keep track of the number of robot decision epochs:
                        n_epochs = 0
                        while True:
                            if not robot.alive:
                                if robot.battery_lvl <= 0:
                                    deaths += 1 # record empty battery death
                                break
                            start_time = time.time()
                            n_epochs += 1
                            # Advance a time step and try to make the robot move:
                            if hasattr(robot_module, 'robot_epoch'):
                                # Use legacy robot_epoch function if available:
                                getattr(robot_module, 'robot_epoch')(robot)
                            else:
                                robot.robot_epoch()  # Use class method otherwise

                            # Calculate some statistics:
                            clean = (grid.cells == 0).sum()
                            dirty = (grid.cells == 1).sum()
                            goal = (grid.cells == 2).sum()
                            clean_percent = (clean / (dirty + clean)) * 100
                            if int(clean_percent) % 10 == 0 and int(clean_percent) != 0:
                                cnt += 1
                                time_tracker.loc[cnt, "Simulation"] = i + 1
                                time_tracker.loc[cnt, "Percent"] = int(clean_percent)
                                time_tracker.loc[cnt, "Time"] = round(time.time() - start_time, 2)
                            # See if the room can be considered clean, if so, stop the simulaiton instance:
                            if clean_percent >= STOPPING_CRITERIA and goal == 0:
                                break
                            # Calculate the effiency score:
                            moves = [(x, y) for (x, y) in zip(robot.history[0], robot.history[1])]
                            u_moves = set(moves)
                            n_revisted_tiles = len(moves) - len(u_moves)
                            efficiency = (100 * n_total_tiles) / (n_total_tiles + n_revisted_tiles)
                        # Keep track of the last statistics for each simulation instance:
                        efficiencies.append(float(efficiency))
                        n_moves.append(len(robot.history[0]))
                        cleaned.append(clean_percent)
                        # Change according to the current experiment
                        experiment.append(experiment_str)

                        grouped_time = time_tracker[["Percent", "Time"]].groupby("Percent")

                        mean_time = grouped_time.mean()

                        mean_time = mean_time.reset_index()

                    df = pd.DataFrame(columns=['Clean', 'Efficiency', 'Experiment'])
                    df['Clean'] = cleaned
                    df['Efficiency'] = efficiencies
                    df['Experiment'] = experiment

                    cleaned_means.append(np.mean(cleaned))
                    cleaned_variances.append(np.var(cleaned))
                    efficiencies_means.append(np.mean(efficiencies))
                    efficiencies_variances.append(np.var(efficiencies))
                    experiments.append(experiment_str)

                    big_df = pd.concat([big_df, df])
except (KeyboardInterrupt, SystemExit):
    print("Process interrupted!")
except Exception as e:
    print('Exception:', e)
    raise
else:
    print('Successfully completed all experiments.')
finally:
    print("Writting results to file...")
    big_df.to_excel(f"policy_iteration_experiment_more_drain_{GRID_FILES[0]}.xlsx", index=False)

    average_df = pd.DataFrame(
        columns=['Clean mean', 'Clean variance', 'Efficiency mean', 'Efficiency variance', 'Experiment'])
    average_df['Clean mean'] = cleaned_means
    average_df['Clean variance'] = cleaned_variances
    average_df['Efficiency mean'] = efficiencies_means
    average_df['Efficiency variance'] = efficiencies_variances
    average_df['Experiment'] = experiments

    average_df.to_excel(f"Overview_policy_iteration_experiment_aggregated_more_drain_{GRID_FILES[0]}.xlsx", index=False)

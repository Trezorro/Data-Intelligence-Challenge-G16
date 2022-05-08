# Import our robot algorithm to use in this simulation:
from robot_configs.value_iteration_robot import robot_epoch
import pickle
from environment import Robot
import matplotlib.pyplot as plt
import pandas as pd
import time
import seaborn as sns
import numpy as np
from tqdm import tqdm
from openpyxl import Workbook, load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# Cleaned tile percentage at which the room is considered 'clean':
stopping_criteria = 100

# Settings
GRID_FILES = ['snake.grid', 'house.grid', 'example-random-house-0.grid']
GAMMAS = [0.2, 0.5, 0.9]
THETAS = [0.1, 0.01, 0.001]
P_MOVES = [0, 0.2]

cleaned_means = []
cleaned_variances = []
efficiencies_means = []
efficiencies_variances = []
experiments = []

# Run 100 times:
big_df = pd.DataFrame(columns=['Clean', 'Efficiency', 'Experiment'])

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
                for i in tqdm(range(20)):
                    # Open the grid file.
                    # (You can create one yourself using the provided editor).
                    with open(f'grid_configs/{grid_file}', 'rb') as f:
                        grid = pickle.load(f)

                    # Calculate the total visitable tiles:
                    n_total_tiles = (grid.cells >= 0).sum()

                    # Spawn the robot at (1,1) facing north with battery drainage enabled:
                    robot = Robot(grid, (1, 1), orientation='n', battery_drain_p=0.5, battery_drain_lam=2,
                                  p_move=p_move_param)

                    # Keep track of the number of robot decision epochs:
                    n_epochs = 0
                    while True:
                        start_time = time.time()
                        n_epochs += 1
                        # Do a robot epoch (basically call the robot algorithm once):
                        robot_epoch(robot, gamma, theta)
                        # Stop this simulation instance if robot died :( :
                        if not robot.alive:
                            deaths += 1
                            break
                        # Calculate some statistics:
                        clean = (grid.cells == 0).sum()
                        dirty = (grid.cells == 1).sum() # edited to only include actual dirty cells
                        goal = (grid.cells == 2).sum()
                        # Calculate the cleaned percentage:
                        clean_percent = (clean / (dirty + clean)) * 100
                        if int(clean_percent) % 10 == 0 and int(clean_percent) != 0:
                            cnt += 1
                            time_tracker.loc[cnt, "Simulation"] = i + 1
                            time_tracker.loc[cnt, "Percent"] = int(clean_percent)
                            time_tracker.loc[cnt, "Time"] = round(time.time() - start_time, 2)
                        # See if the room can be considered clean, if so, stop the simulaiton instance:
                        if clean_percent >= stopping_criteria and goal == 0:
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

                # If the value_iteration or policy_iteration excel files are already generated the uncomment the below

                # wb = load_workbook('policy_iteration.xlsx')
                # ws = wb['Sheet1']
                #
                # # Change the header to True for the first iteration
                # for r in dataframe_to_rows(df, index=False, header=False):
                #     ws.append(r)
                # wb.save('policy_iteration.xlsx')

                # If the value_iteration or policy_iteration excel files are already generated then comment the below
                # df.to_excel("value_iteration.xlsx", index=False)

                big_df = pd.concat([big_df, df])

                # # Make some plots:
                # sns.histplot(data = cleaned, color = 'blue')
                # plt.title('Percentage of tiles cleaned')
                # # Change the suptitle according to the current parameters
                # plt.suptitle('THETA ' + str(theta) + ' | GAMMA ' + str(gamma) + ' | GRID_FILE ' + grid_file)
                # plt.xlabel('% cleaned')
                # plt.ylabel('count')
                # plt.show()
                #
                # sns.histplot(data = efficiencies, color = 'green')
                # plt.title('Efficiency of robot.')
                # # Change the suptitle according to the current parameters
                # plt.suptitle('THETA ' + str(theta) + ' | GAMMA ' + str(gamma) + ' | GRID_FILE ' + grid_file)
                # plt.xlabel('Efficiency %')
                # plt.ylabel('count')
                # plt.show()
                #
                # sns.barplot(x='Percent', y = 'Time', data = mean_time , color = 'orange', ci= None)
                # plt.title('Avg time spent for house cleaning')
                # # Change the suptitle according to the current parameters
                # plt.suptitle('THETA ' + str(theta) + ' | GAMMA ' + str(gamma) + ' | GRID_FILE ' + grid_file)
                # plt.xlabel('percentage of cleaned cells')
                # plt.ylabel('time in seconds')
                # plt.show()

big_df.to_excel("value_iteration.xlsx", index=False)

average_df = pd.DataFrame(
    columns=['Clean mean', 'Clean variance', 'Efficiency mean', 'Efficiency variance', 'Experiment'])
average_df['Clean mean'] = cleaned_means
average_df['Clean variance'] = cleaned_variances
average_df['Efficiency mean'] = efficiencies_means
average_df['Efficiency variance'] = efficiencies_variances
average_df['Experiment'] = experiments

average_df.to_excel("Overview.xlsx", index=False)

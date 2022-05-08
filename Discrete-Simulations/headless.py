# Import our robot algorithm to use in this simulation:
from robot_configs.value_iteration_robot import robot_epoch
import pickle
from environment import Robot
import matplotlib.pyplot as plt
import pandas as pd
import time
import seaborn as sns
from tqdm import tqdm
from openpyxl import Workbook, load_workbook
from openpyxl.utils.dataframe import dataframe_to_rows

# Cleaned tile percentage at which the room is considered 'clean':
stopping_criteria = 100

# Keep track of some statistics:
efficiencies = []
n_moves = []
deaths = 0
cleaned = []

time_tracker = pd.DataFrame(columns = ["Simulation", "Percent", "Time"])
cnt = 0
experiment = []

# Settings
grid_file = 'snake.grid'
gamma = 0.9
theta = 0.001

# Run 100 times:
for i in tqdm(range(20)):
    # Open the grid file.
    # (You can create one yourself using the provided editor).
    with open(f'grid_configs/{grid_file}', 'rb') as f:
        grid = pickle.load(f)
    # Calculate the total visitable tiles:
    n_total_tiles = (grid.cells >= 0).sum()
    # Spawn the robot at (1,1) facing north with battery drainage enabled:
    robot = Robot(grid, (1, 1), orientation='n', battery_drain_p=0.5, battery_drain_lam=2)
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
        dirty = (grid.cells >= 1).sum()
        goal = (grid.cells == 2).sum()
        # Calculate the cleaned percentage:
        clean_percent = (clean / (dirty + clean)) * 100
        if int(clean_percent) % 10 == 0 and int(clean_percent) != 0:
            cnt += 1
            time_tracker.loc[cnt,"Simulation"] = i+1
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
    experiment.append('THETA ' + str(theta) + ' | GAMMA ' + str(gamma) + ' | GRID_FILE ' + grid_file)

    grouped_time = time_tracker[["Percent","Time"]].groupby("Percent")

    mean_time = grouped_time.mean()

    mean_time = mean_time.reset_index()

df = pd.DataFrame(columns = ['Clean', 'Efficiency','Experiment'])
df['Clean'] = cleaned
df['Efficiency'] = efficiencies
df['Experiment'] = experiment

# If the value_iteration or policy_iteration excel files are already generated the uncomment the below

# wb = load_workbook('policy_iteration.xlsx')
# ws = wb['Sheet1']
#
# # Change the header to True for the first iteration
# for r in dataframe_to_rows(df, index=False, header=False):
#     ws.append(r)
# wb.save('policy_iteration.xlsx')

# If the value_iteration or policy_iteration excel files are already generated then comment the below
df.to_excel("value_iteration.xlsx", index=False)

# Make some plots:
sns.histplot(data = cleaned, color = 'blue')
plt.title('Percentage of tiles cleaned')
# Change the suptitle according to the current parameters
plt.suptitle('THETA ' + str(theta) + ' | GAMMA ' + str(gamma) + ' | GRID_FILE ' + grid_file)
plt.xlabel('% cleaned')
plt.ylabel('count')
plt.show()

sns.histplot(data = efficiencies, color = 'green')
plt.title('Efficiency of robot.')
# Change the suptitle according to the current parameters
plt.suptitle('THETA ' + str(theta) + ' | GAMMA ' + str(gamma) + ' | GRID_FILE ' + grid_file)
plt.xlabel('Efficiency %')
plt.ylabel('count')
plt.show()

sns.barplot(x='Percent', y = 'Time', data = mean_time , color = 'orange', ci= None)
plt.title('Avg time spent for house cleaning')
# Change the suptitle according to the current parameters
plt.suptitle('THETA ' + str(theta) + ' | GAMMA ' + str(gamma) + ' | GRID_FILE ' + grid_file)
plt.xlabel('percentage of cleaned cells')
plt.ylabel('time in seconds')
plt.show()

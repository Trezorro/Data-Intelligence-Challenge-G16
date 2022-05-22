from typing import List, Type
import importlib

from flask import Flask, render_template, request
from engineio.payload import Payload

# Increase limit to not drop too many packets:
from flask_socketio import SocketIO
import base64
import numpy as np
from io import BytesIO
from flask_socketio import emit
import pickle
import os
import ast
from matplotlib.figure import Figure
from environment import Grid, RobotBase

Payload.max_decode_packets = 1000

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

grid: Grid = None
robots: List[RobotBase] = None
occupied = False
PATH = os.getcwd()

import logging
logging.basicConfig(level=logging.DEBUG, force=True)
logging.getLogger('matplotlib').setLevel(logging.ERROR)
logging.getLogger('werkzeug').setLevel('WARNING')


def draw_grid(grid_to_draw: Grid):
    """'Helper function for creating a JSON payload which will be displayed in the browser."""
    global robots
    materials = {0: 'cell_clean', -1: 'cell_wall', -2: 'cell_obstacle', -3: 'cell_robot_e', -4: 'cell_robot_s',
                 -5: 'cell_robot_w', -6: 'cell_robot_n', 1: 'cell_dirty', 2: 'cell_goal', 3: 'cell_death'}
    # Setting statistics:
    clean = (grid_to_draw.cells == 0).sum()
    dirty = (grid_to_draw.cells == 1).sum()  # edited to only include actual dirty cells
    goal = (grid_to_draw.cells == 2).sum()
    debug_values = None
    if robots:  # If we have robots on the grid:
        efficiencies = [100 for _ in range(len(robots))]
        batteries = [100 for _ in range(len(robots))]
        alives = [True for _ in range(len(robots))]
        for i, robot in enumerate(robots):
            # Calculating efficiency:
            moves = [(y, x) for (y, x) in zip(robot.history[0], robot.history[1])]
            if len(moves) > 0:
                u_moves = set(moves)
                n_revisited_tiles = len(moves) - len(u_moves)
                n_total_tiles = (grid_to_draw.cells >= 0).sum()
                efficiency = (100 * n_total_tiles) / (n_total_tiles + n_revisited_tiles)
                efficiencies[i] = float(round(efficiency, 2))
            # Min battery level is 0:
            battery = 0 if robot.battery_lvl < 0 else robot.battery_lvl
            # Battery and alive stats:
            batteries[i] = round(battery, 2)
            alives[i] = robot.alive
            if robot.show_debug_values:
                debug_values = robot.debug_values  # show this robot's values in the grid
        return {'grid': render_template('grid.html', height=30, width=30, n_rows=grid_to_draw.n_rows,
                                        n_cols=grid_to_draw.n_cols,
                                        room_config=grid_to_draw.cells, values=debug_values,
                                        materials=materials), 'clean': round((clean / (dirty + clean)) * 100, 2),
                'goal': float(goal), 'efficiency': ','.join([str(i) for i in efficiencies]),
                'battery': ','.join([str(i) for i in batteries]),
                'alive': alives}
    else:  # If we have an empty grid with no robots:
        return {'grid': render_template('grid.html', height=30, width=30, n_rows=grid_to_draw.n_rows,
                                        n_cols=grid_to_draw.n_cols,
                                        room_config=grid_to_draw.cells,
                                        materials=materials), 'clean': round((clean / (dirty + clean)) * 100, 2),
                'goal': float(goal), 'efficiency': ',', 'battery': ',',
                'alive': ','}


# Routes:

@app.route('/')
def home():
    return render_template('home_page.html', files=os.listdir(PATH + '/grid_configs'),
                           rfiles=[i for i in os.listdir(PATH + '/robot_configs') if '__' not in i])


@app.route('/editor')
def editor():
    return render_template('editor.html')


@app.route('/build_grid')
def build_grid():
    """Main route for building a grid. Given a request with the following parameters, a grid
     and accompanying statistics are being constructed.
     Request params:
     height: number of rows in the grid.
     width: number of columns in the grid.
     obstacles: a list of tuples (x,y) of obstacle locations.
     goals: a list of tuples (x,y) of goal locations.
     deaths: a list of tuples (x,y) of death-tile locations.
     save: boolean (true, false) to save the current grid to a file.
     name: filename to save the current grid to.
     """
    global grid, robots
    robots = None
    n_rows = int(request.args.get('height'))
    n_cols = int(request.args.get('width'))
    obstacles = ast.literal_eval(request.args.get('obstacles'))
    goals = ast.literal_eval(request.args.get('goals'))
    deaths = ast.literal_eval(request.args.get('deaths'))
    to_save = False if request.args.get('save') == 'false' else True
    name = str(request.args.get('name'))
    new_grid: Grid = Grid(n_cols, n_rows)
    for (y, x) in obstacles:
        new_grid.put_singular_obstacle(x, y)
    for (y, x) in goals:
        new_grid.put_singular_goal(x, y)
    for (y, x) in deaths:
        new_grid.put_singular_death(x, y)
    if to_save and len(name) > 0:
        pickle.dump(new_grid, open(f'{PATH}/grid_configs/{name}.grid', 'wb'))
        return {'grid': '', 'success': 'true'}
    return draw_grid(new_grid)


@app.route('/get_history')
def get_history():
    """Returns a plot of the history."""
    global robots
    if robots:
        fig = Figure()
        ax = fig.subplots()
        for robot in robots:
            ax.plot(np.array(robot.history[1]) + 0.5, -1 * np.array(robot.history[0]) - 0.5)
            obstacles = [[], []]
            for x in range(robot.grid.n_cols):
                for y in range(robot.grid.n_rows):
                    if (robot.grid.cells[y][x] == -2) or (robot.grid.cells[y][x] == -1):
                        obstacles[0].extend([x, x + 0.5, None])
                        obstacles[1].extend([-1 * y, -1 * y - 0.5, None])
            ax.plot(obstacles[0], obstacles[1])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        # Save it to a temporary buffer.
        buf = BytesIO()
        fig.savefig(buf, format="png")
        # Embed the result in the html output.
        data = base64.b64encode(buf.getbuffer()).decode("ascii")
        return f"<img src='data:image/png;base64,{data}'/>"
    return ''


# Event handlers:

@socketio.on('get_grid')
def handle_browser_new_grid(json):
    """Handles socket event 'get_grid', needs filename of grid config as payload."""
    global grid
    global occupied
    global robots
    robots = None
    occupied = False
    with open(f'{PATH}/grid_configs/{json["data"]}', 'rb') as f:
        grid = pickle.load(f)
    if not hasattr(grid, 'transposed_version'):
        grid.cells = grid.cells.T
    emit('new_grid', draw_grid(grid))


@socketio.on('get_robot')
def handle_browser_spawn_robot(json):
    global robots
    global grid
    p_determ = float(json['determ'])
    x_spawn = json['x_spawns'].split(',')
    y_spawn = json['y_spawns'].split(',')
    orient = json['orient']
    p_drain = float(json['p_drain'])
    lam_drain = float(json['lam_drain'])
    vision = int(json['vision'])
    n_robots = int(json['n_robots'])
    # Dynamically load correct bot class:
    robot_module = importlib.import_module('robot_configs.'+json['robot_file'].split('.py')[0])
    RobotClass: Type[RobotBase] = getattr(robot_module, 'Robot', RobotBase)
    try:
        robots = [
            RobotClass(grid, (int(y_spawn[i]), int(x_spawn[i])),
                       orientation=orient,
                       battery_drain_p=p_drain,
                       battery_drain_lam=lam_drain,
                       p_move=p_determ,
                       vision=vision) for i in range(n_robots)
        ]
    except IndexError:
        emit('new_grid', {'grid': '<h1>Invalid robot coordinates entered!</h1>'})
        print('[ERROR] invalid starting coordinate entered!')
    except ValueError:
        emit('new_grid', {'grid': '<h1>Invalid robot coordinates entered, spot on map is not free!</h1>'})
        print('[ERROR] invalid starting coordinate entered, spot on map is not free!')
    else:
        emit('new_grid', draw_grid(grid))


@socketio.on('get_update')
def handle_browser_update(json):
    global robots
    global occupied
    global grid
    robot_alg = json['robot_file'].split('.py')[0]
    robot_module = importlib.import_module('robot_configs.'+robot_alg)
    if not (hasattr(robot_module, 'Robot') or hasattr(robot_module, 'robot_epoch')):
        raise ImportError(f"No Robot class or robot_epoch function found in {robot_alg}.py!")
    
    if not occupied:
        occupied = True
        for robot in robots:
            # Don't update dead robots:
            if robot.alive:
                # Call the robot epoch method of the selected robot config file:
                if hasattr(robot_module, 'robot_epoch'):  # Try legacy robot_epoch function
                    getattr(robot_module, 'robot_epoch')(robot)
                else:
                    robot.robot_epoch()  # Use method otherwise
        emit('new_grid', draw_grid(grid))
        emit('new_plot', get_history())
        occupied = False
    else:
        pass


if __name__ == '__main__':
    socketio.run(app, debug=True)

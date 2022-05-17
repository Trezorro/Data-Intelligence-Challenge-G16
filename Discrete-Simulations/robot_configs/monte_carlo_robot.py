from typing import Union
import numpy as np
from environment import Robot
import logging
import random
import copy

# Logging settings
from helpers.label_based_reward import get_reward


def robot_epoch(robot: Robot, gamma=0.99, max_episodes = 100, epsilon = 0.1):
    """ Initianlize the attributes needed for the Mote Carlo On Policy implementation

    :param max_episodes: the max number of episodes that we want to compute
    :param q_grid: the action state value, initialzied to 0. For every cell there are 4 possible moves so 4 times 0
    :param moves: list of all actions
    :param Returns: empty array for every possible combination of moves and states
    :param policy: initialize a policy for every possible state
    """

    #Initialization
    max_episodes = max_episodes
    g = gamma
    count_clean = 0
    #number of non wall or obstacle cells
    number_of_cells = 0
    q_grid = {} #q_grid : Q(state, action)
    epsilon = epsilon
    moves = list(robot.dirs.values())
    Returns = {} # Returns(state, action)
    policy = {} # policy(state, action)
    all_possible_moves = []
    possible_moves = {}

    for x in range(robot.grid.n_cols):
        for y in range(robot.grid.n_rows):
            # ignore cells that are walls or obstacles
            if -3 < robot.grid.cells[y][x] < 0 or robot.grid.cells[y][x] == 3:
                continue
            #create a list with possible states
            all_possible_moves.append(tuple(np.array([y, x])))
            possible_moves[(y,x)] = []
            number_of_cells += 1
            if robot.grid.cells[y][x] == 0:
                count_clean += 1
            for move in moves:
                # Calculate the new position the robot would be in after the move
                new_pos = tuple(np.array([y, x]) + move)
                if -3 < robot.grid.cells[new_pos] < 0 or robot.grid.cells[new_pos] == 3:
                    continue
                possible_moves[(y,x)].append(move)
                Returns[(y, x),move] = []
                q_grid[(y, x), move] = 0
            for move in possible_moves[(y,x)]:
                #initialize the policy with equal probability for every action of a certain state
                policy[(y, x), move] = 1 / len(possible_moves)


    """Generate episodes"""
    def generate_episodes(policy):
        """ Generate the episodes based on the current policy\
        :param position: the position of the robot after applying a move based on policy. Initialized as the
        current position
        :param policy: the current policy probabilties

        Returns a list called episodes that includes the state, the move and the reward of the next move i.e.
         [[(1, 1), (1, 0), 2], [(2, 1), (1, 0), -1], [(3, 1), (-1, 0), -1]]: the state (1,1) has next action (1,0)
         that gives 2 as reward, then, the state (2,1) has (1,0) as next action that gives -1 as reward etc.
        """
        #create a deepcopy of robot object because in the simulation
        #we will have an updated board which is the result of simulation and not the actual one
        temp_robot = copy.deepcopy(robot)

        episodes= []
        # TODO random initial move ?
        position = random.choice(all_possible_moves)
        # position = tuple(np.array(temp_robot.pos))
        not_end_episode = True

        #flag for finding a dirty cell in an episode simulation
        found_dirty = False

        step = 0
        # step of the episodes to be proportinal of the grid size
        while step < number_of_cells/3:
            moves = []
            probs = []
            for state, action in policy.keys():
                if state == position:
                    moves.append(action)
                    probs.append(policy[state, action])
            #choose randomly a move but based on action weights
            chosen_move = random.choices(moves, weights=probs, k=1)[0]
            episodes.append([position, chosen_move])
            new_pos = tuple(np.asarray(position) + chosen_move)

            #update the reward in the simulated grid
            reward = get_reward(temp_robot.grid.cells[new_pos])
            episodes[step].append(reward)

            step += 1
            if (get_reward(temp_robot.grid.cells[new_pos]) == 2):
                found_dirty = True


            if (found_dirty and step > number_of_cells/4) or step > count_clean:
            # if (found_clean) or step > count_clean:
                not_end_episode = False

            #update position of the simulated robot
            position = new_pos
            temp_robot.pos = position

        return episodes


    """Implementation"""
    for episode in range(max_episodes):
        # generate episodes

        # gradually reduce the epsilon parameter cause we need less exploration and more exploitation as the episodes increase
        epsilon *= 0.99
        single_episode = generate_episodes(policy)
        G = 0
        for idx, step in enumerate(single_episode[::-1]):
            G = g * G + step[2]
            # first-visit
            if (step[0], step[1]) not in np.array(single_episode[::-1])[:, :2][idx + 1:]:

                #update Returns(state,action) & q_grid(state,action)
                Returns[step[0], step[1]].append(G)
                q_grid[step[0], step[1]] = np.mean(Returns[step[0], step[1]])

                #calculate the best action
                best_action = (0, 0)
                max_value = -100

                count_actions = 0
                # q_grid_lst = list(q_grid)
                for action in possible_moves[step[0]]:
                    # if state == step[0]:
                    count_actions += 1
                    if q_grid[step[0], action] > max_value:
                        best_action = action
                        max_value = q_grid[step[0], action]

                #update the policy matrix of the specific (state,action) pair
                # based on e-soft policy
                for action in possible_moves[step[0]]: # enumerate action space
                    # if state == step[0]:
                    if action == best_action:
                        policy[step[0],action] = 1 - epsilon + epsilon / count_actions
                    else:
                        policy[step[0],action] = epsilon / count_actions

    # when the episodes iteration finishes
    # choose the corresponding robot move based on the updated policy probabilities
    moves = []
    probs = []
    for state, action in policy.keys():
        if state == tuple(np.array(robot.pos)):
            moves.append(action)
            probs.append(policy[state, action])
    corresponding_move = random.choices(moves, weights=probs, k=1)[0]

    current_direction = robot.dirs[robot.orientation]

    while current_direction != corresponding_move:
        robot.rotate('r')
        current_direction = robot.dirs[robot.orientation]

    robot.move()
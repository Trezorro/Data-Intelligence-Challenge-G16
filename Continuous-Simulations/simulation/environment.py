"""Environment.

The simulation environment used for our continuous vacuum cleaner.
"""
from pathlib import Path
from typing import Optional
from random import randint
import numpy as np
from grid_generator import GridBuilder
from simulation.environment_model import EnvironmentModel

import gym
from gym.spaces import Box, Dict, Discrete
import pygame


class ContinuousEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 10}

    def __init__(self, render_mode: Optional[str] = "human"):
        """Creates a continuous environment to run simulations in.

        Args:
            render_mode: The render mode to use for rendering. Choices are
                "human" or None.
        """
        self.grid = None
        self.world = None
        self.window_size = 768  # Pygame window size.

        # Set up the observation space
        # All observations with Discrete(2) represent booleans. 0 for False and
        # 1 for True.
        self.observation_space = Dict({
            "move_succeeded": Discrete(2),
            "hit_wall": Discrete(2),
            "hit_obstacle": Discrete(2),
            "hit_dirt": Box(low=0, high=2048, shape=(1,), dtype=int),
            "hit_death": Discrete(2),
            "is_alive": Discrete(2),
            "world": Box(low=0.,
                         high=1.0,
                         shape=(24, 24, 6))
        })
        # `move` is a represents a boolean. 0 is False, 1 is True.
        self.action_space = Dict({"direction": Box(low=-1.,
                                                   high=1.,
                                                   shape=(1,)),
                                  "move": Discrete(2)})

        self.should_render = render_mode == "human"

        if self.should_render:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size,
                                                   self.window_size))
            pygame.display.set_caption("Reinforcement Learning Vacuum")
            self.clock = pygame.time.Clock()
            self.prev_render_rects = []

    def reset(self, seed=None, return_info=False, options=None):
        """Resets the environment.

        Args:
            seed: The random seed to use.
            return_info: Whether or not to return info from the environment.
            options: An optional dictionary containing run parameters. It can
                contain the following keys
                {"grid_size": int,
                 "num_rooms": int,
                 "num_obstacles": int,
                 "num_dirt": int,
                 "scalar": int,
                 "battery_drain": float,
                 "agent_width": int}

        """
        super().reset(seed=seed)
        # First set defaults
        params = {
            "grid_size": 24,
            "num_rooms": 5,
            "num_obstacles": randint(4, 30),
            "num_dirt": randint(50, 600),
            "scalar": 64,
            "battery_drain": 0.25,
            "agent_width": 48
        }

        self.world = EnvironmentModel(grid = self.grid,
                                      num_obstacles=randint(4, 30),
                                      num_dirt=randint(50, 600),)
        self._initial_render()

    def step(self, action):
        # Asks agent for a move
        # Applies move to the world
        # Asks environment_model for observation
        # Calculate reward from move
        # Provide reward and observation back to agent
        pass

    def _get_obs(self):
        pass

    def _get_info(self):
        pass

    def _initial_render(self):
        """Initial rendering of the environment"""
        if not self.should_render:
            return
        background = pygame.Surface(self.window.get_size())
        background = background.convert()
        background.fill((250, 250, 250))

        # Draw rectangles at every point where there is an object on the grid
        # self.

        # for pos in wall_idxs:
        #     x0 = pos[0] * self.world.scalar
        #     y0 = pos[1] * self.world.scalar
        #     pygame.draw.rect(
        #         surface=background,
        #         color=(10, 10, 10),
        #         rect=[x0, y0, self.world.scalar, self.world.scalar]
        #     )

        # Update the actual display
        update_rect = self.window.blit(background, background.get_rect())
        pygame.display.update(update_rect)

    def render(self, mode="human"):
        if not self.should_render:
            return

        # create surface
        canvas = pygame.Surface(self.window.get_size())
        canvas = canvas.convert()
        canvas.fill((250, 250, 250))

        # Paint each object


        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.should_render:
            pygame.display.quit()
            pygame.quit()

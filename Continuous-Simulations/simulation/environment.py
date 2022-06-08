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

    def __init__(self, grid_params: dict = None,
                 render_mode: Optional[str] = "human"):
        """Creates a continuous environment to run simulations in.

        Args:
            grid_params: Parameters for the Grid Builder as a dictionary.
            render_mode: The render mode to use for rendering. Choices are
                "human" or None.
        """
        try:
            self.grid_size = grid_params["grid_size"]
            grid_builder = GridBuilder(**grid_params)
        except (KeyError, TypeError):
            self.grid_size = 24
            grid_builder = GridBuilder(self.grid_size)

        self.map = grid_builder.generate_grid()
        self.world = None
        self.window_size = self.grid_size * 32  # Pygame window size.

        # Set up the observation space
        self.observation_space = Box(low=0.,
                                     high=1.0,
                                     shape=(self.grid_size, self.grid_size, 5))
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
        """Resets the environment."""
        super().reset(seed=seed)

        self.world = EnvironmentModel(self.map, randint(4, 30), randint(4, 30))
        self._initial_render()

    def step(self, action):
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

        # Draw rectangles at every point where there is an object on the map
        wall_idxs = np.where(self.map > 0)
        wall_idxs = list(zip(wall_idxs[0], wall_idxs[1]))
        for pos in wall_idxs:
            x0 = pos[0] * self.world.scalar
            y0 = pos[1] * self.world.scalar
            pygame.draw.rect(
                surface=background,
                color=(10, 10, 10),
                rect=[x0, y0, self.world.scalar, self.world.scalar]
            )

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

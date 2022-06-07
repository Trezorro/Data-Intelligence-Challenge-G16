"""Environment.

The simulation environment used for our continuous vacuum cleaner.
"""
from pathlib import Path
from typing import Optional
import numpy as np
from grid_generator import GridBuilder
from .environment_model import EnvironmentModel

import gym
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
        self.window_size = self.grid_size * 32  # Pygame window size.

        # Set up the observation space
        self.observation_space = None
        self.action_space = None

        self.should_render = render_mode == "human"

        if self.should_render:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size,
                                                   self.window_size))
            pygame.display.set_caption("Reinforcement Learning Vacuum")
            self.clock = pygame.time.Clock()

    def reset(self, seed=None, return_info=False, options=None):
        """Resets the environment."""
        super().reset(seed=seed)
        self.world = EnvironmentModel(self.map)
        self._initial_render()

    def step(self, action):
        pass

    def _get_obs(self):
        pass

    def _get_info(self):
        pass

    def _initial_render(self):
        """Initial rendering of the environment"""
        background = pygame.Surface(self.window.get_size())
        background = background.convert()
        background.fill((250, 250, 250))

        # Display some text
        font = pygame.font.Font(None, 36)
        text = font.render("Loading environment...", True, (10, 10, 10))
        textpos = text.get_rect()
        textpos.centerx = background.get_rect().centerx
        textpos.centery = background.get_rect().centery
        background.blit(text, textpos)
        update_rect = self.window.blit(background, background.get_rect())
        pygame.display.update(update_rect)

    def render(self, mode="human"):
        if not self.should_render:
            return
        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.should_render:
            pygame.display.quit()
            pygame.quit()

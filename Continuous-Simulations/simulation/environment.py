"""Environment.

The simulation environment used for our continuous vacuum cleaner.
"""
from typing import Optional, Union, Tuple
from random import randint
import numpy as np
import math
from grid_generator import GridBuilder
from simulation.environment_model import EnvironmentModel

import gym
from gym.spaces import Box, Dict, Discrete
import pygame


class ContinuousEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = "human"):
        """Creates a continuous environment to run simulations in.

        Args:
            render_mode: The render mode to use for rendering. Choices are
                "human" or None.
        """
        self.grid: Optional[np.ndarray] = None
        self.world: Optional[EnvironmentModel] = None
        self.grid_size = 24
        self.window_size = 768  # Pygame window size.
        self.agent_speed = 128

        # Set up the observation space
        # All observations with Discrete(2) represent booleans. 0 for False and
        # 1 for True.
        self.observation_space = self._generate_observation_space()
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

    def _generate_observation_space(self) -> Dict:
        return Dict({
            "move_succeeded": Discrete(2),
            "hit_wall": Discrete(2),
            "hit_obstacle": Discrete(2),
            "hit_dirt": Box(low=0, high=2048, shape=(1,), dtype=int),
            "hit_death": Discrete(2),
            "is_alive": Discrete(2),
            "world": Box(low=0.,
                         high=1.0,
                         shape=(self.grid_size, self.grid_size, 6))
        })

    def reset(self, seed=None,
              return_info=False,
              options=None) -> Union[Tuple[dict, dict], dict]:
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
                 "agent_width": int,
                 "agent_speed": int}

        """
        super().reset(seed=seed)
        # First set defaults
        params = {
            "grid_size": 24,
            "num_obstacles": randint(4, 30),
            "num_dirt": randint(50, 600),
            "cell_size": 64,
            "battery_drain": 0.25,
            "agent_width": 96,
            "agent_speed": 10
        }

        possible_keys = {"grid_size", "num_rooms", "num_obstacles", "num_dirt",
                         "cell_size", "battery_drain", "agent_width"}
        if options is not None:
            for key, value in options.items():
                if key not in possible_keys:
                    raise KeyError(f"Provided {key=} is not one of the possible"
                                   f"keys. Possible keys are {possible_keys=}")
                params[key] = value
        self.grid = GridBuilder(params["grid_size"]).generate_grid()
        self.grid_size = self.grid.shape[0]
        self.observation_space = self._generate_observation_space()

        self.agent_speed = params["agent_speed"]

        del params["grid_size"]
        del params["agent_speed"]

        self.world = EnvironmentModel(grid=self.grid, **params)
        self._initial_render()

        if return_info:
            return self._get_obs(), self._get_info()
        else:
            return self._get_obs()

    def step(self, action: dict) -> Tuple[dict, int, bool, dict]:
        """Takes an action space value and returns the result.

        Args:
            action: The action from the action space to do.

        Returns:
            observation, reward, done, and info.
        """
        # Applies move to the world and asks environment_model for observation.
        direction = int(180 * action["direction"])
        self.world.rotate_agent(direction)
        move_distance = action["move"] * self.agent_speed
        events, observation = self.world.move_agent(int(move_distance))

        # Calculate reward from move
        reward = self._get_reward(events)

        # Provide reward and observation back to agent
        observation_dict = dict(**events, world=observation)
        done = not self.world.agent_is_alive
        return observation_dict, reward, done, self._get_info()

    def _get_reward(self, events: dict) -> int:
        """Given the events dict, returns the reward.

        Args:
            events: The dictionary of events that occured during a step.

        Returns:
            The reward value.
        """

    def _get_obs(self) -> dict:
        return self.world.previous_observation

    def _get_info(self) -> dict:
        pass

    def _initial_render(self):
        """Initial rendering of the environment. Displays loading text."""
        if not self.should_render:
            return
        # create surface
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

    @staticmethod
    def _downsample_rect(rect: pygame.Rect, scalar: float) -> pygame.Rect:
        """Downsamples the given rectangle by a scalar."""
        x = rect.x * scalar
        y = rect.y * scalar
        width = rect.width * scalar
        height = rect.height * scalar
        return pygame.Rect(x, y, width, height)


    def render(self, mode="human"):
        if not self.should_render:
            return
        scalar = self.window.get_size()[0]
        scalar /= self.world.grid_size * self.world.cell_size
        # create surface
        background = pygame.Surface(self.window.get_size())
        background = background.convert()
        background.fill((250, 250, 250))

        # Draw walls
        for wall_rect in self.world.wall_rects:
            pygame.draw.rect(
                surface=background,
                color=(69, 71, 82),
                rect=self._downsample_rect(wall_rect, scalar)
            )
        # Draw death tiles
        for death_tile in self.world.death_rects:
            pygame.draw.rect(
                surface=background,
                color=(235, 64, 52),
                rect=self._downsample_rect(death_tile, scalar)
            )
        # Draw obstacles
        for obstacle in self.world.obstacle_rects:
            pygame.draw.rect(
                surface=background,
                color=(240, 175, 36),
                rect=self._downsample_rect(obstacle, scalar)
            )
        # Draw dirt
        for dirt in self.world.dirt_rects:
            dirt_x = dirt.x * scalar
            dirt_y = dirt.y * scalar
            dirt_width = 4
            dirt_height = 4
            pygame.draw.rect(
                surface=background,
                color=(82, 59, 33),
                rect=pygame.Rect(dirt_x, dirt_y, dirt_width, dirt_height),
            )
        # Draw the agent base
        agent_rect = self._downsample_rect(self.world.agent_rect, scalar)
        pygame.draw.rect(
            surface=background,
            color=(36, 83, 138),
            rect=agent_rect
        )
        # Draw the a triangle representing its direction
        # Triangle has three points (a, b, c) with a center at the origin
        #         C
        #         ^
        #        / \
        #       / O \
        #      /_____\
        #     A       B

        rads = self.world.agent_angle / 180 * np.pi
        center = agent_rect.center
        radius = agent_rect.width * 0.3
        points = []

        # For 0, 120, and 240 degrees in radians
        for theta in (4.1887902047863905, 0., 2.0943951023931953, ):
            x = center[0] + radius * math.cos(theta + rads)
            y = center[1] + radius * math.sin(theta + rads)
            points.append((x, y))

        pygame.draw.aalines(
            surface=background,
            color=(242, 242, 242),
            closed=False,
            points=points,
            blend=1
        )
        # Draw a battery indicator
        batt_level = self.world.agent_battery
        font = pygame.font.Font(None, 18)
        text = font.render(f"Battery: {batt_level:.2f} %",
                           True, (250, 250, 250))
        textpos = text.get_rect()
        textpos.x = 10
        textpos.y = 10

        background.blit(text, textpos)
        # Draw the white background rect
        pygame.draw.rect(
            surface=background,
            color=(250, 250, 250),
            rect=pygame.Rect(120, 10, 204, 14)
        )
        # Draw the battery meter
        length = 200 * batt_level / 100
        pygame.draw.rect(
            surface=background,
            color=(43, 227, 52),
            rect=pygame.Rect(122, 12, int(length), 10)
        )

        # Update the actual display
        update_rect = self.window.blit(background, background.get_rect())
        pygame.display.update(update_rect)
        self.clock.tick(self.metadata["render_fps"])

    def close(self):
        if self.should_render:
            pygame.display.quit()
            pygame.quit()

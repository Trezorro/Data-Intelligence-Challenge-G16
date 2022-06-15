"""Environment.

The simulation environment used for our continuous vacuum cleaner.
"""
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict as Dict_py
import sys
from random import randint
import numpy as np
import math
from grid_generator import GridBuilder
from simulation.world_model import WorldModel

import gym
from gym.spaces import Box, Dict, Discrete
import pygame
from PIL import Image

GRID_SIZE = 24
OBSERVATION_SPACE = Dict({
    "agent_center": Box(low=0, high=1536, shape=(2,), dtype=np.int32),
    "world": Box(low=0.,
                 high=3,
                 shape=(5, GRID_SIZE, GRID_SIZE),
                 dtype=np.float64)
})

START_STATS = {"successful_moves": 0,
               "wall_hits": 0,
               "obstacle_hits": 0,
               "dirt_hits": 0,
               "death_hits": 0,
               "is_alive": True,
               "score": 0,
               "fps": "0.0"}


class ContinuousEnv(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, render_mode: Optional[str] = "human"):
        """Creates a continuous environment to run simulations in.

        Args:
            render_mode: The render mode to use for rendering. Choices are
                "human" or None.
        """
        self.grid: Optional[np.ndarray] = None
        self.world: Optional[WorldModel] = None
        self.window_size = (1152, 768)  # Pygame window size.
        self.agent_speed = 128

        # Set up the observation space
        # All observations with Discrete(2) represent booleans. 0 for False and
        # 1 for True.
        self.observation_space = OBSERVATION_SPACE
        # `move` is a represents a boolean. 0 is False, 1 is True.
        self.action_space = Box(
            np.array([-1, +1]),
            np.array([0, 1]),
            dtype=np.float32,
        )

        self.should_render = render_mode == "human"
        self.info = START_STATS.copy()

        if self.should_render:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(self.window_size)
            pygame.display.set_caption("Reinforcement Learning Vacuum")
            self.clock = pygame.time.Clock()
            self.agent_eyes = Image.open(Path(__file__).parent / "eyes.png")

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
        self.info = START_STATS.copy()

        params = {
            "grid_size": 24,
            "num_obstacles": 4,
            "num_dirt": 2000,
            "cell_size": 64,
            "battery_drain": 0.25,
            "agent_width": 60,
            "agent_speed": 60
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

        self.agent_speed = params["agent_speed"]

        del params["grid_size"]
        del params["agent_speed"]

        self.world = WorldModel(grid=self.grid, **params)
        self._initial_render()  # shows loading text

        agent_observation = {
            "agent_center": self.world.agent_rect.center,
            "world": self.world.last_observation
        }
        return agent_observation

    def step(self, action: List) -> Tuple[Dict_py, int, bool, dict]:
        """Takes an action space value and returns the result.

        Args:
            action: The action from the action space to do.

        Returns:
            observation, reward, done, and info.
        """
        # Applies move to the world and asks environment_model for observation.
        direction = int(180 * action[0])
        self.world.rotate_agent(direction)
        move_distance = action[1] * self.agent_speed
        events, observation = self.world.move_agent(int(move_distance))

        # Calculate reward from move and update info
        reward = self._get_reward(events)
        self._update_info(events, reward)

        # Provide reward and observation back to agent
        done = not self.world.agent_is_alive

        return_observation = {
            "agent_center": self.world.agent_rect.center,
            "world": observation
        }
        return return_observation, reward, done, self.info

    def _get_reward(self, events: dict) -> int:
        """Given the events dict, returns the reward.

        Args:
            events: The dictionary of events that occured during a step.

        Returns:
            The reward value.
        """
        # TODO
        return events["hit_dirt"] * 10

    def _update_info(self, events: dict, reward: int):
        """Updates the info dict."""
        self.info["successful_moves"] += events["move_succeeded"]
        self.info["wall_hits"] += events["hit_wall"]
        self.info["obstacle_hits"] += events["hit_obstacle"]
        self.info["dirt_hits"] += events["hit_dirt"]
        self.info["death_hits"] += events["hit_death"]
        self.info["is_alive"] = events["is_alive"]
        self.info["score"] += reward

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

        self.info["fps"] = f"{self.clock.get_fps():.01f}"
        scalar = self.window.get_size()[1]
        scalar /= self.world.grid_size * self.world.cell_size

        # create surface
        background = pygame.Surface(self.window.get_size())
        background = background.convert()
        background.fill((250, 250, 250))

        self._draw_world(background, scalar)
        self._draw_agent(background, scalar)
        self._draw_info(background, 798, 30)
        self._draw_observation(background, 798, 350)
        self._draw_battery(background, 798, 30)

        # Update the actual display
        update_rect = self.window.blit(background, background.get_rect())
        pygame.display.update(update_rect)

        # Get all events to make it interactive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()
        pygame.event.pump()
        # self.clock.tick(self.metadata["render_fps"])
        self.clock.tick(999)

    def _draw_world(self, surface, scalar):
        # Draw walls
        for wall_rect in self.world.wall_rects:
            pygame.draw.rect(
                surface=surface,
                color=(69, 71, 82),
                rect=self._downsample_rect(wall_rect, scalar)
            )
        # Draw death tiles
        for death_tile in self.world.death_rects:
            pygame.draw.rect(
                surface=surface,
                color=(235, 64, 52),
                rect=self._downsample_rect(death_tile, scalar)
            )
        # Draw obstacles
        for obstacle in self.world.obstacle_rects:
            pygame.draw.rect(
                surface=surface,
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
                surface=surface,
                color=(82, 59, 33),
                rect=pygame.Rect(dirt_x, dirt_y, dirt_width, dirt_height),
            )

    def _draw_agent(self, surface, scalar):
        """Draws the agent on the surface."""
        # Draw the agent
        agent_rect = self._downsample_rect(self.world.agent_rect, scalar)

        agent_surface = pygame.Surface((agent_rect.width, agent_rect.height))
        agent_surface = agent_surface.convert()
        agent_surface.fill((196, 223, 155))
        # agent_surface.fill((50, 50, 50))

        img = self.agent_eyes.resize(tuple([int(0.9 * agent_rect.width)] * 2),
                                     Image.Resampling.BICUBIC) \
            .rotate(self.world.agent_angle + 90) \
            .transpose(method=Image.Transpose.FLIP_LEFT_RIGHT)
        img = pygame.image.fromstring(img.tobytes(), img.size, "RGBA")
        img_pos = img.get_rect()
        img_pos.center = agent_surface.get_rect().center
        agent_surface.blit(img, img_pos)
        # Draw the a triangle representing its direction
        # Triangle has three points (a, b, c) with a center at the origin
        #         C
        #         ^
        #        / \
        #       / O \
        #      /_____\
        #     A       B
        #
        rads = self.world.agent_angle / 180 * np.pi
        center = agent_surface.get_rect().center
        radius = (agent_rect.width * 0.2,
                  agent_rect.width * 0.05,
                  agent_rect.width * 0.2)
        angles = (4.1887902047863905, 0., 2.0943951023931953)
        points = []

        # For 0, 120, and 240 degrees in radians
        for theta, r in zip(angles, radius):
            x = center[0] + r * math.cos(theta + rads)
            y = center[1] + r * math.sin(theta + rads)
            points.append((x, y))

        pygame.draw.aalines(
            surface=agent_surface,
            color=(168, 100, 168),
            closed=False,
            points=points,
            blend=1
        )
        surface.blit(agent_surface, agent_rect)

    def _draw_info(self, surface, x, padding):
        """Draws run info."""
        font = pygame.font.Font(None, 38)
        human_names = {"successful_moves": "Moves:",
                       "wall_hits": "Wall:",
                       "obstacle_hits": "Obstacles:",
                       "dirt_hits": "Dirt:",
                       "death_hits": "Death:",
                       "is_alive": "Alive:",
                       "score": "Score:",
                       "fps": "FPS:"}
        for idx, (key, value) in enumerate(self.info.items()):
            y_pos = padding + (idx * 38)

            key_label = font.render(f"{human_names[key]}", True, (10, 10, 10))
            key_pos = key_label.get_rect()
            key_pos.x = x
            key_pos.y = y_pos
            surface.blit(key_label, key_pos)

            val_label = font.render(f"{value}", True, (10, 10, 10))
            val_pos = val_label.get_rect()
            val_pos.x = surface.get_rect().width - padding - val_pos.width
            val_pos.y = y_pos
            surface.blit(val_label, val_pos)

    def _draw_observation(self, surface, x, y):
        """Draws the observation array that the agent receives."""
        obs = self.world.last_observation

        walls = np.zeros([24, 24, 3])
        walls[obs[0, :, :] == 1] = np.array([69, 71, 82])

        death = np.zeros_like(walls)
        death[obs[0, :, :] == 3] = np.array([235, 64, 52])

        visited = np.zeros_like(walls)
        visited[obs[3, :, :] == 1] = np.array([113, 52, 235])

        fow = (1 - obs[4, :, :][:, :, np.newaxis]) * np.array([150, 150, 150])

        obstacles = np.full_like(walls, 255)
        obstacles += obs[1, :, :][:, :, np.newaxis] * np.array([-20, -60, -203])

        dirt = np.full_like(walls, 255)
        dirt += obs[2, :, :][:, :, np.newaxis] * np.array([-173, -196, -222])

        image_base = (walls + death + visited)
        # Subtract image_base from fow so the visualization makes sense
        fow[image_base.any(axis=2)] = np.array([0, 0, 0])
        image_base += fow
        base_mask = image_base.any(axis=2)

        for idx, (text, img) in enumerate(zip(("Obstacles", "Dirt"),
                                              (obstacles, dirt))):
            font = pygame.font.Font(None, 24)
            label = font.render(text, True, (10, 10, 10))
            label_pos = label.get_rect()

            img[base_mask] = image_base[base_mask]
            img = Image.fromarray(img.astype('uint8'), mode='RGB') \
                .resize((96, 96), resample=Image.Resampling.NEAREST) \
                .rotate(90) \
                .transpose(method=Image.Transpose.FLIP_TOP_BOTTOM)

            img = pygame.image.fromstring(img.tobytes(), img.size, "RGB")
            img_pos = img.get_rect()
            img_pos.x = x + (128 * idx)
            img_pos.y = y + 28

            # Center the label, put it above the image
            label_pos.centerx = img_pos.centerx
            label_pos.y = y

            surface.blit(img, img_pos)
            surface.blit(label, label_pos)

    def _draw_battery(self, surface, x, bottom_pad):
        """Draws the battery"""
        # Draw the text label
        font = pygame.font.Font(None, 45)
        batt_label = font.render(f"Battery:", True, (10, 10, 10))
        label_pos = batt_label.get_rect()
        label_pos.x = x
        label_pos.y = surface.get_height() - 2 * (label_pos.height + bottom_pad)
        surface.blit(batt_label, label_pos)

        batt_level = font.render(f"{self.world.agent_battery:.02f} %",
                                 True, (10, 10, 10))
        level_pos = batt_level.get_rect()
        level_pos.x = x + label_pos.width + 10
        level_pos.y = label_pos.y
        surface.blit(batt_level, level_pos)

        # Draw the white surface rect
        top = label_pos.y + 45
        width = 324
        height = surface.get_height() - bottom_pad - top
        pygame.draw.rect(
            surface=surface,
            color=(230, 230, 230),
            rect=pygame.Rect(x, top, width, height)
        )
        # Draw the battery meter
        length = (width - 4) * self.world.agent_battery / 100
        pygame.draw.rect(
            surface=surface,
            color=(52, 201, 93),
            rect=pygame.Rect(x + 5, top + 5, int(length), height - 10)
        )

    def close(self):
        if self.should_render:
            pygame.display.quit()
            pygame.quit()

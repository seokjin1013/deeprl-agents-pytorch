import gymnasium
from gymnasium import spaces
import numpy as np
import pygame
from gymnasium.envs.registration import register

class GridWorldEnvV0(gymnasium.Env):
    metadata = {
        'render_modes': ['rgb_array', 'human'],
        'render_fps': 4,
    }

    def __init__(self, render_mode=None, size=5):
        self.size = size
        self.window_size = 512
        self.observation_space = spaces.Dict(
            {
                'agent': spaces.Box(0, size - 1, shape=(2,), dtype=int),
                'target': spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        self.window = None
        self.clock = None

    def _get_obs(self):
        return {'agent': self._agent_location, 'target': self._target_location}
    
    def _get_info(self):
        return {'disatance': np.linalg.norm(self._agent_location - self._target_location, 1)}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        self._target_location = self._agent_location
        while np.array_equal(self._agent_location, self._target_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()
        return observation, info
    
    def step(self, action):
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size-1)
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == 'human':
            self._render_frame()
        return observation, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == 'rgb_array':
            return self._render_frame()
        
    def _render_frame(self):
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size
        pygame.draw.rect(canvas, (255, 0, 0), pygame.Rect(self._target_location * pix_square_size, (pix_square_size, pix_square_size)))
        pygame.draw.circle(canvas, (0, 0, 255), (self._agent_location + 0.5) * pix_square_size, pix_square_size / 3)
        for x in range(self.size + 1):
            pygame.draw.line(canvas, 0, (0, pix_square_size * x), (self.window_size, pix_square_size * x), width=3)
        for y in range(self.size + 1):
            pygame.draw.line(canvas, 0, (pix_square_size * y, 0), (pix_square_size * y, self.window_size), width=3)
        
        if self.render_mode == 'human':
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata['render_fps'])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            

class GridWorldEnvV1(gymnasium.Env):
    metadata = {
        'render_modes': ['rgb_array', 'human'],
        'render_fps': 4,
    }

    def __init__(self, render_mode=None, size=5):
        self.size = size
        self.window_size = 512
        self.observation_space = spaces.Dict(
            {
                'agent': spaces.Box(0, size - 1, shape=(2,), dtype=int),
                'target': spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )
        self.action_space = spaces.Discrete(4)

        self._action_to_direction = {
            0: np.array([1, 0]),
            1: np.array([0, 1]),
            2: np.array([-1, 0]),
            3: np.array([0, -1]),
        }
        assert render_mode is None or render_mode in self.metadata['render_modes']
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        self._elapsed_steps = None

    def _get_obs(self):
        return {'agent': self._agent_location, 'target': self._target_location}
    
    def _get_info(self):
        return {'disatance': np.linalg.norm(self._agent_location - self._target_location, 1)}
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        self._target_location = self._agent_location
        while np.array_equal(self._agent_location, self._target_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)
        
        observation = self._get_obs()
        info = self._get_info()
        self._elapsed_steps = 0

        if self.render_mode == 'human':
            self._render_frame()
        return observation, info
    
    def step(self, action):
        direction = self._action_to_direction[action]
        self._agent_location = np.clip(self._agent_location + direction, 0, self.size-1)
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = self._get_info()
        self._elapsed_steps += 1

        if self.render_mode == 'human':
            self._render_frame()
        return observation, reward, terminated, False, info
    
    def render(self):
        if self.render_mode == 'rgb_array':
            return self._render_frame()
        
    def _render_frame(self):
        if self.window is None and self.render_mode == 'human':
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == 'human':
            self.clock = pygame.time.Clock()
        
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = self.window_size / self.size
        pygame.draw.rect(canvas, (255, 0, 0), pygame.Rect(self._target_location * pix_square_size, (pix_square_size, pix_square_size)))
        pygame.draw.circle(canvas, (0, 0, 255), (self._agent_location + 0.5) * pix_square_size, pix_square_size / 3)
        for x in range(self.size + 1):
            pygame.draw.line(canvas, 0, (0, pix_square_size * x), (self.window_size, pix_square_size * x), width=3)
        for y in range(self.size + 1):
            pygame.draw.line(canvas, 0, (pix_square_size * y, 0), (pix_square_size * y, self.window_size), width=3)
        font = pygame.font.Font(pygame.font.get_default_font(), 42)
        text = font.render(f'timestep: {self._elapsed_steps}', True, (0, 0, 0))
        canvas.blit(text, text.get_rect(topleft=(10, 10)))
        
        if self.render_mode == 'human':
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata['render_fps'])
        else:
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2)
            )
        
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

register(
    id='GridWorld-v0',
    entry_point='grid_world_env:GridWorldEnvV0',
    max_episode_steps=300,
)
register(
    id='GridWorld-v1',
    entry_point='grid_world_env:GridWorldEnvV1',
    max_episode_steps=300,
)
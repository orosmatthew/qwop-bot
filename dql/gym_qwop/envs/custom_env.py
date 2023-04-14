import gym
from gym import spaces
import numpy as np
from dql.gym_qwop.envs import character_simulation
import torch
import pyray as rl


class CustomEnv(gym.Env):
    def __init__(self):
        self.pygame = character_simulation.CharacterSimulation()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32)
        self.step_count = 0
        self.rendered = False

    def reset(self, **kwargs):
        self.step_count = 0
        del self.pygame
        self.pygame = character_simulation.CharacterSimulation()
        observation = torch.tensor(self.pygame.output_list, dtype=torch.float32)
        return observation

    def step(self, action):

        self.step_count += 1
        observation = torch.tensor([*self.pygame.output_list], dtype=torch.float32)

        self.pygame.action(action)
        self.pygame.step(1.0 / 60.0)

        if self.rendered:
            self.pygame.step_render()

        reward: float = self.pygame.fitness
        # done will be connected to the collusion
        done = self.step_count > 1000
        return observation, reward, done, False, {}

    def render(self, mode='human'):
        self.pygame.render()
        self.rendered = True

    def close(self):
        pass

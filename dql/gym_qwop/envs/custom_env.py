import gym
from gym import spaces
import numpy as np
from dql.gym_qwop.envs import character_simulation
import torch


class CustomEnv(gym.Env):
    def __init__(self):
        self.pygame = character_simulation.CharacterSimulation()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32)
        self.step_count = 0

    def reset(self, **kwargs):
        self.step_count = 0
        del self.pygame
        self.pygame = character_simulation.CharacterSimulation()
        observation = torch.tensor(self.pygame.output_list, dtype=torch.float32)
        return observation

    def step(self, action):
        # self.pygame.step(self.pygame.time_step)
        # self.pygame.sim_time += self.pygame.time_step
        # self.pygame.app_time += self.pygame.time_step
        self.step_count += 1
        observation = torch.tensor([*self.pygame.output_list], dtype=torch.float32)

        self.pygame.action(action)
        self.pygame.step(1.0 / 60.0)

        reward: float = self.pygame.fitness
        # done will be connected to the collusion
        done = self.step_count > 1000
        return observation, reward, done, False, {}

    def render(self, mode='human'):
        self.pygame.render()

    def close(self):
        pass

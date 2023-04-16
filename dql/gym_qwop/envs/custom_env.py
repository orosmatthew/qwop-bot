import gym
from gym import spaces
import numpy as np
from dql.gym_qwop.envs import character_simulation
import torch
import pyray as rl
import random


class CustomEnv(gym.Env):
    def __init__(self):
        self.pygame = character_simulation.CharacterSimulation()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, dtype=np.float32)
        self.loop_back = 0
        self.rendered = False

    def reset(self, **kwargs):
        del self.pygame
        self.loop_back = 0
        self.pygame = character_simulation.CharacterSimulation()
        observation = torch.tensor([*self.pygame.output_list, self.loop_back, random.uniform(0, 1)],
                                   dtype=torch.float32)
        return observation

    def step(self, action):
        observation = torch.tensor([*self.pygame.output_list, self.loop_back, random.uniform(0, 1)],
                                   dtype=torch.float32)
        self.loop_back = 0
        if action < 8:
            self.pygame.action(action)
        elif action == 8:
            self.loop_back = 1
        self.pygame.step(1.0 / 60.0)

        if self.rendered:
            self.pygame.step_render()

        reward: float = (self.pygame.fitness * 0.5 + self.pygame.character_position().x) * 0.01
        # done will be connected to the collusion
        done = self.pygame.fitness <= 0

        return observation, reward, done, False, {}

    def render(self, mode='human'):
        self.pygame.render()
        self.rendered = True

    def close(self):
        pass

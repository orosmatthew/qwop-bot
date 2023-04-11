import gym
from gym import spaces
import numpy as np
from dql.gym_qwop.envs import character_simulation


class CustomEnv(gym.Env):
    def __init__(self):
        self.pygame = character_simulation.CharacterSimulation()
        self.action_space = spaces.Discrete(4)
        #TODO: fix observation space
        self.observation_space = spaces.Box()
        self.state = None

    def reset(self):
        del self.pygame
        self.pygame = character_simulation.CharacterSimulation()
        obs = self.pygame.character.outputs
        return obs

    def step(self, action):
        self.pygame.action(action)
        observation = self.pygame.outputs
        reward = self.pygame.fitness
        done = False
        info = {}
        return observation, reward, done, info

    def render(self, mode='human'):
        self.pygame.render()

    def close(self):
        pass

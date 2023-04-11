import gym
from gym import spaces
import numpy as np
from gym_qwop.envs.qwop_game import QwopGame


class MyGameEnv(gym.Env):
    def __init__(self):
        self.pygame = QwopGame()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]), dtype=np.float32)
        self.state = None

    def reset(self):
        del self.pygame
        self.pygame = QwopGame()
        obs = self.pygame.observe()

        return self.state

    def step(self, action):
        reward = np.sum(action * self.state)
        self.state += action
        observation = self.state + np.random.normal(loc=0, scale=0.1, size=(3,))
        done = False
        info = {}
        return observation, reward, done, info

    def render(self, mode='human'):
        pass

    def close(self):
        pass





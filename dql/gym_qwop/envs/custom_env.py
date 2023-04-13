import gym
from gym import spaces
import numpy as np
from dql.gym_qwop.envs import character_simulation


class CustomEnv(gym.Env):
    def __init__(self):
        self.pygame = character_simulation.CharacterSimulation()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, dtype=np.float64)
        self.state = None

    def reset(self):
        del self.pygame
        self.pygame = character_simulation.CharacterSimulation()
        obs = self.pygame.outputs
        return obs

    def step(self, action):
        # self.pygame.step(self.pygame.time_step)
        # self.pygame.sim_time += self.pygame.time_step
        # self.pygame.app_time += self.pygame.time_step

        self.pygame.action(action)

        obs = self.pygame.outputs
        reward = self.pygame.fitness
        # done will be connected to the collusion
        done = False
        turnicated = False
        info = {}
        return obs, reward, done, turnicated, info

    def render(self, mode='human'):
        self.pygame.render()

    def close(self):
        pass

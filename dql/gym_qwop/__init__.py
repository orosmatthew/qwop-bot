import gym
from gym.envs.registration import register

register(
    id="QWOP",
    entry_point="dql.gym_qwop.envs:CustomEnv",
    max_episode_steps=2000
)
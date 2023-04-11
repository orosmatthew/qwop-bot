from gym.envs.registration import register

register(
    id="QWOP-v0",
    entry_point="gym_qwop.envs:CustomEnv",
    max_episode_steps=2000,
)
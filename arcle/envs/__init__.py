from .arcenv import *
from gymnasium.envs.registration import register
register(
    id='ARCLE/ArcEnv-v0',
    entry_point='arcle.envs.arcenv:ARCEnv',
    max_episode_steps=100,
)

register(
    id='ARCLE/MiniArcEnv-v0',
    entry_point='arcle.envs.arcenv:MiniARCEnv',
    max_episode_steps=100,
)
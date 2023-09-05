from .arcenv import AbstractARCEnv, ARCEnv, MiniARCEnv
from .o2arcenv import O2ARCv2Env
from gymnasium.envs.registration import register

register(
    id='ARCLE/ARCEnv-v0',
    entry_point='arcle.envs.arcenv:ARCEnv',
    max_episode_steps=100,
)

register(
    id='ARCLE/MiniARCEnv-v0',
    entry_point='arcle.envs.arcenv:MiniARCEnv',
    max_episode_steps=100,
)

register(
    id='ARCLE/O2ARCv2Env-v0',
    entry_point='arcle.envs.o2arcenv:O2ARCv2Env',
    max_episode_steps=1000000,
)
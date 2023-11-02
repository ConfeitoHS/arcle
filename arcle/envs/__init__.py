from .base import AbstractARCEnv
from .arcenv import RawARCEnv, ARCEnv
from .o2arcenv import O2ARCv2Env
from gymnasium.envs.registration import register

register(
    id='ARCLE/RawARCEnv-v0',
    entry_point='arcle.envs.arcenv:RawARCEnv',
)

register(
    id='ARCLE/ARCEnv-v0',
    entry_point='arcle.envs.arcenv:ARCEnv',
)

register(
    id='ARCLE/O2ARCv2Env-v0',
    entry_point='arcle.envs.o2arcenv:O2ARCv2Env'
)
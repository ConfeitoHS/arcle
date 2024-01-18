import gymnasium as gym
from gymnasium.core import Env

class SelectionBBoxWrapper(gym.Wrapper):
    def __init__(self, env: Env):
        super().__init__(env)
        
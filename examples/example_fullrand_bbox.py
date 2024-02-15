import arcle
from arcle.wrappers import BBoxWrapper
import gymnasium as gym

env = gym.make('ARCLE/O2ARCEnv', render_mode='ansi')
env = BBoxWrapper(env)

obs, info = env.reset()
action = env.action_space.sample()
print(action) # 5-tuple: (y1, x1, y2, x2, op)
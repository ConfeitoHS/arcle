import arcle
import gymnasium as gym
from arcle.wrappers import BBoxWrapper

env = gym.make('ARCLE/RawARCEnv-v0', render_mode='ansi')
env = BBoxWrapper(env)
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample() # 5-tuple (X1, Y1, X2, Y2, OP)
    obs, reward, term, trunc, info = env.step(action)
    if term or trunc:
        obs, info = env.reset()

env.close()

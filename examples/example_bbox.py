import arcle
from arcle.wrappers import BBoxWrapper
import gymnasium as gym
import time

env = gym.make("ARCLE/O2ARCEnv", render_mode="ansi")
env = BBoxWrapper(env)

obs, info = env.reset()
action = env.action_space.sample()
print(action)  # 5-tuple: (y1, x1, y2, x2, op)

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    if term or trunc:
        obs, info = env.reset()
    time.sleep(0.3)

env.close()

import arcle
import gymnasium as gym

env = gym.make("ARCLE/O2ARCEnv", render_mode="ansi")
obs, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    if term or trunc:
        obs, info = env.reset()

env.close()

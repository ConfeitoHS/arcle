import arcle
import gymnasium as gym
import time

env = gym.make('ARCLE/RawARCEnv-v0',render_mode='ansi')

obs, info = env.reset()

for _ in range(1000):
    
    action = env.action_space.sample()
    
    obs,reward,term,trunc,info = env.step(action)

    if term or trunc:
        obs, info = env.reset()
        
    time.sleep(0.3)

env.close()

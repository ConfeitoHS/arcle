from gymnasium.envs.registration import register
register(
    id='ARCLE/ArcEnv-v0',
    entry_point='arcle.envs.arcenv:ArcEnv',
    max_episode_steps=100,
)

register(
    id='ARCLE/MiniArcEnv-v0',
    entry_point='arcle.envs.arcenv:MiniArcEnv',
    max_episode_steps=100,
)
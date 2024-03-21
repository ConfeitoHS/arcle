from typing import Dict, List, Tuple

from numpy.typing import NDArray
import arcle
import gymnasium as gym
import time
import numpy as np
from arcle.loaders import ARCLoader, Loader


class TestLoader(Loader):
    def get_path(self, **kwargs) -> List[str]:
        return [""]

    def parse(
        self, **kwargs
    ) -> List[Tuple[List[NDArray], List[NDArray], List[NDArray], List[NDArray], Dict]]:
        ti = np.zeros((30, 30), dtype=np.int8)
        to = np.zeros((30, 30), dtype=np.int8)
        ei = np.zeros((30, 30), dtype=np.int8)
        eo = np.zeros((30, 30), dtype=np.int8)

        ti[0:10, 0:10] = np.random.randint(0, 10, size=[10, 10])
        return [([ti], [to], [ei], [eo], {"desc": "just for test"})]


env = gym.make(
    "ARCLE/O2ARCv2Env-v0",
    render_mode="ansi",
    data_loader=TestLoader(),
    max_grid_size=(30, 30),
    colors=10,
    max_episode_steps=None,
)

obs, info = env.reset(options={})
valid_action = np.array([0] * 35, dtype=np.int8)
valid_action[[-1, -2, -3, -4]] = 0  # Critical
valid_action[np.arange(10)] = 0  # Colors
valid_action[[28, 29, 30]] = [0, 0, 0]  # Copy/paste
valid_action[[20, 21, 22, 23, 24, 25, 26, 27]] = 0
valid_action[-2] = 1

sels = [((1, 10), (1, 9)), ((1, 17), (1, 25)), ((2, 6), (2, 6))]

while True:
    sel = np.zeros((30, 30), dtype=np.bool_)
    xch, ych = sels[env.unwrapped.action_steps]
    sel[xch[0] : xch[1] + 1, ych[0] : ych[1] + 1] = 1

    op = env.action_space["operation"].sample(mask=valid_action)

    action = {"selection": sel, "operation": op}

    obs, reward, term, trunc, info = env.step(action)

    if term or trunc:
        obs, info = env.reset()
    # input('\033[F')
    time.sleep(1)


env.close()

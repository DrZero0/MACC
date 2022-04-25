from functools import partial
from smac.env import MultiAgentEnv, StarCraft2Env
from .stag_hunt import StagHunt
from smac_plus import ForagingEnv

import sys
import os


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {
    "sc2": partial(env_fn, env=StarCraft2Env),
    "foraging": partial(env_fn, env=ForagingEnv),
    "stag_hunt": partial(env_fn, env=StagHunt), 
}
# REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))

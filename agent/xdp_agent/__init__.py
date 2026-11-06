import os
import sys

# The competition's evaluation image lacks stable-baselines3; a copy bundled under vendor/, when
# present, is put on sys.path before anything imports it. Its own dependencies (torch, numpy,
# gymnasium, cloudpickle, pandas) are available there.
_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if os.path.isdir(_VENDOR) and _VENDOR not in sys.path:
    sys.path.insert(0, _VENDOR)

from .gym_utils import SurvivalReward as reward  # noqa: E402
from .my_agent import MyAgent, make_agent  # noqa: E402

# `reward` is picked up by the competition ingestion/scoring programs
# (`from submission import reward`) and used as the environment reward during retraining
# of the PPO topology proposer. It has no effect on the rule engine or last resort.
__all__ = ["MyAgent", "make_agent", "reward"]

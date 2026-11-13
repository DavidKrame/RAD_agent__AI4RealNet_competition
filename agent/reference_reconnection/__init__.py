"""Reference controller: reconnects disconnected lines and does nothing else.

Grid2Op's RecoPowerlineAgent reconnects a disconnected line as soon as its cooldown allows. It isolates what
line reconnection alone achieves, without topology changes, redispatch, curtailment or storage.
"""
from grid2op.Agent import RecoPowerlineAgent


def make_agent(env, this_directory_path):
    return RecoPowerlineAgent(env.action_space)

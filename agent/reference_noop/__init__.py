"""Reference controller: never acts.

It shows what the 764 local scenarios yield when the network is left to its own protection and to the
scheduled and adversarial outages, with no control at all.
"""
from grid2op.Agent import DoNothingAgent


def make_agent(env, this_directory_path):
    return DoNothingAgent(env.action_space)

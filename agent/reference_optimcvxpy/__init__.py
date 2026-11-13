"""Reference controller: the OptimCVXPY baseline of L2RPN Baselines v0.8.0, alone.

The optimiser runs with its published default parameters, switching between its own safe and unsafe modes
as distributed. The two files next to this one are the same copies the agent in `agent/xdp_agent/` uses:
the vendored optimiser and the layer that adapts it to the pinned CVXPY version.
"""
try:
    from .optim_cvxpy_compat import make_optimcvxpy
except ImportError:
    from optim_cvxpy_compat import make_optimcvxpy


def make_agent(env, this_directory_path):
    return make_optimcvxpy(env.action_space, env)

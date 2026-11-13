"""Compatibility layer between the OptimCVXPY baseline of l2rpn_baselines 0.8.0 and the pinned packages.

`optimCVXPY.py` is included as `optim_cvxpy_vendor.py` (MPL-2.0, from l2rpn_baselines 0.8.0) and imported
from there, so loading does not go through the l2rpn_baselines package `__init__` and needs only cvxpy,
numpy, grid2op and lightsim2grid. Four incompatibilities of the original file with the pinned versions
are fixed here:

  1. lightsim2grid 0.13.1 renamed `lightsim2grid.gridmodel.init` to `init_from_pandapower`;
  2. numpy 2.3.5 removed `np.NaN`;
  3. cvxpy 1.9.2 rejects float arrays as indices, while the original code indexes with the `.value` of
     integer Parameters, which cvxpy returns as float64;
  4. `_update_topo_param` modifies grid2op's shared load, generator and storage `to_subid` arrays in
     place, which corrupts later observations; it is overridden below to copy them.

Fixes 1 and 2 are applied before the vendored module is imported. Fix 3 replaces each bus-index
parameter, which is only ever used through `.value`, with a small holder backed by an int64 array, so
that both in-place updates and integer indexing work.
"""
import numpy as np
import cvxpy as cp

# --- fixes 1 and 2, applied before importing the vendored module ---
import lightsim2grid.gridmodel as _gm
if not hasattr(_gm, "init"):
    _gm.init = _gm.init_from_pandapower
if not hasattr(np, "NaN"):
    np.NaN = np.nan

try:  # packaged on Codabench (submission.optim_cvxpy_vendor)
    from .optim_cvxpy_vendor import OptimCVXPY as _OptimCVXPY  # noqa: E402
except ImportError:  # run locally as a flat script
    from optim_cvxpy_vendor import OptimCVXPY as _OptimCVXPY  # noqa: E402


class OptimCVXPY(_OptimCVXPY):
    """OptimCVXPY that tries CLARABEL first. The original tries OSQP, then SCS, then SCIPY; on the
    IEEE-118 program CLARABEL solves the same problem in about 160 ms, against about 420 ms for OSQP and
    700 ms for SCS. CLARABEL is among the pinned requirements.
    """
    SOLVER_TYPES = [cp.CLARABEL, cp.OSQP, cp.SCS]

    def _update_topo_param(self, obs):
        """Same as the original, but copies `load_to_subid`, `gen_to_subid` and `storage_to_subid` before
        adding `n_sub`. The original copies the line arrays but not these three, so `tmp_[bus == 2] += n_sub`
        modifies grid2op's shared class-level arrays in place and corrupts every later observation. The bus
        parameters are the int-backed holders installed by `make_optimcvxpy`.
        """
        tmp_ = 1 * obs.line_or_to_subid
        tmp_[obs.line_or_bus == 2] += obs.n_sub
        self.bus_or.value[:] = tmp_
        tmp_ = 1 * obs.line_ex_to_subid
        tmp_[obs.line_ex_bus == 2] += obs.n_sub
        self.bus_ex.value[:] = tmp_
        disc = (obs.line_or_bus == -1) | (obs.line_ex_bus == -1)
        self.bus_ex.value[disc] = 0
        self.bus_or.value[disc] = 0
        tmp_ = 1 * obs.load_to_subid
        tmp_[obs.load_bus == 2] += obs.n_sub
        self.bus_load.value[:] = tmp_
        tmp_ = 1 * obs.gen_to_subid
        tmp_[obs.gen_bus == 2] += obs.n_sub
        self.bus_gen.value[:] = tmp_
        if self.bus_storage is not None:
            tmp_ = 1 * obs.storage_to_subid
            tmp_[obs.storage_bus == 2] += obs.n_sub
            self.bus_storage.value[:] = tmp_


class _IntBusParam:
    """Stand-in for a bus-index cp.Parameter: a mutable int64 array exposed as `.value`, so both
    in-place updates (`.value[:] = ...`) and integer fancy indexing (`theta[.value]`) work under
    cvxpy 1.9.2, which coerces cp.Parameter `.value` to float and then rejects it as an index."""
    __slots__ = ("value",)

    def __init__(self, arr):
        self.value = np.asarray(arr, dtype=np.int64)


def make_optimcvxpy(action_space, env, **kwargs):
    """Build an OptimCVXPY agent whose bus-index params are int-backed (cvxpy-1.9.2 safe)."""
    opt = OptimCVXPY(action_space, env, **kwargs)
    for _name in ("bus_or", "bus_ex", "bus_load", "bus_gen", "bus_storage"):
        _p = getattr(opt, _name)
        if _p is not None:
            setattr(opt, _name, _IntBusParam(_p.value))
    return opt

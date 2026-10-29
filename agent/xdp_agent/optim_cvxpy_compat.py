"""Compatibility layer for l2rpn_baselines 0.8.0 OptimCVXPY against the FROZEN Codabench env.

`optimCVXPY.py` is VENDORED into this submission as `optim_cvxpy_vendor.py` (MPL-2.0, from
l2rpn_baselines 0.8.0) and imported from there rather than via `from l2rpn_baselines.OptimCVXPY
import ...`, so loading never traverses the l2rpn_baselines.OptimCVXPY package `__init__` (which also
imports `evaluate`/`make_agent`) and depends on nothing beyond the packages already in
requirements.lock (cvxpy, numpy, grid2op, lightsim2grid). The vendored file is broken as-shipped
against the LOCKED deps; four issues are fixed here (all verified 2026-07-18/19):

  1. lightsim2grid 0.13.1 renamed `lightsim2grid.gridmodel.init` -> `init_from_pandapower`
     (module-level import in optimCVXPY.py fails).
  2. numpy 2.3.5 removed `np.NaN` (used throughout optimCVXPY.py).
  3. cvxpy 1.9.2 rejects FLOAT arrays as fancy indices; the upstream code indexes cp expressions /
     numpy masks with `integer=True` Parameter `.value` arrays, whose `.value` cvxpy nonetheless
     returns as float64 -> `theta[self.bus_or.value]`, `_mask_theta_zero`, `to_grid2op` all raise
     `IndexError: arrays used as indices must be of integer (or boolean) type`.
  4. `_update_topo_param` mutates grid2op's SHARED load/gen/storage `to_subid` arrays in place
     (missing copy) -> corrupts every later observation; overridden below to copy.

Shims 1-2 are pure-Python module patches applied BEFORE the vendored module is imported. #3 is fixed
structurally: every bus-index parameter (`bus_or/bus_ex/bus_load/bus_gen/bus_storage`) is used ONLY
through `.value` (for fancy indexing, in-place update, or `== bus_id` masks) and never as a cvxpy
expression, so we swap each for a minimal holder backed by a mutable int64 array. This makes both
`.value[:] = ...` (in-place topology update) and `theta[.value]` (integer fancy index) valid, fixing
all sites at once without duplicating any optimizer method.
"""
import numpy as np
import cvxpy as cp

# --- shims 1 & 2: apply BEFORE importing the vendored optimCVXPY ---
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
    """OptimCVXPY that tries CLARABEL first. Upstream defaults to OSQP -> SCS -> SCIPY; on the
    IEEE-118 unsafe OPF, CLARABEL solves the same problem in ~160 ms vs OSQP ~420 ms / SCS ~700 ms
    (measured 2026-07-18). CLARABEL is in requirements.lock, so this is submission-safe."""
    SOLVER_TYPES = [cp.CLARABEL, cp.OSQP, cp.SCS]

    def _update_topo_param(self, obs):
        """Same as upstream, but COPIES `load_to_subid` / `gen_to_subid` / `storage_to_subid`
        before adding `n_sub`. Upstream copies the line arrays (`1 * obs.line_or_to_subid`) but not
        these three, so `tmp_[bus == 2] += n_sub` mutates grid2op's SHARED class-level to_subid
        arrays in place, corrupting every later observation (indices accumulate `+= n_sub` per call
        -> out-of-bounds). The bus params are the int-backed holders installed by `make_optimcvxpy`."""
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

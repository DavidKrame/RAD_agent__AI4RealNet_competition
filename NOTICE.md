# Third-party code

* `agent/xdp_agent/optim_cvxpy_vendor.py` is a copy of `OptimCVXPY` from RTE's
  [L2RPN Baselines](https://github.com/Grid2op/l2rpn-baselines) (v0.8.0), distributed under the Mozilla
  Public License 2.0, with its original header kept. `optim_cvxpy_compat.py` adapts it to the CVXPY version
  pinned in `requirements.txt`.
* The agent runs on [Grid2Op](https://github.com/Grid2Op/grid2op) and
  [LightSim2Grid](https://github.com/Grid2Op/lightsim2grid) (both MPL-2.0), which are installed from
  `requirements.txt`, not copied here.
* The `l2rpn_idf_2023` dataset is downloaded by Grid2Op and is not redistributed in this repository.

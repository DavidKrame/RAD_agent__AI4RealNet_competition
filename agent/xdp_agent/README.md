# `xdp_agent`

The controller described in the paper, packaged in the L2RPN challenge submission format: `__init__.py`
exposes `make_agent(env, path)`, and the agent implements `act`, `reset`, `train`, `save_state` and
`load_state`.

| file | role |
|---|---|
| `my_agent.py` | the controller: control cycle, candidate generation, ranking, safeguards, `train()` |
| `ac_dispatch.py` | AC-anchored convex dispatch problem (redispatch, curtailment, storage) |
| `optim_cvxpy_vendor.py`, `optim_cvxpy_compat.py` | RTE's OptimCVXPY baseline (MPL-2.0) and a compatibility wrapper |
| `action_space.json`, `action_pool.json`, `action_pool_v3.json` | offline topology libraries |
| `build_action_space.py`, `build_action_pool.py` | scripts that build those libraries |
| `fatal_lines.json` | offline record of postures that survive each scheduled outage |
| `gla_library.json` | offline topology postures for a known fragile line |
| `config.json` | thresholds and search sizes |
| `prepare_data.py` | downloads the dataset and creates the 764/34/34 split |
| `gym_utils.py`, `train_ppo.py` | helpers of an early learned baseline, not used by this agent |

`docs/code_map.md` at the repository root maps each mechanism of the paper to its function.

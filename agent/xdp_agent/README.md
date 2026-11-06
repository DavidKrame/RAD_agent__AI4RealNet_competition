# `xdp_agent`

The agent as a self-contained package in the L2RPN competition format: `__init__.py`
exposes `make_agent(env, path)`, and the agent implements `act`, `reset`, `train`, `save_state` and
`load_state`.

| file | role |
|---|---|
| `my_agent.py` | the controller: control cycle, candidate generation, ranking, safeguards, `train()` |
| `ac_dispatch.py` | AC-anchored convex dispatch problem (redispatch, curtailment, storage) |
| `optim_cvxpy_vendor.py`, `optim_cvxpy_compat.py` | RTE's OptimCVXPY baseline (MPL-2.0) and a compatibility wrapper |
| `action_space.json`, `action_pool.json` | offline topology libraries |
| `action_pool_v3.json` | an alternative library included in the evaluated package; not loaded by the agent |
| `build_action_space.py`, `build_action_pool.py` | scripts that build those libraries |
| `fatal_lines.json` | offline record of 246 configurations of substation 79 that survived the scheduled outage of line 175 |
| `gla_library.json` | offline busbar postures for two fragile attackable lines (154 and 180), used by the guarded-line rule |
| `config.json` | thresholds and search sizes |
| `prepare_data.py` | downloads the dataset and creates the 764/34/34 split |
| `gym_utils.py`, `train_ppo.py` | training scripts of an optional PPO topology proposer; this configuration ships no weights |

`docs/code_map.md` at the repository root maps each mechanism to its function.

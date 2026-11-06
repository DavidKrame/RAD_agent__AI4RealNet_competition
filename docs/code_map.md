# Where each mechanism lives in the code

All paths are relative to `agent/xdp_agent/`. Line numbers refer to the files as published. Every mechanism
is on by default; the environment variable in the last column switches it off (`=0`) for an ablation run, for
example `EXP_FLEXFLOOR=0 python evaluation/eval_full.py --submission xdp_agent ...`.

| paper section | mechanism | code | switch |
|---|---|---|---|
| Section 3 | PTDF and LODF, computed by LightSim2Grid for the environment the agent is built on | `ac_dispatch.py`: `_ptdf` (l. 160); `my_agent.py`: `_lodf_for` (l. 1753) | |
| Section 4.2, Fig. 1, eq. (8) | control cycle: early-exit rules, operating regimes, then candidate search and ranking | `my_agent.py`: `act` (l. 1214), `_relief` (l. 2004), `_relief_danger_pick` (l. 2339) | |
| Section 4.1, eq. (5)-(7) | peak loading $F_H$, state cost $\Psi$, and cost-to-go $J_H$, evaluated by forward simulation | `_composite` (l. 3065), `_worst_rho` (l. 2879), `_sim` (l. 1446) | |
| Section 4.3 (i), App. C | offline library of single-substation bus reassignments | `action_pool.json`, built by `build_action_pool.py` | |
| Section 4.3 (ii), App. C | local pool around current congestion | `_build_sub_topo_pool` (l. 2904) and the danger branch of `_relief` | `EXP_BIGPOOL` |
| Section 4.3 (iii), eq. (9) | LODF-screened voluntary line openings | `_lodfcut_candidates` (l. 1775) | `EXP_LODFCUT` |
| Section 4.3, App. C | OptimCVXPY DC optimal power flow, and its emergency variant | `_cvxpy_candidate` (l. 1539), `_cvxpy_emergency_candidate` (l. 1570), `optim_cvxpy_vendor.py` | `EXP_CVXPY`, `EXP_EMERGENCY` |
| Section 4.4, eq. (11)-(14) | AC-anchored convex dispatch QP with per-generator ramp bounds | `ac_dispatch.py`: `ACAnchoredDispatch.solve` (l. 169); called by `_acopf_candidate` (l. 1514) | `EXP_ACOPF` |
| Section 4.4 | pending-outage security constraint inside the QP | `ac_dispatch.py`, block guarded by `self._secn1` | `EXP_SECN1` |
| Section 4.5, eq. (15) | composition of the best topology and continuous candidates | `_relief_danger_pick` (l. 2339), `_fs_compose_pick` (l. 1482) | `EXP_DPCOMP`, `EXP_FS_COMPOSE` |
| Section 4.6, Alg. 1 | preventive withholding of one line per region | `_fullstarve_arm` (l. 1423), `_starve_filter` (l. 2770); withheld lines in `EXP_FULLSTARVE_LINES` (default `106,131,33`) | `EXP_FULLSTARVE` |
| Section 4.7, Alg. 3 | scheduled-outage look-ahead with the offline posture record | `_maint_guard` (l. 1646); record in `fatal_lines.json` | `EXP_MAINT_GUARD` |
| Section 4.7 | margin preservation (downward ramp room) | `_flex_topup_action` (l. 2578), `_flex_room_vec` (l. 2567) | `EXP_FLEXFLOOR` |
| Section 4.7, App. D | extended-horizon safeguard | `_sustain_vet_pick` (l. 2271) | |
| Section 4.7, eq. (16) | revalidation of the withheld lines on new data, run by `train()` | `train` (l. 3244), `_ratchet_train` (l. 1301), `_ratchet_score` (l. 1269) | `EXP_RATCHET`, cap `EXP_RATCHET_CAP_S` (3600 s) |

`train_ppo.py` and the reinforcement-learning helpers in `gym_utils.py` belong to an early learned baseline.
They are not used by the agent, whose `act()` contains no learned component.

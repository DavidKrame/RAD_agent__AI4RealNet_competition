# Where each mechanism lives in the code

All paths are relative to `agent/xdp_agent/`, and line numbers refer to the files as published. Every
mechanism is on by default; the environment variable in the last column switches it off (`=0`), for example
`EXP_FLEXFLOOR=0 python evaluation/eval_full.py --submission xdp_agent ...`. The second column links to the
description in [`how_the_agent_works.md`](how_the_agent_works.md).

| mechanism | described in | code | switch |
|---|---|---|---|
| control cycle: the four stages in order, the operating regimes, candidate search and ranking | [§1](how_the_agent_works.md#1-one-step-of-act), [§5](how_the_agent_works.md#5-stage-3-reactive-control) | `my_agent.py`: `act` (l. 853), `_relief` (l. 1577), `_relief_inner` (l. 1600), `_relief_danger_pick` (l. 1868) | |
| peak loading $F_H$, state cost $\Psi$ and cost $J_H$, by forward simulation | [§2](how_the_agent_works.md#2-how-every-candidate-is-scored) | `_composite` (l. 2560), `_worst_rho` (l. 2378), `_sim` (l. 1066) | |
| scheduled-outage look-ahead with the offline posture record | [§3](how_the_agent_works.md#3-stage-1-scheduled-outage-look-ahead) | `_maint_guard` (l. 1242); record in `fatal_lines.json` | `EXP_MAINT_GUARD` |
| preventive withholding of one line per region | [§4](how_the_agent_works.md#4-stage-2-preventive-withholding) | `_fullstarve_arm` (l. 1047), `_starve_filter` (l. 2277); withheld lines in `EXP_FULLSTARVE_LINES` (default `106,131,33`) | `EXP_FULLSTARVE` |
| relief added to the opening of the withheld lines | [§4](how_the_agent_works.md#4-stage-2-preventive-withholding) | `_fs_compose_pick` (l. 1090) | `EXP_FS_COMPOSE` |
| guarded-line rule | [§5](how_the_agent_works.md#5-stage-3-reactive-control) | `_gla_step` (l. 1403); postures in `gla_library.json` | `EXP_GLA` |
| reconnection of recoverable lines | [§5](how_the_agent_works.md#5-stage-3-reactive-control) | `_reconnection` (l. 2313) | |
| curtailment lift and return of redispatch in the comfortable regime | [§5.1](how_the_agent_works.md#51-comfortable-regime-and-margin-preservation) | `_cancel_curtailment_action` (l. 2004), `_cancel_redispatch_action` (l. 2153) | `EXP_CANCEL_CURTAIL`, `EXP_CANCEL_REDISP` |
| margin preservation (downward ramp room) | [§5.1](how_the_agent_works.md#51-comfortable-regime-and-margin-preservation) | `_flex_topup_action` (l. 2094), `_flex_room_vec` (l. 2083) | `EXP_FLEXFLOOR` |
| OptimCVXPY DC optimal power flow, and its second instance solved while an overflow counter runs | [§5.2](how_the_agent_works.md#52-preventive-regime) | `_cvxpy_candidate` (l. 1145), `_cvxpy_emergency_candidate` (l. 1175), `optim_cvxpy_vendor.py` | `EXP_CVXPY`, `EXP_EMERGENCY` |
| anticipatory dispatch against the likeliest attack | [§5.2](how_the_agent_works.md#52-preventive-regime) | `_anticipatory_opf` (l. 1524) | `EXP_ANTICIP_OPF` |
| offline library of single-substation busbar assignments | [§5.3](how_the_agent_works.md#53-danger-regime-candidate-search) | `action_pool.json`, built by `build_action_pool.py` | |
| local pool around current congestion | [§5.3](how_the_agent_works.md#53-danger-regime-candidate-search) | `_build_sub_topo_pool` (l. 2403) and the danger branch of `_relief` | `EXP_BIGPOOL` |
| composition of the best topology stack and the best continuous candidate | [§5.3](how_the_agent_works.md#53-danger-regime-candidate-search) | `_relief_danger_pick` (l. 1868) | `EXP_DPCOMP` |
| LODF-screened line openings | [§5.4](how_the_agent_works.md#54-line-opening-screen) | `_lodfcut_candidates` (l. 1353), `_lodf_for` (l. 1331) | `EXP_LODFCUT` |
| AC-anchored dispatch with per-generator ramp bounds | [§5.5](how_the_agent_works.md#55-ac-anchored-dispatch) | `ac_dispatch.py`: `ACAnchoredDispatch.solve` (l. 83), PTDF from `_ptdf` (l. 74); called by `_acopf_candidate` (l. 1120) | `EXP_ACOPF` |
| security constraints for scheduled outages inside the dispatch program | [§5.5](how_the_agent_works.md#55-ac-anchored-dispatch) | `ac_dispatch.py`, block guarded by `self._secn1` | `EXP_SECN1` |
| extended-horizon safeguard | [§6](how_the_agent_works.md#6-stage-4-extended-horizon-safeguard) | `_sustain_vet_pick` (l. 1800) | `EXP_SUSTAIN_VET` |
| revalidation of the withheld lines in `train()` | [§7](how_the_agent_works.md#7-revalidation-in-train) | `train` (l. 2733), `_ratchet_train` (l. 930), `_ratchet_score` (l. 907) | `EXP_RATCHET`, cap `EXP_RATCHET_CAP_S` (3600 s) |

PTDF and LODF are computed by LightSim2Grid for the environment on which the agent is constructed (see the
implementation notes in the [README](../README.md#implementation-notes)).

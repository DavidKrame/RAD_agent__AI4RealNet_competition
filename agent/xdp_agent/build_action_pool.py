"""Build a hub-INCLUSIVE, usefulness-SCREENED per-substation topology pool for the candidate search.

Rejects two free assumptions of the baseline / v0.1:
  1. `build_reduced_actions` excludes >8-element hubs (subs 48/99/...) via max_elems=8.
  2. v0.1 kept the *first* 40 configs per sub (enumeration order, not usefulness).

Here, for every substation with >=4 elements, we take its configs (for hubs, a RANDOM SAMPLE of the
full 2^(k-1) space, not the first slice), and keep only those that *measurably relieve congestion*:
a config is kept if, on at least one stressed state, its 1-step simulated max-rho is below do-nothing's
by > `margin`. Configs are ranked by how many stressed states they help, capped per sub. The result is
indexed by substation -> action_pool.json; the agent localizes to the congestion at runtime.

Screening states are do-nothing-driven high-rho states (a proxy for attack-driven congestion; noted).
"""
import argparse
import json
import os
import random
import warnings

warnings.filterwarnings("ignore")
import numpy as np
import grid2op

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="l2rpn_idf_2023_train")
    ap.add_argument("--n-states", type=int, default=30)
    ap.add_argument("--rho-stress", type=float, default=0.85)
    ap.add_argument("--hub-sample", type=int, default=500, help="random configs sampled per hub (>this many configs)")
    ap.add_argument("--keep-per-sub", type=int, default=60)
    ap.add_argument("--margin", type=float, default=0.01)
    ap.add_argument("--out", default=os.path.join(HERE, "action_pool.json"))
    a = ap.parse_args()

    try:
        from lightsim2grid import LightSimBackend as BK
        env = grid2op.make(a.env, backend=BK())
    except Exception:
        env = grid2op.make(a.env)
    dn = env.action_space({})
    rng = random.Random(0)

    # 1) collect stressed states + their do-nothing 1-step worst rho
    env.seed(0); obs = env.reset(); states = []
    guard = 0
    while len(states) < a.n_states and guard < 200000:
        guard += 1
        if float(obs.rho.max()) > a.rho_stress:
            try:
                sim, _, d, _ = obs.simulate(dn, time_step=1)
                if not d:
                    states.append((obs.copy(), float(sim.rho.max())))
            except Exception:
                pass
        obs, _, done, _ = env.step(dn)
        if done:
            obs = env.reset()
    print(f"collected {len(states)} stressed states (rho>{a.rho_stress})", flush=True)

    # 2) per-sub: sample configs, screen for congestion relief
    A = env.action_space
    sub_info = env.sub_info
    pool = {}
    for s in range(int(env.n_sub)):
        if int(sub_info[s]) < 4:
            continue
        try:
            acts = list(A.get_all_unitary_topologies_set(A, sub_id=s))
        except Exception:
            continue
        if not acts:
            continue
        if len(acts) > a.hub_sample:
            acts = rng.sample(acts, a.hub_sample)
        scored = []
        for act in acts:
            hits = 0
            for st, dn_rho in states:
                try:
                    sim, _, d, _ = st.simulate(act, time_step=1)
                    if (not d) and float(sim.rho.max()) < dn_rho - a.margin:
                        hits += 1
                except Exception:
                    pass
            if hits > 0:
                scored.append((hits, act))
        scored.sort(key=lambda t: -t[0])
        kept = [act for _, act in scored[: a.keep_per_sub]]
        if kept:
            pool[s] = kept
        print(f"  sub {s:3d} ({int(sub_info[s])} elems, {len(acts)} tried): kept {len(pool.get(s, []))}", flush=True)

    # 3) serialize by substation
    out = {str(s): [act.as_serializable_dict() for act in acts] for s, acts in pool.items()}
    with open(a.out, "w") as f:
        json.dump(out, f)
    total = sum(len(v) for v in pool.values())
    print(f"WROTE {total} screened actions across {len(pool)} subs "
          f"(hubs {[s for s in [48,95,99,31] if s in pool]}) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()

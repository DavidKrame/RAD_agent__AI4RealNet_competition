"""Recompute the reported results from the stored records in `results/`.

    python analysis/reproduce_results.py

No simulation is run: the script only reads the stored per-scenario records, so anyone can check that the
reported figures follow from the data. To regenerate the data themselves, see `evaluation/README.md`.

Sources
-------
* `results/competition/{validation,final_test}/` are the files the competition's evaluation returned for
  the agent (scores.txt, score_episodes.csv, res_agent.json). A scenario is completed when the number of steps played
  equals the scenario length.
* `results/local_764/*.json` are local full-set runs on the 764 scenarios listed in `data/chronics_764.txt`,
  one record per scenario as written by `evaluation/eval_full.py`. `xdp_agent.json` is the agent as released;
  each `without_*.json` is the same agent with one component switched off, so each comparison differs in that
  component only; each `reference_*.json` is a reference controller from `agent/reference_*`.
"""
import json
import os
from math import comb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")


def competition(split):
    folder = os.path.join(RES, "competition", split)
    scores = {}
    with open(os.path.join(folder, "scores.txt")) as fh:
        for line in fh:
            key, _, value = line.partition(":")
            scores[key.strip()] = float(value)
    with open(os.path.join(folder, "res_agent.json")) as fh:
        res = json.load(fh)
    done = sum(int(p == t) for p, t in zip(res["n_played"], res["total_ts"]))
    return scores, done, len(res["n_played"])


def rows(name):
    with open(os.path.join(RES, "local_764", name + ".json")) as fh:
        return {r["ep"]: r for r in json.load(fh)["rows"]}


def mcnemar(n01, n10):
    """Exact two-sided binomial test on the scenarios whose completion differs between two runs."""
    n = n01 + n10
    if n == 0:
        return 1.0
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(min(n01, n10) + 1)) / 2.0 ** n)


def local(name):
    recs = list(rows(name).values())
    return {
        "completed": sum(r["completed"] for r in recs),
        # A non-convergent power flow, typically after an uncontrolled cascade, is reported by grid2op as a
        # backend error, called a network-divergence failure here.
        "divergence": sum(r["end_cause"] == "BackendError" for r in recs if not r["completed"]),
        # Counted among the failed scenarios only: a run that ends with an error on its last step still counts
        # as completed under the harness's rule (steps played >= scenario length - 1).
        "infeasible_redispatch": sum(r["end_cause"] == "ImpossibleRedispatching" for r in recs if not r["completed"]),
        # Attacks counted once per line at their onset (`n_attack_onsets`). Runs recorded before that field
        # existed carry only `n_attacks`, which counts changes of the attacked set and is not used here.
        "realized_contingencies": (sum(r["n_attack_onsets"] for r in recs)
                                   if all("n_attack_onsets" in r for r in recs) else None),
        "n": len(recs),
    }


# Table 1 of the paper, in the order of its rows: (label, file, counter of the component). The counter is the
# per-scenario count of the component's actions; it must be zero in the run where the component is switched
# off. Two components have no counter of their own, and their runs are identified by their flags only.
ABLATIONS = (
    ("dispatch program (Sec. 3.3)", "without_dispatch_program", "_acopf_fired"),
    ("OptimCVXPY, all uses (Sec. 3.3)", "without_optimcvxpy", "_cvx_computed"),
    ("secured outages Q (Sec. 3.3)", "without_secured_outages", "_secn1_armed"),
    ("local pool (Sec. 3.4)", "without_local_pool", None),
    ("LODF line openings (Sec. 3.4)", "without_lodf_openings", "_lodfcut_offered"),
    ("opening of withheld lines (Sec. 3.5)", "without_withholding", "_fs_forced"),
    ("scheduled-outage look-ahead (Sec. 3.6)", "without_lookahead", "_mg_fired"),
    ("margin preservation (Sec. 3.6)", "without_margin_preservation", "_flex_fired"),
    ("extended-horizon safeguard (Sec. 3.6)", "without_safeguard", None),
)
REFERENCES = (
    ("reference: no-op", "reference_noop"),
    ("reference: reconnection only", "reference_reconnection"),
    ("reference: OptimCVXPY alone", "reference_optimcvxpy"),
)


def paired(full, other):
    """Scenarios completed only by the full agent (lost) and only by the other run (gained)."""
    lost = sum(1 for e in full if full[e]["completed"] and not other[e]["completed"])
    gained = sum(1 for e in full if other[e]["completed"] and not full[e]["completed"])
    return lost, gained


def main():
    print("Competition results")
    for split, label in (("validation", "validation set"), ("final_test", "sequestered final test set")):
        s, done, n = competition(split)
        print(f"  {label:28s} score {s['score']:.2f}  (operations {s['op_score_retraining']:.2f}, "
              f"low-carbon {s['nres_score_retraining']:.2f})  completed {done}/{n}  "
              f"total agent time {s['duration']:.0f} s")
        print(f"  {'':28s} without the training step {s['score_no_retraining']:.2f}; decision time: "
              f"95th percentile {s['decision_time_95']:.3f} s, 99th {s['decision_time_99']:.3f} s, "
              f"maximum {s['decision_time_max']:.2f} s")

    full = rows("xdp_agent")
    print("\nTable 1: paired runs on 764 local scenarios")
    print(f"  {'controller':50s} {'completed':>9s} {'lost':>5s} {'gained':>6s} {'p':>9s} "
          f"{'backend':>7s} {'infeasible':>10s}")

    def line(label, name, lost=None, gained=None, p=None):
        x = local(name)
        cols = (f"{lost:5d} {gained:6d} {p:9.2g}" if lost is not None else f"{'':5s} {'':6s} {'':9s}")
        print(f"  {label:50s} {x['completed']:9d} {cols} {x['divergence']:7d} {x['infeasible_redispatch']:10d}")

    line("full controller", "xdp_agent")
    for label, name, counter in ABLATIONS:
        abl = rows(name)
        if counter is not None:
            assert all(r["fired"].get(counter, 0) == 0 for r in abl.values()), f"{counter} acted in {name}"
        lost, gained = paired(full, abl)
        line("without " + label, name, lost, gained, mcnemar(lost, gained))
    for label, name in REFERENCES:
        line(label, name)
    print("  backend: episodes ended by a backend failure (non-convergent power flow); infeasible: episodes ended "
          "by infeasible redispatch")

    print("\nFigures quoted in the text of Section 4")
    att = {name: local(name)["realized_contingencies"]
           for name in ("xdp_agent", "without_withholding", "reference_noop", "reference_reconnection")}
    print(f"  realized attacks: full agent {att['xdp_agent']}, without opening the withheld lines "
          f"{att['without_withholding']}; no-op {att['reference_noop']}, reconnection only "
          f"{att['reference_reconnection']}")
    look = rows("without_lookahead")
    changed = [e for e in full
               if (full[e]["steps"], full[e]["end_cause"]) != (look[e]["steps"], look[e]["end_cause"])]
    acted = {e for e, r in full.items() if r["fired"].get("_mg_fired", 0) > 0}
    lost, gained = paired(full, look)
    print(f"  look-ahead: acts in {len(acted)} scenarios; switching it off changes the length or end cause of "
          f"{len(changed)} ({sum(e in acted for e in changed)} of them among those) and the completion of "
          f"{lost + gained}")
    flex = rows("without_margin_preservation")

    def ir(r):
        return r["end_cause"] == "ImpossibleRedispatching"

    only_without = sum(1 for e in full if ir(flex[e]) and not ir(full[e]))
    only_with = sum(1 for e in full if ir(full[e]) and not ir(flex[e]))
    print(f"  margin preservation: infeasible redispatch only without it {only_without}, only with it {only_with} "
          f"(exact McNemar p = {mcnemar(only_without, only_with):.2f})")
    cvx = local("reference_optimcvxpy")
    print(f"  OptimCVXPY alone: {cvx['completed']} completed, {cvx['infeasible_redispatch']} of its "
          f"{cvx['n'] - cvx['completed']} failures by infeasible redispatch")


if __name__ == "__main__":
    main()

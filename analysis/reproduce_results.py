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
  each `without_*.json` is the same agent with one mechanism switched off, so each comparison differs in that
  mechanism only.
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
        "divergence": sum(r["end_cause"] == "BackendError" for r in recs),
        "infeasible_redispatch": sum(r["end_cause"] == "ImpossibleRedispatching" for r in recs),
        "realized_contingencies": sum(r["n_attacks"] for r in recs),
        "n": len(recs),
    }


def main():
    print("Competition results")
    for split, label in (("validation", "validation set"), ("final_test", "sequestered final test set")):
        s, done, n = competition(split)
        print(f"  {label:28s} score {s['score']:.2f}  (operations {s['op_score_retraining']:.2f}, "
              f"low-carbon {s['nres_score_retraining']:.2f})  completed {done}/{n}  "
              f"total agent time {s['duration']:.0f} s")
        print(f"  {'':28s} without the training step {s['score_no_retraining']:.2f}")

    print("\nAblations on 764 scenarios (one mechanism switched off per run)")
    full = rows("xdp_agent")
    ablations = (
        ("preventive withholding", "without_withholding", "_fs_forced"),
        ("scheduled-outage look-ahead", "without_lookahead", "_mg_fired"),
        ("margin preservation", "without_margin_preservation", "_flex_fired"),
    )
    for label, name, counter in ablations:
        a, b = local(name), local("xdp_agent")
        abl = rows(name)
        lost = sum(1 for e in full if full[e]["completed"] and not abl[e]["completed"])
        gained = sum(1 for e in full if abl[e]["completed"] and not full[e]["completed"])
        print(f"  {label} (without -> with):")
        print(f"    completed episodes        {a['completed']} -> {b['completed']} of {a['n']}  "
              f"({lost} completed only with it, {gained} only without it; exact McNemar p = "
              f"{mcnemar(lost, gained):.3f})")
        print(f"    network-divergence        {a['divergence']} -> {b['divergence']}")
        print(f"    infeasible redispatch     {a['infeasible_redispatch']} -> {b['infeasible_redispatch']}")
        print(f"    realized contingencies    {a['realized_contingencies']} -> {b['realized_contingencies']}")
        # The switched-off mechanism must never act in its ablation run. Where the mechanism acted in the full
        # agent is reported for all scenarios and for those whose outcome differs, since the second share is
        # only informative against the first.
        assert all(r["fired"].get(counter, 0) == 0 for r in abl.values()), f"{counter} acted in {name}"
        changed = [e for e in full
                   if (full[e]["steps"], full[e]["end_cause"]) != (abl[e]["steps"], abl[e]["end_cause"])]
        acted_all = sum(1 for r in full.values() if r["fired"].get(counter, 0) > 0)
        acted = sum(1 for e in changed if full[e]["fired"].get(counter, 0) > 0)
        print(f"    mechanism acted in {acted_all} of {len(full)} scenarios, and in {acted} of the "
              f"{len(changed)} whose outcome differs")

    x = local("xdp_agent")
    print("\nThe agent, local 764-scenario run")
    print(f"  completed {x['completed']}/{x['n']}, network-divergence {x['divergence']}, "
          f"infeasible redispatch {x['infeasible_redispatch']}")


if __name__ == "__main__":
    main()

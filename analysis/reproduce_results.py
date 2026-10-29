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
  one record per scenario as written by `evaluation/eval_full.py`. The six `pair_*` runs form three paired
  comparisons; each pair differs in one mechanism only.
"""
import json
import os

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


def local(name):
    with open(os.path.join(RES, "local_764", name + ".json")) as fh:
        rows = json.load(fh)["rows"]
    return {
        "completed": sum(r["completed"] for r in rows),
        # A non-convergent power flow, typically after an uncontrolled cascade, is reported by grid2op as a
        # backend error, called a network-divergence failure here.
        "divergence": sum(r["end_cause"] == "BackendError" for r in rows),
        "infeasible_redispatch": sum(r["end_cause"] == "ImpossibleRedispatching" for r in rows),
        "realized_contingencies": sum(r["n_attacks"] for r in rows),
        "n": len(rows),
    }


def main():
    print("Competition results")
    for split, label in (("validation", "validation set"), ("final_test", "sequestered final test set")):
        s, done, n = competition(split)
        print(f"  {label:28s} score {s['score']:.2f}  (operations {s['op_score_retraining']:.2f}, "
              f"low-carbon {s['nres_score_retraining']:.2f})  completed {done}/{n}  "
              f"total agent time {s['duration']:.0f} s")

    print("\nPaired comparisons on 764 scenarios (one mechanism switched per pair)")
    pairs = (
        ("preventive withholding", "pair_withholding_off", "pair_withholding_on", "added"),
        ("scheduled-outage look-ahead", "pair_lookahead_on", "pair_lookahead_off", "removed"),
        ("margin preservation", "pair_margin_off", "pair_margin_on", "added"),
    )
    for label, before, after, verb in pairs:
        a, b = local(before), local(after)
        print(f"  {label} ({verb}):")
        print(f"    completed episodes        {a['completed']} -> {b['completed']} of {a['n']}")
        print(f"    network-divergence        {a['divergence']} -> {b['divergence']}")
        print(f"    infeasible redispatch     {a['infeasible_redispatch']} -> {b['infeasible_redispatch']}")
        if label == "preventive withholding":
            drop = 100.0 * (1 - b["realized_contingencies"] / a["realized_contingencies"])
            print(f"    realized contingencies    {a['realized_contingencies']} -> "
                  f"{b['realized_contingencies']}  ({drop:.1f}% fewer)")

    x = local("xdp_agent")
    print("\nThe agent, local 764-scenario run")
    print(f"  completed {x['completed']}/{x['n']}, network-divergence {x['divergence']}, "
          f"infeasible redispatch {x['infeasible_redispatch']}")


if __name__ == "__main__":
    main()

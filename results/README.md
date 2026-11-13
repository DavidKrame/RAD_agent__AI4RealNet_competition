# Stored results

* `competition/validation/` and `competition/final_test/`: the files the competition's evaluation returned for
  the agent on its validation set and its final test set, 208 scenarios each. `res_agent.json` holds the steps
  played (`n_played`) and the scenario lengths (`total_ts`), so completions can be counted per scenario.
  `training_log.txt` holds the agent's own log of its training step on each set.
* `local_764/`: our runs on the 764 scenarios listed in `data/chronics_764.txt`, one JSON per run with one
  record per scenario, as written by `evaluation/eval_full.py`.
  * `xdp_agent.json`: the agent in this repository.
  * `without_*.json`: the same agent with one component switched off; the `flags` field records the switch.
    `without_withholding.json` switches off the opening of the withheld lines (`EXP_FULLSTARVE=0`) and keeps
    the reconnection hold.
  * `reference_*.json`: the three reference controllers of `agent/reference_*/` (no-op, reconnection only,
    and OptimCVXPY alone with its published settings).
  * Every record carries `attack_onsets`, each attacked line counted once at the step its attack starts, and
    `n_attack_onsets`, their number. The realized attacks reported in the paper and by `reproduce_results.py`
    are these counts, except in `without_lookahead.json` and `without_margin_preservation.json`, which were
    recorded before the field existed.
  * A scenario counts as completed when the agent plays all its steps; the harness also accepts a run that ends
    on its last step. This happens in four records: two of OptimCVXPY alone (infeasible redispatch) and one each
    without margin preservation and without OptimCVXPY (illegal action), none in `xdp_agent.json`. Failures are
    counted by end cause among the scenarios that are not completed.

`python analysis/reproduce_results.py` recomputes the reported results from these files.

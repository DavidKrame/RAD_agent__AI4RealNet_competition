# Stored results

* `competition/validation/` and `competition/final_test/`: the files the competition's evaluation returned for
  the agent on its validation set and its final test set, 208 scenarios each. `res_agent.json` holds the steps
  played (`n_played`) and the scenario lengths (`total_ts`), so completions can be counted per scenario.
  `training_log.txt` holds the agent's own log of its training step on each set.
* `local_764/`: our runs on the 764 scenarios listed in `data/chronics_764.txt`, one JSON per run with one
  record per scenario, as written by `evaluation/eval_full.py`.
  * `xdp_agent.json`: the agent in this repository.
  * `without_*.json`: the same agent with one mechanism switched off (withholding, scheduled-outage
    look-ahead, margin preservation); the `flags` field records the switch.

`python analysis/reproduce_results.py` recomputes the reported results from these files.

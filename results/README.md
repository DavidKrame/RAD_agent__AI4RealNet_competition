# Stored results

* `platform/validation/` and `platform/final_test/`: the files the challenge platform returned for the
  submitted agent on the validation set and on the sequestered final test set (208 scenarios each).
  `res_agent.json` holds the steps played (`n_played`) and the scenario lengths (`total_ts`) per scenario.
* `local_764/`: local runs on the 764 scenarios of `data/chronics_764.txt`, one JSON per run with one record
  per scenario as written by `evaluation/eval_full.py`.
  * `xdp_agent.json`: the published agent.
  * `table1_*.json`: the three paired comparisons of the paper's Table 1. They were run with development
    versions of the agent that differ only in the mechanism named in the file. The same mechanisms can be
    switched off in the published agent (`docs/code_map.md`); such a run measures them on the final agent and
    gives different numbers.

`python analysis/reproduce_paper_numbers.py` recomputes the reported figures from these files.

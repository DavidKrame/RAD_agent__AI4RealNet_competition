# Evaluation harness

`eval_full.py` plays the agent on a range or list of scenarios and writes one record per scenario: steps
played, how the episode ended, the operating cost, the low-carbon term, the attacks it suffered, and the
counters of each mechanism. Each episode is seeded by its scenario index, so a run is reproducible record by
record on the same software stack. With `--reference`, it also plays a do-nothing agent on the same scenario.

```sh
# a few scenarios on a laptop
python evaluation/eval_full.py --submission xdp_agent --eps "0,5" --mode fast --reference --out results/my_run/check.json

# the 764 scenarios, split across workers (laptop: WORKERS parallel processes)
WORKERS=8 SUBMISSION=xdp_agent OUTDIR=results/my_run bash evaluation/run_eval.sh

# the same on a SLURM cluster: one node, every core used; resubmitting resumes where it stopped
sbatch --export=ALL,SUBMISSION=xdp_agent,OUTDIR=results/my_run,VENV=$HOME/venvs/xdp evaluation/run_eval.sh
```

`run_eval.sh` skips any worker whose output file already exists, so give each run its own `OUTDIR`.
To switch a mechanism off, export its variable before launching (see `docs/code_map.md`); a value
containing commas must be exported in the shell, not passed inside `--export`, which splits on commas.

# RAD agent for grid topology and dispatch control

This repository contains the agent developed by team RAD for the L2RPN AI4RealNet 2026 competition. The agent
controls the 118-bus transmission network of the Grid2Op environment `l2rpn_idf_2023` under adversarial line
outages, acting through bus-bar splitting, line switching, redispatch, curtailment, and storage. It contains
no neural network and no learned policy: at every decision step, a bounded set of candidate actions is
simulated a few steps ahead, and the candidate with the smallest constraint violation is applied.

## Components

The controller combines six components, each of which can be studied or reused on its own.

1. *Preventive withholding.* In this environment, an outage attempt in a region proceeds only if every line of
   that region is in service. Keeping one line per region out of service therefore cancels these attempts,
   which are consumed rather than postponed. Over the 764 local scenarios, realized outages fall from 2,360 to
   64.
2. *AC-anchored dispatch.* A convex quadratic program sets redispatch, curtailment, and storage jointly. It is
   anchored on the observed AC flows, uses the power-transfer distribution factors (PTDF) only for the flow
   change it induces, and preserves a reserve of downward ramping capacity.
3. *Line-opening screen.* The loading of every line after a candidate opening is estimated with line-outage
   distribution factors (LODF), at the cost of one matrix-vector product per candidate.
4. *Scheduled-outage look-ahead.* Before an announced outage, the agent applies a configuration recorded
   offline as surviving that outage.
5. *Margin preservation.* Before redispatch is returned toward its economic set point, part of it is shifted,
   at zero net change, toward generators close to their minimum output, so that downward ramping capacity
   remains available for later corrections.
6. *Revalidation of fitted parameters.* A parameter fitted offline is replaced only when an alternative
   performs significantly better on new data.

### Withholding mechanism

```mermaid
flowchart LR
    T([Outage attempt<br/>in a region]) --> Q{Every line of the<br/>region in service?}
    Q -- yes --> X[A heavily loaded line<br/>is taken out]
    Q -- no --> Y[Attempt canceled<br/>and consumed]
    K[[One line per region<br/>held out of service]] -.-> Q
```

The withheld line of each region is selected offline so that the network remains secure without it.

### Decision cycle

The environment advances in five-minute steps. At each step, the agent proceeds as follows:

```mermaid
flowchart LR
    O([Observe the grid]) --> M{Scheduled outage<br/>imminent?}
    M -- "yes, and severe" --> P[Apply a posture recorded<br/>to survive that outage]
    M -- no --> W{Withholding needed<br/>and safe?}
    W -- yes --> H[Hold one line per region<br/>out of service]
    W -- no --> G["Generate candidates:<br/>topology templates, local pool,<br/>LODF line openings,<br/>dispatch QP, DC optimal power flow"]
    G --> R[Simulate each a few steps ahead,<br/>rank by constraint violation]
    R --> C[Combine the best topology<br/>and dispatch candidates]
    P --> A([Apply the action])
    H --> A
    C --> A
```

The function implementing each step, and the environment variable that disables it, are listed in
[`docs/code_map.md`](docs/code_map.md).

## Results

In the competition, the agent was evaluated on two sets of 208 scenarios not available during development:

| | completed | score | operations | low-carbon | total agent time |
|---|---|---|---|---|---|
| Validation set | 184 / 208 | 72.93 | 66.97 | 96.76 | 17,384 s |
| Final test set | 189 / 208 | 64.39 | 56.32 | 96.68 | 13,177 s |

Among the five highest-ranked entries, it achieved the second-best ratio of score to computation time. On the
local evaluation set of 764 scenarios, it completes 706.

The contribution of each mechanism was measured by paired runs that differ only in that mechanism, on the
same 764 scenarios:

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/figures/mechanisms_dark.png">
  <img alt="Scenarios completed out of 764 without and with each mechanism: preventive withholding 630 to 659, scheduled-outage look-ahead 672 to 693, margin preservation 669 to 680." src="docs/figures/mechanisms_light.png" width="760">
</picture>

All reported values are recomputed from the records in [`results/`](results/) by
`python analysis/reproduce_results.py`.

## Installation and usage

The code requires Python 3.11 or later.

```sh
pip install -r requirements.txt
python agent/xdp_agent/prepare_data.py
```

The second command downloads `l2rpn_idf_2023` and splits it into 764 training, 34 validation, and 34 test
scenarios; the scenario lists in [`data/`](data/) allow the split to be verified.

A run on two scenarios takes a few minutes on a laptop:

```sh
python evaluation/eval_full.py --submission xdp_agent --eps "0,5" --mode fast --reference \
       --out results/my_run/check.json
```

A complete evaluation of the 764 scenarios takes about half an hour on a 128-core node; laptop and SLURM
instructions are given in [`evaluation/README.md`](evaluation/README.md).

## Repository structure

```
agent/xdp_agent/     the agent, a self-contained package exposing make_agent(env, path)
evaluation/          evaluation harness used for all local results
analysis/            reproduce_results.py and make_figures.py
results/             competition result files and per-scenario local records
data/                scenario lists of the 764 / 34 / 34 split
docs/                code map and figures
```

## Implementation notes

The code is the version evaluated in the competition. Only comments and docstrings were edited for this
release; the Python syntax tree of every file is unchanged.

PTDF and LODF are computed with LightSim2Grid for the environment on which the agent is constructed. In the
competition, that environment was never stepped, so the factors corresponded to the nominal topology. In the
local evaluation harness, the agent is constructed on the environment it plays, so the factors follow the
current topology.

## Citation

```bibtex
@inproceedings{rad2026rollout,
  title     = {A Rollout-Based Rule-and-Optimization Framework for Real-Time Topology and Dispatch
               Control Under Contingency Uncertainty, Developed and Tested in the {L2RPN AI4RealNet}
               2026 Challenge},
  author    = {AUTHOR_1 and AUTHOR_2},
  booktitle = {VENUE},
  year      = {2026}
}
```

## License

The code is released under the MIT license ([`LICENSE`](LICENSE)). The vendored OptimCVXPY file retains its
MPL-2.0 license ([`NOTICE.md`](NOTICE.md)).

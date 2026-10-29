"""
One-time generation of the curated topology action space shared by the PPO proposer
and the rule engine.

Builds the competition grid locally (from the starting-kit input data by default),
generates a bounded list of substation ``set_bus`` actions with
``gym_utils.build_reduced_actions``, optionally screens them against stressed states
with ``gym_utils.screen_actions_by_simulation``, and serializes the result to
``action_space.json`` (by element name, so it rebuilds correctly regardless of the
platform's internal ordering). Both ``MyAgent``'s rule-based candidate search and the
PPO ``DiscreteActSpace`` used for training/inference are built from this same file.

Usage
-----
    python prepare_data.py                             # once, downloads/splits the dataset
    python build_action_space.py                       # uses l2rpn_idf_2023_train
    python build_action_space.py --env l2rpn_idf_2023 --screen
"""

import argparse
import json
import os
import warnings

warnings.filterwarnings("ignore")

import grid2op

try:
    from .gym_utils import build_reduced_actions, screen_actions_by_simulation, serialize_actions
except ImportError:
    from gym_utils import build_reduced_actions, screen_actions_by_simulation, serialize_actions


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ENV_NAME = "l2rpn_idf_2023_train"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--env",
        default=DEFAULT_ENV_NAME,
        help="grid2op env name or path used only to enumerate topologies "
        "(default: the starting-kit input data).",
    )
    parser.add_argument("--min-elems", type=int, default=4)
    parser.add_argument("--max-elems", type=int, default=8)
    parser.add_argument("--max-per-sub", type=int, default=30)
    parser.add_argument("--max-actions", type=int, default=400)
    parser.add_argument(
        "--screen",
        action="store_true",
        help="keep only actions that help on stressed states (simulate-based pruning).",
    )
    parser.add_argument("--n-states", type=int, default=40, help="stressed states for screening")
    parser.add_argument("--rho-low", type=float, default=0.85, help="rho threshold for a 'stressed' state")
    parser.add_argument(
        "--out", default=os.path.join(HERE, "action_space.json")
    )
    args = parser.parse_args()

    print(f"Loading env from: {args.env}")
    try:
        from lightsim2grid import LightSimBackend

        env = grid2op.make(args.env, backend=LightSimBackend())
    except Exception:
        env = grid2op.make(args.env)
    print(f"n_sub={env.n_sub}, n_line={env.n_line}, dim_topo={env.dim_topo}")

    actions = build_reduced_actions(
        env,
        min_elems=args.min_elems,
        max_elems=args.max_elems,
        max_per_sub=args.max_per_sub,
        max_actions=args.max_actions,
    )
    print(f"Generated {len(actions)} candidate actions (index 0 = do-nothing).")

    if args.screen:
        print(f"Screening actions on stressed states (n_states={args.n_states}, rho_low={args.rho_low}) ...")
        actions, _ = screen_actions_by_simulation(
            env, actions, n_states=args.n_states, rho_low=args.rho_low
        )

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(serialize_actions(actions), f)
    print(f"Wrote {len(actions)} actions -> {args.out}")

    print(
        "\nNOTE: regenerating this file changes the menu MyAgent's rule engine searches "
        "AND the PPO DiscreteActSpace's action ids -> retrain (MyAgent.train / save_state) "
        "afterwards, otherwise saved_model/ppo_model.zip's ids no longer match this menu."
    )


if __name__ == "__main__":
    main()

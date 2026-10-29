"""
Stand-alone script to (re)train the PPO topology proposer used by MyAgent.

This is a beginner-friendly wrapper around ``MyAgent.train()``: it loads the grid2op
environment, builds the agent the same way Codabench does (via ``make_agent``), trains for
a wall-clock budget you choose, and saves the result to ``saved_model/`` -- backing up
whatever weights were already there first, so a bad run never destroys a working one.

The rule engine (reconnection, danger gating, storage, curtailment, last resort) is NOT
affected by this script or by training: it is fully defined by ``config.json`` and
``action_space.json``. Training only updates the PPO topology proposer -- one of the three
sources of topology candidates the agent searches over at inference time.

Usage
-----
    # (once) download the data and create the train/val/test split -- see prepare_data.py
    python prepare_data.py

    # from inside this submission's folder:
    python train_ppo.py --minutes 15      # quick smoke test: confirms nothing crashes
    python train_ppo.py --minutes 180     # a few hours -- start of a meaningful policy
    python train_ppo.py --minutes 15 --env educ_case14_storage --test   # try on a tiny bundled grid first

Prerequisites
-------------
    pip install grid2op stable-baselines3 gymnasium
    pip install lightsim2grid      # optional, but training is much faster with it

You also need the ``l2rpn_idf_2023_train`` dataset available locally. Run
``python prepare_data.py`` once first -- it downloads ``l2rpn_idf_2023`` via grid2op's
built-in dataset registry and creates the train/val/test split, writing to grid2op's own
data directory (``grid2op.get_current_local_dir()``), independent of where you run either
script from. Pass ``--env`` to train on a different dataset name or a local path instead.
"""

import argparse
import importlib
import os
import shutil
import sys
import warnings

warnings.filterwarnings("ignore")

HERE = os.path.dirname(os.path.abspath(__file__))
PKG_NAME = os.path.basename(HERE)
DEFAULT_ENV_NAME = "l2rpn_idf_2023_train"


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--env", default=DEFAULT_ENV_NAME,
        help="grid2op env name or path to train on (default: l2rpn_idf_2023_train, "
             "created by prepare_data.py)",
    )
    parser.add_argument(
        "--minutes", type=float, default=15.0,
        help="wall-clock training budget, in minutes (default: 15, a smoke test)",
    )
    parser.add_argument(
        "--workers", type=int, default=1,
        help="accepted for interface compliance; training is single-process regardless",
    )
    parser.add_argument(
        "--no-backup", action="store_true",
        help="skip backing up the existing saved_model/ppo_model.zip before overwriting it",
    )
    parser.add_argument(
        "--test", action="store_true",
        help="use grid2op's small bundled test grids (e.g. --env educ_case14_storage --test) "
             "instead of the full competition data -- handy for a first dry run",
    )
    args = parser.parse_args()

    import grid2op

    print(f"Loading environment from: {args.env}")
    try:
        from lightsim2grid import LightSimBackend

        env = grid2op.make(args.env, backend=LightSimBackend(), test=args.test)
        print("Using LightSimBackend (fast).")
    except Exception:
        env = grid2op.make(args.env, test=args.test)
        print("lightsim2grid not available -- using the default (much slower) backend. "
              "`pip install lightsim2grid` is strongly recommended for real training runs.")

    # Import this submission as a regular package (not a relative import) so its own
    # internal `from .gym_utils import ...` imports resolve correctly, whatever this
    # folder happens to be named.
    sys.path.insert(0, os.path.dirname(HERE))
    pkg = importlib.import_module(PKG_NAME)

    agent = pkg.make_agent(env, HERE)

    saved_model_dir = os.path.join(HERE, "saved_model")
    existing = os.path.join(saved_model_dir, "ppo_model.zip")
    if not args.no_backup and os.path.exists(existing):
        backup_dir = os.path.join(HERE, "saved_model_backup")
        os.makedirs(backup_dir, exist_ok=True)
        shutil.copy2(existing, os.path.join(backup_dir, "ppo_model.zip"))
        print(f"Backed up existing weights to {backup_dir}/ppo_model.zip "
              f"(restore by copying it back over {existing} if the new run is worse).")

    budget_seconds = args.minutes * 60.0
    print(f"Training for up to {args.minutes:.1f} minutes ({budget_seconds:.0f}s) ...")
    # checkpoint_dir enables the mid-training safety net: MyAgent.train() mirrors the
    # best-evaluated checkpoint into saved_model_dir/ppo_model.zip every time it improves,
    # not only once training finishes -- so an interrupted run (crash, closed laptop,
    # Ctrl+C) still leaves you the best weights found up to that point.
    agent.train(
        env, num_workers=args.workers, max_training_time=budget_seconds,
        checkpoint_dir=saved_model_dir,
    )

    agent.save_state(saved_model_dir)
    print(f"Saved trained weights to {saved_model_dir}/ppo_model.zip")


if __name__ == "__main__":
    main()

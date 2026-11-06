"""Download the l2rpn_idf_2023 dataset and split it into 764 training, 34 validation and 34 test scenarios.

Run it once, from anywhere: it writes to grid2op's data directory, not to the current folder.

    python3 prepare_data.py
"""

import os
import warnings

warnings.filterwarnings("ignore")
import grid2op

BASE_ENV = "l2rpn_idf_2023"
TRAIN_ENV = "l2rpn_idf_2023_train"


def main():
    data_dir = grid2op.get_current_local_dir()
    train_path = os.path.join(data_dir, TRAIN_ENV)
    if os.path.isdir(train_path):
        print(f"'{TRAIN_ENV}' already present in {data_dir} - nothing to do.")
        return
    print(
        f"Downloading '{BASE_ENV}' into {data_dir} (this can take several minutes)..."
    )
    env = grid2op.make(BASE_ENV)
    print("Creating the train/val/test split...")
    env.seed(1)
    env.reset()
    nm_train, nm_val, nm_test = env.train_val_split_random(
        add_for_test="test", pct_val=4.2, pct_test=4.2
    )
    print(f"Created: {nm_train}, {nm_val}, {nm_test}")
    print(f"Data ready in {data_dir}. You can now run train_ppo.py.")


if __name__ == "__main__":
    main()

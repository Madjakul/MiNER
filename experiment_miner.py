# experiment_miner.py

import argparse
import logging

from miner.utils import logging_config
from experiments import tune_miner, benchmark_kb


logging_config()


if __name__=="__main__":
    benchmark_kb(wandb_=True)


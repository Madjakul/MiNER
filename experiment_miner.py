# experiment_miner.py

import argparse
import logging

from miner.utils import logging_config
from experiments import benchmark_kb, benchmark_gamma


logging_config()


if __name__=="__main__":
    # benchmark_kb(wandb_=True)
    benchmark_gamma()


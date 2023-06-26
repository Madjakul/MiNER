# experiment_miner.py

from miner.utils import logging_config
from experiments import benchmark_kb


logging_config()


if __name__=="__main__":
    benchmark_kb(wandb_=True)


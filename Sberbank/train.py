import sys, os
import argparse
import numpy as np
import pandas as pd
from utils import DataProcessor

def parse_args():
    parser = argparse.ArgumentParser(description='Sberbank Housing Market')
    parser.add_argument('--train', required=True)
    parser.add_argument('--test', required=True)
    return parser.parse_args()


def main(args):
    data = DataProcessor()
    data.read_data(args.train, args.test)
    data.preprocess()


if __name__ == '__main__':
    args = parse_args()
    main(args)

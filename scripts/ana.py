from typing import List
import argparse
import pandas
import os
import numpy as np


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analysis')
    parser.add_argument('--task', type=str, choices=['logprob'])
    parser.add_argument('--inp', type=str, help='input')
    parser.add_argument('--out', type=str, help='output')
    args = parser.parse_args()

    if args.task == 'logprob':
        gold_dir, pred_dir = args.inp.split(':')
        ratios: List[float] = []
        for root, dirs, files in os.walk(gold_dir):
            for file in files:
                gold = pandas.read_csv(os.path.join(root, file))['log_prob']
                pred = pandas.read_csv(os.path.join(pred_dir, file))['log_prob']
                pred_better = (pred >= gold).sum() / len(gold)
                ratios.append(pred_better)
        print('on average {} predictions have higher or equal prob than golds'.format(np.mean(ratios)))

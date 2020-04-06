import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from typing import List, Dict, Tuple
import argparse
import pandas
import os
import json
from random import shuffle
from glob import glob
import numpy as np
from probe import tokenizer_wrap, LamaPredictions, EvalContext, CsvLogFileContext


def load_result(filename: str) -> List[LamaPredictions]:
    result: List[Dict] = []
    with open(filename, 'r') as fin:
        for l in fin:
            result.append(LamaPredictions.from_str(l))
    return result


def compute_acc(filename: str, eval: EvalContext) -> float:
    result: List[LamaPredictions] = load_result(filename)
    correct = total = 0
    for r in result:
        correct += int(r.eval(eval))
        total += 1
    return correct / (total or 1)


def prettify(in_file: str, out_file: str, eval: EvalContext):
    headers = ['sentence', 'prediction', 'gold', 'is_same']
    result: List[LamaPredictions] = load_result(in_file)
    with CsvLogFileContext(out_file, headers=headers) as csv_file:
        for r in result:
            r.eval(eval)
            r.prettify(csv_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analysis')
    parser.add_argument('--task', type=str, choices=['logprob', 'acc', 'compare', 'multi_eval', 'prettify'])
    parser.add_argument('--lang', type=str, help='language')
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
                pred_better = (pred >= gold).sum() / (len(gold) + 1e-10)
                ratios.append(pred_better)
        print('on average {} predictions have higher or equal prob than golds'.format(np.mean(ratios)))

    elif args.task == 'compare':
        show = 0
        sys1_dir, sys2_dir = args.inp.split(':')
        better1n = better2n = 0
        better1l, better2l = [], []
        for root, dirs, files in os.walk(sys1_dir):
            for file in files:
                diff: List[Tuple[int, str, str, str, int]] = []
                result1 = load_result(os.path.join(root, file))
                result2 = load_result(os.path.join(sys2_dir, file))
                for r1, r2 in zip(result1, result2):
                    r1p = r1['pred'][r1['num_mask']-1]
                    r2p = r2['pred'][r2['num_mask']-1]
                    r1c = is_correct(r1p, r1['tokenized_obj_label_inflection'])
                    r2c = is_correct(r2p, r2['tokenized_obj_label_inflection'])
                    if r1c and not r2c:
                        diff.append((1, r1['sentence'], r1p, r2p, r1['num_mask']))
                    elif not r1c and r2c:
                        diff.append((2, r1['sentence'], r1p, r2p, r2['num_mask']))
                b1 = list(filter(lambda x: x[0] == 1, diff))
                b1n = len(b1)
                b2 = list(filter(lambda x: x[0] == 2, diff))
                b2n = len(b2)

                better1n += b1n
                better2n += b2n

                b1l = np.mean([r[-1] for r in b1]) if b1n else 0
                b2l = np.mean([r[-1] for r in b2]) if b2n else 0

                if b1l:
                    better1l.append(b1l)
                if b2l:
                    better2l.append(b2l)
                print(file,
                      1, '#', b1n, 'len', b1l,
                      2, '#', b2n, 'len', b2l)
                if show and (b1n > 0 or b2n > 0):
                    shuffle(diff)
                    for d in diff[:show]:
                        print(d)
                    input()
        print(1, '#', better1n, 'len', np.mean(better1l),
              2, '#', better2n, 'len', np.mean(better2l))


    elif args.task == 'acc':
        result_dirs = args.inp

        for result_dir in sorted(glob(result_dirs + '/*'), key=lambda x: os.path.getmtime(x)):
            if not os.path.isdir(result_dir):
                continue
            acc_li: List[float] = []
            for root, dirs, files in os.walk(result_dir):
                for file in files:
                    acc = compute_acc(os.path.join(root, file), norm=False)
                    acc_li.append(acc)
            print('{}\tacc {}'.format(result_dir, np.mean(acc_li)))

    elif args.task == 'multi_eval':
        eval = EvalContext(lang=args.lang)
        acc_li: List[float] = []
        for root, dirs, files in os.walk(args.inp):
            for file in files:
                if not file.endswith('.jsonl'):
                    continue
                acc = compute_acc(os.path.join(root, file), eval)
                acc_li.append(acc)
        print('acc {}'.format(np.mean(acc_li)))

    elif args.task == 'prettify':
        eval = EvalContext(lang=args.lang)
        for root, dirs, files in os.walk(args.inp):
            for file in files:
                if not file.endswith('.jsonl'):
                    continue
                prettify(os.path.join(root, file),
                         os.path.join(root, file.rsplit('.', 1)[0] + '.csv'),
                         eval)

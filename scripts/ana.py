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
    pid = filename.rsplit('/', 1)[1].rsplit('.', 1)[0]
    with open(filename, 'r') as fin:
        for l in fin:
            result.append(LamaPredictions.from_str(l, pid))
    return result


def compute_acc(in_file: str, eval: EvalContext, prettify_out_file: str=None) \
        -> Tuple[float, float, float, int, int, int]:
    headers = ['sentence', 'prediction', 'gold', 'is_same']
    result: List[LamaPredictions] = load_result(in_file)
    correct = total = 0
    correct_single = total_single = 0
    correct_multi = total_mutli = 0
    with CsvLogFileContext(prettify_out_file, headers=headers) as csv_file:
        for r in result:
            right = int(r.eval(eval))
            if csv_file:
                r.prettify(csv_file)
            correct += right
            total += 1
            if len(r.result['tokenized_obj_label_inflection']) <= 1:
                correct_single += right
                total_single += 1
            else:
                correct_multi += right
                total_mutli += 1
    return correct / (total or 1), \
           correct_single / (total_single or 1), \
           correct_multi / (total_mutli or 1), \
           total, total_single, total_mutli


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
    parser.add_argument('--probe', type=str, help='probe dataset',
                        choices=['lama', 'lama-uhn', 'mlama'], default='lama')
    parser.add_argument('--model', type=str, help='LM to probe file', default='mbert_base')
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
        headers = ['sentence', 'prediction', 'gold', 'is_same']
        eval = EvalContext(lang=args.lang, lm=args.model, probe=args.probe)
        sys1_dir, sys2_dir = args.inp.split(':')
        os.makedirs(args.out, exist_ok=True)
        better1n = better2n = 0
        for root, dirs, files in os.walk(sys1_dir):
            for file in files:
                if not file.endswith('.jsonl'):
                    continue
                rel = file.split('.', 1)[0]
                result1 = load_result(os.path.join(root, file))
                result2 = load_result(os.path.join(sys2_dir, file))
                r1s: List[LamaPredictions] = []
                r2s: List[LamaPredictions] = []
                for r1, r2 in zip(result1, result2):
                    r1c = r1.eval(eval)
                    r2c = r2.eval(eval)
                    if r1c and not r2c:
                        r1s.append(r1)
                        r2s.append(r2)
                        better1n += 1
                    elif not r1c and r2c:
                        r1s.append(r1)
                        r2s.append(r2)
                        better2n += 1
                if len(r1s) > 0:
                    with CsvLogFileContext(os.path.join(args.out, rel + '.1.csv'), headers=headers) as csv1_file, \
                            CsvLogFileContext(os.path.join(args.out, rel + '.2.csv'), headers=headers) as csv2_file:
                        for r1 in r1s:
                            r1.prettify(csv1_file)
                        for r2 in r2s:
                            r2.prettify(csv2_file)

        print(1, '#', better1n, 2, '#', better2n)

    elif args.task == 'multi_eval':
        eval = EvalContext(lang=args.lang, lm=args.model, probe=args.probe)
        acc_li: List[float] = []
        acc_single_li: List[float] = []
        acc_multi_li: List[float] = []
        total_li: List[int] = []
        total_single_li: List[int] = []
        total_multi_li: List[int] = []
        for root, dirs, files in os.walk(args.inp):
            for file in files:
                if not file.endswith('.jsonl'):
                    continue
                in_file = os.path.join(root, file)
                out_file = os.path.join(root, file.rsplit('.', 1)[0] + '.csv')
                acc, acc_single, acc_multi, total, total_single, total_multi = \
                    compute_acc(in_file, eval, prettify_out_file=out_file)
                acc_li.append(acc)
                acc_single_li.append(acc_single)
                acc_multi_li.append(acc_multi)
                total_li.append(total)
                total_single_li.append(total_single)
                total_multi_li.append(total_multi)
        print('no alias {}'.format(eval.alias_manager.no_alias_count))
        print('{}\t{}\t{}'.format(np.mean(acc_li), np.mean(acc_single_li), np.mean(acc_multi_li)))
        print('{}\t{}\t{}'.format(np.sum(total_li), np.sum(total_single_li), np.sum(total_multi_li)))

    elif args.task == 'prettify':
        eval = EvalContext(lang=args.lang, lm=args.lm, probe=args.probe)
        for root, dirs, files in os.walk(args.inp):
            for file in files:
                if not file.endswith('.jsonl'):
                    continue
                prettify(os.path.join(root, file),
                         os.path.join(root, file.rsplit('.', 1)[0] + '.csv'),
                         eval)

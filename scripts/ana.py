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


def compute_acc(in_file: str, eval: EvalContext, prettify_out_file: str=None, only_count: bool=False) \
        -> Tuple[float, float, float, int, int, int]:
    headers = ['sentence', 'prediction', 'gold', 'is_same', 'is_single_word']
    result: List[LamaPredictions] = load_result(in_file)
    correct = total = 0
    correct_single = total_single = 0
    correct_multi = total_mutli = 0
    with CsvLogFileContext(prettify_out_file, headers=headers) as csv_file:
        for r in result:
            if eval.skip_cate and r.is_cate(eval.entity2iscate):
                continue
            right = 0
            if not only_count:
                right = int(r.eval(eval))
                if csv_file:
                    r.prettify(csv_file, eval)
            correct += right
            total += 1
            if r.is_single_word:
                correct_single += right
                total_single += 1
            else:
                correct_multi += right
                total_mutli += 1
    return correct / (total or 1), \
           correct_single / (total_single or 1), \
           correct_multi / (total_mutli or 1), \
           total, total_single, total_mutli


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analysis')
    parser.add_argument('--task', type=str, choices=['logprob', 'compare', 'multi_eval'], default='multi_eval')
    parser.add_argument('--lang', type=str, help='language')
    parser.add_argument('--probe', type=str, help='probe dataset',
                        choices=['lama', 'lama-uhn', 'mlama', 'mlamaf'], default='mlamaf')
    parser.add_argument('--model', type=str, help='LM to probe file', default='mbert_base')
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--use_multi_lang', action='store_true')
    parser.add_argument('--skip_cate', action='store_true')
    parser.add_argument('--only_count', action='store_true')
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
        headers = ['sentence', 'prediction1', 'prediction2', 'gold', 'is_same1', 'is_same2', 'is_single_word']
        eval = EvalContext(args)
        sys1_dir, sys2_dir = args.inp.split(':')
        if args.out:
            os.makedirs(args.out, exist_ok=True)
        better1ns: List[float] = []
        better2ns: List[float] = []
        for root, dirs, files in os.walk(sys1_dir):
            for file in files:
                if not file.endswith('.jsonl'):
                    continue
                better1n: List[int] = []
                better2n: List[int] = []
                is_single: List[int] = []
                rel = file.split('.', 1)[0]
                result1: List[LamaPredictions] = load_result(os.path.join(root, file))
                result2: List[LamaPredictions] = load_result(os.path.join(sys2_dir, file))
                rs: List[LamaPredictions] = []
                single_count: int = 0
                for r1, r2 in zip(result1, result2):
                    r1c = r1.eval(eval)
                    r2c = r2.eval(eval)
                    r1.add_prediction(r2.pred, r2.correct)
                    is_single.append(r1.is_single_word)
                    if r1c and not r2c:
                        better1n.append(1)
                        better2n.append(0)
                        rs.append(r1)
                    elif not r1c and r2c:
                        better1n.append(0)
                        better2n.append(1)
                        rs.append(r1)
                    else:
                        better1n.append(0)
                        better2n.append(0)
                    if r1.is_single_word:
                        '''
                        assert r1.single_word_pred[0] == r2.single_word_pred[0], \
                            'single-word prediction should be the same {}, {}'.format(
                                r1.single_word_pred[0], r2.single_word_pred[0])
                        '''
                        if r1c and not r2c and r1.is_use_single_word_pred and not r2.is_use_single_word_pred:
                            single_count += 1
                better1n, better2n, is_single = np.array(better1n), np.array(better2n), np.array(is_single)
                print(rel,
                      'single', '#1', np.sum(better1n * is_single), '#2', np.sum(better2n * is_single),
                      '#1 better with single-word pred', single_count / (np.sum(better1n * is_single) or 1),
                      'multi', '#1', np.sum(better1n * (1 - is_single)), '#2', np.sum(better2n * (1 - is_single)),
                      sep='\t')
                better1ns.append(np.mean(better1n))
                better2ns.append(np.mean(better2n))
                if len(rs) > 0 and args.out:
                    with CsvLogFileContext(os.path.join(args.out, rel + '.csv'), headers=headers) as csv_file:
                        for r in rs:
                            r.prettify(csv_file, eval)
        print('#1', np.mean(better1ns), '#2', np.mean(better2ns), sep='\t')

    elif args.task == 'multi_eval':
        eval = EvalContext(args)
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
                    compute_acc(in_file, eval, prettify_out_file=out_file, only_count=args.only_count)
                acc_li.append(acc)
                acc_single_li.append(acc_single)
                acc_multi_li.append(acc_multi)
                total_li.append(total)
                total_single_li.append(total_single)
                total_multi_li.append(total_multi)
                print(file.rsplit('.', 1)[0], acc, acc_single, acc_multi)
        print('no alias {}'.format(eval.alias_manager.no_alias_count))
        print('overall acc {}\t{}\t{}'.format(np.mean(acc_li), np.mean(acc_single_li), np.mean(acc_multi_li)))
        print('overall number {}\t{}\t{}'.format(np.sum(total_li), np.sum(total_single_li), np.sum(total_multi_li)))

import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from typing import List, Dict, Tuple, Set
import argparse
from operator import itemgetter
import pandas
import csv
from collections import defaultdict
from tqdm import tqdm
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import PercentFormatter
from probe import tokenizer_wrap, LamaPredictions, EvalContext, CsvLogFileContext, load_entity_lang, \
    DATASET, PROMPT_LANG_PATH


def load_result(filename: str) -> List[LamaPredictions]:
    result: List[Dict] = []
    pid = filename.rsplit('/', 1)[1].rsplit('.', 1)[0]
    with open(filename, 'r') as fin:
        for l in fin:
            result.append(LamaPredictions.from_str(l, pid))
    return result


def compute_acc(in_file: str, eval: EvalContext, prettify_out_file: str=None, only_count: bool=False) \
        -> Tuple[float, float, float, int, int, int]:
    headers = ['sentence', 'prediction', 'gold', 'is_same', 'confidence', 'is_single_word', 'sub_uri', 'obj_uri']
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
    parser.add_argument('--task', type=str,
                        choices=['logprob', 'compare', 'multi_eval', 'reliability', 'rank', 'error', 'overlap', 'plot'],
                        default='multi_eval')
    parser.add_argument('--lang', type=str, help='language', default='en')
    parser.add_argument('--probe', type=str, help='probe dataset',
                        choices=['lama', 'lama-uhn', 'mlama', 'mlamaf'], default='mlamaf')
    parser.add_argument('--model', type=str, help='LM to probe file', default='mbert_base')
    parser.add_argument('--norm', action='store_true')
    parser.add_argument('--multi_lang', type=str, help='use additional language in evaluation', default=None)
    parser.add_argument('--skip_cate', action='store_true')
    parser.add_argument('--gold_len', action='store_true', help='use the number of tokens in ground truth')
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
        headers = ['sentence', 'prediction1', 'prediction2', 'gold', 'is_same1', 'is_same2',
                   'confidence1', 'confidence2', 'is_single_word']
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

    elif args.task == 'reliability':
        csv_file_name = None
        headers = ['sentence', 'prediction', 'gold', 'is_same', 'confidence', 'is_single_word', 'sub_uri', 'obj_uri']
        num_bins = 10
        margin = 1 / num_bins
        xind = np.array([margin * (i + 0.5) for i in range(num_bins)])
        eval = EvalContext(args)
        acc_li: List[float] = []
        conf_li: List[float] = []
        num_token_li: List[int] = []
        pred_li: List[LamaPredictions] = []

        with CsvLogFileContext(csv_file_name, headers=headers) as csv_file:
            for root, dirs, files in os.walk(args.inp):
                for file in tqdm(files):
                    if not file.endswith('.jsonl'):
                        continue
                    in_file = os.path.join(root, file)
                    result: List[LamaPredictions] = load_result(in_file)
                    for r in result:
                        right = int(r.eval(eval))
                        acc_li.append(right)
                        conf_li.append(r.confidence)
                        num_token_li.append(r.num_tokens)
                        pred_li.append(r)
                        if csv_file:
                            r.prettify(csv_file, eval)

        bins = [[] for _ in range(num_bins)]
        for acc, conf, nt, r in zip(acc_li, conf_li, num_token_li, pred_li):
            assert conf >= 0 and conf <= 1, 'confidence out of range'
            ind = min(int(conf / margin), num_bins - 1)
            bins[ind].append((conf, acc, nt, r))

        all_bins = bins
        single_bins = [list(filter(lambda x: x[-1].is_single_word, bin)) for bin in bins]
        multi_bins = [list(filter(lambda x: not x[-1].is_single_word, bin)) for bin in bins]
        for bins, name in [(all_bins, 'all.png'), (single_bins, 'single.png'), (multi_bins, 'multi.png')]:
            eces = [(len(bin), np.mean(list(map(itemgetter(0), bin))), np.mean(list(map(itemgetter(1), bin)))) for bin in bins]
            print(eces)
            ece, total = 0, 0
            for c, conf, acc in eces:
                ece += c * np.abs(conf - acc)
                total += c
            ece /= total
            plt.bar(xind, [np.mean(list(map(itemgetter(1), bin))) for bin in bins], margin)
            plt.plot([0, 1], color='red')
            #plt.bar(xind + margin / 4, [np.mean(list(map(itemgetter(0), bin))) for bin in bins], margin / 2)
            plt.title(ece)
            plt.ylabel('accuracy')
            plt.xlabel('confidence')
            plt.ylim(0.0, 1.0)
            plt.xlim(0.0, 1.0)
            plt.savefig(name)
            plt.close()

    elif args.task == 'rank':
        correct_ranks = []
        for root, dirs, files in os.walk(args.inp):
            for file in files:
                if not file.endswith('.csv'):
                    continue
                in_file = os.path.join(root, file)
                df = pandas.read_csv(in_file)
                correct_mask = (df['is_same'] == True).tolist()
                rank = np.arange(len(correct_mask)) / len(correct_mask)
                correct_rank = rank[correct_mask]
                correct_ranks.extend(correct_rank.tolist())
        plt.hist(correct_ranks, 10)
        plt.savefig('rank.png')

    elif args.task == 'error':
        sample_per_relation = 10
        correct_ranks = []
        merge_df = []
        for root, dirs, files in os.walk(args.inp):
            for file in files:
                if not file.endswith('.csv') or file.startswith('.'):
                    continue
                in_file = os.path.join(root, file)
                df = pandas.read_csv(in_file)
                df = df[df['is_same'] == False]
                r = np.random.choice(len(df), min(sample_per_relation, len(df)), replace=False)
                df = df.iloc[r]
                merge_df.append(df)
        merge_df = pandas.concat(merge_df, axis=0, ignore_index=True)
        merge_df.to_csv(args.out)

    elif args.task == 'plot':
        df = pandas.read_csv('report/token.csv')
        cols = list(df.columns[1:])
        max_len = 15
        ymax = 0.52
        row_count = 12

        plt.rcParams.update({'font.size': 14, 'font.family': 'serif'})
        fig = plt.figure(figsize=(15, 4))
        gs = gridspec.GridSpec(int(np.ceil(len(cols) / row_count)), row_count)
        gs.update(wspace=0.08, hspace=0.15)  # set the spacing between axes.

        first = True
        for i, col in enumerate(cols):
            d = df[col].tolist()
            axx, axy = i // row_count, i % row_count
            ax = fig.add_subplot(gs[axx, axy])
            ax.plot(range(1, max_len + 1), d[:max_len], label=col)
            ax.set_ylim(ymin=0, ymax=ymax)
            ax.set_yticks(np.arange(0, ymax + 0.1, 0.1) if axy == 0 or first else [])
            first = False
            ax.set_xlim(xmin=1, xmax=max_len)
            ax.set_xticks([] if axx == 0 else [1] + list(range(0, max_len + 1, 5))[1:])
            ax.set_title(col, x=0.5, y=0.7)
            dx = 0.1
            dy = 0.0
            xtls = list(ax.xaxis.get_majorticklabels())
            if len(xtls) > 0:
                xtls[-1].set_transform(
                    xtls[-1].get_transform() + matplotlib.transforms.ScaledTranslation(-dx, dy, fig.dpi_scale_trans))

        plt.savefig('report/token.pdf')

    elif args.task == 'overlap':
        dirs = args.inp.split(':')
        corrects: List[Set[Tuple[str, str, str]]] = []
        alls: List[Set[Tuple[str, str, str]]] = []
        fact2sent: Dict[Tuple[str, str, str], List[Tuple[str, str]]] = defaultdict(list)
        for dir in dirs:
            corrects.append(set())
            alls.append(set())
            for root, _, files in os.walk(dir):
                for file in files:
                    if not file.endswith('.csv') or file.startswith('.'):
                        continue
                    rel = file.split('.', 1)[0]
                    df = pandas.read_csv(os.path.join(root, file))
                    for r in df[df['is_same'] == True].iterrows():
                        corrects[-1].add((r[1]['sub_uri'], rel, r[1]['obj_uri']))
                        fact2sent[(r[1]['sub_uri'], rel, r[1]['obj_uri'])].append(
                            (r[1]['sentence'], r[1]['prediction']))
                        alls[-1].add((r[1]['sub_uri'], rel, r[1]['obj_uri']))
                    for r in df[df['is_same'] == False].iterrows():
                        alls[-1].add((r[1]['sub_uri'], rel, r[1]['obj_uri']))

        print('pairwise correlation')
        sdirs = [dir.rsplit('/', 1)[1] for dir in dirs]
        for i in range(len(corrects)):
            print(sdirs[i], end='')
            for j in range(i + 1):
                print('\t{:.3f}'.format(0), end='')
            for j in range(i + 1, len(corrects)):
                com = alls[i] & alls[j]
                join = corrects[i] & corrects[j] & com
                all = (corrects[i] | corrects[j]) & com
                crr = len(join) / (len(all) or 1)
                print('\t{:.3f}'.format(crr), end='')
            print('\n')

        print('count histogram')
        all_cs = list(set.union(*corrects))
        all_cs_lang = [[sdirs[i] for i, correct in enumerate(corrects) if f in correct] for f in all_cs]
        all_count = [len(langs) for langs in all_cs_lang]
        print('#correct facts {}'.format(len(all_cs)))
        plt.rcParams.update({'font.size': 18, 'font.family': 'serif',
                             'font.weight': 'bold', 'axes.labelweight': 'bold'})
        plt.figure(figsize=(7.5, 4.5))
        plt.hist(all_count, bins=np.arange(0, len(sdirs) + 1) + 0.5,
                 weights=np.ones(len(all_count)) / len(all_count), rwidth=0.8)
        plt.xticks(range(len(sdirs) + 1))
        plt.xlabel('number of languages')
        plt.ylim(ymin=0)
        plt.xlim(xmin=0)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
        plt.tight_layout()
        plt.savefig('count.pdf')

        label = load_entity_lang(DATASET[args.probe]['entity_lang_path'])
        prompt_lang = pandas.read_csv(PROMPT_LANG_PATH)
        pid2prompt = lambda pid: prompt_lang[prompt_lang['pid'] == pid]['en'].iloc[0]

        def get_label(uri: str):
            if uri not in label:
                return uri
            if 'en' in label[uri]:
                return label[uri]['en']
            return uri

        with open(args.out, 'w') as fout:
            fout.write(','.join(['fact', 'label', 'sentence', 'langs', 'number of langs']) + '\n')
            csv_file = csv.writer(fout)
            for i, (langs, f) in enumerate(sorted(zip(all_cs_lang, all_cs), key=lambda x: -len(x[0]))):
                csv_file.writerow([
                    f,
                    (get_label(f[0]), pid2prompt(f[1]), get_label(f[2])),
                    fact2sent[f][0] if len(fact2sent[f]) == 1 else None,
                    langs,
                    len(langs)])

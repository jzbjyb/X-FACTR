from typing import Tuple, List, Dict, Iterable, Set
import os
import logging
import re
import random
import numpy as np
from tqdm import tqdm
import argparse
from collections import defaultdict
from operator import itemgetter
import json
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.nn.utils.rnn import pad_sequence
from transformers import *


logger = logging.getLogger(__name__)
SEED = 2020
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


class CodeSwitchDataset(object):
    def __init__(self, filepath: str):
        self.filepath = filepath

    def load(self):
        self.mid2count: Dict[str, int] = defaultdict(lambda: 0)
        self.sentences: List[str] = []
        for tokens, mentions in self.iter():
            for m in mentions:
                self.mid2count[m[0]] += 1
            sent = self.fill(tokens, mentions, replace=True, sorted=True)
            self.sentences.append(sent)


    def iter(self) -> Iterable[Tuple[str, List[Tuple[str, str, str]]]]:
        with open(self.filepath, 'r') as fin:
            for l in tqdm(fin, disable=False):
                tokens, mentions = self.load_line(l)
                yield tokens, mentions


    def format(self, tokens: str, mentions: List[Tuple[str, str, str]], fill_in: Set[int]=None):
        tokens = self.fill(tokens, mentions, fill_inds=fill_in, replace=False, sorted=True)
        mentions = [m for i, m in enumerate(mentions) if i not in fill_in]
        return '{}\t{}'.format(tokens, '\t'.join([' ||| '.join(m) for m in mentions]))


    def load_line(self, line: str) -> Tuple[str, List[Tuple[str, str, str]]]:
        l = line.strip().split('\t')
        tokens, mentions = l[0], l[1:]  # TODO: tokens has already been tokenized with space
        mentions = [tuple(m.split(' ||| ')) for m in mentions]
        return tokens, mentions


    def fill(self,
             tokens: str,
             mentions: List[Tuple[str, str, str]],
             fill_inds: Set[int]=None,
             replace: bool=False,
             sorted: bool=True) -> str:
        '''
        :param replace: If true, use targets to fill in blanks.
        :param sorted: If true, assume the mentions are aligned with the blanks.
        '''
        if sorted:
            new_tokens: List[str] = []
            prev_pos = 0
            for i, match in enumerate(re.finditer('\[\[[^\[\]]+\]\]', tokens)):
                if prev_pos < match.start(0):
                    new_tokens.append(tokens[prev_pos:match.start(0)])
                mid, source, target = mentions[i]
                if fill_inds and i not in fill_inds:  # not fill
                    new_tokens.append(match.group(0))
                else:
                    new_tokens.append(target if replace else source)
                prev_pos = match.end(0)
            new_tokens.append(tokens[prev_pos:])
            return ''.join(new_tokens)
        else:
            # TODO: problematic because mid might appears multiple times in the sentence
            for i, (mid, source, target) in enumerate(mentions):
                if fill_inds and i not in fill_inds:  # not fill
                    continue
                anchor = '[[{}]]'.format(mid)
                anchor_start = tokens.find(anchor)
                tokens[anchor_start:anchor_start + len(anchor)] = target if replace else source
        return tokens


    def save_sentences(self, filename: str):
        with open(filename, 'w') as fout:
            for sent in self.sentences:
                fout.write(sent + '\n')


class ReplaceTextDataset(Dataset):
    def __init__(self, filepath: str, tokenizer: PreTrainedTokenizer, max_length: int=512):
        assert os.path.isfile(filepath)
        logger.info('read from {}'.format(filepath))
        self.cs_dataset = CodeSwitchDataset(filepath)
        #self.examples: List[List[int]] = \
        #    tokenizer.batch_encode_plus(self.sentences, add_special_tokens=True, max_length=max_length)['input_ids']

    def __len__(self):
        return len(self.examples)


    def __getitem__(self, i):
        return torch.tensor(self.examples[i], dtype=torch.long)


def train(args, dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int, float]:
    def collate(examples: List[torch.Tensor]):
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id or 0)

    sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=args.batch_size, collate_fn=collate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fine-tune multilingual PLM')
    parser.add_argument('--task', type=str, choices=['filter', 'gen'])
    parser.add_argument('--out', type=str, help='output')
    parser.add_argument('--replace', action='store_true')
    args = parser.parse_args()
    '''
    LM = 'bert-base-multilingual-cased'
    print('load data')
    tokenizer = AutoTokenizer.from_pretrained(LM)
    en_el_dataset = CodeSwitchDataset('data/cs/el_en/en_el.txt')
    en_el_dataset.load()
    en_el_dataset.save_sentences('data/cs/el_en/en_el_el.txt')

    model = AutoModelWithLMHead.from_pretrained(LM)
    model.to('cuda')
    model.train()
    '''

    if args.task == 'gen':
        for split in ['train', 'test']:
            with open('data/cs/el_en_filter/{}_raw.txt'.format(split), 'w') as fout:
                for source, target in [('en', 'el'), ('el', 'en')]:
                    # TODO: the ratio between en and el is not balanced
                    dataset = CodeSwitchDataset('data/cs/el_en_filter/{}_{}.{}.txt'.format(source, target, split))
                    for tokens, mentions in dataset.iter():
                        sent_source = dataset.fill(tokens, mentions, replace=False)
                        fout.write(sent_source + '\n')
                        if args.replace:
                            sent_target = dataset.fill(tokens, mentions, replace=True)
                            fout.write(sent_target + '\n')

    elif args.task == 'filter':
        with open('data/lang/el_en_fact.json', 'r') as fin:
            facts: List[Tuple[str, str]] = json.load(fin)['join']
            facts: Set[Tuple[str, str]] = set(tuple(f) for f in facts)

        # count fact occurrence in the wikipedia

        en_el_fact2count: Dict[Tuple[str, str], int] = defaultdict(lambda: 0)
        el_en_fact2count: Dict[Tuple[str, str], int] = defaultdict(lambda: 0)

        for source, target in [('en', 'el'), ('el', 'en')]:
            dataset = CodeSwitchDataset('data/cs/el_en/{}_{}.txt'.format(source, target))
            fact2count = eval('{}_{}_fact2count'.format(source, target))
            for tokens, mentions in dataset.iter():
                for i in range(len(mentions)):
                    for j in range(i + 1, len(mentions)):
                        e1, e2 = mentions[i][0], mentions[j][0]
                        if (e1, e2) in facts:
                            fact2count[(e1, e2)] += 1
                        if (e2, e1) in facts:
                            fact2count[(e2, e1)] += 1

        print('#facts in en: {}'.format(len(en_el_fact2count)))
        print('#facts in el: {}'.format(len(el_en_fact2count)))

        # keep frequent ones

        fact2count = defaultdict(lambda: 0)  # harmonic mean
        for k in en_el_fact2count.keys() | el_en_fact2count.keys():
            # TODO: counts might be of different scales for different langauges
            fact2count[k] = 2 * en_el_fact2count[k] * el_en_fact2count[k] / (en_el_fact2count[k] + el_en_fact2count[k])

        thres = 50
        fact2count = sorted(fact2count.items(), key=lambda x: -x[1])
        fact2count_kept = [(f, c) for f, c in fact2count if c >= thres]
        print('#facts: {} #facts >{}: {}'.format(len(fact2count), thres, len(fact2count_kept)))
        print(fact2count_kept[:10])
        print(fact2count_kept[-10:])

        # filter data

        fact_kept = set(map(itemgetter(0), fact2count_kept))

        num_mention_kept = num_sent_kept = 0
        test_ratio = 0.1
        for source, target in [('en', 'el'), ('el', 'en')]:
            dataset = CodeSwitchDataset('data/cs/el_en/{}_{}.txt'.format(source, target))
            fact_kept2count: Dict[Tuple[str, str], int] = defaultdict(lambda: 0)
            with open(os.path.join(args.out, '{}_{}.train.txt'.format(source, target)), 'w') as fout_train, \
                    open(os.path.join(args.out, '{}_{}.test.txt'.format(source, target)), 'w') as fout_test:
                for tokens, mentions in dataset.iter():
                    seen = True
                    kept: Set[int] = set()
                    for i in range(len(mentions)):
                        for j in range(i + 1, len(mentions)):
                            e1, e2 = mentions[i][0], mentions[j][0]
                            if (e2, e1) in fact_kept:
                                e1, e2 = e2, e1
                            if (e1, e2) in fact_kept:
                                fact_kept2count[(e1, e2)] += 1
                                kept.add(i)
                                kept.add(j)
                                if fact_kept2count[(e1, e2)] <= 1:
                                    seen = False
                    if len(kept) > 0:
                        num_mention_kept += len(kept)
                        num_sent_kept += 1
                        fill_in = set(range(len(mentions))) - kept
                        line = dataset.format(tokens, mentions, fill_in=fill_in)
                        if seen and random.random() <= test_ratio:
                            fout_test.write(line + '\n')
                        else:
                            fout_train.write(line + '\n')
        print('#sentences {}, #mentions {}'.format(num_sent_kept, num_mention_kept))

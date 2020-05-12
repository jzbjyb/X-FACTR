from typing import Tuple, List, Dict, Iterable, Set
import os
import logging
import re
import random
import json
import numpy as np
from tqdm import tqdm
import argparse


logger = logging.getLogger(__name__)
SEED = 2020
random.seed(SEED)
np.random.seed(SEED)


class CodeSwitchDataset(object):
    def __init__(self, filepath: str):
        self.filepath = filepath


    def load_line(self, line: str) -> Tuple[str, List[Tuple[str, str, str]]]:
        l = line.strip().split('\t')
        tokens, mentions = l[0], l[1:]  # TODO: tokens has already been tokenized with space
        mentions = [tuple(m.split(' ||| ')) for m in mentions]
        return tokens, mentions


    def iter(self) -> Iterable[Tuple[str, List[Tuple[str, str, str]]]]:
        with open(self.filepath, 'r') as fin:
            for l in tqdm(fin, disable=False):
                tokens, mentions = self.load_line(l)
                yield tokens, mentions, l


    def format(self, tokens: str, mentions: List[Tuple[str, str, str]], fill_in: Set[int]=None):
        tokens = self.fill(tokens, mentions, fill_inds=fill_in, replace=False, sorted=True)
        mentions = [m for i, m in enumerate(mentions) if i not in fill_in]
        return '{}\t{}'.format(tokens, '\t'.join([' ||| '.join(m) for m in mentions]))


    def fill(self,
             tokens: str,
             mentions: List[Tuple[str, str, str]],
             fill_inds: Set[int]=None,
             replace: bool=False,
             alias: Dict[str, Tuple[List[str], List[float]]] = None,
             sorted: bool=True,
             tab_for_filled_mention: bool=False) -> str:
        '''
        :param replace: If true, use targets to fill in blanks.
        :param sorted: If true, assume the mentions are aligned with the blanks.
        :param tab_for_filled_mention: If true, mentions are separated with the context using tab
            (mentions always occur in even positions)
        '''
        if sorted:
            new_tokens: List[str] = []
            prev_pos = 0
            for i, match in enumerate(re.finditer('\[\[[^\[\]]+\]\]', tokens)):
                if prev_pos < match.start(0):
                    new_tokens.append(tokens[prev_pos:match.start(0)])
                mid, source, target = mentions[i]
                if fill_inds is not None and i not in fill_inds:  # not fill
                    new_tokens.append(match.group(0))
                else:
                    if tab_for_filled_mention:
                        new_tokens.append('\t')
                    if alias is not None and mid in alias:
                        target = np.random.choice(alias[mid][0], size=1, replace=False, p=alias[mid][1])[0]
                    new_tokens.append(target if replace else source)
                    if tab_for_filled_mention:
                        new_tokens.append('\t')
                prev_pos = match.end(0)
            new_tokens.append(tokens[prev_pos:])
            return ''.join(new_tokens)
        else:
            raise NotImplementedError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='fine-tune multilingual PLM')
    parser.add_argument('--task', type=str, choices=['gen'], default='gen')
    parser.add_argument('--inp', type=str, help='output')
    parser.add_argument('--suffix', type=str, help='input file suffix', default='')
    parser.add_argument('--out', type=str, help='output')
    parser.add_argument('--lang', type=str, help='language to probe')
    parser.add_argument('--replace', action='store_true')
    parser.add_argument('--random_alias', action='store_true', help='use random alias to do the replacement')
    args = parser.parse_args()

    if args.task == 'gen':
        os.makedirs(args.out, exist_ok=True)
        filename = os.path.join(
            args.out, '{}_{}.txt'.format('cs' if args.replace else 'raw',
                                         'random' if args.random_alias else 'fix'))
        with open(filename, 'w') as fout:
            for source, target in [(args.lang, 'en'), ('en', args.lang)]:
                # load data
                dataset = CodeSwitchDataset(
                    os.path.join(args.inp, '{}_{}{}.txt'.format(source, target, args.suffix)))
                # load alias
                if args.random_alias:
                    with open(os.path.join(args.inp, '{}_alias.txt'.format(target))) as fin:
                        _id2alias: Dict[str, Dict[str, int]] = json.load(fin)
                    id2alias: Dict[str, Tuple[List[str], List[float]]] = {}
                    for id, alias in _id2alias.items():
                        alias_value, alias_prob = list(zip(*list(alias.items())))
                        alias_prob = np.array(alias_prob)
                        alias_prob = alias_prob / np.sum(alias_prob)
                        id2alias[id] = (alias_value, alias_prob)
                else:
                    id2alias = None
                for tokens, mentions, _ in dataset.iter():
                    # raw sentence
                    sent_source = dataset.fill(
                        tokens, mentions, replace=False, tab_for_filled_mention=True)
                    # cs or raw data
                    sent_target = dataset.fill(
                        tokens, mentions, replace=args.replace, alias=id2alias, tab_for_filled_mention=True)
                    fout.write('{}\n{}\n'.format(sent_source, sent_target))

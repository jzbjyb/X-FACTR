from typing import Tuple, List
import os
import logging
import re
import random
import numpy as np
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


class ReplaceTextDataset(Dataset):
    def __init__(self, file_path: str, tokenizer: PreTrainedTokenizer, max_length: int=512):
        assert os.path.isfile(file_path)
        logger.info('read from {}'.format(file_path))
        with open(file_path, 'r') as fin:
            self.sentences: List[str] = [self.fill(*self.load_line(l), replace=True, sorted=True) for l in fin]
        self.examples: List[List[int]] = \
            tokenizer.batch_encode_plus(self.sentences, add_special_tokens=True, max_length=max_length)['input_ids']


    def load_line(self, line: str) -> Tuple[str, List[Tuple[str, str, str]]]:
        l = line.strip().split('\t')
        tokens, mentions = l[0], l[1:]  # TODO: tokens has already been tokenized with space
        mentions = [tuple(m.split(' ||| ')) for m in mentions]
        return tokens, mentions


    def fill(self, tokens: str, mentions: List[Tuple[str, str, str]], replace: bool=False, sorted: bool=False):
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
                new_tokens.append(target if replace else source)
                prev_pos = match.end(0)
            return ''.join(new_tokens)
        else:
            for mid, source, target in mentions:
                anchor = '[[{}]]'.format(mid)
                anchor_start = tokens.find(anchor)
                tokens[anchor_start:anchor_start + len(anchor)] = target if replace else source
        return tokens


    def save_sentences(self, filename: str):
        with open(filename, 'w') as fout:
            for sent in self.sentences:
                fout.write(sent + '\n')


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
    LM = 'bert-base-multilingual-cased'
    print('load data')
    tokenizer = AutoTokenizer.from_pretrained(LM)
    dataset = ReplaceTextDataset('data/cs/el_en/en_el.txt', tokenizer, max_length=512)
    dataset.save_sentences('data/cs/el_en/en_el_el.txt')

    # model = AutoModelWithLMHead.from_pretrained(LM)
    # model.to('cuda')
    # model.train()

from typing import List, Dict
import torch
from transformers import *
import json
import numpy as np
import os
from tqdm import tqdm
import argparse
import logging
from collections import defaultdict
import pandas

logger = logging.getLogger('mLAMA')
logger.setLevel(logging.ERROR)

NUM_MASK = 5
BATCH_SIZE = 8
MASK_LABEL = '[MASK]'
PREFIX_DATA = '../LAMA/'
VOCAB_PATH = PREFIX_DATA + 'pre-trained_language_models/common_vocab_cased_mbert.txt'
RELATION_PATH = PREFIX_DATA + 'data/relations.jsonl'
ENTITY_PATH = PREFIX_DATA + 'data/TREx/{}.jsonl'
PROMPT_LANG_PATH = 'data/TREx_prompts.csv'
ENTITY_LANG_PATH = 'data/TREx_unicode_escape.txt'
LM_NAME = {
    'mbert_base': 'bert-base-multilingual-cased',
    'bert_base': 'bert-base-cased'
}

def batcher(data: List, batch_size: int):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def load_entity_lang(filename: str) -> Dict[str, Dict[str, str]]:
    entity2lang = defaultdict(lambda: {})
    with open(filename, 'r') as fin:
        for l in fin:
            l = l.strip().split('\t')
            entity = l[0]
            for lang in l[1:]:
                label ,lang = lang.rsplit('@', 1)
                entity2lang[entity][lang] = label.strip('"')
    return entity2lang

parser = argparse.ArgumentParser(description='probe LMs with multilingual LAMA')
parser.add_argument('--model', type=str, help='LM to probe file',
                    choices=['mbert_base', 'bert_base'], default='mbert_base')
parser.add_argument('--lang', type=str, help='language to probe',
                    choices=['en', 'zh'], default='en')
args = parser.parse_args()
lm = LM_NAME[args.model]
lang = args.lang

# load relations and templates
patterns = []
with open(RELATION_PATH) as fin:
    patterns.extend([json.loads(l) for l in fin])
entity2lang = load_entity_lang(ENTITY_LANG_PATH)
prompt_lang = pandas.read_csv(PROMPT_LANG_PATH)

# load model
print('load model')
tokenizer = BertTokenizer.from_pretrained(lm)
MASK = tokenizer.convert_tokens_to_ids(MASK_LABEL)
model = BertForMaskedLM.from_pretrained(lm)
model.to('cuda')
model.eval()

# load vocab
with open(VOCAB_PATH) as fin:
    allowed_vocab = [l.strip() for l in fin]
    allowed_vocab = set(allowed_vocab)
#restrict_vocab = [tokenizer.vocab[w] for w in tokenizer.vocab if not w in allowed_vocab]
# TODO: add a shared vocab for all LMs?
restrict_vocab = []

all_queries = []
for pattern in patterns:
    relation = pattern['relation']
    prompt = prompt_lang[prompt_lang['pid'] == relation][lang].iloc[0]

    f = ENTITY_PATH.format(relation)
    if not os.path.exists(f):
        continue

    queries: List[Dict] = []
    with open(f) as fin:
        for l in fin:
            l = json.loads(l)
            if lang not in entity2lang[l['sub_uri']] or lang not in entity2lang[l['obj_uri']]:
                # TODO: support entities without translations
                continue
            l['sub_label'] = entity2lang[l['sub_uri']][lang]
            l['obj_label'] = entity2lang[l['obj_uri']][lang]
            queries.append(l)

    acc, len_acc = [], []
    for query_li in tqdm(batcher(queries, BATCH_SIZE), desc=relation, disable=False):
        obj_li: List[np.ndarray] = []
        inp_tensor: List[torch.Tensor] = []
        batch_size = len(query_li)
        for query in query_li:
            obj = np.array(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(query['obj_label']))).reshape(-1)
            if len(obj) > NUM_MASK:
                logger.warning('{} is splitted into {} tokens'.format(query['obj_label'], len(obj)))
            obj_li.append(obj)
            instance_x: str = prompt.replace('[X]', query['sub_label'])
            for nm in range(NUM_MASK):
                inp: str = instance_x.replace('[Y]', ' '.join(['[MASK]'] * (nm + 1)))
                inp: List[int] = tokenizer.encode(inp)
                inp_tensor.append(torch.tensor(inp))

        # SHAPE: (batch_size * num_mask, seq_len)
        inp_tensor: torch.Tensor = torch.nn.utils.rnn.pad_sequence(inp_tensor, batch_first=True, padding_value=0).cuda()
        attention_mask: torch.Tensor = inp_tensor.ne(0).long().cuda()
        # SHAPE: (batch_size, num_mask, seq_len)
        mask_ind: torch.Tensor = inp_tensor.eq(MASK).float().cuda().view(batch_size, NUM_MASK, -1)

        # SHAPE: (batch_size * num_mask, seq_len, vocab_size)
        logit = model(inp_tensor, attention_mask=attention_mask)[0]
        logit[:, :, restrict_vocab] = float('-inf')
        logprob = logit.log_softmax(-1)
        # SHAPE: (batch_size * num_mask, seq_len)
        logprob, rank = logprob.max(-1)
        # SHAPE: (batch_size, num_mask, seq_len)
        logprob, rank = logprob.view(batch_size, NUM_MASK, -1), rank.view(batch_size, NUM_MASK, -1)

        # find the best setting
        for i, best_num_mask in enumerate(((logprob * mask_ind).sum(-1) / mask_ind.sum(-1)).max(1)[1]):
            obj = obj_li[i]
            pred: np.ndarray = rank[i, best_num_mask].masked_select(mask_ind[i, best_num_mask].eq(1))\
                .detach().cpu().numpy().reshape(-1)
            acc.append(int((len(pred) == len(obj)) and (pred == obj).all()))
            len_acc.append(int((len(pred) == len(obj))))
            '''
            if len(pred) == len(obj):
                print(tokenizer.convert_ids_to_tokens(pred))
                print(tokenizer.convert_ids_to_tokens(obj))
                input()
            '''

    print('pid {}\t#fact {}\tacc {}\tlen_acc {}\n'.format(relation, len(queries), np.mean(acc), np.mean(len_acc)))

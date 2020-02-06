import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

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
from prompt import Prompt
from check_gender import load_entity_gender

logger = logging.getLogger('mLAMA')
logger.setLevel(logging.ERROR)

NUM_MASK = 5
BATCH_SIZE = 4
MASK_LABEL = '[MASK]'
UNK_LABEL = '[UNK]'
PREFIX_DATA = '../LAMA/'
VOCAB_PATH = PREFIX_DATA + 'pre-trained_language_models/common_vocab_cased.txt'
RELATION_PATH = PREFIX_DATA + 'data/relations.jsonl'
ENTITY_PATH = PREFIX_DATA + 'data/TREx/{}.jsonl'
PROMPT_LANG_PATH = 'data/TREx_prompts.csv'
ENTITY_LANG_PATH = 'data/TREx_unicode_escape.txt'
ENTITY_GENDER_PATH = 'data/TREx_gender.txt'
LM_NAME = {
    'mbert_base': 'bert-base-multilingual-cased',
    'bert_base': 'bert-base-cased',
    'zh_bert_base': 'bert-base-chinese'
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
                    choices=['mbert_base', 'bert_base', 'zh_bert_base'], default='mbert_base')
parser.add_argument('--lang', type=str, help='language to probe',
                    choices=['en', 'zh-cn', 'el'], default='en')
parser.add_argument('--portion', type=str, choices=['all', 'trans', 'non'], default='all',
                    help='which portion of facts to use')
parser.add_argument('--sub_obj_same_lang', action='store_true',
                    help='use the same language for sub and obj')
args = parser.parse_args()
lm = LM_NAME[args.model]
lang = args.lang

# load relations and templates
patterns = []
with open(RELATION_PATH) as fin:
    patterns.extend([json.loads(l) for l in fin])
entity2lang = load_entity_lang(ENTITY_LANG_PATH)
entity2gender = load_entity_gender(ENTITY_GENDER_PATH)
prompt_lang = pandas.read_csv(PROMPT_LANG_PATH)

# load model
print('load model')
tokenizer = BertTokenizer.from_pretrained(lm)
MASK = tokenizer.convert_tokens_to_ids(MASK_LABEL)
UNK = tokenizer.convert_tokens_to_ids(UNK_LABEL)
model = BertForMaskedLM.from_pretrained(lm)
model.to('cuda')
model.eval()

# load promp rendering model
prompt_model = Prompt.from_lang(lang)

# load vocab
'''
with open(VOCAB_PATH) as fin:
    allowed_vocab = [l.strip() for l in fin]
    allowed_vocab = set(allowed_vocab)
restrict_vocab = [tokenizer.vocab[w] for w in tokenizer.vocab if not w in allowed_vocab]
'''
# TODO: add a shared vocab for all LMs?
restrict_vocab = []

all_queries = []
for pattern in patterns:
    relation = pattern['relation']
    prompt = prompt_lang[prompt_lang['pid'] == relation][lang].iloc[0]

    '''
    if relation != 'P413':
        continue
    '''

    f = ENTITY_PATH.format(relation)
    if not os.path.exists(f):
        continue

    queries: List[Dict] = []
    num_skip = 0
    not_exist = 0
    with open(f) as fin:
        for l in fin:
            l = json.loads(l)
            sub_exist = lang in entity2lang[l['sub_uri']]
            obj_exist = lang in entity2lang[l['obj_uri']]
            exist = sub_exist and obj_exist
            if args.portion == 'trans' and not exist:
                num_skip += 1
                continue
            elif args.portion == 'non' and exist:
                num_skip += 1
                continue
            # load gender
            l['sub_gender'] = entity2gender[l['sub_uri']]
            l['obj_gender'] = entity2gender[l['obj_uri']]
            # resort to English label
            if args.sub_obj_same_lang:
                l['sub_label'] = entity2lang[l['sub_uri']][lang if exist else 'en']
                l['obj_label'] = entity2lang[l['obj_uri']][lang if exist else 'en']
            else:
                l['sub_label'] = entity2lang[l['sub_uri']][lang if sub_exist else 'en']
                l['obj_label'] = entity2lang[l['obj_uri']][lang if obj_exist else 'en']
            if UNK in tokenizer.convert_tokens_to_ids(tokenizer.tokenize(l['sub_label'])) or \
                    UNK in tokenizer.convert_tokens_to_ids(tokenizer.tokenize(l['sub_label'])):
                not_exist += 1
                continue
            queries.append(l)

    acc, len_acc = [], []
    for query_li in tqdm(batcher(queries, BATCH_SIZE), desc=relation, disable=False):
        obj_li: List[np.ndarray] = []
        inp_tensor: List[torch.Tensor] = []
        batch_size = len(query_li)
        for query in query_li:
            # fill in subject and masks
            instance_x, _ = prompt_model.fill_x(
                prompt, query['sub_uri'], query['sub_label'], gender=query['sub_gender'])
            for nm in range(NUM_MASK):
                inp, obj_label = prompt_model.fill_y(
                    instance_x, query['obj_uri'], query['obj_label'], gender=query['obj_gender'],
                    num_mask=nm + 1, mask_sym='[MASK]')
                inp: List[int] = tokenizer.encode(inp)
                inp_tensor.append(torch.tensor(inp))

            # tokenize gold object
            obj = np.array(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj_label))).reshape(-1)
            if len(obj) > NUM_MASK:
                logger.warning('{} is splitted into {} tokens'.format(obj_label, len(obj)))
            obj_li.append(obj)

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
                print('pred {}\tgold {}'.format(
                    tokenizer.convert_ids_to_tokens(pred), tokenizer.convert_ids_to_tokens(obj)))
                input()
            '''

    print('pid {}\t#fact {}\t#skip {}\t#notexist {}\tacc {}\tlen_acc {}'.format(
        relation, len(queries), num_skip, not_exist, np.mean(acc), np.mean(len_acc)))

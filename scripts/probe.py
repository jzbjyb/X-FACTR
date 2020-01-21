from typing import List
import torch
from transformers import *
import json
import numpy as np
import os
from tqdm import tqdm
import argparse

NUM_MASK = 5
BATCH_SIZE = 8
MASK_LABEL = '[MASK]'
PREFIX_DATA = '../LAMA/'
VOCAB_PATH = 'pre-trained_language_models/common_vocab_cased_mbert.txt'
LM_NAME = {
    'mbert_base': 'bert-base-multilingual-cased',
    'bert_base': 'bert-base-cased'
}

def batcher(data: List, batch_size):
    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

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
with open(PREFIX_DATA + 'data/relations.jsonl') as fin:
    patterns.extend([json.loads(l) for l in fin])

# load model
print('load model')
tokenizer = BertTokenizer.from_pretrained(lm)
MASK = tokenizer.convert_tokens_to_ids(MASK_LABEL)
model = BertForMaskedLM.from_pretrained(lm)
model.to('cuda')
model.eval()

# load vocab
with open(PREFIX_DATA + VOCAB_PATH) as fin:
    allowed_vocab = [l.strip() for l in fin]
    allowed_vocab = set(allowed_vocab)
restrict_vocab = [tokenizer.vocab[w] for w in tokenizer.vocab if not w in allowed_vocab]

all_queries = []
for pattern in patterns:
    relation = pattern['relation']
    template = pattern['template']

    f = PREFIX_DATA + 'data/TREx/{}.jsonl'.format(relation)
    if not os.path.exists(f):
        continue

    with open(f) as fin:
        queries = [json.loads(l) for l in fin]

    acc, len_acc = [], []
    for query_li in tqdm(batcher(queries, BATCH_SIZE), desc=relation, disable=False):
        obj_li: List[np.ndarray] = []
        inp_tensor: List[torch.Tensor] = []
        batch_size = len(query_li)
        for query in query_li:
            obj_li.append(np.array(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(query['obj_label']))).reshape(-1))
            instance_x: str = template.replace('[X]', query['sub_label'])
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

    print('pid {} acc {} len_acc {}\n'.format(relation, np.mean(acc), np.mean(len_acc)))

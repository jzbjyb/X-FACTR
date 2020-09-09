import sys
from os.path import dirname, abspath
sys.path.insert(0, dirname(dirname(dirname(abspath(__file__)))))

from typing import List, Dict, Tuple, Set, Union
import traceback
import torch
from transformers import *
import transformers
import json
import numpy as np
import os
from tqdm import tqdm
import argparse
import logging
from collections import defaultdict
import pandas
import csv
import time
import re
from prompt import Prompt
from check_gender import load_entity_gender, Gender
from check_instanceof import load_entity_instance, load_entity_is_cate
from entity_lang import Alias, MultiRel
from tokenization_kobert import KoBertTokenizer


logger = logging.getLogger('mLAMA')
logger.setLevel(logging.ERROR)

SUB_LABEL = '##'
PREFIX_DATA = '../LAMA/'
VOCAB_PATH = PREFIX_DATA + 'pre-trained_language_models/common_vocab_cased.txt'
RELATION_PATH = 'data/TREx-relations.jsonl'
PROMPT_LANG_PATH = 'data/TREx_prompts.csv'
LM_NAME = {
    # multilingual model
    'mbert_base': 'bert-base-multilingual-cased',
    'xlm_base': 'xlm-mlm-100-1280',
    'xlmr_base': 'xlm-roberta-base',
    # language-specific model
    'bert_base': 'bert-base-cased',
    'fr_roberta_base': 'camembert-base',
    'nl_bert_base': 'bert-base-dutch-cased',
    'es_bert_base': 'dccuchile/bert-base-spanish-wwm-cased',
    'ru_bert_base': 'pretrain/rubert_cased_L-12_H-768_A-12_v2',
    # 'DeepPavlov/rubert-base-cased' doesn't include lm head
    'zh_bert_base': 'bert-base-chinese',
    'tr_bert_base': 'dbmdz/bert-base-turkish-cased',
    'ko_bert_base': 'monologg/kobert-lm',
    # 'monologg/kobert' doesn't include lm head
    'el_bert_base': 'nlpaueb/bert-base-greek-uncased-v1'
}
DATASET = {
    'lama': {
        'entity_path': 'data/TREx/{}.jsonl',
        'entity_lang_path': 'data/TREx_unicode_escape.txt',
        'entity_gender_path': 'data/TREx_gender.txt',
        'entity_instance_path': 'data/TREx_instanceof.txt',
        'alias_root': 'data/alias/TREx',
        'multi_rel': 'data/TREx_multi_rel.txt',
        'is_cate': 'data/TREx_is_cate.txt',
    },
    'lama-uhn': {
        'entity_path': 'data/TREx_UHN/{}.jsonl',
        'entity_lang_path': 'data/TREx_unicode_escape.txt',
        'entity_gender_path': 'data/TREx_gender.txt',
        'entity_instance_path': 'data/TREx_instanceof.txt',
        'alias_root': 'data/alias/TREx',
        'multi_rel': 'data/TREx_multi_rel.txt',
        'is_cate': 'data/TREx_is_cate.txt',
    },
    'mlama': {
        'entity_path': 'data/mTREx/sub/{}.jsonl',
        'entity_lang_path': 'data/mTREx_unicode_escape.txt',
        'entity_gender_path': 'data/mTREx_gender.txt',
        'entity_instance_path': 'data/mTREx_instanceof.txt',
        'alias_root': 'data/alias/mTREx',
        'multi_rel': 'data/mTREx_multi_rel.txt',
        'is_cate': 'data/mTREx_is_cate.txt',
    },
    'mlamaf': {
        'entity_path': 'data/mTRExf/sub/{}.jsonl',
        'entity_lang_path': 'data/mTRExf_unicode_escape.txt',
        'entity_gender_path': 'data/mTRExf_gender.txt',
        'entity_instance_path': 'data/mTRExf_instanceof.txt',
        'alias_root': 'data/alias/mTRExf',
        'multi_rel': 'data/mTRExf_multi_rel.txt',
        'is_cate': 'data/mTRExf_is_cate.txt',
    }
}


_tie_breaking: Dict[int, torch.Tensor] = {}
def get_tie_breaking(dim: int):
    if dim not in _tie_breaking:
        _tie_breaking[dim] = torch.zeros(dim).uniform_(0, 1e-5)
    return _tie_breaking[dim]


def get_tokenizer(lang: str, name: str):
    if lang == 'ko' and name in {'monologg/kobert-lm'}:
        return KoBertTokenizer.from_pretrained(name)
    return AutoTokenizer.from_pretrained(name)


def model_prediction_wrap(model, inp_tensor, attention_mask):
    logit = model(inp_tensor, attention_mask=attention_mask)[0]
    if transformers.__version__ in {'2.4.1', '2.4.0'}:
        if hasattr(model, 'cls'):  # bert
            bias = model.cls.predictions.bias
        elif hasattr(model, 'lm_head'):  # roberta
            bias = model.lm_head.bias
        elif hasattr(model, 'pred_layer'):  # xlm
            bias = 0.0
        else:
            raise Exception('not sure whether the bias is correct')
        logit = logit - bias
    elif transformers.__version__ in {'2.3.0'}:
        pass
    else:
        raise Exception('not sure whether version {} is correct'.format(transformers.__version__))
    return logit


def tokenizer_wrap(tokenizer, lang: str, encode: bool, *args, **kwargs):
    params = dict()
    if type(tokenizer) is transformers.tokenization_xlm.XLMTokenizer:
        if lang.startswith('zh-'):
            lang = 'zh'
        params = {'lang': lang}
    if encode:
        return tokenizer.encode(*args, **kwargs, **params)
    else:
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(*args, **kwargs, **params))


class EvalContext(object):
    def __init__(self, args):
        self.norm: bool = args.norm
        self.use_alias: bool = True
        self.uncase: bool = True
        self.use_multi_rel: bool = True
        self.use_period: bool = True
        self.multi_lang: str = args.multi_lang
        self.skip_cate: bool = args.skip_cate
        self.lang: str = args.lang
        self.gold_len: bool = args.gold_len

        for k, v in DATASET[args.probe].items():
            setattr(self, k, v)
        self.lm = LM_NAME[args.model] if args.model in LM_NAME else args.model
        self.entity2gender = load_entity_gender(self.entity_gender_path)
        self.entity2instance = load_entity_instance(self.entity_instance_path)
        self.entity2iscate = load_entity_is_cate(self.is_cate)
        self.prompt_model_dict: Dict[str, Prompt] = \
            {self.lang: Prompt.from_lang(self.lang, self.entity2gender, self.entity2instance)}
        self.tokenizer = get_tokenizer(self.lang, self.lm)
        self.alias_manager = Alias(self.alias_root)
        self.multi_rel_manager = MultiRel(self.multi_rel)


    def get_prompt_model(self, lang: str) -> Prompt:
        if lang not in self.prompt_model_dict:
            self.prompt_model_dict[lang] = Prompt.from_lang(lang, self.entity2gender, self.entity2instance)
        return self.prompt_model_dict[lang]


class CsvLogFileContext:
    def __init__(self, filename: str=None, headers: List[str]=None):
        self.filename = filename
        self.headers = headers


    def __enter__(self):
        if self.filename:
            self.file = open(self.filename, 'w')
            self.file.write(','.join(self.headers) + '\n')
            csv_file = csv.writer(self.file)
            return csv_file
        return None


    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.filename:
            self.file.close()


class LamaPredictions(object):
    greek_unstress = str.maketrans('άόίέύώή', 'αοιευωη')


    def __init__(self, result: Dict, pid: str=None):
        self.result = result
        self.pid = pid


    def __str__(self):
        return json.dumps(self.result)


    @classmethod
    def from_str(cls, str, pid: str=None):
        return cls(json.loads(str), pid)


    @staticmethod
    def prettify_tokens(tokens: List[str], tokenizer):
        return tokenizer.convert_tokens_to_string(tokens)


    @staticmethod
    def is_y_followed_by_at_end(prompt: str, last_char: str) -> bool:
        assert len(last_char) == 1, 'should be a char'
        if prompt[-1] == last_char and re.match('\[.*Y.*\]$', prompt[:-1].rstrip()):
            return True
        return False


    def eval(self, eval: EvalContext) -> bool:
        scores: List[float] = []
        for p in self.result['pred_log_prob']:
            scores.append(np.mean(p) if eval.norm else np.sum(p))
        best = int(np.argmax(scores))
        correct, pred, pred_log_prob, golds = self.match_with_gold(
            self.result['pred'], best, self.pid, self.result,
            use_alias=eval.use_alias, lang=eval.lang, uncase=eval.uncase,
            use_multi_rel=eval.use_multi_rel, use_period=eval.use_period, multi_lang=eval.multi_lang,
            gold_len=eval.gold_len, tokenizer=eval.tokenizer, alias_manager=eval.alias_manager,
            multi_rel_manager=eval.multi_rel_manager, eval=eval)
        self.pred = pred
        self.pred_log_prob = pred_log_prob
        self.correct = correct
        self.golds = golds
        return correct


    @property
    def is_single_word(self) -> bool:
        return len(self.result['tokenized_obj_label_inflection']) <= 1


    @property
    def single_word_pred(self) -> Tuple[str, float]:
        return (self.result['pred'][0][0], self.result['pred_log_prob'][0][0])


    @property
    def num_tokens(self) -> int:
        return len(self.pred)
    
    
    @property
    def confidence(self) -> float:
        return np.exp(np.mean(self.pred_log_prob))


    @property
    def is_use_single_word_pred(self) -> bool:
        return self.num_tokens == 1


    def is_cate(self, entity2iscate: Dict[str, bool]) -> bool:
        if entity2iscate[self.result['sub_uri']]:
            print(self.result['sub_label'])
        return entity2iscate[self.result['sub_uri']]


    def add_prediction(self, pred: List[str], correct: bool):
        self.pred2 = pred
        self.correct2 = correct


    def prettify(self, csv_file: CsvLogFileContext, eval: EvalContext):
        csv_file.writerow([self.prettify_tokens(self.result['sentence'], eval.tokenizer)] +
            [self.prettify_tokens(self.pred, eval.tokenizer)] +
            ([self.prettify_tokens(self.pred2, eval.tokenizer)] if hasattr(self, 'pred2') else []) +
            [' | '.join([self.prettify_tokens(g, eval.tokenizer) for g in self.golds])] +
            [self.correct] + ([self.correct2] if hasattr(self, 'correct2') else []) +
            [self.confidence] + ([self.confidence2] if hasattr(self, 'confidence2') else []) +  # TODO: confidence2
            [self.is_single_word, self.result['sub_uri'], self.result['obj_uri']])


    @staticmethod
    def match_with_gold(pred: List[List[str]],
                        best: int,
                        pid: str,
                        result: Dict,
                        use_alias: bool=False,
                        lang: str=None,
                        uncase: bool=False,
                        use_multi_rel: bool=False,
                        use_period: bool=False,
                        multi_lang: str=None,
                        gold_len: bool=False,
                        tokenizer=None,
                        alias_manager: Alias=None,
                        multi_rel_manager: MultiRel=None,
                        eval: EvalContext=None
                        ) -> Tuple[bool, List[str], List[float], List[List[str]]]:
        if use_multi_rel and not use_alias:
            raise NotImplementedError
        if multi_lang and not use_alias:
            raise NotImplementedError

        raw_lang = lang
        if multi_lang and lang != multi_lang:  # TODO: use all languages?
            langs = [lang, multi_lang]
        else:
            langs = [lang]

        all_golds: List[List[str]] = []
        for lang in langs:
            casify = lambda x: x.lower() if uncase else x
            unstress = lambda x: x.translate(LamaPredictions.greek_unstress) if lang == 'el' else x
            golds: List[List[str]] = [result['tokenized_obj_label_inflection']]
            if use_alias:
                for alias in alias_manager.get_alias(result['obj_uri'], langs=lang):
                    if lang == raw_lang:  # only do inflection for the raw language
                        _, alias = eval.get_prompt_model(lang).fill_y(result['prompt'], result['obj_uri'], alias)
                    golds.append(tokenizer.convert_ids_to_tokens(tokenizer_wrap(tokenizer, lang, False, alias)))
                if use_multi_rel:
                    for obj in multi_rel_manager.get_objects(result['sub_uri'], pid):
                        for alias in alias_manager.get_alias(obj, langs=lang):
                            if lang == raw_lang:  # only do inflection for the raw language
                                _, alias = eval.get_prompt_model(lang).fill_y(result['prompt'], obj, alias)
                            golds.append(tokenizer.convert_ids_to_tokens(tokenizer_wrap(tokenizer, lang, False, alias)))
            all_golds.extend(golds)

            for gold in golds:
                _gold: str = unstress(casify(tokenizer.convert_tokens_to_string(gold)))
                if gold_len and len(gold) <= len(pred):
                    choices = [len(gold) - 1]
                    if lang == 'en' and len(gold) > 1:
                        choices.append(len(gold) - 2)  # the period issue
                else:
                    choices = [best]
                for choice in choices:
                    _pred: str = unstress(casify(tokenizer.convert_tokens_to_string(pred[choice])))
                    if _pred == _gold:
                        return True, pred[choice], result['pred_log_prob'][choice], all_golds
                    if use_period and lang == 'en' and LamaPredictions.is_y_followed_by_at_end(result['prompt'], '.'):
                        if len(_gold) > 0 and _gold[-1] == '.' and _pred.rstrip() == _gold[:-1].rstrip():
                            return True, pred[choice], result['pred_log_prob'][choice], all_golds
        return False, pred[best], result['pred_log_prob'][best], all_golds


class JsonLogFileContext:
    def __init__(self, filename: str=None):
        self.filename = filename


    def __enter__(self):
        if self.filename:
            self.file = open(self.filename, 'w')
            return self.file
        return None


    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.filename:
            self.file.close()


class ProbeIterator(object):
    def __init__(self, args: argparse.Namespace, tokenizer):
        if args.use_gold:
            args.num_mask = 1

        self.args = args
        self.tokenizer = tokenizer

        # special tokens
        self.mask_label = tokenizer.mask_token
        self.unk_label = tokenizer.unk_token
        self.pad_label = tokenizer.pad_token
        self.mask = tokenizer.convert_tokens_to_ids(self.mask_label)
        self.unk = tokenizer.convert_tokens_to_ids(self.unk_label)
        self.pad = tokenizer.convert_tokens_to_ids(self.pad_label)

        # load vocab
        # TODO: add a shared vocab for all LMs?
        # TODO: not work with RoBERTa
        ''' 
        with open(VOCAB_PATH) as fin:
            allowed_vocab = [l.strip() for l in fin]
            allowed_vocab = set(allowed_vocab)
        self.restrict_vocab = [tokenizer.vocab[w] for w in tokenizer.vocab if not w in allowed_vocab]
        '''
        self.restrict_vocab = []

        # prepare path to data
        self.relation_path = RELATION_PATH
        self.prompt_lang_path = PROMPT_LANG_PATH
        for k, v in DATASET[args.probe].items():
            setattr(self, k, v)

        # load data
        self.patterns = []
        with open(self.relation_path) as fin:
            self.patterns.extend([json.loads(l) for l in fin])
        self.entity2lang = load_entity_lang(self.entity_lang_path)
        self.entity2gender: Dict[str, Gender] = load_entity_gender(self.entity_gender_path)
        self.entity2instance: Dict[str, str] = load_entity_instance(self.entity_instance_path)
        self.prompt_lang = pandas.read_csv(self.prompt_lang_path)

        # load facts
        self.restricted_facts = None
        if args.facts is not None:
            filename, part = args.facts.split(':')
            with open(filename, 'r') as fin:
                self.restricted_facts = set(map(tuple, json.load(fin)[part]))
                print('#restricted facts {}'.format(len(self.restricted_facts)))

        # log
        if args.log_dir and not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        if args.pred_dir and not os.path.exists(args.pred_dir):
            os.makedirs(args.pred_dir)

        # prompt model
        self.prompt_model = Prompt.from_lang(
            args.prompt_model_lang or args.lang, self.entity2gender, self.entity2instance,
            args.disable_inflection, args.disable_article)

        # summary
        self.summary = {
            'num_max_mask': 0,  # number of facts where the object has more tokens than the max number of masks
            'numtoken2count': defaultdict(lambda: 0),  # number of token (gold) to count
        }


    def relation_iter(self, pids: Set[str]=None) -> Tuple[Dict, str]:
        for pattern in self.patterns:
            relation = pattern['relation']
            if pids is not None and relation not in pids:
                continue
            fact_path = self.entity_path.format(relation)
            if not os.path.exists(fact_path):
                continue
            yield pattern, fact_path


    def get_queries(self, fact_path: str) -> Tuple[List[Dict], List[Union[int, float]]]:
        LANG = self.args.lang

        queries: List[Dict] = []
        num_skip = not_exist = num_multi_word = num_single_word = 0
        with open(fact_path) as fin:
            for l in fin:
                l = json.loads(l)
                sub_exist = LANG in self.entity2lang[l['sub_uri']]
                obj_exist = LANG in self.entity2lang[l['obj_uri']]
                if self.restricted_facts is not None and \
                        (l['sub_uri'], l['obj_uri']) not in self.restricted_facts:
                    continue
                exist = sub_exist and obj_exist
                if self.args.portion == 'trans' and not exist:
                    num_skip += 1
                    continue
                elif self.args.portion == 'non' and exist:
                    num_skip += 1
                    continue
                # resort to English label
                if self.args.sub_obj_same_lang:
                    l['sub_label'] = self.entity2lang[l['sub_uri']][LANG if exist else 'en']
                    l['obj_label'] = self.entity2lang[l['obj_uri']][LANG if exist else 'en']
                else:
                    l['sub_label'] = self.entity2lang[l['sub_uri']][LANG if sub_exist else 'en']
                    l['obj_label'] = self.entity2lang[l['obj_uri']][LANG if obj_exist else 'en']
                sub_label_t = tokenizer_wrap(self.tokenizer, LANG, False, l['sub_label'])
                obj_label_t = tokenizer_wrap(self.tokenizer, LANG, False, l['obj_label'])
                if self.unk in sub_label_t or self.unk in obj_label_t:
                    not_exist += 1
                    continue
                if len(obj_label_t) <= 1:
                    num_single_word += 1
                    if self.args.skip_single_word:
                        continue
                if len(obj_label_t) > 1:
                    num_multi_word += 1
                    if self.args.skip_multi_word:
                        continue
                queries.append(l)

        return queries, [num_skip, not_exist, num_multi_word, num_single_word]


    def batcher(self, queries: List[Dict], prompt: str) -> Tuple[List, Tuple, Tuple]:
        LANG = self.args.lang
        NUM_MASK = self.args.num_mask

        if self.args.dry_run and self.args.dry_run <= 50:
            queries = queries[:self.args.dry_run]
            print('')

        for b in tqdm(range(0, len(queries), self.args.batch_size), disable=True):
            query_batch = queries[b:b + self.args.batch_size]

            obj_li: List[np.ndarray] = []
            obj_ori_li: List[np.ndarray] = []
            inp_tensor: List[torch.Tensor] = []
            gold_with_mask_tensor: List[torch.Tensor] = []

            for query in query_batch:
                # fill in subjects
                instance_x, _ = self.prompt_model.fill_x(
                    prompt, query['sub_uri'], query['sub_label'])

                # fill in objects
                instance_xys: List[str] = []
                if self.args.use_gold:
                    instance_xy, obj_label = self.prompt_model.fill_y(
                        instance_x, query['obj_uri'], query['obj_label'])
                    if self.args.dry_run and self.args.dry_run <= 50:
                        print(instance_xy)
                    instance_xys.append(instance_xy)
                    nt_obj = len(tokenizer_wrap(self.tokenizer, LANG, False, obj_label))
                    instance_xy_, _ = self.prompt_model.fill_y(
                        instance_x, query['obj_uri'], query['obj_label'],
                        num_mask=nt_obj, mask_sym=self.mask_label)
                    gold_with_mask_tensor.append(
                        torch.tensor(tokenizer_wrap(self.tokenizer, LANG, True, instance_xy_)))
                else:
                    for nm in range(NUM_MASK):
                        if self.args.sent:
                            instance_x = self.args.sent
                        instance_xy, obj_label = self.prompt_model.fill_y(
                            instance_x, query['obj_uri'], query['obj_label'],
                            num_mask=nm + 1, mask_sym=self.mask_label)
                        instance_xys.append(instance_xy)

                # tokenize sentences
                for instance_xy in instance_xys:
                    # TODO: greek BERT does not seem to need this
                    '''
                    if self.args.model == 'el_bert_base':
                        instance_xy = self.prompt_model.normalize(instance_xy, mask_sym=self.mask_label)
                        obj_label = self.prompt_model.normalize(obj_label)
                    '''
                    if re.match('\[.*X.*\]', instance_xy) or re.match('\[.*Y.*\]', instance_xy):
                        raise Exception('inflection missing from "{}"'.format(instance_xy))
                    if not self.args.use_gold and instance_xy.find(self.mask_label) == -1:
                        raise Exception('not contain mask tokens "{}"'.format(instance_xy))
                    inp_tensor.append(torch.tensor(tokenizer_wrap(self.tokenizer, LANG, True, instance_xy)))

                # tokenize gold object
                obj = np.array(tokenizer_wrap(self.tokenizer, LANG, False, obj_label)).reshape(-1)
                obj_li.append(obj)

                # tokenize gold object (before inflection)
                obj_ori = np.array(tokenizer_wrap(self.tokenizer, LANG, False, query['obj_label'])).reshape(-1)
                obj_ori_li.append(obj_ori)

                self.summary['numtoken2count'][len(obj)] += 1
                if len(obj) > NUM_MASK or len(obj_ori) > NUM_MASK:
                    self.summary['num_max_mask'] += 1
                    logger.warning('{} is splitted into {}/{} tokens'.format(obj_label, len(obj), len(obj_ori)))

            # SHAPE: (batch_size * num_mask, seq_len)
            inp_tensor: torch.Tensor = torch.nn.utils.rnn.pad_sequence(
                inp_tensor, batch_first=True, padding_value=self.pad)
            attention_mask: torch.Tensor = inp_tensor.ne(self.pad).long()
            if self.args.use_gold:
                mask_ind: torch.Tensor = torch.nn.utils.rnn.pad_sequence(
                    gold_with_mask_tensor, batch_first=True, padding_value=self.pad).eq(self.mask).long()
            else:
                mask_ind: torch.Tensor = inp_tensor.eq(self.mask).long()

            if torch.cuda.is_available() and not self.args.no_cuda:
                inp_tensor = inp_tensor.cuda()
                attention_mask = attention_mask.cuda()
                mask_ind = mask_ind.cuda()

            yield query_batch, (inp_tensor, attention_mask, mask_ind), (obj_li, obj_ori_li)

        if self.args.dry_run and self.args.dry_run <= 50:
            print('')


    def iter(self, pids: Set[str]=None):
        LANG = self.args.lang
        NUM_MASK = self.args.num_mask

        num_fact = 0
        num_correct_fact = 0
        acc_li: List[float] = []
        iters: List[int] = []

        for pattern, fact_path in self.relation_iter(pids=pids):
            relation = pattern['relation']

            try:
                log_filename = headers = None
                if self.args.log_dir:
                    log_filename = os.path.join(self.args.log_dir, relation + '.csv')
                    headers = ['sentence', 'prediction', 'gold_inflection', 'is_same',
                               'gold_original', 'is_same', 'log_prob']
                json_log_filename = None
                if self.args.pred_dir:
                    json_log_filename = os.path.join(self.args.pred_dir, relation + '.jsonl')
                with CsvLogFileContext(log_filename, headers=headers) as csv_file, \
                        JsonLogFileContext(json_log_filename) as json_file:
                    start_time = time.time()

                    # get queries
                    queries, (num_skip, not_exist, num_multi_word, num_single_word) = self.get_queries(fact_path)

                    # get prompt
                    if self.args.prompts:
                        with open(os.path.join(self.args.prompts, relation + '.jsonl'), 'r') as fin:
                            prompts = [json.loads(l)['template'] for l in fin][:50]  # TODO: top 50
                    else:
                        prompts = [self.prompt_lang[self.prompt_lang['pid'] == relation][LANG].iloc[0]]

                    correct_facts: Set[Tuple[str, str]] = set()

                    for prompt in prompts:
                        acc, len_acc, acc_ori, len_acc_ori = [], [], [], []
                        for qbi, \
                            (query_batch,
                             (inp_tensor, attention_mask, mask_ind),
                             (obj_li, obj_ori_li)) in tqdm(enumerate(self.batcher(queries, prompt)), disable=True):

                            if self.args.dry_run:
                                continue

                            batch_size = len(query_batch)
                            inp_tensor = inp_tensor.view(batch_size, NUM_MASK, -1)
                            attention_mask = attention_mask.view(batch_size, NUM_MASK, -1)
                            mask_ind = mask_ind.view(batch_size, NUM_MASK, -1)

                            out_tensors: List[torch.LongTensor] = []
                            logprobs: List[torch.Tensor] = []
                            for nm in range(NUM_MASK):
                                # decoding
                                # SHAPE: (batch_size * num_mask, seq_len)
                                out_tensor, logprob, iter = iter_decode_beam_search(
                                    model, inp_tensor[:, nm, :], mask_ind[:, nm, :], attention_mask[:, nm, :],
                                    restrict_vocab=self.restrict_vocab, mask_value=self.mask,
                                    max_iter=self.args.max_iter, tokenizer=self.tokenizer,
                                    init_method=self.args.init_method, iter_method=self.args.iter_method,
                                    reprob=self.args.reprob, beam_size=args.beam_size)
                                out_tensors.append(out_tensor)
                                logprobs.append(logprob)
                                iters.append(iter)
                                if self.args.sent:
                                    print('=== #mask {} ==='.format(nm + 1))
                                    print(self.tokenizer.convert_ids_to_tokens(out_tensor[0].cpu().numpy()))
                                    print((logprob[0] * mask_ind[0, nm].float()).sum().cpu().numpy())

                            if self.args.sent:
                                break

                            # SHAPE: (batch_size, num_mask, seq_len)
                            mask_ind = mask_ind.float()
                            logprob = torch.stack(logprobs, 1)
                            out_tensor = torch.stack(out_tensors, 1)

                            # mask len norm
                            mask_len = mask_ind.sum(-1)
                            mask_len_norm = 1.0 if self.args.no_len_norm else mask_len

                            # find the best setting
                            for i, avg_log in enumerate((logprob * mask_ind).sum(-1) / mask_len_norm):
                                lp, best_num_mask = avg_log.max(0)
                                pred: np.ndarray = out_tensor[i, best_num_mask].masked_select(
                                    mask_ind[i, best_num_mask].eq(1)).detach().cpu().numpy().reshape(-1)
                                inp: np.ndarray = inp_tensor[i, best_num_mask].detach().cpu().numpy()

                                obj = obj_li[i]
                                obj_ori = obj_ori_li[i]

                                is_correct = int((len(pred) == len(obj)) and (pred == obj).all())
                                is_correct_ori = int((len(pred) == len(obj_ori)) and (pred == obj_ori).all())

                                len_acc.append(int((len(pred) == len(obj))))
                                len_acc_ori.append(int((len(pred) == len(obj_ori))))

                                acc.append(is_correct)
                                acc_ori.append(is_correct_ori)

                                if is_correct:
                                    correct_facts.add((query_batch[i]['sub_uri'], query_batch[i]['obj_uri']))

                                '''
                                print('===', tokenizer.convert_ids_to_tokens(obj), is_correct, '===')
                                for j in range(NUM_MASK):
                                    print(tokenizer.convert_ids_to_tokens(inp_tensor[i, j].detach().cpu().numpy()))
                                    tpred = out_tensor[i, j].masked_select(mask_ind[i, j].eq(1)).detach().cpu().numpy().reshape(-1)
                                    print(tokenizer.convert_ids_to_tokens(tpred), avg_log[j])
                                input()
                                '''

                                if self.args.log_dir:
                                    csv_file.writerow([
                                        load_word_ids(inp, self.tokenizer, self.pad_label),
                                        load_word_ids(pred, self.tokenizer, self.pad_label),
                                        load_word_ids(obj, self.tokenizer, self.pad_label), is_correct,
                                        load_word_ids(obj_ori, self.tokenizer, self.pad_label), is_correct_ori,
                                        '{:.5f}'.format(lp.item())])

                                def get_all_pred_score():
                                    results: List[str] = []
                                    for nm in range(NUM_MASK):
                                        pred = logprob[i, nm].masked_select(
                                            mask_ind[i, nm].eq(1)).detach().cpu().numpy().reshape(-1)
                                        results.append(pred.tolist())
                                    return results

                                def get_all_pred():
                                    results: List[str] = []
                                    for nm in range(NUM_MASK):
                                        pred = out_tensor[i, nm].masked_select(
                                            mask_ind[i, nm].eq(1)).detach().cpu().numpy().reshape(-1)
                                        results.append(merge_subwords(pred, tokenizer, merge=False))
                                    return results

                                if self.args.pred_dir:
                                    json_file.write(str(LamaPredictions({
                                        # raw data
                                        'relation': relation,
                                        'sub_uri': query_batch[i]['sub_uri'],
                                        'obj_uri': query_batch[i]['obj_uri'],
                                        'sub_label': query_batch[i]['sub_label'],
                                        'obj_label': query_batch[i]['obj_label'],
                                        'prompt': prompt,
                                        # tokenized data
                                        'num_mask': best_num_mask.item() + 1,
                                        'sentence': merge_subwords(inp, tokenizer, merge=False),
                                        'tokenized_obj_label_inflection': merge_subwords(obj, tokenizer, merge=False),
                                        'tokenized_obj_label': merge_subwords(obj_ori, tokenizer, merge=False),
                                        # predictions
                                        'pred': get_all_pred(),
                                        'pred_log_prob': get_all_pred_score(),
                                    })) + '\n')

                                '''
                                if len(pred) == len(obj):
                                    print('pred {}\tgold {}'.format(
                                        tokenizer.convert_ids_to_tokens(pred), tokenizer.convert_ids_to_tokens(obj)))
                                    input()
                                '''

                        print('pid {}\tacc {:.4f}/{:.4f}\tlen_acc {:.4f}/{:.4f}\tprompt {}'.format(
                            relation, np.mean(acc), np.mean(acc_ori), np.mean(len_acc), np.mean(len_acc_ori), prompt))

                    num_fact += len(queries)
                    num_correct_fact += len(correct_facts)
                    acc_for_rel = len(correct_facts) / (len(queries) + 1e-10)
                    acc_li.append(acc_for_rel)

                    print('pid {}\t#fact {}\t'
                          '#notrans {}\t#notexist {}\t#multiword {},{}\t#singleword {},{}\t'
                          'oracle {:.4f}\ttime {:.1f}'.format(
                        relation, len(queries),
                        num_skip, not_exist, num_multi_word, self.args.skip_multi_word,
                        num_single_word, self.args.skip_single_word,
                        acc_for_rel, time.time() - start_time))

            except Exception as e:
                print('bug for pid {}'.format(relation))
                print(e)
                traceback.print_exc()
                raise e

        print('acc per fact {}/{}={:.4f}\tacc per relation {}\tavg iter {}\tnum_max_mask {}'.format(
            num_correct_fact, num_fact, num_correct_fact / (num_fact + 1e-10),
            np.mean(acc_li), np.mean(iters), self.summary['num_max_mask']))
        if args.dry_run:
            for nt in range(1, np.max(list(self.summary['numtoken2count'].keys())) + 1):
                _ = self.summary['numtoken2count'][nt]
            print('numtoken2count', sorted(self.summary['numtoken2count'].items(), key=lambda x: x[0]))


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


def load_word_ids(ids: Union[np.ndarray, List[int]], tokenizer, pad_label: str) -> str:
    tokens: List[Tuple[str, int]] = []
    for t in tokenizer.convert_ids_to_tokens(ids):
        if t == pad_label:
            continue
        if t.startswith(SUB_LABEL) and len(tokens) > 0:
            tokens[-1][0] += t[len(SUB_LABEL):]
            tokens[-1][1] += 1
        else:
            tokens.append([t, 1])
    return ' '.join(map(lambda t: '{}:{}'.format(*t) if t[1] > 1 else t[0], tokens))


def merge_subwords(ids: Union[np.ndarray, List[int]], tokenizer, merge: bool=False) -> str:
    if not merge:
        return list(tokenizer.convert_ids_to_tokens(ids))
    return NotImplementedError


def iter_decode_beam_search(model,
                            inp_tensor: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
                            raw_mask: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
                            attention_mask: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
                            restrict_vocab: List[int] = None,
                            mask_value: int = 0,  # indicate which value is used for mask
                            max_iter: int = None,  # max number of iteration
                            tokenizer = None,
                            init_method: str='all',
                            iter_method: str='none',
                            reprob: bool = False,  # recompute the prob finally
                            beam_size: int = 5,
                            ) -> Tuple[torch.LongTensor, torch.Tensor, int]:  # HAPE: (batch_size, seq_len)
    '''
    Masks must be consecutive.
    '''
    assert init_method in {'all', 'left', 'confidence'}
    assert iter_method in {'none', 'left', 'confidence', 'confidence-multi'}
    bs, sl = inp_tensor.size(0), inp_tensor.size(1)
    init_mask = inp_tensor.eq(mask_value).long()  # SHAPE: (batch_size, seq_len)
    init_has_mask = init_mask.sum().item() > 0

    if iter_method == 'confidence-multi':
        number_to_mask = torch.unique(init_mask.sum(-1))
        assert number_to_mask.size(0) == 1, 'this batch has different numbers of mask tokens'
        number_to_mask = number_to_mask[0].item() - 1
        assert max_iter == 0, 'do not need to set max_iter in confidence-multi setting'
    elif iter_method == 'left':
        leftmost_mask = init_mask * torch.cat([init_mask.new_ones((bs, 1)), 1 - init_mask], 1)[:, :-1]
        number_to_mask = torch.unique(init_mask.sum(-1))
        assert number_to_mask.size(0) == 1, 'this batch has different numbers of mask tokens'
        number_to_mask: int = number_to_mask[0].item()
        mask_offset: int = 0
        has_modified: bool = False  # track wether modification happens during a left-to-right pass

    # SHAPE: (<=beam_size, batch_size, seq_len)
    out_tensors: List[torch.LongTensor] = inp_tensor.unsqueeze(0)
    # tokens not considered have log prob of zero
    out_logprobs: List[torch.Tensor] = torch.zeros_like(inp_tensor).float().unsqueeze(0)
    iter: int = 0
    stop: bool = False
    while True and init_has_mask:  # skip when there is not mask initially
        next_out_tensors = []
        next_out_logprobs = []

        # enumerate over all previous result
        for out_tensor, out_logprob in zip(out_tensors, out_logprobs):
            #print(tokenizer.convert_ids_to_tokens(out_tensor[0].cpu().numpy()))

            # get input
            if iter > 0:
                if iter_method == 'none':
                    inp_tensor = out_tensor
                    if inp_tensor.eq(mask_value).long().sum().item() == 0:  # no mask
                        stop = True
                        break
                elif iter_method == 'confidence':
                    has_mask = out_tensor.eq(mask_value).any(-1).unsqueeze(-1).long()  # SHAPE: (batch_size, 1)
                    inp_tensor = out_tensor.scatter(1, out_logprob.min(-1)[1].unsqueeze(-1), mask_value)
                    # no need to insert mask when there are masks
                    inp_tensor = out_tensor * has_mask + inp_tensor * (1 - has_mask)
                elif iter_method == 'confidence-multi':
                    has_mask = out_tensor.eq(mask_value).any(-1).unsqueeze(-1) # SHAPE: (batch_size, 1)
                    all_has_mask = has_mask.all().item()
                    assert all_has_mask == has_mask.any().item(), 'some samples have masks while the others do not'
                    if not all_has_mask:
                        if number_to_mask <= 0:
                            stop = True
                            break
                        inp_tensor = out_tensor.scatter(1, (-out_logprob).topk(number_to_mask, dim=-1)[1], mask_value)
                        init_method = 'all'
                        number_to_mask -= 1
                    else:
                        inp_tensor = out_tensor
                elif iter_method == 'left':
                    has_mask = out_tensor.eq(mask_value).any(-1).unsqueeze(-1)  # SHAPE: (batch_size, 1)
                    all_has_mask = has_mask.all().item()
                    any_has_mask = has_mask.any().item()
                    assert all_has_mask == any_has_mask, \
                        'some samples have masks while the others do not'
                    if not all_has_mask:  # no mask, should do refinement
                        if mask_offset >= number_to_mask:
                            mask_offset = 0
                        if mask_offset == 0:  # restart when starting from the beginning
                            has_modified = False
                        cur_mask = torch.cat([leftmost_mask.new_zeros((bs, mask_offset)), leftmost_mask], 1)[:, :sl]
                        cur_mask = cur_mask * init_mask
                        inp_tensor = out_tensor * (1 - cur_mask) + mask_value * cur_mask
                        mask_offset += 1
                    else:
                        inp_tensor = out_tensor
                else:
                    raise NotImplementedError

            # predict
            # SHAPE: (batch_size, seq_len)
            mask_mask = inp_tensor.eq(mask_value).long()
            logit = model_prediction_wrap(model, inp_tensor, attention_mask)
            if restrict_vocab is not None:
                logit[:, :, restrict_vocab] = float('-inf')
            # SHAPE: (batch_size, seq_len, beam_size)
            new_out_logprobs, new_out_tensors = logit.log_softmax(-1).topk(beam_size, dim=-1)

            if init_method == 'confidence':
                # mask out non-mask positions
                new_out_logprobs = new_out_logprobs + mask_mask.unsqueeze(-1).float().log()
                new_out_logprobs = new_out_logprobs.view(-1, sl * beam_size)
                new_out_tensors = new_out_tensors.view(-1, sl * beam_size)

            for b in range(beam_size):
                if init_method == 'all':
                    new_out_logprob = new_out_logprobs[:, :, b]
                    new_out_tensor = new_out_tensors[:, :, b]
                    # SHAPE: (batch_size, seq_len)
                    changes = (out_tensor * mask_mask).ne(new_out_tensor * mask_mask)
                elif init_method == 'left':  # only modify the left-most one.
                    new_out_logprob = new_out_logprobs[:, :, b]
                    new_out_tensor = new_out_tensors[:, :, b]
                    # SHAPE: (batch_size, seq_len)
                    changes = (out_tensor * mask_mask).ne(new_out_tensor * mask_mask)
                    changes = changes & torch.cat([changes.new_ones((bs, 1)), ~changes], 1)[:, :-1]
                elif init_method == 'confidence':  # only modify the most confident one.
                    # SHAPE: (batch_size,)
                    raw_lp, raw_ind = new_out_logprobs.max(-1)
                    # SHAPE: (batch_size, 1)
                    raw_lp, raw_ind = raw_lp.unsqueeze(-1), raw_ind.unsqueeze(-1)
                    seq_ind = raw_ind // beam_size
                    changes = mask_mask & torch.zeros_like(mask_mask).scatter(1, seq_ind, True)
                    new_out_tensor = torch.zeros_like(out_tensor).scatter(1, seq_ind, new_out_tensors.gather(1, raw_ind))
                    new_out_logprob = torch.zeros_like(out_logprob).scatter(1, seq_ind, raw_lp)
                    changes = (out_tensor * changes.long()).ne(new_out_tensor * changes.long())
                    # max for the next max in beam search
                    new_out_logprobs = new_out_logprobs.scatter(1, raw_ind, float('-inf'))
                else:
                    raise NotImplementedError

                # only modify tokens that have changes
                changes = changes.long()
                _out_tensor = out_tensor * (1 - changes) + new_out_tensor * changes
                _out_logprob = out_logprob * (1 - changes.float()) + new_out_logprob.detach() * changes.float()

                # involves heavy computation, where we re-compute probabilities for beam_size * beam_size samples
                if reprob:
                    _out_logprob = compute_likelihood(
                        model, _out_tensor, _out_logprob,
                        init_mask, attention_mask, restrict_vocab, mask_value=mask_value)
                    _out_logprob = _out_logprob * (1 - _out_tensor.eq(mask_value).float())  # skip mask tokens

                next_out_tensors.append(_out_tensor)
                next_out_logprobs.append(_out_logprob)

                '''
                for i in range(bs):
                    print(tokenizer.convert_ids_to_tokens(inp_tensor[i].cpu().numpy()))
                    print(tokenizer.convert_ids_to_tokens(_out_tensor[i].cpu().numpy()))
                input()
                '''

        if stop:
            break

        next_out_tensors = torch.stack(next_out_tensors, 0)
        next_out_logprobs = torch.stack(next_out_logprobs, 0)
        # tie breaking
        next_out_logprobs = next_out_logprobs + \
                            get_tie_breaking(int(next_out_logprobs.size(0))).view(-1, 1, 1).to(next_out_logprobs.device)

        # dedup
        not_dups = []
        for i in range(bs):
            abs = next_out_tensors.size(0)
            # SHAPE: (all_beam_size, seq_len)
            one_sample = next_out_tensors[:, i, :]
            # SHAPE: (all_beam_size,)
            inv = torch.unique(one_sample, dim=0, return_inverse=True)[1]
            # SHAPE: (all_beam_size, all_beam_size)
            not_dup = inv.unsqueeze(-1).ne(inv.unsqueeze(0)) | \
                      (torch.arange(abs).unsqueeze(-1) <= torch.arange(abs).unsqueeze(0)).to(inv.device)
            # SHAPE: (all_beam_size,)
            not_dup = not_dup.all(-1)
            not_dups.append(not_dup)
        # SHAPE: (all_beam_size, batch_size)
        not_dups = torch.stack(not_dups, -1)

        # select top
        # SHAPE: (all_beam_size, batch_size)
        beam_score = (next_out_logprobs * init_mask.unsqueeze(0).float() +
                      not_dups.unsqueeze(-1).float().log()).sum(-1)
        # SHAPE: (beam_size, batch_size, seq_len)
        beam_top = beam_score.topk(beam_size, dim=0)[1].view(-1, bs, 1).repeat(1, 1, sl)
        next_out_logprobs = torch.gather(next_out_logprobs, 0, beam_top)
        next_out_tensors = torch.gather(next_out_tensors, 0, beam_top)

        # stop condition for other type of iter
        if next_out_tensors.size(0) == out_tensors.size(0) and next_out_tensors.eq(out_tensors).all():
            if iter_method != 'left':
                stop = True
        else:
            if iter_method == 'left':
                has_modified = True
        # stop condition for 'left' iter
        if iter_method == 'left' and not has_modified and mask_offset == number_to_mask:
            # reach the last position and no modification happens during this iteration
            stop = True

        #print(next_out_tensors.ne(out_tensors).any(-1).any(0).nonzero())

        out_tensors = next_out_tensors
        out_logprobs = next_out_logprobs

        iter += 1
        if max_iter and iter >= max_iter:  # max_iter can be zero
            stop = True
        if stop:
            break

    out_tensor = out_tensors[0]
    final_out_logprob = out_logprobs[0]

    return out_tensor, final_out_logprob, iter


def compute_likelihood(model,
                       inp_tensor: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
                       lp_tensor: torch.Tensor,  # SHAPE: (batch_size, seq_len)
                       mask_tensor: torch.LongTensor,  # SHAPE: (batch_size, seq_len)
                       attention_mask: torch.LongTensor,  # SHAPE: (batch_size, seq_len))
                       restrict_vocab: List[int] = None,
                       mask_value: int=0,  # indicate which value is used for mask
                       ) -> torch.Tensor:  # SHAPE: (batch_size, seq_len)
    '''
    Masks must be consecutive.
    '''
    bs, seq_len = inp_tensor.size(0), inp_tensor.size(1)
    max_num_masks = mask_tensor.sum(-1).max().item()
    leftmost_mask = mask_tensor * torch.cat([mask_tensor.new_ones((bs, 1)), 1 - mask_tensor], 1)[:, :-1]
    logits = None
    for i in range(max_num_masks):
        # SHAPE: (batch_size, seq_len)
        cur_mask = torch.cat([leftmost_mask.new_zeros((bs, i)), leftmost_mask], 1)[:, :seq_len] * mask_tensor
        inp_tensor_ = (1 - cur_mask) * inp_tensor + cur_mask * mask_value
        logit = model_prediction_wrap(model, inp_tensor_, attention_mask)
        cur_mask = cur_mask.unsqueeze(-1).float()
        if logits is None:
            logits = (logit * cur_mask).detach()
        else:
            logits = (logits * (1 - cur_mask) + logit * cur_mask).detach()
    if restrict_vocab is not None:
        logits[:, :, restrict_vocab] = float('-inf')
    lp = logits.log_softmax(-1)
    lp = torch.gather(lp.view(-1, lp.size(-1)), 1, inp_tensor.view(-1).unsqueeze(-1)).view(bs, seq_len)
    lp_tensor = (1 - mask_tensor).float() * lp_tensor + mask_tensor.float() * lp
    return lp_tensor.detach()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='probe LMs with multilingual LAMA')
    parser.add_argument('--model', type=str, help='LM to probe file', default='mbert_base')
    parser.add_argument('--lm_layer_model', type=str,
                        help='LM from which the final lm layer is used', default=None)
    parser.add_argument('--lang', type=str, help='language to probe',
                        choices=['en', 'fr', 'nl', 'es', 'zh',
                                 'mr', 'vi', 'ko', 'he', 'yo',
                                 'el', 'tr', 'ru'], default='en')
    parser.add_argument('--sent', type=str, help='actual sentence with [Y]', default=None)

    # dataset-related flags
    parser.add_argument('--probe', type=str, help='probe dataset',
                        choices=['lama', 'lama-uhn', 'mlama', 'mlamaf'], default='mlamaf')
    parser.add_argument('--pids', type=str, help='pids to run', default=None)
    parser.add_argument('--portion', type=str, choices=['all', 'trans', 'non'], default='trans',
                        help='which portion of facts to use')
    parser.add_argument('--facts', type=str, help='file path to facts', default=None)
    parser.add_argument('--prompts', type=str, default=None,
                        help='directory where multiple prompts are stored for each relation')
    parser.add_argument('--sub_obj_same_lang', action='store_true',
                        help='use the same language for sub and obj')
    parser.add_argument('--skip_multi_word', action='store_true',
                        help='skip objects with multiple words (not sub-words)')
    parser.add_argument('--skip_single_word', action='store_true',
                        help='skip objects with a single word')

    # inflection-related flags
    parser.add_argument('--prompt_model_lang', type=str, help='prompt model to use',
                        choices=['en', 'el', 'ru', 'es', 'mr'], default=None)
    parser.add_argument('--disable_inflection', type=str, choices=['x', 'y', 'xy'])
    parser.add_argument('--disable_article', action='store_true')

    # decoding-related flags
    parser.add_argument('--num_mask', type=int, help='the maximum number of masks to insert', default=5)
    parser.add_argument('--max_iter', type=int, help='the maximum number of iteration in decoding', default=1)
    parser.add_argument('--init_method', type=str, help='iteration method', default='all')
    parser.add_argument('--iter_method', type=str, help='iteration method', default='none')
    parser.add_argument('--no_len_norm', action='store_true', help='not use length normalization')
    parser.add_argument('--reprob', action='store_true', help='recompute the prob finally')
    parser.add_argument('--beam_size', type=int, help='beam search size', default=1)

    # others
    parser.add_argument('--use_gold', action='store_true', help='use gold objects')
    parser.add_argument('--dry_run', type=int, help='dry run the probe to show inflection results', default=None)
    parser.add_argument('--log_dir', type=str, help='directory to vis prediction results', default=None)
    parser.add_argument('--pred_dir', type=str, help='directory to store prediction results', default=None)
    parser.add_argument('--batch_size', type=int, help='the real batch size is this times num_mask', default=20)
    parser.add_argument('--no_cuda', action='store_true', help='not use cuda')
    args = parser.parse_args()

    if (args.init_method != 'all' or args.iter_method != 'none') and args.max_iter:
        assert args.max_iter >= args.num_mask, 'the results will contain mask'
    if args.sent:
        args.batch_size = 1
        args.pids = 'P19'

    LM = LM_NAME[args.model] if args.model in LM_NAME else args.model  # use pre-defined models or path

    # load data
    print('load data')
    tokenizer = get_tokenizer(args.lang, LM)
    probe_iter = ProbeIterator(args, tokenizer)

    # load model
    print('load model')
    model = AutoModelWithLMHead.from_pretrained(LM)
    if args.lm_layer_model is not None:
        llm = LM_NAME[args.lm_layer_model] if args.lm_layer_model in LM_NAME else args.lm_layer_model
        llm = AutoModelWithLMHead.from_pretrained(llm)
        model.cls = llm.cls
    model.eval()
    if torch.cuda.is_available() and not args.no_cuda:
        model.to('cuda')

    probe_iter.iter(pids=set(args.pids.strip().split(',')) if args.pids is not None else None)

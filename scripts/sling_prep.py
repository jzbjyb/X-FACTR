from typing import Tuple, Iterable, List, Dict, Any, Set
import os
import json
import argparse
from termcolor import colored
from collections import defaultdict
import numpy as np
import glob
from tqdm import tqdm
import random
import string
import importlib
import sling
from distantly_supervise import SlingExtractor
from probe import load_entity_lang
from ft import CodeSwitchDataset


SEED = 2020
random.seed(SEED)
np.random.seed(SEED)


# inspect a record file
# bazel-bin/tools/codex --frames local/data/e.bak/ner/fi/documents-00000-of-00010.rec | less

# sling python API
# https://github.com/google/sling/blob/master/doc/guide/pyapi.md


commons = sling.Store()
DOCSCHEMA = sling.DocumentSchema(commons)
commons.freeze()
PUNCT = set(list(string.punctuation))
LANG2STOPWORDS: Dict[str, Set[str]] = {}


def get_stopwords(lang: str):
    if lang not in LANG2STOPWORDS:
        LANG2STOPWORDS[lang] = importlib.import_module('spacy.lang.{}'.format(lang)).stop_words.STOP_WORDS
    return LANG2STOPWORDS[lang]


def get_metadata(frame: sling.Frame) -> Tuple[int, str, str]:
    pageid = frame.get('/wp/page/pageid')  # wikiPEDIA page ID
    title = frame.get('/wp/page/title')  # article title
    item = frame.get('/wp/page/item')  # wikidata ID associated to the article
    return pageid, title, item


class SlingExtractorForQualifier(SlingExtractor):
    def __init__(self, *args, **kwargs):
        super(SlingExtractorForQualifier).__init__(*args, **kwargs)


    def iter_mentions_position(self, wid_set: Set[str]=None) -> Iterable[Tuple[str, Dict[str, List[int]]]]:
        for n, (doc_wid, doc_raw) in enumerate(self.corpus.input):
            doc_wid = str(doc_wid, 'utf-8')
            if wid_set is not None and doc_wid not in wid_set:
                continue
            store = sling.Store(self.commons)
            frame = store.parse(doc_raw)
            document = sling.Document(frame, store, self.docschema)
            sorted_mentions = sorted(document.mentions, key=lambda m: m.begin)
            m2pos: Dict[str, List[int]] = defaultdict(list)
            for ii, mention in enumerate(sorted_mentions):
                linked_entity = self.get_linked_entity(mention)
                m2pos[linked_entity].append(mention.begin)
            yield (doc_wid, m2pos)


    def iter_mentions(self, wid_set: Set[str]=None, only_entity: bool=False, split_by: str=None) -> \
            Iterable[Tuple[str, sling.Document, List[Tuple[str, int, int]]]]:
        assert split_by in {'sentence', None}, 'not supported split_by'
        split_by = {'sentence': 3, None: None}[split_by]
        for n, (doc_wid, doc_raw) in enumerate(self.corpus.input):
            doc_wid = str(doc_wid, 'utf-8')
            if wid_set is not None and doc_wid not in wid_set:
                continue
            store = sling.Store(self.commons)
            frame = store.parse(doc_raw)
            document = sling.Document(frame, store, self.docschema)
            sorted_mentions = sorted(document.mentions, key=lambda m: m.begin)
            tokens = [t.word for t in document.tokens]
            split_start = [0] + [i for i, t in enumerate(document.tokens) if t.brk == split_by]
            split_ind = 0
            mentions: List[Tuple[str, int, int]] = []
            for mention in sorted_mentions:
                while len(split_start) > split_ind + 1 and mention.begin >= split_start[split_ind + 1]:
                    if len(mentions) > 0:
                        yield (doc_wid, tokens[split_start[split_ind]:split_start[split_ind + 1]], mentions)
                        mentions = []
                    split_ind += 1
                if len(split_start) > split_ind + 1 and mention.end > split_start[split_ind + 1]:
                    # skip mentions beyond the boundary
                    continue
                linked_entity = self.get_linked_entity(mention)
                if only_entity and (type(linked_entity) is not str or not linked_entity.startswith('Q')):
                    continue
                mentions.append((linked_entity,
                                 mention.begin - split_start[split_ind],
                                 mention.end - split_start[split_ind]))
            if len(mentions) > 0:
                yield (doc_wid, tokens[split_start[split_ind]:], mentions)


    def find_all_mentions(self):
        for n, (doc_wid, doc_raw) in enumerate(self.corpus.input):
            doc_wid = str(doc_wid, 'utf-8')
            store = sling.Store(self.commons)
            frame = store.parse(doc_raw)
            document = sling.Document(frame, store, self.docschema)
            doc_title = get_metadata(frame)[1]
            if len(document.tokens) == 0:
                continue
            sorted_mentions = sorted(document.mentions, key=lambda m: m.begin)
            all_mentions: List[Tuple[int, int, Any]] = []
            for ii, mention in enumerate(sorted_mentions):
                linked_entity = self.get_linked_entity(mention)
                all_mentions.append((mention.begin, mention.end, linked_entity))

            tokens = [t.word for t in document.tokens]
            prev_e = 0
            colored_tokens: List[str] = []
            for s, e, wid in all_mentions:
                colored_tokens.append(' '.join(tokens[prev_e:s]))
                colored_tokens.append(colored('{}||{}'.format(' '.join(tokens[s:e]), wid), 'green'))
                prev_e = e
            colored_text = ' '.join(colored_tokens)
            yield doc_wid, doc_title, colored_text


    def find_date_mentions(self) -> Iterable:
        for n, (doc_wid, doc_raw) in enumerate(self.corpus.input):
            doc_wid = str(doc_wid, 'utf-8')
            store = sling.Store(self.commons)
            frame = store.parse(doc_raw)
            document = sling.Document(frame, store, self.docschema)
            doc_title = get_metadata(frame)[1]
            if len(document.tokens) == 0:
                continue
            sorted_mentions = sorted(document.mentions, key=lambda m: m.begin)
            postion2time: Dict[int, int] = {}
            for ii, mention in enumerate(sorted_mentions):
                linked_entity = self.get_linked_entity(mention)
                if type(linked_entity) is not int:
                    continue
                for i in range(mention.begin, mention.end):
                    postion2time[i] = linked_entity
            colored_tokens: List[str] = []
            for i, tok in enumerate(document.tokens):
                if i in postion2time:
                    colored_tokens.append(colored('{}:{}'.format(tok.word, postion2time[i]), 'green'))
                else:
                    if tok.word.isnumeric():
                        colored_tokens.append(colored(tok.word, 'red'))
                    else:
                        colored_tokens.append(tok.word)
            colored_text = ' '.join(colored_tokens)
            yield doc_wid, doc_title, colored_text


def locate_entity(entities: Set[str], sling_recfiles: List[str]) -> set:
    s = SlingExtractorForQualifier()
    for sr in sling_recfiles:
        s.load_corpus(sr)
        # TODO: only find sentences in articles corresponding to these entities
        for wid, tokens, mentions in tqdm(s.iter_mentions(wid_set=entities, only_entity=True, split_by='sentence')):
            mentions = [mention for mention in mentions if mention[0] in entities]
            if len(mentions) == 0:
                continue
            yield tokens, mentions


def locate_fact(facts: Set[Tuple[str, str]], sling_recfiles: List[str], thres: int) -> set:
    s = SlingExtractorForQualifier()
    notfound = set(facts)
    fact_entities = set(e for f in facts for e in f)
    for sr in sling_recfiles:
        s.load_corpus(sr)
        for wid, m2pos in tqdm(s.iter_mentions_position(wid_set=fact_entities)):
            found = set()
            for sub, obj in notfound:
                if sub not in m2pos or obj not in m2pos:
                    continue
                dist = np.min(np.abs(np.array(m2pos[sub]).reshape(1, -1) - np.array(m2pos[obj]).reshape(-1, 1)))
                if dist <= thres:
                    found.add((sub, obj))
            notfound -= found
    return facts - notfound


def check_prompt(prompt: str, lang: str) -> bool:
    stopwords = get_stopwords(lang)
    prompt = prompt.replace('[X]', '').replace('[Y]', '').strip().lower()
    if len(prompt) <= 0:  # empty prompt
        return False
    if prompt in PUNCT:  # remove punct
        return False
    if prompt[0] in PUNCT:  # start with punct
        return False
    tokens = prompt.split(' ')
    if len(tokens) > 10:  # too long
        return False
    if np.all([t in stopwords for t in tokens]):  # all tokens are stopwords
        return False
    return True


def distant_supervision_sentences(fact2pid: Dict[Tuple[str, str], Set[str]], lang: str, dist_thres: int):
    s = SlingExtractorForQualifier()
    pid2prompts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(lambda: 0))
    for sr in glob.glob('data/sling/{}/*.rec'.format(lang)):
        s.load_corpus(sr)
        for wid, tokens, mentions in tqdm(s.iter_mentions(wid_set=None, only_entity=True, split_by='sentence')):
            for i, left_m in enumerate(mentions):
                for j in range(i + 1, min(i + 10, len(mentions))):
                    right_m = mentions[j]
                    if right_m[1] - left_m[2] < 0:
                        continue
                    if right_m[1] - left_m[2] > dist_thres:
                        break
                    if (left_m[0], right_m[0]) in fact2pid:
                        to_insert = {
                            left_m[1]: '[[',
                            left_m[2]: ']]_x_' + left_m[0],
                            right_m[1]: '[[',
                            right_m[2]: ']]_y_' + right_m[0],
                        }
                        new_tokens = []
                        for ti, t in enumerate(tokens):
                            if ti in to_insert:
                                new_tokens.append(to_insert[ti])
                            new_tokens.append(t)
                        for pid in fact2pid[(left_m[0], right_m[0])]:
                            yield new_tokens, pid

                    if (right_m[0], left_m[0]) in fact2pid:
                        to_insert = {
                            left_m[1]: '[[',
                            left_m[2]: ']]_y_' + left_m[0],
                            right_m[1]: '[[',
                            right_m[2]: ']]_x_' + right_m[0],
                        }
                        new_tokens = []
                        for ti, t in enumerate(tokens):
                            if ti in to_insert:
                                new_tokens.append(to_insert[ti])
                            new_tokens.append(t)
                        for pid in fact2pid[(right_m[0], left_m[0])]:
                            yield new_tokens, pid


def distant_supervision(fact2pid: Dict[Tuple[str, str], Set[str]],
                        lang: str, dist_thres: int, count_thres: int, outdir: str):
    s = SlingExtractorForQualifier()
    pid2prompts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(lambda: 0))
    for sr in glob.glob('data/sling/{}/*.rec'.format(lang)):
        s.load_corpus(sr)
        for wid, tokens, mentions in tqdm(s.iter_mentions(wid_set=None, only_entity=True)):
            for i, left_m in enumerate(mentions):
                for j in range(i + 1, min(i + 10, len(mentions))):
                    right_m = mentions[j]
                    if right_m[1] - left_m[2] < 0:
                        continue
                    if right_m[1] - left_m[2] > dist_thres:
                        break
                    prompt = ' '.join(tokens[left_m[2]:right_m[1]])
                    if (left_m[0], right_m[0]) in fact2pid:
                        for pid in fact2pid[(left_m[0], right_m[0])]:
                            pid2prompts[pid]['[X] ' + prompt + ' [Y]'] += 1
                    if (right_m[0], left_m[0]) in fact2pid:
                        for pid in fact2pid[(right_m[0], left_m[0])]:
                            pid2prompts[pid]['[Y] ' + prompt + ' [X]'] += 1
    if not os.path.exists(outdir):
        os.makedirs(outdir, exist_ok=True)
    for pid, prompts in pid2prompts.items():
        prompts = [(p, c) for p, c in sorted(prompts.items(), key=lambda x: -x[1])
                   if c >= count_thres and check_prompt(p, lang=lang)]
        with open(os.path.join(outdir, '{}.jsonl'.format(pid)), 'w') as fout:
            for p, c in prompts:
                p_ = {
                    'relation': pid,
                    'template': p + ' .',
                    'wikipedia_count': c
                }
                fout.write(json.dumps(p_).encode('utf8').decode('unicode_escape') + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SLING-related preprocessing')
    parser.add_argument('--task', type=str, choices=['inspect', 'ds', 'cw', 'cw_split'])
    parser.add_argument('--lang', type=str, help='language to probe',
                        choices=['en', 'el', 'fr', 'nl', 'ru'], default='en')
    parser.add_argument('--dir', type=str, help='data dir')
    parser.add_argument('--inp', type=str, help='input file', default=None)
    parser.add_argument('--out', type=str, help='output file', default=None)
    parser.add_argument('--down_sample', type=float, help='down sample ratio', default=None)
    parser.add_argument('--balance_lang', action='store_true', help='balance the data between two languages')
    args = parser.parse_args()

    if args.task == 'inspect':
        s = SlingExtractorForQualifier()
        s.load_corpus(corpus_file=args.inp)
        for wid, title, text in s.find_all_mentions():
            print('=' * 30)
            print(wid, title)
            print(text[:10000])
            input('press any key to continue ...')

    elif args.task == 'ds':
        print('load triples ...')
        fact2pid = defaultdict(set)
        with open('data/triple_subset.npy', 'rb') as fin:
            for s, r, o in np.load(fin):
                fact2pid[(s, o)].add(r)
        print('#facts {}'.format(len(fact2pid)))
        #distant_supervision(fact2pid, lang=args.lang, dist_thres=20, count_thres=5, outdir=args.out)
        with open(args.out, 'w') as fout:
            for tokens, pid in distant_supervision_sentences(fact2pid, lang=args.lang, dist_thres=10):
                if len(tokens) > 128:
                    continue
                fout.write(pid + '\t' + ' '.join(tokens) + '\n')

    elif args.task == 'cw':
        entity_lang_path = 'data/mTRExf_unicode_escape.txt'
        fact_path = 'data/mTRExf/sub'

        # load entities
        entity2lang: Dict[str, Dict[str, str]] = load_entity_lang(entity_lang_path)

        # load facts we want to identify
        all_facts: List[Tuple[str, str, str]] = []
        entities: Set[str] = set()
        for root, dirs, files in os.walk(fact_path):
            for file in files:
                with open(os.path.join(root, file), 'r') as fin:
                    pid = file.split('.', 1)[0]
                    for l in fin:
                        f = json.loads(l)
                        s, o = f['sub_uri'], f['obj_uri']
                        if args.lang not in entity2lang[s] or 'en' not in entity2lang[s] or \
                                args.lang not in entity2lang[o] or 'en' not in entity2lang[o]:
                            continue
                        f = (s, pid, o)
                        all_facts.append(f)
                        entities.add(s)
                        entities.add(o)

        facts = all_facts
        oov_facts = []
        if args.down_sample is not None:
            facts = [all_facts[i] for i in
                     np.random.choice(len(all_facts), int(len(all_facts) * args.down_sample), replace=False)]
            entities = set(e for f in facts for e in [f[0], f[2]])
            oov_facts = [f for f in all_facts if f[0] not in entities and f[2] not in entities]
        print('#facts {}, #entities {}, #oov facts {} #all facts {}'.format(
            len(facts), len(entities), len(oov_facts), len(all_facts)))

        if not os.path.exists(args.out):
            os.makedirs(args.out, exist_ok=True)

        with open(os.path.join(args.out, 'facts.txt'), 'w') as fout, \
                open(os.path.join(args.out, 'all_facts.txt'), 'w') as fout2:
            for s, p, o in facts:
                fout.write('{}\t{}\t{}\n'.format(s, p, o))
            for s, p, o in all_facts:
                fout2.write('{}\t{}\t{}\n'.format(s, p, o))

        for lang_from, lang_to in [(args.lang, 'en'), ('en', args.lang)]:
            # code-switching for two directions
            id2alias: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(lambda: 0))
            with open(os.path.join(args.out, '{}_{}.txt'.format(lang_from, lang_to)), 'w') as fout:
                for sent, mentions in locate_entity(
                        entities, glob.glob('data/sling/{}/*.rec'.format(lang_from))):
                    if len(sent) >= 128:  # TODO: focus on short sentence
                        continue

                    pos2mentind: Dict[int, int] = {}
                    for i, (entity, start, end) in enumerate(mentions):
                        overlap: bool = False
                        for j in range(start, end):
                            if j in pos2mentind:
                                overlap = True
                                break
                        if overlap:
                            continue
                        for j in range(start, end):
                            pos2mentind[j] = i

                    entity_id: List[str] = []
                    surface_from: List[str] = []
                    surface_to: List[str] = []
                    tokens: List[str] = []

                    for i in range(len(sent)):
                        if i not in pos2mentind:
                            tokens.append(sent[i])
                        elif i in pos2mentind and (i - 1) in pos2mentind and pos2mentind[i] == pos2mentind[i - 1]:
                            continue
                        else:
                            entity, start, end = mentions[pos2mentind[i]]
                            tokens.append('[[' + entity + ']]')
                            entity_id.append(entity)
                            raw_word = ' '.join(sent[start:end])
                            translation = entity2lang[entity][lang_to]  # too simplistic
                            surface_from.append(raw_word)
                            surface_to.append(translation)
                            id2alias[entity][raw_word] += 1

                    fout.write('{}\t{}\n'.format(
                        ' '.join(tokens),
                        '\t'.join(['{} ||| {} ||| {}'.format(e, f, t)
                                   for e, f, t in zip(entity_id, surface_from, surface_to)])))

            with open(os.path.join(args.out, '{}_alias.txt'.format(lang_from)), 'w') as fout:
                json.dump(id2alias, fout, indent=2)

    elif args.task == 'cw_split':
        source_lang, target_lang = args.lang, 'en'

        facts: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
        with open(os.path.join(args.inp, 'facts.txt'), 'r') as fin:
            for l in fin:
                s, p, o = l.strip().split('\t')
                facts[(s, o)].add(p)

        numentity2count1: Dict[int, int] = defaultdict(lambda: 0)
        numentity2count2: Dict[int, int] = defaultdict(lambda: 0)
        entity2count1: Dict[str, int] = defaultdict(lambda: 0)
        entity2count2: Dict[str, int] = defaultdict(lambda: 0)
        fact2sent1: Dict[Tuple[str, str], Set[int]] = defaultdict(set)
        fact2sent2: Dict[Tuple[str, str], Set[int]] = defaultdict(set)

        prev_total = None
        for ind, (source, target) in enumerate([(source_lang, target_lang), (target_lang, source_lang)]):
            numentity2count = eval('numentity2count{}'.format(ind + 1))
            fact2sent = eval('fact2sent{}'.format(ind + 1))
            entity2count = eval('entity2count{}'.format(ind + 1))
            dataset = CodeSwitchDataset(os.path.join(args.inp, '{}_{}.txt'.format(source, target)))
            total = sum(1 for _ in dataset.iter())
            ds = args.down_sample
            if ds and args.balance_lang and prev_total is not None:
                ds = args.down_sample / (total / prev_total)
            ds_file = None
            if ds and args.out:
                if args.balance_lang:
                    ds_file = open(os.path.join(args.out, '{}_{}.eq_ds.txt'.format(source, target)), 'w')
                else:
                    ds_file = open(os.path.join(args.out, '{}_{}.ds.txt'.format(source, target)), 'w')
            prev_total = total
            remain = 0
            for sent_ind, (tokens, mentions, raw_line) in enumerate(dataset.iter()):
                if ds is not None:
                    cond = np.all([[i == j or
                                    (mentions[i][0], mentions[j][0]) not in facts or
                                    len(fact2sent[(mentions[i][0], mentions[j][0])]) > 0
                                    for j in range(len(mentions))] for i in range(len(mentions))])
                    if cond and random.random() > ds:
                        continue
                remain += 1
                if ds_file is not None:
                    ds_file.write(raw_line)
                numentity2count[len(mentions)] += 1
                for i in range(len(mentions)):
                    entity2count[mentions[i][0]] += 1
                    for j in range(len(mentions)):
                        if i == j:
                            continue
                        e1, e2 = mentions[i][0], mentions[j][0]
                        if (e1, e2) in facts:
                            fact2sent[(e1, e2)].add(sent_ind)
            print('{} down sample with ratio {} from {} to {}'.format(source, ds, total, remain))

        print('#facts {}, #facts {} {}, #facts {} {}'.format(
            len(facts), source_lang, len(fact2sent1), target_lang, len(fact2sent2)))
        print('#entites per sent in {} {}'.format(
            source_lang, sorted(numentity2count1.items(), key=lambda x: -x[1])[:5]))
        print('most freq entities in {} {}'.format(
            source_lang, sorted(entity2count1.items(), key=lambda x: -x[1])[:5]))
        print('most freq facts in {} {}'.format(
            source_lang, sorted(map(lambda x: (x[0], len(x[1])), fact2sent1.items()), key=lambda x: -x[1])[:3]))
        print('#entites per sent in {} {}'.format(
            target_lang, sorted(numentity2count2.items(), key=lambda x: -x[1])[:5]))
        print('most freq entities in {} {}'.format(
            target_lang, sorted(entity2count2.items(), key=lambda x: -x[1])[:5]))
        print('most freq facts in {} {}'.format(
            target_lang, sorted(map(lambda x: (x[0], len(x[1])), fact2sent2.items()), key=lambda x: -x[1])[:3]))

        join_fact = set(fact2sent1.keys()) & set(fact2sent2.keys())
        fact_only1 = set(fact2sent1.keys()) - join_fact
        fact_only2 = set(fact2sent2.keys()) - join_fact
        fact_none = set(facts.keys()) - join_fact - fact_only1 - fact_only2

        print('join {}, {} {}, {} {}, none {}'.format(
            len(join_fact), source_lang, len(fact_only1), target_lang, len(fact_only2), len(fact_none)))

        if args.out:
            with open(os.path.join(args.out, 'split.json'), 'w') as fout:
                json.dump({
                    'join': list(join_fact),
                    source_lang: list(fact_only1),
                    target_lang: list(fact_only2),
                    'none': list(fact_none),
                }, fout)

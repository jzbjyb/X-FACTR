from typing import Tuple, Iterable, List, Dict, Any, Set
import sys
import os
import json
import argparse
from termcolor import colored
from collections import defaultdict
import numpy as np
import glob
from tqdm import tqdm
import random
from random import shuffle
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
    parser.add_argument('--task', type=str, choices=['inspect', 'filter', 'ds', 'cw_gen_data',
                                                     'partition_entity', 'partition_fact',
                                                     'cw_gen_data_control', 'cw_gen_data_control2'])
    parser.add_argument('--lang', type=str, help='language to probe', choices=['el', 'fr', 'nl'], default='en')
    parser.add_argument('--dir', type=str, help='data dir')
    parser.add_argument('--inp', type=str, default=None)
    parser.add_argument('--out', type=str, help='output')
    args = parser.parse_args()

    if args.task == 'inspect':
        s = SlingExtractorForQualifier()
        s.load_corpus(corpus_file=args.inp)
        for wid, title, text in s.find_all_mentions():
            print('=' * 30)
            print(wid, title)
            print(text[:10000])
            input('press any key to continue ...')

    elif args.task == 'filter':
        max_dist = 20
        entity2lang = load_entity_lang(ENTITY_LANG_PATH)
        facts = set()
        for root, dirs, files in os.walk(ENTITY_PATH.rsplit('/', 1)[0]):
            for file in files:
                with open(os.path.join(root, file), 'r') as fin:
                    for l in fin:
                        l = json.loads(l)
                        sub_exist = args.lang in entity2lang[l['sub_uri']]
                        obj_exist = args.lang in entity2lang[l['obj_uri']]
                        exist = sub_exist and obj_exist
                        if not exist:
                            continue
                        facts.add((l['sub_uri'], l['obj_uri']))
        found_lang = locate_fact(facts, glob.glob('data/sling/{}/*.rec'.format(args.lang)), max_dist)
        found_en = locate_fact(facts, glob.glob('data/sling/{}/*.rec'.format('en')), max_dist)
        result = {
            args.lang: list(found_lang - found_en),
            'en': list(found_en - found_lang),
            'join': list(found_lang & found_en),
            'none': list(facts - found_lang - found_en)
        }
        print('#facts {}, join {}, {} {}, {} {}, none {}'.format(
            len(facts), len(result['join']),
            args.lang, len(result[args.lang]),
            'en', len(result['en']),
            len(result['none'])
        ))
        with open(args.out, 'w') as fout:
            json.dump(result, fout, indent=2)

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

    elif args.task == 'cw_gen_data':
        # load entities we want to identify
        with open('data/lang/{}_en_fact.json'.format(args.lang), 'r') as fin:
            entities: Set[str] = set(e for f in json.load(fin)['join'] for e in f)
            print('#entities {}'.format(len(entities)))
        # load entities' translations
        entity2lang: Dict[str, Dict[str, str]] = load_entity_lang(ENTITY_LANG_PATH)

        if not os.path.exists(args.out):
            os.makedirs(args.out, exist_ok=True)
        for lang_from, lang_to in [(args.lang, 'en'), ('en', args.lang)]:
            # code-switching for two directions
            with open(os.path.join(args.out, '{}_{}.txt'.format(lang_from, lang_to)), 'w') as fout:
                for sent, mentions in locate_entity(
                        entities, glob.glob('data/sling/{}/*.rec'.format(lang_from))):
                    if len(sent) >= 128:  # TODO: focus on short sentence
                        continue

                    mentions = [m for m in mentions if m[0] in entity2lang and lang_to in entity2lang[m[0]]]
                    pos2mentind: Dict[int, int] = {}
                    for i, (entity, start, end) in enumerate(mentions):
                        for j in range(start, end):
                            if j in pos2mentind:
                                break  # avoid overlapping mentions
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
                            surface_from.append(' '.join(sent[start:end]))
                            surface_to.append(entity2lang[entity][lang_to])

                    fout.write('{}\t{}\n'.format(
                        ' '.join(tokens),
                        '\t'.join(['{} ||| {} ||| {}'.format(e, f, t)
                                   for e, f, t in zip(entity_id, surface_from, surface_to)])))

    elif args.task == 'partition_entity':
        # load entities we want to identify
        with open('data/lang/{}_en_fact.json'.format(args.lang), 'r') as fin:
            data = json.load(fin)
        entities: Set[str] = set(e for f in data['join'] for e in f)
        print('#entities {}'.format(len(entities)))
        fact_overlap: List[Tuple[str, str]] = []
        one_overlap: List[Tuple[str, str]] = []
        no_overlap: List[Tuple[str, str]] = []
        for part in ['en', args.lang, 'none']:
            for sub, obj in data[part]:
                if sub in entities and obj in entities:
                    fact_overlap.append((sub, obj))
                elif sub in entities or obj in entities:
                    one_overlap.append((sub, obj))
                else:
                    no_overlap.append((sub, obj))
        print('#fact {}, #fact ol {}, #one ov {}, #no ol {}'.format(
            len(data['join']), len(fact_overlap), len(one_overlap), len(no_overlap)))
        with open(args.out, 'w') as fout:
            json.dump({'fact': data['join'],
                       'fact_overlap': fact_overlap,
                       'one_overlap': one_overlap,
                       'no_overlap': no_overlap}, fout, indent=True)

    elif args.task == 'partition_fact':
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

    elif args.task == 'cw_gen_data_control':
        entity_lang_path = 'data/mTREx_unicode_escape.txt'
        fact_path = 'data/mTREx'

        # load entities' translations
        entity2lang: Dict[str, Dict[str, str]] = load_entity_lang(entity_lang_path)

        # load facts we want to identify
        facts: List[Tuple[str, str, str]] = []
        entities: Set[str] = set()
        for root, dirs, files in os.walk(fact_path):
            for file in files:
                with open(os.path.join(root, file), 'r') as fin:
                    pid = file.split('.', 1)[0]
                    for l in fin:
                        f = json.loads(l)
                        s, o = f['sub_uri'], f['obj_uri']
                        if args.lang not in entity2lang[s] or \
                                'en' not in entity2lang[s] or \
                                args.lang not in entity2lang[o] or \
                                'en' not in entity2lang[o]:
                            continue
                        f = (s, pid, o)
                        facts.append(f)
                        entities.add(s)
                        entities.add(o)

        print('#facts {}, #entities {}'.format(len(facts), len(entities)))

        if not os.path.exists(args.out):
            os.makedirs(args.out, exist_ok=True)

        with open(os.path.join(args.out, 'facts.txt'), 'w') as fout:
            for s, p, o in facts:
                fout.write('{}\t{}\t{}\n'.format(s, p, o))

        for lang_from, lang_to in [(args.lang, 'en'), ('en', args.lang)]:
            # code-switching for two directions
            with open(os.path.join(args.out, '{}_{}.txt'.format(lang_from, lang_to)), 'w') as fout:
                # TODO: restrict to articles of these entites
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
                            surface_from.append(' '.join(sent[start:end]))
                            surface_to.append(entity2lang[entity][lang_to])

                    fout.write('{}\t{}\n'.format(
                        ' '.join(tokens),
                        '\t'.join(['{} ||| {} ||| {}'.format(e, f, t)
                                   for e, f, t in zip(entity_id, surface_from, surface_to)])))

    elif args.task == 'cw_gen_data_control2':
        source_lang, target_lang = 'en', args.lang

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

        for ind, (source, target) in enumerate([(source_lang, target_lang), (target_lang, source_lang)]):
            numentity2count = eval('numentity2count{}'.format(ind + 1))
            fact2sent = eval('fact2sent{}'.format(ind + 1))
            entity2count = eval('entity2count{}'.format(ind + 1))
            dataset = CodeSwitchDataset(os.path.join(args.inp, '{}_{}.train.txt'.format(source, target)))
            for sent_ind, (tokens, mentions) in enumerate(dataset.iter()):
                numentity2count[len(mentions)] += 1
                for i in range(len(mentions)):
                    entity2count[mentions[i][0]] += 1
                    for j in range(i + 1, len(mentions)):
                        e1, e2 = mentions[i][0], mentions[j][0]
                        if (e1, e2) in facts:
                            fact2sent[(e1, e2)].add(sent_ind)
                        if (e2, e1) in facts:
                            fact2sent[(e2, e1)].add(sent_ind)

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

        with open(args.out, 'w') as fout:
            json.dump({
                'join': list(join_fact),
                source_lang: list(fact_only1),
                target_lang: list(fact_only2),
                'none': list(fact_none),
            }, fout)

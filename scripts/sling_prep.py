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
import sling
from distantly_supervise import SlingExtractor
from probe import load_entity_lang, ENTITY_LANG_PATH, ENTITY_PATH


# inspect a record file
# bazel-bin/tools/codex --frames local/data/e.bak/ner/fi/documents-00000-of-00010.rec | less

# sling python API
# https://github.com/google/sling/blob/master/doc/guide/pyapi.md


commons = sling.Store()
DOCSCHEMA = sling.DocumentSchema(commons)
commons.freeze()


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SLING-related preprocessing')
    parser.add_argument('--task', type=str, choices=['inspect', 'filter'])
    parser.add_argument('--lang', type=str, help='language to probe', choices=['zh-cn', 'el', 'fr'], default='en')
    parser.add_argument('--dir', type=str, help='data dir')
    parser.add_argument('--out', type=str, help='output')
    args = parser.parse_args()

    if args.task == 'inspcet':
        s = SlingExtractorForQualifier()
        s.load_corpus(corpus_file=sys.argv[1])
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

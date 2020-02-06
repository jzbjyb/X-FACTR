from typing import Tuple, Iterable, List, Dict, Any
import sys
from termcolor import colored
import sling
from distantly_supervise import SlingExtractor


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


if __name__ == '__main__':
    s = SlingExtractorForQualifier()
    s.load_corpus(corpus_file=sys.argv[1])
    for wid, title, text in s.find_all_mentions():
        print('=' * 30)
        print(wid, title)
        print(text[:10000])
        input('press any key to continue ...')

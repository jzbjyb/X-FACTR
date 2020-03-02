from typing import Set, List, Tuple, Iterable, Dict
import os
from collections import defaultdict
import argparse
import json
import uuid
from tqdm import tqdm


def filter_by_relations(filename: str, relations: Set[str]) -> Iterable[Tuple]:
    with open(filename, 'r') as fin:
        for doc in tqdm(json.load(fin)):
            for triple in doc['triples']:
                fact: List[Tuple[str, str]] = []
                for field in ['subject', 'predicate', 'object']:
                    uri = triple[field]['uri'].rsplit('/', 1)[1]
                    label = triple[field]['surfaceform']
                    fact.append((uri, label))
                if not fact[1][0].startswith('P') or \
                        not fact[0][0].startswith('Q') or \
                        not fact[2][0].startswith('Q'):
                    continue
                if fact[1][0] not in relations:
                    continue
                yield fact


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='build multi-entity TREx')
    #parser.add_argument('--task', type=str, choices=['build'])
    parser.add_argument('--inp', type=str, help='input file')
    parser.add_argument('--out', type=str, help='output file')
    args = parser.parse_args()

    with open('../LAMA/data/relations.jsonl', 'r') as fin:
        relations: Set[str] = set([json.loads(l)['relation'] for l in fin])

    os.makedirs(args.out, exist_ok=True)
    pid2files: Dict[str, object] = {}
    pid2facts: Dict[str, Set[Tuple[str, str]]] = defaultdict(set)
    try:
        for root, dirs, files in os.walk(args.inp):
            print('#files {}'.format(len(files)))
            for file in files:
                if not file.endswith('.json'):
                    continue
                for (su, sl), (pu, pl), (ou, ol) in filter_by_relations(os.path.join(root, file), relations):
                    if (su, ou) in pid2facts[pu]:
                        continue
                    pid2facts[pu].add((su, ou))
                    if pu not in pid2files:
                        pid2files[pu] = open(os.path.join(args.out, pu + '.jsonl'), 'w')
                    pid2files[pu].write(json.dumps({
                        'uuid': str(uuid.uuid1()),
                        'predicate_id': pu,
                        'sub_uri': su,
                        'sub_label': sl,
                        'obj_uri': ou,
                        'obj_label': ol
                    }) + '\n')
    finally:
        for pid, f in pid2files.items():
            f.close()

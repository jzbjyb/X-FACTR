from typing import List, Dict, Set
from collections import defaultdict
import json
import argparse
from entity_lang import get_result, get_qid_from_uri, handle_redirect, TRExDataset


GET_MULTI_OBJECTS = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?item ?value WHERE
{
VALUES ?item { %s }
?item wdt:%s ?value
SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}
"""


@handle_redirect(debug=True, disable=False)
def get_multi_objects(uris: List[str], pid: str) -> Dict[str, Set[str]]:
    results = get_result(GET_MULTI_OBJECTS % (' '.join(map(lambda x: 'wd:' + x, uris)), pid))
    sub2objs: Dict[str, Set[str]] = defaultdict(set)
    for result in results['results']['bindings']:
        sub = get_qid_from_uri(result['item']['value'])
        try:
            obj = get_qid_from_uri(result['value']['value'])
            sub2objs[sub].add(obj)
        except:
            print('subject "{}" with relation "{}" has "{}"'.format(sub, pid, result['value']['value']))
    return sub2objs


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='retrieve multiple objects for N-M relations')
    parser.add_argument('--inp', type=str, help='input file')
    parser.add_argument('--out', type=str, help='output file')
    args = parser.parse_args()

    rel_file, fact_dir = args.inp.split(':')

    batch_size = 300
    data = TRExDataset(fact_dir)
    pid2sub2objs: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    num_subrel = num_objs = 0
    with open(rel_file, 'r') as rel_fin:
        for l in rel_fin:
            l = json.loads(l)
            if l['type'] != 'N-M':
                continue
            pid: str = l['relation']
            print('=== relation {} ==='.format(pid))
            subs: Set[str] = set()
            for fact in data.iter(pid + '.jsonl'):
                subs.add(fact['sub_uri'])
            subs: List[str] = list(subs)
            for b in range(0, len(subs), batch_size):
                pid2sub2objs[pid].update(get_multi_objects(subs[b:b + batch_size], pid))
            for fact in data.iter(pid + '.jsonl'):  # add original object if not exist
                if fact['obj_uri'] not in pid2sub2objs[pid][fact['sub_uri']]:
                    pid2sub2objs[pid][fact['sub_uri']].add(fact['obj_uri'])
            num_subrel += len(pid2sub2objs[pid])
            num_objs += sum(len(v) for k, v in pid2sub2objs[pid].items())

    print('#subject-relation pairs {}, #objects {}'.format(num_subrel, num_objs))

    with open(args.out, 'w') as fout:
        for pid, sub2objs in sorted(pid2sub2objs.items(), key=lambda x: int(x[0][1:])):
            for sub, objs in sorted(sub2objs.items(), key=lambda x: int(x[0][1:])):
                fout.write('{}\t{}\t{}\n'.format(sub, pid, ' '.join(objs)))

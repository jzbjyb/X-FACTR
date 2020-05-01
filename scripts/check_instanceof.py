from typing import List, Dict, Set, Tuple
import argparse
from collections import defaultdict
from tqdm import tqdm
from entity_lang import get_result, get_qid_from_uri, handle_redirect, load_qid_from_lang_file


GET_INSTANCEOF = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?item ?value ?valueLabel
{
VALUES ?item { %s }
?item p:P31 ?statement.
?statement ps:P31 ?value.
SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}
"""

CHECK_IS_CATEGORY = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?value ?valueLabel
{
VALUES ?value { %s }
FILTER EXISTS {?item wdt:P31 ?value}
SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}
"""


@handle_redirect(debug=False, disable=False)
def get_instanceof(uris: List[str]) -> Dict[str, Set[Tuple[str, str]]]:
    results = get_result(GET_INSTANCEOF % ' '.join(map(lambda x: 'wd:' + x, uris)))
    instanceofs = defaultdict(set)
    for result in results['results']['bindings']:
        uri = get_qid_from_uri(result['item']['value'])
        inst = get_qid_from_uri(result['value']['value'])
        inst_label = result['valueLabel']['value']
        instanceofs[uri].add((inst, inst_label))
    return instanceofs


def load_entity_instance(filename: str):
    entity2instance: Dict[str, str] = {}
    with open(filename, 'r') as fin:
        for l in fin:
            l = l.strip().split('\t')
            entity2instance[l[0]] = ','.join(l[1:])
    return entity2instance


@handle_redirect(debug=False, disable=False)
def check_is_category(uris: List[str]) -> Dict[str, int]:
    uris = set(uris)
    results = get_result(CHECK_IS_CATEGORY % ' '.join(map(lambda x: 'wd:' + x, uris)))
    is_category: Dict[str, int] = dict((uri, 0) for uri in uris)
    for result in results['results']['bindings']:
        uri = get_qid_from_uri(result['value']['value'])
        assert uri in uris, 'should introduce new entities'
        is_category[uri] = 1
    return is_category


def load_entity_is_cate(filename: str) -> Dict[str, bool]:
    is_cate: Dict[str, bool] = {}
    with open(filename, 'r') as fin:
        for l in fin:
            l = l.strip().split('\t')
            is_cate[l[0]] = bool(int(l[1]))
    return is_cate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='retrieve the instance-of relation of entities')
    parser.add_argument('--task', type=str, help='task')
    parser.add_argument('--inp', type=str, help='input file')
    parser.add_argument('--out', type=str, help='output file')
    args = parser.parse_args()

    if args.task == 'instanceof':
        batch_size = 300
        qids = load_qid_from_lang_file(args.inp)
        instanceofs = {}
        for b in tqdm(range(0, len(qids), batch_size)):
            instanceofs.update(get_instanceof(qids[b:b + batch_size]))

        print('#entities {}, #results {}'.format(len(qids), len(instanceofs)))

        for qid in qids:
            if qid in instanceofs:
                continue
            instanceofs[qid] = set()

        with open(args.out, 'w') as fout:
            for k, v in sorted(instanceofs.items(), key=lambda x: int(x[0][1:])):
                fout.write('{}\t{}\n'.format(k, '\t'.join(map(lambda x: x[0] + ',' + x[1], v))))

    elif args.task == 'is_category':
        batch_size = 100
        qids = load_qid_from_lang_file(args.inp)

        cate: Dict[str, int] = {}
        for b in tqdm(range(0, len(qids), batch_size)):
            cate.update(check_is_category(qids[b:b + batch_size]))

        with open(args.out, 'w') as fout:
            for qid in qids:
                fout.write('{}\t{}\n'.format(qid, cate[qid]))

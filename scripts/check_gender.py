from typing import List, Dict
import argparse
from tqdm import tqdm
from entity_lang import get_result, get_qid_from_uri, handle_redirect, load_qid_from_lang_file


GET_GENDER = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?item ?valueLabel WHERE
{
VALUES ?item { %s }
?item wdt:P21 ?value
SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}
"""


class Gender:
    NONE = 'none'
    MALE = 'male'
    FEMALE = 'female'


    @staticmethod
    def parse(text: str):
        if text.lower() == Gender.MALE:
            return Gender.MALE
        if text.lower() == Gender.FEMALE:
            return Gender.FEMALE
        return Gender.NONE


@handle_redirect(debug=False, disable=False)
def get_gender(uris: List[str]) -> Dict[str, str]:
    results = get_result(GET_GENDER % ' '.join(map(lambda x: 'wd:' + x, uris)))
    genders = {}
    for result in results['results']['bindings']:
        uri = get_qid_from_uri(result['item']['value'])
        gender = Gender.parse(result['valueLabel']['value'])
        genders[uri] = gender
    return genders


def load_entity_gender(filename: str) -> Dict[str, Gender]:
    result: Dict[str, Gender] = {}
    with open(filename, 'r') as fin:
        for l in fin:
            uri, gender = l.strip().split('\t')
            result[uri] = Gender.parse(gender)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='retrieve the gender of entities')
    parser.add_argument('--inp', type=str, help='input file')
    parser.add_argument('--out', type=str, help='output file')
    args = parser.parse_args()

    batch_size = 300
    qids = load_qid_from_lang_file(args.inp)
    genders = {}
    for b in tqdm(range(0, len(qids), batch_size)):
        genders.update(get_gender(qids[b:b + batch_size]))

    print('#entities {}, #results {}'.format(len(qids), len(genders)))

    for qid in qids:
        if qid in genders:
            continue
        genders[qid] = Gender.NONE

    with open(args.out, 'w') as fout:
        for k, v in sorted(genders.items(), key=lambda x: int(x[0][1:])):
            fout.write('{}\t{}\n'.format(k, v))

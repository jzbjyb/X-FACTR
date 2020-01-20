from typing import List, Dict
import argparse
from tqdm import tqdm
from SPARQLWrapper import SPARQLWrapper, JSON
from entity_lang import get_result

GET_GENDER = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?item ?valueLabel WHERE
{
VALUES ?item { %s }
?item wdt:P21 ?value
SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}"""


class Gender:
    NONE = 'none'
    MALE = 'male'
    FEMALE = 'female'


def get_gender(uris: List[str]) -> Dict[str: str]:
    results = get_result(GET_GENDER % ' '.join(map(lambda x: 'wd:' + x, uris)))
    genders = {}
    for result in results['results']['bindings']:
        uri = result['item']['value'].rsplit('/', 1)[1]
        gender = Gender.MALE if result['valueLabel']['value'] == 'male' else Gender.FEMALE
        genders[uri] = gender
    for uri in uris:
        if uri not in genders:
            genders[uri] = Gender.NONE
    return genders


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='retrieve the gender of entities')
    parser.add_argument('--inp', type=str, help='input file')
    parser.add_argument('--out', type=str, help='output file')
    args = parser.parse_args()

    '''
    # usage example, which returns an dict {'Q31': 'none', 'Q76': 'male', 'Q36153': 'female'}
    print(get_gender(['Q31', 'Q76', 'Q36153']))
    '''

    num_entity_per_query = 1000
    uris = []
    genders = {}
    with open(args.inp, 'r') as fin:
        for l in tqdm(fin):
            uri = l.strip().split('\t', 1)[0]
            uris.append(uri)
            if len(uris) >= num_entity_per_query:
                genders.update(get_gender(uris))
                uris = []
        if len(uris) > 0:
            genders.update(get_gender(uris))
            uris = []

    print(len(genders))

    with open(args.out, 'w') as fout:
        for k, v in genders.items():
            fout.write('{}\t{}\n'.format(k, v))

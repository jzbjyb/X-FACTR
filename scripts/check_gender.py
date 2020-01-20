import argparse
from SPARQLWrapper import SPARQLWrapper, JSON
from entity_lang import get_result

GET_GENDER = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?valueLabel WHERE
{
wd:%s wdt:P21 ?value
SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
}"""


def get_gender(uri: str):
    print(GET_GENDER % uri)
    return get_result(GET_GENDER % uri)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='retrieve the gender of entities')
    parser.add_argument('--inp', type=str, help='input file')
    args = parser.parse_args()

    print(get_gender('Q76'))

import os
import sys
from tqdm import tqdm
from collections import defaultdict
from SPARQLWrapper import SPARQLWrapper, JSON


GET_REDIRECT = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?label {wd:%s owl:sameAs ?label}"""

GET_LANG = """PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?label (lang(?label) as ?label_lang) {wd:%s rdfs:label ?label}"""

prefix = '<http://www.wikidata.org/entity/'


def get_lang(uri):
	return get_result(GET_LANG % uri)


def get_redirect(uri):
	return get_result(GET_REDIRECT % uri)


def get_result(query, timeout=None):
	sparql = SPARQLWrapper('https://query.wikidata.org/sparql')
	if timeout:
		sparql.setTimeout(timeout)
	sparql.setQuery(query)
	sparql.setReturnFormat(JSON)
	return sparql.query().convert()


def sup(uris):
	results = []
	for uri in uris:
		real_uri = get_redirect(uri)['results']['bindings'][0]['label']['value'].rsplit('/', 1)[1]
		lang = [(l['label']['value'], l['label_lang']['value']) for l in get_lang(real_uri)['results']['bindings']]
		results.append(uri + '\t' + '\t'.join(map(lambda x: '"{}"@{}'.format(*x), lang)))
	return results


if __name__ == '__main__':
	task = sys.argv[1]

	if task == 'filter':
		inp_rdf, inp_uri, out = sys.argv[2:]
		with open(inp_uri, 'r') as fin:
			uris = set(fin.read().strip().split('\n'))

		print('#uri {}'.format(len(uris)))
		seen_uris = set()

		prev_eid = None
		with open(inp_rdf, 'r') as fin, open(out, 'w') as fout:
			for l in tqdm(fin):
				l = l.strip()
				if not l.startswith(prefix):
					continue
				try:
					eid, _, label = l.split('> ', 2)
				except:
					print()
					print(l)
					print()
					raise
				eid = eid[len(prefix):].rstrip('>')
				if eid not in uris:
					continue
				seen_uris.add(eid)
				label = label.rstrip().rstrip(' .')
				if eid != prev_eid:
					if prev_eid is not None:
						fout.write('\n')
					fout.write(eid)
				fout.write('\t')
				fout.write(label)
				prev_eid = eid
			fout.write('\n')
			print('#miss {}'.format(len(uris - seen_uris)))
			fout.write('\n'.join(sup(uris - seen_uris)))

	elif task == 'redirect':
		uris = ['Q3292203', 'Q7790052', 'Q367143', 'Q7579156', 'Q4881051',
				'Q504802', 'Q1539426', 'Q1323442', 'Q17518425', 'Q3630470',
				'Q5422636', 'Q6794459', 'Q16255398', 'Q3761669', 'Q3812948',
				'Q18913178', 'Q5380327', 'Q7427317', 'Q6203279', 'Q17004641',
				'Q64347', 'Q1379239', 'Q28775', 'Q412609', 'Q7259490']
		print('\n'.join(sup(uris)))

	elif task == 'ana':
		inp_label, = sys.argv[2:]
		lang2count = defaultdict(lambda: 0)
		count = 0
		with open(inp_label, 'r') as fin:
			for l in fin:
				count += 1
				l = l.strip().split('\t')
				eid = l[0]
				lang = [la.rsplit('@', 1)[1] for la in l[1:]]
				for la in lang:
					lang2count[la] += 1
		for k in lang2count:
			lang2count[k] /= count
		print(len(lang2count))
		for k, v in sorted(lang2count.items(), key=lambda x: -x[1]):
			print('{}\t{:.3f}'.format(k, v))

	elif task == 'convert':
		inp_label, out = sys.argv[2:]
		with open(inp_label, 'r') as fin, open(out, 'w') as fout:
			for l in fin:
				fout.write(l.encode('utf8').decode('unicode_escape'))

	elif task == 'miss':
		inp_uri, inp_label = sys.argv[2:]
		with open(inp_uri, 'r') as fin:
			uri1 = set(fin.read().strip().split('\n'))
		with open(inp_label, 'r') as fin:
			uri2 = set([l.strip().split('\t', 1)[0] for l in fin])
		print(len(uri1 - uri2))
		print(uri1 - uri2)

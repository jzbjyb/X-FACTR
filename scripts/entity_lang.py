from typing import Dict, Iterable, Set, List, Union, Tuple
import os
import json
from tqdm import tqdm
from collections import defaultdict
import functools
import argparse
from SPARQLWrapper import SPARQLWrapper, JSON
import socket
import time
import urllib

# https://stackoverflow.com/questions/44374215/how-do-i-specify-url-resolution-in-pythons-requests-library-in-a-similar-fashio
# capture a dict of hostname and their IPs to override with
dns_cache = {}
activate_dns_cache: bool = False
def override_dns(domain, ip):
	dns_cache[domain] = ip
prv_getaddrinfo = socket.getaddrinfo
# override default socket.getaddrinfo() and pass ip instead of host if override is detected
def new_getaddrinfo(*args):
	if activate_dns_cache and args[0] in dns_cache:
		print('Forcing FQDN: {} to IP: {}'.format(args[0], dns_cache[args[0]]))
		return prv_getaddrinfo(dns_cache[args[0]], *args[1:])
	else:
		return prv_getaddrinfo(*args)
socket.getaddrinfo = new_getaddrinfo
override_dns('query.wikidata.org', '208.80.154.224')


GET_REDIRECT = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?item ?label 
{
VALUES ?item { %s } 
?item owl:sameAs ?label
}
"""

GET_LANG = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?label (lang(?label) as ?label_lang) {wd:%s rdfs:label ?label}
"""

GET_LANG_MULTI = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?item ?label (lang(?label) as ?label_lang)
{
VALUES ?item { %s } 
?item rdfs:label ?label
}
"""

GET_ALIAS_IN_LANG = """
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
SELECT ?item ?itemLabel ?alt WHERE
{
VALUES ?item { %s }
OPTIONAL {
?item skos:altLabel ?alt .
FILTER (lang(?alt)='%s')
}
SERVICE wikibase:label { bd:serviceParam wikibase:language "%s" }
}
"""

prefix = '<http://www.wikidata.org/entity/'


class TRExDataset(object):
	def __init__(self, directory: str):
		self.directory = directory


	def iter(self, file: str=None) -> Iterable[Dict]:
		if file is None:
			for root, dirs, files in os.walk(self.directory):
				for file in files:
					if not file.endswith('.jsonl'):
						continue
					with open(os.path.join(root, file), 'r') as fin:
						for l in fin:
							yield json.loads(l)
		else:
			filepath = os.path.join(self.directory, file)
			if not os.path.exists(filepath):
				return
			with open(filepath, 'r') as fin:
				for l in fin:
					yield json.loads(l)


class Alias(object):
	def __init__(self, directory: str):
		self.directory = directory
		self.alias: Dict[str, Dict[str, List[str]]] = {}
		self.no_alias_count: int = 0


	@staticmethod
	def load_alias_from_file(filename: str) -> Dict[str, List[str]]:
		id2alias: Dict[str, List[str]] = defaultdict(list)
		with open(filename, 'r') as fin:
			for l in fin:
				l = l.strip().split('\t')
				id2alias[l[0]].extend(l[1:])
		return id2alias


	def load_alias(self, lang: str):
		if lang in self.alias:
			return
		self.alias[lang] = self.load_alias_from_file(os.path.join(self.directory, lang + '.txt'))


	def get_alias(self, uri: str, langs: Union[str, List[str]]) -> List[str]:
		if type(langs) is not list:
			langs = [langs]
		result: List[str] = []
		for lang in langs:
			self.load_alias(lang)
			if uri in self.alias[lang]:
				result.extend(self.alias[lang][uri])
			else:
				self.no_alias_count += 1
		return result


class MultiRel(object):
	def __init__(self, filename: str):
		self.filename = filename
		self.subpid2objs = self.load_multi_objects(filename)


	@staticmethod
	def load_multi_objects(filename: str) -> Dict[Tuple[str, str], List[str]]:
		subpid2objs: Dict[Tuple[str, str], List[str]] = {}
		with open(filename, 'r') as fin:
			for l in fin:
				sub, rel, objs = l.strip().split('\t')
				subpid2objs[(sub, rel)] = objs.split(' ')
		return subpid2objs


	def get_objects(self, sub: str, pid: str) -> List[str]:
		if (sub, pid) in self.subpid2objs:
			return self.subpid2objs[(sub, pid)]
		return []


def get_lang(uri):
	return get_result(GET_LANG % uri)


def handle_redirect(debug: bool=False, disable: bool=False, retry_with_host: bool=False):
	def decorator(func):
		@functools.wraps(func)
		def new_func(uris: Union[List[str], Set[str]], *args, **kwargs) -> Dict:
			global activate_dns_cache
			result: Dict = func(uris, *args, **kwargs)
			if disable:
				return result
			if len(uris) == len(result):
				pass
			elif len(uris) < len(result):
				raise Exception('impossible')
			else:
				miss = set(uris) - set(result.keys())
				if retry_with_host:
					activate_dns_cache = True
					result.update(func(uris, *args, **kwargs))
					miss = set(uris) - set(result.keys())
					if len(miss) <= 0:
						return result
					activate_dns_cache = False
				fid2tid = get_redirects(miss)
				tid2fid = dict((v, k) for k, v in fid2tid.items())
				missto = set(fid2tid.values())
				if debug and len(fid2tid) < len(miss):
					print('no redirect for {}'.format(miss - set(fid2tid.keys())))
				if len(missto) > 0:
					sup = new_func(missto, *args, **kwargs)
					if debug and len(sup) < len(missto):
						print('cant find targets after redirect for {}'.format(missto - set(sup.keys())))
					for k, v in sup.items():
						result[tid2fid[k]] = v
			return result
		return new_func
	return decorator


def handle_http_error(errors: Set[int]={429}, retry_after: int=10, max_retry=5):
	def decorator(func):
		@functools.wraps(func)
		def new_func(*args, **kwargs):
			count = kwargs['count'] if 'count' in kwargs else 0
			if 'count' in kwargs:
				del kwargs['count']
			if count >= max_retry:
				raise Exception('too many retries')
			try:
				return func(*args, **kwargs)
			except urllib.error.HTTPError as e:
				if e.code not in errors:
					raise e
				if 'Retry-after' in e.headers:
					retry_after = int(e.headers['Retry-after'])
				retry_after += 60 * count
				print('retry after {} sec'.format(retry_after))
				time.sleep(retry_after)
				return new_func(*args, **kwargs, count=count + 1)
		return new_func
	return decorator


@handle_redirect(debug=True)
def get_langs(uris:  Union[List[str], Set[str]]) -> Dict[str, Dict[str, str]]:
	id2lang: Dict[str, Dict[str, str]] = defaultdict(lambda: {})
	results = get_result(GET_LANG_MULTI % ' '.join(map(lambda x: 'wd:' + x, uris)))
	for result in results['results']['bindings']:
		uri = get_qid_from_uri(result['item']['value'])
		lang = result['label_lang']['value']
		label = result['label']['value']
		id2lang[uri][lang] = label
	return id2lang


@handle_http_error({429}, retry_after=10, max_retry=5)
@handle_redirect(debug=True)
def get_alias(uris: Union[List[str], Set[str]], lang: str) -> Dict[str, Set[str]]:
	id2alias: Dict[str, Set[str]] = defaultdict(set)
	results = get_result(GET_ALIAS_IN_LANG % (' '.join(map(lambda x: 'wd:' + x, uris)), lang, lang))
	for result in results['results']['bindings']:
		uri = get_qid_from_uri(result['item']['value'])
		label = result['itemLabel']['value']
		if label == uri:  # redirected entity
			continue
		id2alias[uri].add(label)
		if 'alt' in result:
			alias = result['alt']['value']
			id2alias[uri].add(alias)
	return id2alias


def get_redirects(uris: Union[List[str], Set[str]]) -> Dict[str, str]:
	id2id: Dict[str, str] = {}
	results = get_result(GET_REDIRECT % ' '.join(map(lambda x: 'wd:' + x, uris)))
	for result in results['results']['bindings']:
		id2id[get_qid_from_uri(result['item']['value'])] = get_qid_from_uri(result['label']['value'])
	return id2id


def get_result(query, timeout=None):
	sparql = SPARQLWrapper('https://query.wikidata.org/sparql', agent=
	'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36')
	if timeout:
		sparql.setTimeout(timeout)
	sparql.setQuery(query)
	sparql.setReturnFormat(JSON)
	return sparql.query().convert()


def get_qid_from_uri(uri: str) -> str:
	return uri.rsplit('/', 1)[1]


def sup(uris):
	results = []
	for uri in uris:
		real_uri = list(get_redirects([uri]).values())[0]
		lang = [(l['label']['value'], l['label_lang']['value']) for l in get_lang(real_uri)['results']['bindings']]
		results.append(uri + '\t' + '\t'.join(map(lambda x: '"{}"@{}'.format(*x), lang)))
	return results


def load_qid_from_lang_file(filename: str) -> List[str]:
	qids = []
	with open(filename, 'r') as fin:
		for l in tqdm(fin):
			qids.append(l.strip().split('\t', 1)[0])
	return qids


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='retrieve multiple objects for N-M relations')
	parser.add_argument('--task', type=str, help='task')
	parser.add_argument('--inp', type=str, help='input file')
	parser.add_argument('--out', type=str, help='output file')
	parser.add_argument('--lang', type=str, help='language')
	args = parser.parse_args()

	if args.task == 'get_lang':
		inp_dir, multi_rel_file = args.inp.split(':')
		batch_size = 300
		data = TRExDataset(inp_dir)
		multi_rel = MultiRel.load_multi_objects(multi_rel_file)
		entities: Set[str] = set()
		for query in data.iter():
			for entity in [query['sub_uri'], query['obj_uri']]:
				entities.add(entity)
		for k, v in multi_rel.items():
			entities.update(v)
		print('#entities {}'.format(len(entities)))
		entities = list(entities)

		results: Dict[str, Dict[str, str]] = defaultdict(lambda: {})
		for b in tqdm(range(0, len(entities), batch_size)):
			results.update(get_langs(entities[b:b + batch_size]))

		# use specified host for the missing ones
		activate_dns_cache = True
		miss = list(set(entities) - set(results.keys()))
		for b in tqdm(range(0, len(miss), batch_size)):
			results.update(get_langs(miss[b:b + batch_size]))
			time.sleep(1)
		activate_dns_cache = False

		with open(args.out, 'w') as fout:
			for eid in sorted(results.keys(), key=lambda x: int(x[1:])):
				fout.write('{}\t{}\n'.format(
					eid, '\t'.join(map(lambda x: '"{}"@{}'.format(x[1], x[0]), results[eid].items()))))

	elif args.task == 'get_alias':
		batch_size = 300
		batch_size = int(batch_size)
		entities: List[str] = load_qid_from_lang_file(args.inp)
		print('#entities {}'.format(len(entities)))

		results: Dict[str, Set[str]] = {}
		for b in tqdm(range(0, len(entities), batch_size)):
			results.update(get_alias(entities[b:b + batch_size], lang=args.lang))
			time.sleep(1)

		'''
		# use specified host for the missing ones
		activate_dns_cache = True
		miss = list(set(entities) - set(results.keys()))
		for b in tqdm(range(0, len(miss), batch_size)):
			results.update(get_alias(miss[b:b + batch_size], lang=args.lang))
			time.sleep(1)
		activate_dns_cache = False
		'''

		with open(args.out, 'w') as fout:
			for eid in sorted(results, key=lambda x: int(x[1:])):
				fout.write('{}\t{}\n'.format(eid, '\t'.join(sorted(results[eid]))))

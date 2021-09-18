from typing import Set, List, Tuple, Iterable, Dict
import os
from collections import defaultdict
import argparse
import json
import uuid
import pickle
import pandas
import numpy as np
import random
from tqdm import tqdm
import matplotlib.pyplot as plt

SEED = 2020
random.seed(SEED)
np.random.seed(SEED)


def filter_by_relations(filename: str, relations: Set[str]) -> Iterable[Tuple]:
    with open(filename, 'r') as fin:
        for doc in json.load(fin):
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
    parser.add_argument('--task', type=str, choices=['gen', 'filter', 'compose', 'gen_question', 'sample', 'predict'])
    parser.add_argument('--inp', type=str, help='input file')
    parser.add_argument('--out', type=str, help='output file')
    parser.add_argument('--prop', action='store_true', help='sample proportionally to the frequency')
    args = parser.parse_args()

    if args.task == 'gen':

        with open('data/TREx-relations.jsonl', 'r') as fin:
            relations: Set[str] = set([json.loads(l)['relation'] for l in fin])

        os.makedirs(args.out, exist_ok=True)
        pid2facts: Dict[str, Dict[Tuple[str, str], Tuple[Dict[str, int], Dict[str, int], int]]] = \
            defaultdict(lambda: defaultdict(lambda: [defaultdict(lambda: 0), defaultdict(lambda: 0), 0]))
        for root, dirs, files in os.walk(args.inp):
            print('#files {}'.format(len(files)))
            for file in tqdm(files):
                if not file.endswith('.json'):
                    continue
                for (su, sl), (pu, pl), (ou, ol) in filter_by_relations(os.path.join(root, file), relations):
                    pid2facts[pu][(su, ou)][0][sl] += 1
                    pid2facts[pu][(su, ou)][1][ol] += 1
                    pid2facts[pu][(su, ou)][2] += 1
        for pid, subobj2count in pid2facts.items():
            with open(os.path.join(args.out, pid + '.jsonl'), 'w') as fout:
                for (sub, obj), count in sorted(subobj2count.items(), key=lambda x: -x[1][2]):
                    sl: str = sorted(count[0].items(), key=lambda x: -x[1])[0][0]
                    ol: str = sorted(count[1].items(), key=lambda x: -x[1])[0][0]
                    count: int = count[2]
                    fout.write(json.dumps({
                        'uuid': str(uuid.uuid1()),
                        'predicate_id': pid,
                        'sub_uri': sub,
                        'sub_label': sl,
                        'obj_uri': obj,
                        'obj_label': ol,
                        'count': count
                    }) + '\n')

    elif args.task == 'filter':
        freq_thres = 10
        max_num_alias = 10

        with open('data/TREx-relations.jsonl', 'r') as fin:
            relations: List[Dict] = [json.loads(l) for l in fin]
            relations: Set[str] = set(r['relation'] for r in relations if r['type'] in {'N-1', '1-1'})


        id2names: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(lambda: 0))
        sub2pid2obj: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))

        print('build index')
        for root, dirs, files in os.walk(args.inp):
            for file in tqdm(files):
                if not file.endswith('.json'):
                    continue
                for (su, sl), (pu, pl), (ou, ol) in filter_by_relations(os.path.join(root, file), relations):
                    id2names[su][sl] += 1
                    id2names[ou][ol] += 1
                    sub2pid2obj[su][pu][ou] += 1

        print('filter by frequency')
        used_ids: Set[str] = set()
        with open(args.out, 'w') as fout:
            for sub, pid2obj in sub2pid2obj.items():
                for pid, obj2count in pid2obj.items():
                    objs = list(obj2count.keys())
                    for obj in objs:
                        count = obj2count[obj]
                        if count >= freq_thres:
                            used_ids.add(sub)
                            used_ids.add(obj)
                            fout.write(f'{sub}\t{pid}\t{obj}\t{count}\n')
        with open(args.out + '.id2name', 'w') as fout:
            for id, names in id2names.items():
                if id not in used_ids:
                    continue
                names = sorted(names.items(), key=lambda x: -x[1])[:max_num_alias]
                names = '\t'.join([n[0] for n in names])
                fout.write(f'{id}\t{names}\n')

    elif args.task == 'compose':
        combine_sym = '@'
        num_instance_per_pid = 100
        max_entity_count = 20

        id2names: Dict[str, List[str]] = {}
        sub2pid2obj: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))
        with open(args.inp, 'r') as fin, open(args.inp + '.id2name', 'r') as nfin:
            for l in fin:
                s, p, o, c = l.strip().split('\t')
                c = int(c)
                sub2pid2obj[s][p][o] = c
            for l in nfin:
                ns = l.strip().split('\t')
                id2names[ns[0]] = ns[1:]

        paths = []
        pidpath2count = defaultdict(lambda: 0)
        for sub, pid2obj in tqdm(sub2pid2obj.items()):
            for pid, obj2count in pid2obj.items():
                for obj, count in obj2count.items():
                    if obj in sub2pid2obj:
                        for npid, nobj2count in sub2pid2obj[obj].items():
                            for nobj, ncount in nobj2count.items():
                                paths.append((sub, pid, obj, npid, nobj, count, ncount))
                                pidpath2count[(pid, npid)] += 1

        '''
        prompt_lang = pandas.read_csv('data/TREx_prompts.csv')
        for pp in sorted(pidpath2count.items(), key=lambda x: -x[1]):
            pnames = combine_sym.join([prompt_lang[prompt_lang['pid'] == p]['relation'].iloc[0] for p in pp[0]])
            prompts = combine_sym.join([prompt_lang[prompt_lang['pid'] == p]['en'].iloc[0] for p in pp[0]])
            print(f'{combine_sym.join(pp[0])}\t{pnames}\t{prompts}\t{pp[1]}')
        print(len(paths))
        '''

        pid2question: Dict[str, Tuple[str, str]] = {}
        for index, row in pandas.read_csv('data/multihop.csv').iterrows():
            pid2question[row['pid']] = (row['single-question'], row['multi-question'])
        pid2paths: Dict[str, List[Tuple[str, str, str, int, int]]] = defaultdict(list)
        for sub, pid, obj, npid, nobj, count, ncount in paths:
            pidkey = f'{pid}@{npid}'
            pid2paths[pidkey].append((sub, obj, nobj, count, ncount))
        index = 0
        hm = lambda x: 2 * x[-1] * x[-2] / (x[-1] + x[-2])
        with open(args.out, 'w') as fout, open(args.out + '.source', 'w') as sfout, open(args.out + '.target', 'w') as tfout,\
          open(args.out + '.id', 'w') as ifout, open(args.out + '.freq', 'w') as ffout:
            for pidkey, tuples in pid2paths.items():
                used_count = 0
                entity2count: Dict[str, int] = defaultdict(lambda: 0)
                if pidkey not in pid2question:
                    continue
                tuples = sorted(tuples, key=lambda x: -hm(x))
                for sub, obj, nobj, count, ncount in tuples:
                    skip = False
                    for e in [sub, obj, nobj]:
                        if entity2count[e] >= max_entity_count:
                            skip = True
                            break
                    if skip:
                        continue
                    else:
                        for e in [sub, obj, nobj]:
                            entity2count[e] += 1
                    single_q, multi_q = pid2question[pidkey]
                    first_q, sec_q = single_q.split(combine_sym)
                    first_q = first_q.replace('[X]', id2names[sub][0])
                    sec_q = sec_q.replace('[X]', id2names[obj][0])
                    multi_q = multi_q.replace('[X]', id2names[sub][0])
                    fout.write(f'{index}\t{first_q}\t{id2names[obj][0]}\t0\n')
                    sfout.write(f'{first_q}\n')
                    a = '\t'.join(id2names[obj])
                    tfout.write(f'{a}\n')
                    fout.write(f'{index + 1}\t{sec_q}\t{id2names[nobj][0]}\t0\n')
                    sfout.write(f'{sec_q}\n')
                    a = '\t'.join(id2names[nobj])
                    tfout.write(f'{a}\n')
                    fout.write(f'{index + 2}\t{multi_q}\t{id2names[nobj][0]}\t0\n')
                    sfout.write(f'{multi_q}\n')
                    a = '\t'.join(id2names[nobj])
                    tfout.write(f'{a}\n')
                    ifout.write(f'{pidkey}\n')
                    ifout.write(f'{pidkey}\n')
                    ifout.write(f'{pidkey}\n')
                    ffout.write(f'{count}\n')
                    ffout.write(f'{ncount}\n')
                    ffout.write(f'{hm([count, ncount])}\n')
                    index += 3
                    used_count += 1
                    if used_count >= num_instance_per_pid:
                        break

    elif args.task == 'predict':
        model_name = 'allenai/unifiedqa-t5-3b'
        batch_size = 4
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        print('load model ...')
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        model.cuda()
        model.eval()
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        questions = []
        results = []
        with open(args.inp, 'r') as fin:
            for l in tqdm(fin):
                _, question, answer, _ = l.strip().split('\t')
                questions.append(question)
                if len(questions) >= batch_size:
                    batch = tokenizer(questions, padding=True, return_tensors='pt')
                    outputs = model.generate(batch.input_ids.cuda())
                    for output in outputs:
                        results.append(tokenizer.decode(output, skip_special_tokens=True))
                    questions = []
            if len(questions) > 0:
                batch = tokenizer(questions, return_tensors='pt')
                outputs = model.generate(batch.input_ids)
                for output in outputs:
                    results.append(tokenizer.decode(output, skip_special_tokens=True))
                questions = []

        with open(args.out, 'w') as fout:
            for r in results:
                fout.write(f'{r}\n')

    elif args.task == 'sample':
        count = 100
        pid2dist: Dict[str, List[float]] = {}
        for root, dirs, files in os.walk(args.inp):
            for file in tqdm(files):
                if not file.endswith('.jsonl'):
                    continue
                with open(os.path.join(root, file), 'r') as fin:
                    probs: List[int] = []
                    facts: List[str] = []
                    min_freq: int = 1e10
                    max_freq: int = 0
                    for l in fin:
                        f = json.loads(l)
                        c = f['count']
                        facts.append(l)
                        probs.append(c)
                        if c < min_freq:
                            min_freq = c
                        if c > max_freq:
                            max_freq = c
                    total = np.sum(probs)
                    probs = np.array(probs) / total
                    choices = sorted(np.random.choice(len(probs), count, replace=False, p=probs if args.prop else None))

                inter = probs[0] - probs[-1]
                dist: List[float] = sorted([(probs[i] - probs[-1]) / (inter or 1) for i in choices])
                pid2dist[file.rsplit('.', 1)[0]] = dist

                if args.out:
                    with open(os.path.join(args.out, file), 'w') as fout:
                        for i in choices:
                            fout.write(facts[i])
            break

        x = []
        y = []
        for i, (k, v) in enumerate(pid2dist.items()):
            x.extend(v)
            y.extend([i / len(pid2dist)] * len(v))
        plt.scatter(x, y)
        plt.ylabel('Relation')
        plt.xlabel('Frequency')
        plt.savefig('dist.png')


import jsonlines

def read_necessary_data():

	# Read list of entity IDs with their genders, if known
	with open("../TREx_gender.txt") as inp:
		lines = inp.readlines()

	genders = {}
	for l in lines:
		l = l.strip().split('\t')
		ent_id = l[0]
		ent_gender = l[1]
		genders[ent_id] = ent_gender

	# Read list of entity IDs with their instanceof or subclassof information, if known
	with open("../TREx_instanceof.txt") as inp:
		lines = inp.readlines()

	instanceof = {}
	for l in lines:
		l = l.strip().split('\t')
		ent_id = l[0]
		instanceof[ent_id] = ','.join(l[1:])


	# Read list of vietnamese entities
	with open("TREx_yoruba.txt") as inp:
		lines = inp.readlines()

	entities = {}
	for l in lines:
		l = l.strip().split('\t')
		ent_id = l[0]
		ent_form = l[1]
		entities[ent_id] = ent_form
	
	return entities


def fil_y(words, ent_form):
	if '[Y]' in words:
		i = words.index('[Y]')
		words[i] = ent_form

	
	return words

def fil_x(words, ent_form):
	if '[X]' in words:
		i = words.index('[X]')
		words[i] = ent_form

	return words

def print_examples_for_relation(relation, entities):
	rel_id = relation['relation']
	template = relation['template']
	print(template)
	label = relation['label']
	desc = relation['description']
	tp = relation['type']
	
	words = template.strip().split(' ')
	words = [w.strip() for w in words if w.strip()]
	
	exfiles = f"/Users/antonis/research/lama/data/TREx/{rel_id}.jsonl"
	try:
		with jsonlines.open(exfiles) as reader:
			count = 0
			count_double = 0
			for obj in reader:
				X_ent_id = obj["sub_uri"]
				Y_ent_id = obj["obj_uri"]
				if X_ent_id in entities and Y_ent_id in entities:
					#if ' ' not in entities[X_ent_id][0] and ' ' not in entities[Y_ent_id][0]:
					#if ' ' not in entities[Y_ent_id][0]:
					sentence = fil_x(list(words), entities[X_ent_id])
					sentence = fil_y(sentence, entities[Y_ent_id])
					print("\t", ' '.join(sentence))
					count += 1
				if count == 10:
					break
			print(f"Found {count_double} entries with more than one word.")
	except:
		pass


entities = read_necessary_data()
# Read all relation templates
with jsonlines.open('relations.yo.jsonl') as reader:
	count = 0
	for obj in reader:
		print_examples_for_relation(obj, entities)
		count += 1
		#if count == 4:
		#	break
		


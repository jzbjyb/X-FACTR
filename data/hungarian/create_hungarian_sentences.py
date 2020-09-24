
from unimorph_inflect import inflect
import jsonlines


# Two functions needed to check if the entity has latin characters
# in which case we shouldn't inflect it in Greek
import unicodedata as ud
latin_letters= {}
def is_latin(uchr):
    try: return latin_letters[uchr]
    except KeyError:
         return latin_letters.setdefault(uchr, 'LATIN' in ud.name(uchr))

def some_roman_chars(unistr):
    return any(is_latin(uchr)
           for uchr in unistr
           if uchr.isalpha()) # isalpha suggested by John Machin


def gender_heuristic(w):
	w = w.strip()
	if ' ' not in w:
		return "Neut"
		#if w[-1] == "η" or w[-1] == "ή" or w[-1] == "α":
		#	return "Fem"
		#elif w[-2:] == "ος" or w[-2:] == "ης" or w[-2:] == "ής" or w[-2:] == "ας" or w[-2:] == "άς" or w[-2:] == "ός" or w[-2:] == "ήλ":
		#	return "Masc"
		#else:
		#	return "Neut"
	else:
		w2 = w.split(' ')[0]
		return "Neut"
		#if w2[-1] == "η" or w2[-1] == "ή" or w2[-1] == "α":
		#	return "Fem"
		#elif w2[-2:] == "ος" or w2[-2:] == "ης" or w2[-2:] == "ής" or w2[-2:] == "ας" or w2[-2:] == "άς" or w2[-2:] == "ός" or w2[-2:] == "ήλ":
		#	return "Masc"
		#else:
		#	return "Neut"



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


	# Read list of hungarian entities
	with open("TREx_hungarian.txt") as inp:
		lines = inp.readlines()

	entities = {}
	for l in lines:
		l = l.strip().split('\t')
		ent_id = l[0]
		ent_form = l[1]
		ent_gender = gender_heuristic(ent_form).upper()
		ent_number = "SG"
		is_human = False
		if ent_id in genders:
			if genders[ent_id] == "male":
				ent_gender = "MASC"
			elif genders[ent_id] == "female":
				ent_gender = "FEM"
			else:
				if ent_id in instanceof:
					if 'state' in instanceof[ent_id] or 'country' in instanceof[ent_id]:
						is_human = False
					elif 'human' in instanceof[ent_id]:
						is_human = True

		entities[ent_id] = (ent_form, ent_number, is_human)	
		entities[ent_id] = (ent_form, ent_number, is_human)	
	
	return entities


def fil_y(words, ent_form, ent_number, is_human):
	
	if some_roman_chars(ent_form) or ent_form.isupper():
		do_not_inflect = True
	else:
		do_not_inflect = False


	if '[Y]' in words:
		i = words.index('[Y]')
		words[i] = ent_form
		ent_case = "NOM"
	elif "[Y.IN+ESS]" in words:
		i = words.index('[Y.IN+ESS]')
		if do_not_inflect:
			words[i] = ent_form
		else:
			words[i] = inflect(ent_form, f"N;IN+ESS;{ent_number}", language='hun')[0]
		ent_case = "IN+ESS"
	elif "[Y.IN+ABL]" in words:
		i = words.index('[Y.IN+ABL]')
		if do_not_inflect:
			words[i] = ent_form
		else:
			words[i] = inflect(ent_form, f"N;IN+ABL;{ent_number}", language='hun')[0]
		ent_case = "IN+ABL"
	elif "[Y.ON+ESS]" in words:
		i = words.index('[Y.ON+ESS]')
		if do_not_inflect:
			words[i] = ent_form
		else:
			words[i] = inflect(ent_form, f"N;ON+ESS;{ent_number}", language='hun')[0]
		ent_case = "ON+ESS"
	elif "[Y.DAT]" in words:
		i = words.index('[Y.DAT]')
		if do_not_inflect:
			words[i] = ent_form
		else:
			words[i] = inflect(ent_form, f"N;DAT;{ent_number}", language='hun')[0]
		ent_case = "DAT"
	elif "[Y.ACC]" in words:
		i = words.index('[Y.ACC]')
		if do_not_inflect:
			words[i] = ent_form
		else:
			words[i] = inflect(ent_form, f"N;ACC;{ent_number}", language='hun')[0]
		ent_case = "ACC"
	elif "[Y.INST]" in words:
		i = words.index('[Y.INST]')
		if do_not_inflect:
			words[i] = ent_form
		else:
			words[i] = inflect(ent_form, f"N;INST;{ent_number}", language='hun')[0]
		ent_case = "INST"

	# Now check for the ones that we have a fixed suffix:
	for i,w in enumerate(words):
		if w[:3] == '[Y]' and len(w) > 3:
			words[i] = ent_form+w[3:]
	
	return words

def fil_x(words, ent_form, ent_number, is_human):

	if some_roman_chars(ent_form) or ent_form.isupper():
		do_not_inflect = True
	else:
		do_not_inflect = False

	if '[X]' in words:
		i = words.index('[X]')
		words[i] = ent_form
		ent_case = "NOM"
	elif "[X.ACC]" in words:
		i = words.index('[X.ACC]')
		if do_not_inflect:
			words[i] = ent_form
		else:
			words[i] = inflect(ent_form, f"N;ACC;{ent_number}", language='hun')[0]
			#words[i] = ent_form+'त'
		ent_case = "ACC"
	elif "[X.DAT]" in words:
		i = words.index('[X.DAT]')
		if do_not_inflect:
			words[i] = ent_form
		else:
			words[i] = inflect(ent_form, f"N;DAT;{ent_number}", language='hun')[0]
			#words[i] = ent_form+'त'
		ent_case = "DAT"
	elif "[X.ON+ESS]" in words:
		i = words.index('[X.ON+ESS]')
		if do_not_inflect:
			words[i] = ent_form
		else:
			words[i] = inflect(ent_form, f"N;ON+ESS;{ent_number}", language='hun')[0]
		ent_case = "ON+ESS"

	return words

def print_examples_for_relation(relation, entities):
	rel_id = relation['relation']
	template = relation['template']
	print("Template: ", template)
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
					if ' ' not in entities[Y_ent_id][0]:
						sentence = fil_x(list(words), entities[X_ent_id][0], entities[X_ent_id][1].upper(), entities[X_ent_id][2])
						sentence = fil_y(sentence, entities[Y_ent_id][0], entities[Y_ent_id][1].upper(), entities[Y_ent_id][2])
						print("\t", ' '.join(sentence))
						count += 1
					else:
						count_double += 1
				if count == 10:
					break
			print(f"Found {count_double} entries with more than one word.")
	except:
		pass



entities = read_necessary_data()
# Read all relation templates
with jsonlines.open('relations.hun.jsonl') as reader:
	count = 0
	for obj in reader:
		print_examples_for_relation(obj, entities)
		count += 1
		#if count == 4:
		#	break
		


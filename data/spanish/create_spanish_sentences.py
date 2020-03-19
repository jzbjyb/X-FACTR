
from unimorph_inflect import inflect
import jsonlines


# Decide on Spanish gender for the unknown entities based on the endings
# Based on http://www.study-languages-online.com/russian-nouns-gender.html
def gender_heuristic(w):
	w = w.strip()
	if ' ' not in w:
		if w[-1] == "o":
			return "Masc"
		elif w[-1] == "a" or w[-4:] in ["ción", "sión"] or w[-3:] in ["dad", "tad"] or w[-5:] in ["umbre"]:
			return "Fem"
		else:
			return "Masc"
	else:
		w2 = w.split(' ')[0]
		if w2[-1] == "o":
			return "Masc"
		elif w2[-1] == "a" or w2[-4:] in ["ción", "sión"] or w2[-3:] in ["dad", "tad"] or w2[-5:] in ["umbre"]:
			return "Fem"
		else:
			return "Masc"




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


	# Read list of russian entities
	with open("TREx_spanish.txt") as inp:
		lines = inp.readlines()

	entities = {}
	for l in lines:
		l = l.strip().split('\t')
		ent_id = l[0]
		ent_form = l[1]
		ent_number = "SG"
		ent_gender = gender_heuristic(ent_form).upper()
		if ent_id in genders:
			if genders[ent_id] == "male":
				ent_gender = "MASC"
			elif genders[ent_id] == "female":
				ent_gender = "FEM"
			else:
				if ent_id in instanceof:
					# ARGH WHAT TO DO HERE : Using MASC because it is the most common one :(
					if 'human' in instanceof[ent_id]:
						ent_gender = "MASC" 
					if 'state' in instanceof[ent_id] or 'country' in instanceof[ent_id]:
						ent_gender = "FEM"
					#elif 'business' in instanceof[ent_id]:
					#	ent_gender = "FEM"
					#elif 'enterprise' in instanceof[ent_id]:
					#	ent_gender = "FEM"
					#elif 'city' in instanceof[ent_id]:
					#	ent_gender = "FEM"
					#elif 'island' in instanceof[ent_id]:
					#	ent_gender = "FEM"
					#elif 'literary work' in instanceof[ent_id]:
					#	ent_gender = "NEUT"
					elif 'musical group' in instanceof[ent_id]:
						ent_gender = "MASC"
						ent_number = "PL"
					elif 'record label' in instanceof[ent_id]:
						ent_gender = "FEM"
					#elif 'language' in instanceof[ent_id]:
					#	ent_gender = "NEUT"
					#	ent_number = "PL"
					#elif 'sports team' in instanceof[ent_id]:
					#	ent_gender = "FEM"
					#elif 'automobile manufacturer' in instanceof[ent_id]:
					#	ent_gender = "FEM"
					elif 'football club' in instanceof[ent_id]:
						ent_gender = "FEM"
					#elif '' in instanceof[ent_id]:
					#	ent_gender = "FEM"
					#elif '' in instanceof[ent_id]:
					#	ent_gender = "FEM"


		entities[ent_id] = (ent_form, ent_gender, ent_number)
	
	return entities


def fil_y(words, ent_form, ent_gender, ent_number):
	#ent_form = entities[ent_id][0]
	#ent_gender = entities[ent_id][1].upper()
	#ent_number = "SG"

	if ent_form.isupper():
		do_not_inflect = True
	else:
		do_not_inflect = False

	if '[Y]' in words:
		i = words.index('[Y]')
		words[i] = ent_form

	if '[ART;Y-Gender]' in words:
		i = words.index('[ART;Y-Gender]')
		if ent_gender == "MASC":
			words[i] = 'un'
		elif ent_gender == "FEM":
			words[i] = 'una'
		else:
			words[i] = 'un'

	if '[DEF;Y]' in words:
		i = words.index('[DEF;Y]')
		if ent_gender == "MASC":
			words[i] = 'el'
		elif ent_gender == "FEM":
			words[i] = 'la'
		else:
			words[i] = 'un'


	# Now also check the correponsing articles, if the exist
	for i,w in enumerate(words):
		if w[0] == '[' and 'Y-Gender' in w:
			if '|' in w:
				options = w.strip()[1:-1].split('|')
				if ent_number == "PL" and len(options) == 3:
					form = options[2].strip().split(';')[0]
					words[i] = form
				elif ent_gender == "MASC":
					form = options[0].strip().split(';')[0]
					words[i] = form
				elif ent_gender == "FEM":
					form = options[1].strip().split(';')[0]
					words[i] = form
				else:
					form = options[0].strip().split(';')[0]
					words[i] = form

	
	return words

def fil_x(words, ent_form, ent_gender, ent_number):
	#ent_form = entities[ent_id][0]
	#ent_gender = entities[ent_id][1].upper()
	#ent_number = "SG"

	if ent_form.isupper():
		do_not_inflect = True
	else:
		do_not_inflect = False


	if '[X]' in words:
		i = words.index('[X]')
		words[i] = ent_form

	if '[ART;X-Gender]' in words:
		i = words.index('[ART;X-Gender]')
		if ent_gender == "MASC":
			words[i] = 'un'
		elif ent_gender == "FEM":
			words[i] = 'una'
		else:
			words[i] = 'un'

	# Now also check the corresponding verbs, if they exist
	for i,w in enumerate(words):
		#if w[0] == '[' and 'X-Gender' in w:
		if w[0] == '[':
			#print(w, ent_gender)
			if '|' in w and 'X-Gender' in w:
				options = w.strip()[1:-1].split('|')
				#print(w, ent_gender, options)
				if ent_number == "PL" and len(options) == 3:
					form = options[2].strip().split(';')[0]
					words[i] = form
				elif ent_gender == "MASC":
					form = options[0].strip().split(';')[0]
					words[i] = form
				elif ent_gender == "FEM":
					form = options[1].strip().split(';')[0]
					words[i] = form
				else:
					form = options[0].strip().split(';')[0]
					words[i] = form
			elif '|' in w and 'X-Number' in w:
				options = w.strip()[1:-1].split('|')
				if ent_number == "PL":
					form = options[1].strip().split(';')[0]
					words[i] = form
				elif ent_number == "SG":
					form = options[0].strip().split(';')[0]
					words[i] = form
				else:
					form = options[0].strip().split(';')[0]
					words[i] = form
		
	return words

def print_examples_for_relation(relation, entities):
	rel_id = relation['relation']
	template = relation['template']
	print(template)
	label = relation['label']
	desc = relation['description']
	tp = relation['type']
	
	words = template.strip().split(' ')
	
	exfiles = f"/Users/antonis/research/lama/data/TREx/{rel_id}.jsonl"
	try:
	#if rel_id != "P166" and rel_id!="P69" and rel_id!= "P54": 
		with jsonlines.open(exfiles) as reader:
			count = 0
			count_double = 0
			for obj in reader:
				X_ent_id = obj["sub_uri"]
				Y_ent_id = obj["obj_uri"]
				if X_ent_id in entities and Y_ent_id in entities:
					#if ' ' not in entities[X_ent_id][0] and ' ' not in entities[Y_ent_id][0]:
					if ' ' not in entities[Y_ent_id][0]:
						sentence = fil_x(list(words), entities[X_ent_id][0], entities[X_ent_id][1].upper(), entities[X_ent_id][2].upper())
						sentence = fil_y(sentence, entities[Y_ent_id][0], entities[Y_ent_id][1].upper(), entities[Y_ent_id][2].upper())
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
with jsonlines.open('relations.es.jsonl') as reader:
	count = 0
	for obj in reader:
		print_examples_for_relation(obj, entities)
		count += 1
		#if count == 13:
		#	break
		


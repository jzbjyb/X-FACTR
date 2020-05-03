
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


	# Read list of russian entities
	with open("TREx_hebrew.txt") as inp:
		lines = inp.readlines()

	entities = {}
	for l in lines:
		l = l.strip().split('\t')
		ent_id = l[0]
		ent_form = l[1]
		ent_gender = "FEM"
		if ent_id in genders:
			if genders[ent_id] == "male":
				ent_gender = "MASC"
			elif genders[ent_id] == "female":
				ent_gender = "FEM"
			else:
				if ent_id in instanceof and ent_gender == "NEUT":
					#if 'state' in instanceof[ent_id] or 'country' in instanceof[ent_id]:
					#	ent_gender = "FEM"
					#elif 'business' in instanceof[ent_id]:
					#	ent_gender = "FEM"
					#elif 'enterprise' in instanceof[ent_id]:
					#	ent_gender = "FEM"
					#elif 'city' in instanceof[ent_id]:
					#	ent_gender = "FEM"
					# ARGH WHAT TO DO HERE : Using MASC because it is the most common one :(
					if 'human' in instanceof[ent_id]:
						ent_gender = "MASC" 
					#elif 'island' in instanceof[ent_id]:
					#	ent_gender = "FEM"
					#elif 'literary work' in instanceof[ent_id]:
					#	ent_gender = "NEUT"
					#elif 'musical group' in instanceof[ent_id]:
					#	ent_gender = "MASC"
					#	ent_number = "PL"
					#elif 'record label' in instanceof[ent_id]:
					#	ent_gender = "FEM"
					#elif 'language' in instanceof[ent_id]:
					#	ent_gender = "NEUT"
					#	ent_number = "PL"
					#elif 'sports team' in instanceof[ent_id]:
					#	ent_gender = "FEM"
					#elif 'automobile manufacturer' in instanceof[ent_id]:
					#	ent_gender = "FEM"
					#elif 'football club' in instanceof[ent_id]:
					#	ent_gender = "FEM"
					#elif '' in instanceof[ent_id]:
					#	ent_gender = "FEM"
					#elif '' in instanceof[ent_id]:
					#	ent_gender = "FEM"


		entities[ent_id] = (ent_form, ent_gender)
	
	return entities


def fil_y(words, ent_form, ent_gender):
	if '[Y]' in words:
		i = words.index('[Y]')
		words[i] = ent_form
		ent_case = "NOM"

	for i,w in enumerate(words):
		if '[Y]' in w:
			form = w.replace('[Y]', ent_form)
			words[i] = form

	# Now also check the correponsing articles, if the exist
	for i,w in enumerate(words):
		if w[0] == '[' and 'Y-Gender' in w:
			if '|' in w:
				options = w.strip()[1:-1].split('|')
				if ent_gender == "MASC":
					form = options[0].strip().split(';')[0]
					words[i] = form
				elif ent_gender == "FEM":
					form = options[1].strip().split(';')[0]
					words[i] = form
				else:
					form = options[0].strip().split(';')[0]
					words[i] = form
	
	return words

def fil_x(words, ent_form, ent_gender):

	if '[X]' in words:
		i = words.index('[X]')
		words[i] = ent_form

	for i,w in enumerate(words):
		if '[X]' in w:
			form = w.replace('[X]', ent_form)
			words[i] = form


	# Now also check the corresponfing verbs, if they exist
	for i,w in enumerate(words):
		if w[0] == '[' and 'X-Gender' in w:
			if '|' in w:
				options = w.strip()[1:-1].split('|')
				if ent_gender == "MASC":
					form = options[0].strip().split(';')[0]
					words[i] = form
				elif ent_gender == "FEM":
					form = options[1].strip().split(';')[0]
					words[i] = form
				else:
					form = options[0].strip().split(';')[0]
					words[i] = form
			#else:
			#	lemma = w.strip()[1:-1].split('.')[0]
			#	if "Pst" in w:
			#		form2 = inflect(lemma, f"V;PST;SG;{ent_gender}", language='rus')[0]
			#	elif "Lgspec1" in w:
			#		form2 = inflect(lemma, f"ADJ;{ent_gender};SG;LGSPEC1", language='rus')[0]
			#	words[i] = form2
		
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
		with jsonlines.open(exfiles) as reader:
			count = 0
			count_double = 0
			for obj in reader:
				X_ent_id = obj["sub_uri"]
				Y_ent_id = obj["obj_uri"]
				if X_ent_id in entities and Y_ent_id in entities:
					#if ' ' not in entities[X_ent_id][0] and ' ' not in entities[Y_ent_id][0]:
					if ' ' not in entities[Y_ent_id][0]:
						sentence = fil_x(list(words), entities[X_ent_id][0], entities[X_ent_id][1].upper())
						sentence = fil_y(sentence, entities[Y_ent_id][0], entities[Y_ent_id][1].upper())
						print("\t\t", ' '.join(sentence))
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
with jsonlines.open('relations.he.jsonl') as reader:
	count = 0
	for obj in reader:
		print_examples_for_relation(obj, entities)
		count += 1
		#if count == 4:
		#	break
		



from unimorph_inflect import inflect
import jsonlines


# Two functions needed to check if the entity has latin characters
# in which case we shouldn't inflect it in Greek
import unicodedata as ud
latin_letters= {}
def is_latin(uchr):
	try:
		return latin_letters[uchr]
	except KeyError:
		 return latin_letters.setdefault(uchr, 'LATIN' in ud.name(uchr))

def some_roman_chars(unistr):
	return any(is_latin(uchr) for uchr in unistr if uchr.isalpha()) # isalpha suggested by John Machin

# Decide on Russian gender for the unknown entities based on the endings
# Based on http://www.study-languages-online.com/russian-nouns-gender.html
def gender_heuristic(w):
	w = w.strip()
	if w[-1] == "a" or w[-1] == "я":
		return "Fem"
	elif w[-1] == "о" or w[-1] == "е" or w[-1] == "ё":
		return "Neut"
	elif w[-1] == "й" or w[-1] in ["б", "в", "г", "д", "ж", "з", "к", "л", "м", "н", "п", "р", "с", "т", "ф", "х", "ц", "ч", "ш", "щ"]:
		return "Masc"
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
	with open("TREx_russian.txt") as inp:
		lines = inp.readlines()

	entities = {}
	for l in lines:
		l = l.strip().split('\t')
		ent_id = l[0]
		ent_form = l[1]
		ent_gender = gender_heuristic(ent_form).upper()
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
	#ent_form = entities[ent_id][0]
	#ent_gender = entities[ent_id][1].upper()
	ent_number = "SG"

	if some_roman_chars(ent_form) or ent_form.isupper():
		do_not_inflect = True
	else:
		do_not_inflect = False


	if '[Y]' in words:
		i = words.index('[Y]')
		words[i] = ent_form
		ent_case = "NOM"
	elif "[Y.Nom]" in words:
		# In Greek the default case is Nominative so we don't need to try to inflect it
		i = words.index('[Y.Nom]')
		words[i] = ent_form
		ent_case = "NOM"
	elif "[Y.Gen]" in words:
		i = words.index('[Y.Gen]')
		if do_not_inflect:
			words[i] = ent_form
		else:
			words[i] = inflect(ent_form, f"N;GEN;{ent_number}", language='rus')[0]
		ent_case = "GEN"
	elif "[Y.Acc]" in words:
		i = words.index('[Y.Acc]')
		if do_not_inflect:
			words[i] = ent_form
		else:
			words[i] = inflect(ent_form, f"N;ACC;{ent_number}", language='rus')[0]
		ent_case = "ACC"
	elif "[Y.Dat]" in words:
		i = words.index('[Y.Dat]')
		if do_not_inflect:
			words[i] = ent_form
		else:
			words[i] = inflect(ent_form, f"N;DAT;{ent_number}", language='rus')[0]
		ent_case = "DAT"
	elif "[Y.Ess]" in words:
		i = words.index('[Y.Ess]')
		if do_not_inflect:
			words[i] = ent_form
		else:
			words[i] = inflect(ent_form, f"N;ESS;{ent_number}", language='rus')[0]
		ent_case = "ESS"
	elif "[Y.Ins]" in words:
		i = words.index('[Y.Ins]')
		if do_not_inflect:
			words[i] = ent_form
		else:
			words[i] = inflect(ent_form, f"N;INS;{ent_number}", language='rus')[0]
		ent_case = "INS"

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
				elif ent_gender == "NEUT":
					form = options[2].strip().split(';')[0]
					words[i] = form
			if "Pst" in w:
				form2 = inflect(lemma, f"V;PST;SG;{ent_gender}", language='rus')[0]
			elif "Lgspec1" in w:
				form2 = inflect(lemma, f"ADJ;{ent_gender};SG;LGSPEC1", language='rus')[0]
			words[i] = form2

	
	return words

def fil_x(words, ent_form, ent_gender):
	#ent_form = entities[ent_id][0]
	#ent_gender = entities[ent_id][1].upper()
	ent_number = "SG"

	if some_roman_chars(ent_form) or ent_form.isupper():
		do_not_inflect = True
	else:
		do_not_inflect = False


	if '[X]' in words:
		i = words.index('[X]')
		words[i] = ent_form
		ent_case = "NOM"
	elif "[X.Nom]" in words:
		# In Greek the default case is Nominative so we don't need to try to inflect it
		i = words.index('[X.Nom]')
		words[i] = ent_form
		ent_case = "NOM"
	elif "[X.Masc.Nom]" in words:
		# In Greek the default case is Nominative so we don't need to try to inflect it
		i = words.index('[X.Masc.Nom]')
		words[i] = ent_form
		ent_case = "NOM"
		ent_gender = "Masc"
	elif "[X.Gen]" in words:
		i = words.index('[X.Gen]')
		if do_not_inflect:
			words[i] = ent_form
		else:
			words[i] = inflect(ent_form, f"N;GEN;{ent_number}", language='rus')[0]
		ent_case = "GEN"
	elif "[X.Ess]" in words:
		i = words.index('[X.Ess]')
		if do_not_inflect:
			words[i] = ent_form
		else:
			words[i] = inflect(ent_form, f"N;ESS;{ent_number}", language='rus')[0]
		ent_case = "ESS"

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
				elif ent_gender == "NEUT":
					form = options[2].strip().split(';')[0]
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
with jsonlines.open('relations.ru.jsonl') as reader:
	count = 0
	for obj in reader:
		print_examples_for_relation(obj, entities)
		count += 1
		if count == 4:
			break
		


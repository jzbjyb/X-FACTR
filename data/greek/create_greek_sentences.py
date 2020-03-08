
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
		if w[-1] == "η" or w[-1] == "ή" or w[-1] == "α":
			return "Fem"
		elif w[-2:] == "ος" or w[-2:] == "ης" or w[-2:] == "ής" or w[-2:] == "ας" or w[-2:] == "άς" or w[-2:] == "ός" or w[-2:] == "ήλ":
			return "Masc"
		else:
			return "Neut"
	else:
		w2 = w.split(' ')[0]
		if w2[-1] == "η" or w2[-1] == "ή" or w2[-1] == "α":
			return "Fem"
		elif w2[-2:] == "ος" or w2[-2:] == "ης" or w2[-2:] == "ής" or w2[-2:] == "ας" or w2[-2:] == "άς" or w2[-2:] == "ός" or w2[-2:] == "ήλ":
			return "Masc"
		else:
			return "Neut"



def read_necessary_data():

	# Read list of entity IDs with their genders, if known
	with open("../mTREx_gender.txt") as inp:
		lines = inp.readlines()

	genders = {}
	for l in lines:
		l = l.strip().split('\t')
		ent_id = l[0]
		ent_gender = l[1]
		genders[ent_id] = ent_gender

	# Read list of greek entities
	with open("TREx_greek.txt") as inp:
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
		entities[ent_id] = (ent_form, ent_gender)	

	'''
	# Read list of greek entities
	with open("TREx_greek_tagged.txt") as inp:
		lines = inp.readlines()

	entities = {}
	for l in lines:
		l = l.strip().split('\t')
		ent_id = l[0]
		ent_form = l[1]
		ent_gender = l[2]
		entities[ent_id] = (ent_form, ent_gender)
	'''

	with open("articles.el.txt") as inp:
		lines = inp.readlines()

	article = {}
	for l in lines:
		l = l.strip().split('\t')
		article[l[1]] = l[0]
	
	return entities, article


def fil_y(words, ent_form, ent_gender, article):
	#ent_form = entities[ent_id][0]
	#ent_gender = entities[ent_id][1].upper()
	ent_number = "SG"
	if ent_form[-2:] == "ες":
		ent_number = "PL"
		ent_gender = "FEM"
	elif ent_form[-1] == "ά":
		ent_number = "PL"
		ent_gender = "NEUT"

	if some_roman_chars(ent_form) or ent_form.isupper() or ent_form[-1] in ['β','γ','δ','ζ','κ','λ','μ','ν','ξ','π','ρ','τ','φ','χ','ψ']:
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
			words[i] = inflect(ent_form, f"N;GEN;{ent_number}", language='ell2')[0]
		ent_case = "GEN"
	elif "[Y.Acc]" in words:
		i = words.index('[Y.Acc]')
		if do_not_inflect:
			words[i] = ent_form
		else:
			words[i] = inflect(ent_form, f"N;ACC;{ent_number}", language='ell2')[0]
		ent_case = "ACC"
	# Now also check the correponsing articles, if they exist
	if "[DEF;Y]" in words:
		i = words.index('[DEF;Y]')
		words[i] = article[f"ART;DEF;{ent_gender};{ent_number};{ent_case}"]
	if "[DEF.Gen;Y]" in words:
		i = words.index('[DEF.Gen;Y]')
		words[i] = article[f"ART;DEF;{ent_gender};{ent_number};GEN"]
	if "[PREPDEF;Y]" in words:
		i = words.index('[PREPDEF;Y]')
		words[i] = article[f"ART;PREPDEF;{ent_gender};{ent_number};{ent_case}"]
	if "[INDEF;Y]" in words:
		i = words.index('[INDEF;Y]')
		#print(f"ART;INDEF;{ent_gender};{ent_number};{ent_case}")
		#print(article[f"ART;INDEF;{ent_gender};{ent_number};{ent_case}"])
		words[i] = article[f"ART;INDEF;{ent_gender};{ent_number};{ent_case}"]
	if "[DEF;Y.Fem]" in words:
		i = words.index('[DEF;Y.Fem]')
		words[i] = article[f"ART;DEF;FEM;{ent_number}"]
		
	
	return words

def fil_x(words, ent_form, ent_gender, article):
	#ent_form = entities[ent_id][0]
	#ent_gender = entities[ent_id][1].upper()
	ent_number = "SG"
	if ent_form[-2:] == "ες":
		ent_number = "PL"
		ent_gender = "FEM"
	elif ent_form[-1] == "ά":
		ent_number = "PL"
		ent_gender = "NEUT"

	if some_roman_chars(ent_form) or ent_form.isupper() or ent_form[-1] in ['β','γ','δ','ζ','κ','λ','μ','ν','ξ','π','ρ','τ','φ','χ','ψ']:
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
	elif "[X.Gen]" in words:
		i = words.index('[X.Gen]')
		if do_not_inflect:
			words[i] = ent_form
		else:
			words[i] = inflect(ent_form, f"N;GEN;{ent_number}", language='ell2')[0]
		ent_case = "GEN"
	elif "[X.Acc]" in words:
		i = words.index('[X.Acc]')
		if do_not_inflect:
			words[i] = ent_form
		else:
			words[i] = inflect(ent_form, f"N;ACC;{ent_number}", language='ell2')[0]
		ent_case = "ACC"

	# Now also check the correponsing articles, if the exist
	if "[DEF;X]" in words:
		i = words.index('[DEF;X]')
		words[i] = article[f"ART;DEF;{ent_gender};{ent_number};{ent_case}"]
	if "[DEF.Gen;X]" in words:
		i = words.index('[DEF.Gen;X]')
		words[i] = article[f"ART;DEF;{ent_gender};{ent_number};GEN"]
	if "[PREPDEF;X]" in words:
		i = words.index('[PREPDEF;X]')
		words[i] = article[f"ART;PREPDEF;{ent_gender};{ent_number};{ent_case}"]
		
	return words

def print_examples_for_relation(relation, entities, article):
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
						sentence = fil_x(list(words), entities[X_ent_id][0], entities[X_ent_id][1].upper(), article)
						sentence = fil_y(sentence, entities[Y_ent_id][0], entities[Y_ent_id][1].upper(), article)
						print("\t\t\t\t\t\t", ' '.join(sentence))
						count += 1
					else:
						count_double += 1
				if count == 10:
					break
			print(f"Found {count_double} entries with more than one word.")
	except:
		pass



entities, article = read_necessary_data()
# Read all relation templates
with jsonlines.open('relations.el.jsonl') as reader:
	count = 0
	for obj in reader:
		print_examples_for_relation(obj, entities, article)
		count += 1
		#if count == 4:
		#	break
		


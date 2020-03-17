
from unimorph_inflect import inflect
import jsonlines

def starts_with_vowel(w):
	w = w.strip()
	if w[0] in ['a','o','i','e','u', 'y', 'à', 'è', 'ù', 'é', 'â', 'ê', 'î', 'ô', 'û']:
		return True
	return False

def gender_heuristic(w):
	# Based on this: https://frenchtogether.com/french-nouns-gender/
	w = w.strip()
	if ' ' not in w:
		if w[-6:] in ["ouille"]:
			return "Fem"
		elif w[-5:] in ["aisse","ousse","aille","eille","ouche","anche"]:
			return "Fem"
		elif w[-4:] in ["asse","esse","isse","ance","anse","ence","once","enne","onne","aine","eine","erne","ande","ende","onde","arde","orde","euse","ouse","aise","oise","ache","iche","ehce","oche","uche","iere","eure","atte","otte","oute","orte","ante","ente","inte","onte","alle","elle","ille","olle","appe","ampe","ombe","igue"]:
			return "Fem"
		elif w[-4:] in ["aume","isme","ours","euil","ueil"]:
			return "Masc"
		elif w[-3:] in ["aie","oue","eue","ace","ece","ice","une","ine","ade","ase","ese","ise","yse","ose","use","ave","eve","ive","ete","ête"]:
			return "Fem"
		elif w[-3:] in ["and","ant","ent","int","ond","ont","eau","aud","aut","ais","ait","out","oux","age","ege","ème","ome","est","eul","all","air","erf","ert","arc","ars","art","our","ord","ors","ort","oir","eur","ail","eil","ing"]:
			return "Masc"
		elif w[-2:] in ["te","ée","ie","ue",]:
			return "Fem"
		elif w[-2:] in ["an","in","om","on","au","os","ot","ai","es","et","ou","il","it","is","at","as","us","ex","al","el","ol","if","ef","ac","ic","oc","uc","um","am","en"]:
			return "Masc"
		elif w[-1] == "o" or w[-1] == "i" or w[-1] == "y"  or w[-1] == "u":
			return "Masc"
		else:
			return "Masc"
			#ARGH
	else:
		w2 = w.split(' ')[0]
		if w2[-6:] in ["ouille"]:
			return "Fem"
		elif w2[-5:] in ["aisse","ousse","aille","eille","ouche","anche"]:
			return "Fem"
		elif w2[-4:] in ["asse","esse","isse","ance","anse","ence","once","enne","onne","aine","eine","erne","ande","ende","onde","arde","orde","euse","ouse","aise","oise","ache","iche","ehce","oche","uche","iere","eure","atte","otte","oute","orte","ante","ente","inte","onte","alle","elle","ille","olle","appe","ampe","ombe","igue"]:
			return "Fem"
		elif w2[-4:] in ["aume","isme","ours","euil","ueil"]:
			return "Masc"
		elif w2[-3:] in ["aie","oue","eue","ace","ece","ice","une","ine","ade","ase","ese","ise","yse","ose","use","ave","eve","ive","ete","ête"]:
			return "Fem"
		elif w2[-3:] in ["and","ant","ent","int","ond","ont","eau","aud","aut","ais","ait","out","oux","age","ege","ème","ome","est","eul","all","air","erf","ert","arc","ars","art","our","ord","ors","ort","oir","eur","ail","eil","ing"]:
			return "Masc"
		elif w2[-2:] in ["te","ée","ie","ue",]:
			return "Fem"
		elif w2[-2:] in ["an","in","om","on","au","os","ot","ai","es","et","ou","il","it","is","at","as","us","ex","al","el","ol","if","ef","ac","ic","oc","uc","um","am","en"]:
			return "Masc"
		elif w2[-1] == "o" or w2[-1] == "i" or w2[-1] == "y"  or w2[-1] == "u":
			return "Masc"
		else:
			return "Masc"
			#ARGH


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


	# Read list of greek entities
	with open("TREx_french.txt") as inp:
		lines = inp.readlines()

	entities = {}
	for l in lines:
		l = l.strip().split('\t')
		ent_id = l[0]
		ent_form = l[1]
		ent_gender = gender_heuristic(ent_form).upper()
		ent_number = "SG"
		ent_country = False
		ent_city = False
		ent_proper = True
		if ent_id in genders:
			if genders[ent_id] == "male":
				ent_gender = "MASC"
			elif genders[ent_id] == "female":
				ent_gender = "FEM"
			else:
				if ent_id in instanceof:
					if 'state' in instanceof[ent_id] or 'country' in instanceof[ent_id]:
						ent_country=True
						ent_proper = True
					elif 'business' in instanceof[ent_id]:
						ent_gender = "FEM"
						ent_proper = True
					elif 'enterprise' in instanceof[ent_id]:
						ent_gender = "FEM"
						ent_proper = True
					elif 'city' in instanceof[ent_id]:
						ent_city = True
						ent_proper = True
					#	if 'ι' != ent_form[-1] and 'ο' != ent_form[-1]:
					#		ent_gender = "FEM"
					# ARGH WHAT TO DO HERE
					elif 'human' in instanceof[ent_id]:
						ent_gender = "MASC" 
					elif 'island' in instanceof[ent_id]:
						ent_proper = True
					#	ent_gender = "FEM"
					#elif 'literary work' in instanceof[ent_id]:
					#	ent_gender = "NEUT"
					elif 'musical group' in instanceof[ent_id]:
						ent_proper = True
					#	ent_gender = "MASC"
					elif 'record label' in instanceof[ent_id]:
						ent_proper = True
					#	ent_gender = "FEM"
					elif 'language' in instanceof[ent_id]:
						ent_gender = "MASC"
					elif 'sports team' in instanceof[ent_id]:
						ent_gender = "FEM"
						ent_proper = True
					elif 'automobile manufacturer' in instanceof[ent_id]:
						ent_gender = "FEM"
						ent_proper = True
					elif 'football club' in instanceof[ent_id]:
						ent_gender = "FEM"
						ent_proper = True
					#elif '' in instanceof[ent_id]:
					#	ent_gender = "FEM"
					#elif '' in instanceof[ent_id]:
					#	ent_gender = "FEM"


		entities[ent_id] = (ent_form, ent_gender, ent_number, ent_country, ent_city, ent_proper)	


	with open("../lang_resource/fr/articles.txt") as inp:
		lines = inp.readlines()

	article = {}
	for l in lines:
		l = l.strip().split('\t')
		article[l[1]] = l[0]
	
	return entities, article


def fil_y(words, ent_form, ent_gender, ent_number, ent_country, ent_city, ent_proper, article):

	if ent_form.isupper():
		do_not_inflect = True
	else:
		do_not_inflect = False


	if '[Y]' in words:
		i = words.index('[Y]')
		words[i] = ent_form

	# Now also check the correponsing articles, if they exist
	# Now also check the correponding articles, if they exist
	if "[ARTDEF;Y]" in words:
		i = words.index('[ARTDEF;Y]')
		vowel = starts_with_vowel(words[i+1])
		if ent_proper and not ent_country:
			#Paul: If X is a proper noun (but not a country), then drop the article altogether
			del words[i]
		elif vowel:
			words[i] = article["ARTDEF;VOWEL"]
		elif ent_number == "SG":
			words[i] = article[f"ARTDEF;{ent_gender};{ent_number}"]
		else:
			words[i] = article[f"ARTDEF;{ent_number}"]
	if "[ARTIND;Y]" in words:
		i = words.index('[ARTIND;Y]')
		if ent_number == "SG":
			words[i] = article[f"ARTIND;{ent_gender};{ent_number}"]
		else:
			words[i] = article[f"ARTIND;{ent_number}"]
	if "[PREPDEF;Y]" in words:
		i = words.index('[PREPDEF;Y]')
		if ent_proper and not ent_country:
			words[i] = article[f"PREPDEF;PROPN"]
		elif ent_number == "SG":
			words[i] = article[f"PREPDEF;{ent_gender};{ent_number}"]
		else:
			words[i] = article[f"PREPDEF;{ent_number}"]
	if "[PREPDEF-à;Y]" in words:
		i = words.index('[PREPDEF-à;Y]')
		if ent_proper and not ent_country:
			#Paul: If Y is a proper noun (but not a country), then just à 
			words[i] = article[f"PREPDEF-à;PROPN"]
		elif ent_number == "SG":
			words[i] = article[f"PREPDEF-à;{ent_gender};{ent_number}"]
		else:
			words[i] = article[f"PREPDEF-à;{ent_number}"]
	if "[PREPLOC;Y]" in words:
		i = words.index('[PREPLOC;Y]')
		if ent_city:
			words[i] = article[f"PREPLOC;CITY"]
		elif ent_country:
			words[i] = article[f"PREPLOC;COUNTRY"]
		else:
			words[i] = article[f"PREPLOC;COUNTRY"]		
	
	return words

def fil_x(words, ent_form, ent_gender, ent_number, ent_country, ent_city, ent_proper, article):

	if ent_form.isupper():
		do_not_inflect = True
	else:
		do_not_inflect = False


	if '[X]' in words:
		i = words.index('[X]')
		words[i] = ent_form

	# Now also check the correponding articles, if they exist
	if "[ARTDEF;X]" in words:
		i = words.index('[ARTDEF;X]')
		vowel = starts_with_vowel(words[i+1])
		if ent_proper and not ent_country:
			#Paul: If X is a proper noun (but not a country), then drop the article altogether
			del words[i]
		elif vowel:
			words[i] = article["ARTDEF;VOWEL"]
		elif ent_number == "SG":
			words[i] = article[f"ARTDEF;{ent_gender};{ent_number}"]
		else:
			words[i] = article[f"ARTDEF;{ent_number}"]
	if "[PREPDEF;X]" in words:
		i = words.index('[PREPDEF;X]')
		if ent_proper and not ent_country:
			words[i] = article[f"PREPDEF;PROPN"]
		elif ent_number == "SG":
			words[i] = article[f"PREPDEF;{ent_gender};{ent_number}"]
		else:
			words[i] = article[f"PREPDEF;{ent_number}"]
	if "[PREPDEF-à;X]" in words:
		i = words.index('[PREPDEF-à;X]')
		if ent_proper and not ent_country:
			#Paul: If Y is a proper noun (but not a country), then just à 
			words[i] = article[f"PREPDEF-à;PROPN"]
		elif ent_number == "SG":
			words[i] = article[f"PREPDEF-à;{ent_gender};{ent_number}"]
		else:
			words[i] = article[f"PREPDEF-à;{ent_number}"]

	# Now also check the corresponfing verbs, if they exist.
	# Needed for subject-verb agreement
	for i,w in enumerate(words):
		if w[0] == '[' and 'X-Gender' in w:
			if '|' in w:
				options = w.strip()[1:-1].split('|')
				if ent_gender == "MASC":
					form = options[0].strip().split(';')[0]
					words[i] = form
				else:
					form = options[1].strip().split(';')[0]
					words[i] = form
		
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
						sentence = fil_x(list(words), entities[X_ent_id][0], entities[X_ent_id][1].upper(), entities[X_ent_id][2].upper(), entities[X_ent_id][3], entities[X_ent_id][4], entities[X_ent_id][5], article)
						sentence = fil_y(sentence, entities[Y_ent_id][0], entities[Y_ent_id][1].upper(), entities[Y_ent_id][2].upper(), entities[Y_ent_id][3], entities[Y_ent_id][4], entities[Y_ent_id][5], article)
						print("\t", ' '.join(sentence))
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
with jsonlines.open('relations.fr.jsonl') as reader:
	count = 0
	for obj in reader:
		print_examples_for_relation(obj, entities, article)
		count += 1
		#if count == 4:
		#	break
		


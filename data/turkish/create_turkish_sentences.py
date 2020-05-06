
from unimorph_inflect import inflect
import jsonlines


def read_necessary_data():

	# Read list of entity IDs with their instanceof or subclassof information, if known
	with open("../TREx_instanceof.txt") as inp:
		lines = inp.readlines()

	instanceof = {}
	for l in lines:
		l = l.strip().split('\t')
		ent_id = l[0]
		instanceof[ent_id] = ','.join(l[1:])


	# Read list of greek entities
	with open("TREx_turkish.txt") as inp:
		lines = inp.readlines()

	entities = {}
	for l in lines:
		l = l.strip().split('\t')
		ent_id = l[0]
		ent_form = l[1]
		ent_number = "SG"
		'''
		if ent_id in genders:
			if genders[ent_id] == "male":
				ent_gender = "MASC"
			elif genders[ent_id] == "female":
				ent_gender = "FEM"
			else:
				if ent_id in instanceof and ent_gender == "NEUT":
					if 'state' in instanceof[ent_id] or 'country' in instanceof[ent_id]:
						if 'ν' != ent_form[-1] and 'ο' != ent_form[-1]  and 'ό' != ent_form[-1]:
							ent_gender = "FEM"
					elif 'business' in instanceof[ent_id]:
						ent_gender = "FEM"
					elif 'enterprise' in instanceof[ent_id]:
						ent_gender = "FEM"
					elif 'city' in instanceof[ent_id]:
						if 'ι' != ent_form[-1] and 'ο' != ent_form[-1]:
							ent_gender = "FEM"
					# ARGH WHAT TO DO HERE
					elif 'human' in instanceof[ent_id]:
						ent_gender = "MASC" 
					elif 'island' in instanceof[ent_id]:
						ent_gender = "FEM"
					elif 'literary work' in instanceof[ent_id]:
						ent_gender = "NEUT"
					elif 'musical group' in instanceof[ent_id]:
						ent_gender = "MASC"
						ent_number = "PL"
					elif 'record label' in instanceof[ent_id]:
						ent_gender = "FEM"
					elif 'language' in instanceof[ent_id]:
						ent_gender = "NEUT"
						ent_number = "PL"
					elif 'sports team' in instanceof[ent_id]:
						ent_gender = "FEM"
					elif 'automobile manufacturer' in instanceof[ent_id]:
						ent_gender = "FEM"
					elif 'football club' in instanceof[ent_id]:
						ent_gender = "FEM"
					#elif '' in instanceof[ent_id]:
					#	ent_gender = "FEM"
					#elif '' in instanceof[ent_id]:
					#	ent_gender = "FEM"
		'''

		entities[ent_id] = (ent_form, ent_number)	
	
	return entities

def fix_up(w, inds):
	out = ''
	for i,c in enumerate(w):
		if i in inds:
			out += c.upper()
		else:
			out += c
	return out

def add_be(w, number):
	change = 'ç f h k p s ş t'.split()
	vowels = "a,o,u,ı,e,ü,ö,i".split(',')
	undotted = "a,o,u,ı".split(',')
	dotted = "e,ü,ö,i".split(',')
	
	last_vowel = None
	i = len(w)-1
	while i > 0:
		if w[i] in vowels:
			last_vowel = w[i]
			break
		else:
			i -= 1

	first = '\'d'
	if w[-1] in change:
		first = '\'t'

	end = first + 'ir'
	if number == "PL":
		end += "lar"

	if last_vowel in undotted:
		if last_vowel == "a" or last_vowel == 'ı':
			end = first + "ır"
			if number == "PL":
				end += "lar"
		elif last_vowel == "o" or last_vowel == 'u':
			end = first + "ur"
			if number == "PL":
				end += "lar"
	elif last_vowel in dotted:
		elif last_vowel == "e" or last_vowel == 'i':
			end = first + "ir"
			if number == "PL":
				end += "ler"
		elif last_vowel == "ö" or last_vowel == 'ü':
			end = first + "ür"
			if number == "PL":
				end += "ler"
	return w+end

	


def fil_y(words, ent_form, ent_number):

	inds = [i for i,x in enumerate(ent_form) if x.isupper()]
	if inds:
		ent_form2 = ent_form.lower()
	else:
		ent_form2 = ent_form


	if '[Y]' in words:
		i = words.index('[Y]')
		words[i] = ent_form
		ent_case = "NOM"
	elif "[Y.Loc]" in words:
		i = words.index('[Y.Loc]')
		temp = inflect(ent_form2, f"N;LOC;{ent_number}", language='tur')[0]
		if inds:
			words[i] = fix_up(temp, inds)
		else:
			words[i] = temp
		ent_case = "LOC"
	elif "[Y.Gen]" in words:
		i = words.index('[Y.Gen]')
		temp = inflect(ent_form2, f"N;GEN;{ent_number}", language='tur')[0]
		if inds:
			words[i] = fix_up(temp, inds)
		else:
			words[i] = temp
		ent_case = "GEN"
	elif "[Y.Acc]" in words:
		i = words.index('[Y.Acc]')
		temp = inflect(ent_form2, f"N;ACC;{ent_number}", language='tur')[0]
		if inds:
			words[i] = fix_up(temp, inds)
		else:
			words[i] = temp
		ent_case = "ACC"
	elif "[Y.Dat]" in words:
		i = words.index('[Y.Dat]')
		temp = inflect(ent_form2, f"N;DAT;{ent_number}", language='tur')[0]
		if inds:
			words[i] = fix_up(temp, inds)
		else:
			words[i] = temp
		ent_case = "DAT"
	elif "[Y.Abl]" in words:
		i = words.index('[Y.Abl]')
		temp = inflect(ent_form2, f"N;ABL;{ent_number}", language='tur')[0]
		if inds:
			words[i] = fix_up(temp, inds)
		else:
			words[i] = temp
		ent_case = "ABL"

	if '[Y;be]' in words:
		i = words.index('[Y;be]')
		words[i] = add_be(ent_form, ent_number)

	
	return words

def fil_x(words, ent_form, ent_number):
	
	inds = [i for i,x in enumerate(ent_form) if x.isupper()]
	if inds:
		ent_form2 = ent_form.lower()
	else:
		ent_form2 = ent_form



	if '[X]' in words:
		i = words.index('[X]')
		words[i] = ent_form
		ent_case = "NOM"
	elif "[X.Loc]" in words:
		i = words.index('[X.Loc]')
		temp = inflect(ent_form2, f"N;LOC;{ent_number}", language='tur')[0]
		if inds:
			words[i] = fix_up(temp, inds)
		else:
			words[i] = temp
		ent_case = "LOC"
	elif "[X.Gen]" in words:
		i = words.index('[X.Gen]')
		temp = inflect(ent_form2, f"N;GEN;{ent_number}", language='tur')[0]
		if inds:
			words[i] = fix_up(temp, inds)
		else:
			words[i] = temp
		ent_case = "GEN"
	elif "[X.Acc]" in words:
		i = words.index('[X.Acc]')
		temp = inflect(ent_form2, f"N;ACC;{ent_number}", language='tur')[0]
		if inds:
			words[i] = fix_up(temp, inds)
		else:
			words[i] = temp
		ent_case = "ACC"
	elif "[X.Dat]" in words:
		i = words.index('[X.Dat]')
		temp = inflect(ent_form2, f"N;DAT;{ent_number}", language='tur')[0]
		if inds:
			words[i] = fix_up(temp, inds)
		else:
			words[i] = temp
		ent_case = "DAT"
	elif "[X.Abl]" in words:
		i = words.index('[X.Abl]')
		temp = inflect(ent_form2, f"N;ABL;{ent_number}", language='tur')[0]
		if inds:
			words[i] = fix_up(temp, inds)
		else:
			words[i] = temp
		ent_case = "ABL"

	if '[X;be]' in words:
		i = words.index('[X;be]')
		words[i] = add_be(ent_form, ent_number)
	
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
with jsonlines.open('relations.tr.jsonl') as reader:
	count = 0
	for obj in reader:
		print_examples_for_relation(obj, entities)
		count += 1
		#if count == 4:
		#	break
		


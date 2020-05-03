import sys

with open("../TREx_unicode_escape.txt") as inp:
	lines = inp.readlines()


with open("TREx_vietnamese.txt", 'w') as inp:
	for l in lines:
		if "@vi" in l:
			l = l.strip().split('\t')
			ent = l[0]
			for entry in l:
				if entry[-3:] == "@vi":
					form = entry[1:-4]
					inp.write(f"{ent}\t{form}\n")


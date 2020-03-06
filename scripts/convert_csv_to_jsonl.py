import jsonlines
import sys

inpfile = sys.argv[1]
outfile = sys.argv[2]

with open(inpfile) as inp:
	lines = inp.readlines()


with jsonlines.open(outfile, mode='w') as writer:
	for line in lines[1:]:
		line = line.strip().split('\t')
		obj = {}
		obj['relation'] = line[0]
		obj['label'] = line[1]
		obj['description'] = line[2]
		obj['type'] = line[3]
		obj['template'] = line[4]
		writer.write(obj)
		#print(f"{relation}\t{label}\t{desc}\t{tp}\t{template}")

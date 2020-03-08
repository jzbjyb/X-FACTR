
import sys

inpfile = sys.argv[1]
goalsfile = sys.argv[2]
cleanfile = sys.argv[3]


with open(inpfile) as inp:
	lines = inp.readlines()


with open(goalsfile, 'w') as inp:
	with open(cleanfile, 'w') as outp:
		for ll in lines:
			l = ll.strip().split('\t')[1]
			rel = ll.strip().split('\t')[0]

			first = 0
			second = 0
			found_first = False
			found_second = False

			if l:
				words = l.split()
				indexes = []
				for i,w in enumerate(words):
					if w == '[[':
						indexes.append(i)
						if not first:
							first = i
						elif not second:
							second = i-2
					#elif  w == ']]':
					#	indexes.append(i)
					if w.startswith(']]_x'):
						indexes.append(i)
						if first and not second:
							if i-2 == first:
								goal_x = f"{first+1}"
							else:
								goal_x = f"{first+1}-{i-1}"
						if second:
							if i-4 == second:
								goal_x = f"{second+1}"
							else:
								goal_x = f"{second+1}-{i-3}"
					elif w.startswith(']]_y'):
						indexes.append(i)
						if first and not second:
							if i-2 == first:
								goal_y = f"{first+1}"
							else:
								goal_y = f"{first+1}-{i-1}"
						if second:
							if i-4 == second:
								goal_y = f"{second+1}"
							else:
								goal_y = f"{second+1}-{i-3}"

				for i in reversed(sorted(indexes)):
					del words[i]
				outp.write(' '.join(words) + '\n')
			inp.write(f"{rel}\t{goal_x}\t{goal_y}\n")





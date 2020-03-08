import sys
import pyconll

def read_goals(f):
	with open(f) as inp:
		lines = inp.readlines()
	relations = []
	x = []
	y = []
	for l in lines:
		if l.strip():
			l = l.strip().split('\t')
			relations.append(l[0])
			x.append(l[1])
			y.append(l[2])
	return relations,x,y

def convert_tree_to_graph(sentence):
	d = {}
	for token in sentence:
		token_id = token.id
		token_head = token.head
		if token_head:
			if token_id in d:
				d[token_id].append(token_head)
			else:
				d[token_id] = [token_head]
			if token_head in d:
				d[token_head].append(token_id)
			else:
				d[token_head] = [token_id]
	return d

def get_all_phrase_ids(sentence, range_start, range_end):
	search_space = list(range(range_start,range_end+1))
	for i,s in enumerate(search_space):
		search_space[i] = str(s)
	return search_space

def find_head_of_phrase(sentence, range_start, range_end):
	search_space = list(range(range_start,range_end+1))
	for i,s in enumerate(search_space):
		search_space[i] = str(s)
	#print(f"Search Space: {search_space}")
	possible_answers = []
	try:
		for i in search_space:
			#print(i,sentence[i].id, sentence[i].form, sentence[i].head)
			if sentence[i].head not in search_space:
				possible_answers.append(i)
		if len(possible_answers) > 1:
			return min(possible_answers)
		elif len(possible_answers) == 0:
			return -1
		return possible_answers[0]
	except:
		return -1

def find_immediate_dependents(sentence, pos):
	deps = []
	for token in sentence:
		if token.head:
			if token.head == pos:
				deps.append(token.id)
	return deps



# finds shortest path between 2 nodes of a graph using BFS
def bfs_shortest_path(graph, start, goal):
	# keep track of explored nodes
	explored = []
	# keep track of all the paths to be checked
	queue = [[start]]
	# return path if start is goal
	if start == goal:
		return -1
	# keeps looping until all possible paths have been checked
	while queue:
		# pop the first path from the queue
		path = queue.pop(0)
		# get the last node from the path
		node = path[-1]
		if node not in explored:
			neighbours = graph[node]
			# go through all neighbour nodes, construct a new path and
			# push it into the queue
			for neighbour in neighbours:
				new_path = list(path)
				new_path.append(neighbour)
				queue.append(new_path)
				# return path if neighbour is goal
				if neighbour == goal:
					return new_path
			# mark node as explored
			explored.append(node)
	# in case there's no path between the 2 nodes
	return -1


PARSED_FILE = sys.argv[1]
GOALS_FILE = sys.argv[2]
OUTPUT_FILE = sys.argv[3]

parsed_data = pyconll.load_from_file(PARSED_FILE)
relations, x_start, y_goal = read_goals(GOALS_FILE)
DEBUG = False

with open(OUTPUT_FILE, 'w') as outp:
	for i,sentence in enumerate(parsed_data):
		try:
			# Convert the tree to a graph representation for BFS
			graph = convert_tree_to_graph(sentence)
			if DEBUG:
				print(f"Sentence {i}")
				print("Graph:")
				print(graph)

			# Get the start (X) and the goal (Y)
			start = x_start[i]
			target = y_goal[i]
			# These will keep all the ids of the entity if more than one
			s_start = []
			s_target = []
			if '-' in start:
				temp = start.strip().split('-')
				start = find_head_of_phrase(sentence, int(temp[0]), int(temp[1]))
				s_start = get_all_phrase_ids(sentence, int(temp[0]), int(temp[1]))
			if '-' in target:
				temp = target.strip().split('-')
				target = find_head_of_phrase(sentence, int(temp[0]), int(temp[1]))
				s_target = get_all_phrase_ids(sentence, int(temp[0]), int(temp[1]))
			if DEBUG:
				print(f"\tStart: {start} -- {sentence[start].form}")
				print(f"\tTarget: {target} -- {sentence[target].form}")
			
			# Find the shortest path between (x) and (Y)
			path = bfs_shortest_path(graph, start, target)
			if DEBUG:
				print(path)

			# If we managed to find a path...
			if path != -1:
				# Get the immediate dependents of the entities
				start_deps = find_immediate_dependents(sentence, start)
				target_deps = find_immediate_dependents(sentence, target)

				# We want to keep:
				# a) the shortest path
				path2 = [int(p) for p in path]
				# b) all the ids of the entities and their immediaate dependents
				path2 += list(set([int(p) for p in s_start + s_target + start_deps + target_deps if int(p) not in path2]))
				# Sort them for presentation
				path2 = [str(p) for p in sorted(path2)]
				if DEBUG:
					for k in path2:
						print(f"\t\t{sentence[k].id}\t{sentence[k].form}\t{sentence[k].head}")

				# Create template form
				template = []
				for k in path2:
					if sentence[k].id in s_start or sentence[k].id == start:
						if "[X]" not in template:
							template.append("[X]")
					elif sentence[k].id in s_target or sentence[k].id == target:
						if "[Y]" not in template:
							template.append("[Y]")
					else:
						template.append(sentence[k].form)

				outp.write(f"{relations[i]}\t{' '.join(template)}\n")
			else:
				outp.write('\n')
		except:
			outp.write('\n')

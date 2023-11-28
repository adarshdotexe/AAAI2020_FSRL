import json
import random
import numpy as np

def train_generate(datapath, batch_size, few, symbol2id, ent2id, e1rel_e2):
	train_tasks = json.load(open(datapath + '/train_tasks.json'))
	rel2candidates = json.load(open(datapath + '/rel2candidates_all.json'))
	ent_embed = np.loadtxt(datapath + '/embed/entity2vec.' + 'TransE')
	# for i in ['DistMult', 'TransE', 'ComplEx', 'RESCAL']:
	# 	ent_embed.append(np.loadtxt(datapath + '/embed/entity2vec.' + i))
	task_pool = list(train_tasks.keys())
	#print (task_pool[0])

	num_tasks = len(task_pool)
	t = 0
	# for query_ in train_tasks.keys():
	# 	print len(train_tasks[query_])
	# 	if len(train_tasks[query_]) < 4:
	# 		print len(train_tasks[query_])

	# data contains score of each noise entity for each e1 in each relation
	# data[rel][e1][noise] = score
	data = json.load(open(datapath + '/data.json'))

	rel_idx = 0
	while True:
		if rel_idx % num_tasks == 0:
			random.shuffle(task_pool)
		query = task_pool[rel_idx % num_tasks]
		#print (query)
		rel_idx += 1

		#query_rand = random.randint(0, (num_tasks - 1))
		#query = task_pool[query_rand]

		candidates = rel2candidates[query]
		#print rel_idx

		if len(candidates) <= 20:
			continue

		train_and_test = train_tasks[query]
		random.shuffle(train_and_test)

		support_triples = train_and_test[:few]
		support_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in support_triples]

		support_left = [ent2id[triple[0]] for triple in support_triples]
		support_right = [ent2id[triple[2]] for triple in support_triples]

		all_test_triples = train_and_test[few:]

		if len(all_test_triples) == 0:
			continue

		# Instead of randomly sampling, we use the hardest negative sample
		# if len(all_test_triples) < batch_size:
		# 	query_triples = [random.choice(all_test_triples) for _ in range(batch_size)]
		# else:
		# 	query_triples = random.sample(all_test_triples, batch_size)

		if len(all_test_triples) < batch_size:
			query_triples = all_test_triples.extend(random.sample(candidates, batch_size - len(all_test_triples)))
		else:
			query_triples = all_test_triples.sort(key=lambda x: sum(data[query][x[0]]), reverse=True)[:batch_size]

		query_pairs = [[symbol2id[triple[0]], symbol2id[triple[2]]] for triple in query_triples]

		query_left = [ent2id[triple[0]] for triple in query_triples]
		query_right = [ent2id[triple[2]] for triple in query_triples]

		false_pairs = []
		false_left = []
		false_right = []
		for triple in query_triples:
			e_h = triple[0]
			noise = random.choice(list(data[query][e_h].keys()))
			false_pairs.append([symbol2id[e_h], symbol2id[noise]])
			false_left.append(ent2id[e_h])
			false_right.append(ent2id[noise])

		yield support_pairs, query_pairs, false_pairs, support_left, support_right, query_left, query_right, false_left, false_right
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
	data = dict()
	for rel in task_pool:
		print("\t\tRel: ", rel)
		data[rel] = dict()
		for triple in train_tasks[rel]:
			e1 = triple[0]
			rel = triple[1]
			e2 = triple[2]
			for noise in rel2candidates[rel]:
				if data[rel].get(e1) is None:
					data[rel][e1] = dict()
				if noise not in e1rel_e2[e1+rel] and noise != e2:
					e1_embed = list(ent_embed[ent2id[e1],:])
					noise_embed = list(ent_embed[ent2id[noise],:])
					curr = np.dot(e1_embed, noise_embed)
					if curr > t:
						data[rel][e1][noise] = curr
			# print(data[rel][e1])

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

		if rel_idx % 10000 == 0:
			t+=0.05
			print("\t\tThreshold: ", t)
			for rel in task_pool:
				for e1 in data[rel].keys():
					for noise in list(data[rel][e1].keys()):
						e1_embed = list(ent_embed[ent2id[e1],:])
						noise_embed = list(ent_embed[ent2id[noise],:])
						curr = np.dot(e1_embed, noise_embed)
						if curr > t:
							data[rel][e1][noise] = curr
						else:
							if len(data[rel][e1]) > 1:
								del data[rel][e1][noise]

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

		if len(all_test_triples) < batch_size:
			query_triples = [random.choice(all_test_triples) for _ in range(batch_size)]
		else:
			query_triples = random.sample(all_test_triples, batch_size)

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
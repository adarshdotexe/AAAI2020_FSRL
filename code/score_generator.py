import json
import numpy as np
from collections import defaultdict
from tqdm import tqdm

datapath = "/content/AAAI2020_FSRL/data/NELL"
train_tasks = json.load(open(datapath + '/train_tasks.json'))
rel2candidates = json.load(open(datapath + '/rel2candidates_all.json'))
ent_embed = np.loadtxt(datapath + '/embed/entity2vec.' + 'TransE')
rel2id = json.load(open(datapath + '/relation2ids'))
ent2id = json.load(open(datapath + '/ent2ids'))
e1rel_e2 = json.load(open(datapath + '/e1rel_e2.json'))

task_pool = list(train_tasks.keys())
num_tasks = len(task_pool)
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
        e2_embed = list(ent_embed[ent2id[e2],:])
        curr = np.dot(e2_embed, noise_embed)
        for e3 in e1rel_e2[e1+rel]:
          e3_embed = list(ent_embed[ent2id[e3],:])
          noise_embed = list(ent_embed[ent2id[noise],:])
          curr += np.dot(e3_embed, noise_embed)
        data[rel][e1][noise] = curr

json.dump(data, open(datapath + '/data.json', 'w'))

# Sort data for each e1 and relation
for rel in task_pool:
  print("\t\tRel: ", rel)
  for e1 in tqdm(data[rel].keys()):
    data[rel][e1] = sorted(data[rel][e1].items(), key=lambda kv: kv[1], reverse=True)
json.dump(data, open(datapath + '/data.json', 'w'))
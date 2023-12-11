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
      if noise not in e1rel_e2[e1+rel] and noise != e2 and noise not in list(data[rel][e1].keys()):
        e1_embed = np.array(ent_embed[ent2id[e1],:])
        e2_embed = np.array(ent_embed[ent2id[e2],:])
        curr = 0
        noise_embed = np.array(ent_embed[ent2id[noise],:])
        temp = np.dot(e1_embed, noise_embed)
        for e3 in e1rel_e2[e1+rel]:
          e3_embed = np.array(ent_embed[ent2id[e3],:])
          curr += float(max(float(np.dot(noise_embed, e3_embed)), float(temp)))
        data[rel][e1][noise] = curr

json.dump(data, open(datapath + '/data.json', 'w'))
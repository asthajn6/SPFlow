

import numpy as np

from spn.algorithms.StructureLearning import get_next_operation, learn_structure
from spn.algorithms.CnetStructureLearning import get_next_operation_cnet, learn_structure_cnet
from spn.algorithms.Validity import is_valid
from spn.algorithms.Statistics import get_structure_stats_dict

from spn.structure.Base import Sum, assign_ids

from spn.structure.leaves.histogram.Histograms import create_histogram_leaf
from spn.structure.leaves.parametric.Parametric import create_parametric_leaf
from spn.structure.leaves.piecewise.PiecewiseLinear import create_piecewise_leaf
from spn.structure.leaves.cltree.CLTree import create_cltree_leaf
from spn.algorithms.splitting.Conditioning import (
	get_split_rows_naive_mle_conditioning,
	get_split_rows_random_conditioning,
)

from spn.algorithms.splitting.Clustering import get_split_rows_XMeans
from spn.algorithms.splitting.RDC import get_split_cols_single_RDC_py, get_split_cols_distributed_RDC_py

import logging

logger = logging.getLogger(__name__)


import warnings

warnings.filterwarnings('ignore')



import pandas as pd
from spn.structure.Base import Context
from spn.structure.StatisticalTypes import MetaType
import matplotlib.pyplot as plt
from os import path as pth
import sys, os
import random

cols="rdc"
rows="kmeans"
min_instances_slice=200
threshold=0.3
ohe=False
leaves = create_histogram_leaf
rand_gen=None
cpus=-1


datasets = ["nltcs", "plants", "baudio", "jester", "bnetflix"]
#datasets = ["nltcs","msnbc", "plants", "kdd", "baudio", "jester", "bnetflix"]
#datasets = ["kdd"]
path = "test1"



for dataset in datasets:
	
	print(f"\n\n\n{dataset}\n\n\n")
	plot_path = f"{path}/{dataset}"
	if not pth.exists(plot_path):
		try:
			os.makedirs(plot_path)
		except OSError:
			print ("Creation of the directory %s failed" % plot_path)
			sys.exit()
			
	df = pd.read_csv(f"spn/data/binary/{dataset}.ts.data", sep=',')
	data = df.values
	print(data.shape)
	max_iter = data.shape[1]
	samples, var = data.shape
	ds_context = Context(meta_types=[MetaType.DISCRETE]*var)
	ds_context.add_domains(data)

	df2 = pd.read_csv(f"spn/data/binary/{dataset}.test.data", sep=',')
	test = random.sample(list(df2.values), 1500)
	print(test.shape)

	ll = list()
	nodes = list()
	k_limit = 2 #[i for i in range(1,5)]
	past3 = list()
	
	n = int(max_iter**0.5)  #[i for i in range(int(max_iter**0.5),max_iter+1,2)]
	step = (max_iter - (max_iter**0.5))/20

	i = 0
	while True:
		split_cols = get_split_cols_distributed_RDC_py(rand_gen=rand_gen, ohe=ohe, n_jobs=cpus, n=round(n))
		split_rows = get_split_rows_XMeans(limit=k_limit, returnk=False)
		nextop = get_next_operation(min_instances_slice)

		spn = learn_structure(data, ds_context, split_rows, split_cols, leaves, nextop)

		nodes.append(get_structure_stats_dict(spn)["nodes"])
		from spn.io.Graphics import plot_spn

		plot_spn(spn, f'{path}/{dataset}/spn{i}.png')

		from spn.algorithms.Inference import log_likelihood
		total_ll = 0
		for j, instance in enumerate(test):
			test_data = np.array(instance).reshape(-1, var)
			total_ll += log_likelihood(spn, test_data)[0][0]
			printProgressBar(j+1, len(test), prefix = 'Evaluation Progress:', suffix = 'Complete', length = 50)
		ll.append(total_ll/len(test))

		
		print("\n\n\n\n\n")
		print(f"X-Means Limit: {k_limit}, \tVariables for splitting: {round(n)}")
		print("#Nodes: ",nodes[i])
		print("Log-likelihood: ",ll[i])
		print(ll)
		print(nodes)
		print("\n\n\n\n\n")
		
		plt.close()
		# plot line 
		plt.plot(ll, marker="o") 
		plt.title(f"{dataset} Log Likelihood")
		plt.savefig(f"{path}/{dataset}/ll.png", dpi=100)
		plt.close()
		plt.plot(nodes, marker="o") 
		plt.title(f"{dataset} Nodes")
		plt.savefig(f"{path}/{dataset}/nodes.png", dpi=100)
		plt.close()

		f = open(f"{path}/{dataset}/stats.txt", "w")
		f.write(f"\n\tLog Likelihood: {ll}")
		f.write(f"\n\t\tNodes: {nodes}")
		f.close()
		
		past3 = ll[-min(len(ll),3):]
				
		if n>=max_iter and round(np.std(past3), 3) <= 0.001:
			break
		
		i+=1
		n = min(n+step, max_iter)
		k_limit += 1

	print("Log Likelihood",ll)
	print("Nodes",nodes)

	plt.close()
	# plot line 
	plt.plot(ll, marker="o") 
	#plt.show()
	plt.title(f"{dataset} Log Likelihood")
	plt.savefig(f"{path}/{dataset}/ll.png", dpi=100)
	plt.close()
	plt.plot(nodes, marker="o") 
	#plt.show()
	plt.title(f"{dataset} Nodes")
	plt.savefig(f"{path}/{dataset}/nodes.png", dpi=100)
	plt.close()



def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
   
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()
"""
Created on March 28, 2019
@author: Hari Teja Tatavarti

"""
from spn.structure.Base import Sum, Product, Max
from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up
from spn.algorithms.LearningWrappers import learn_parametric_aspmn, learn_mspn_for_aspmn
from spn.algorithms.splitting.RDC import get_split_cols_distributed_RDC_py1, get_split_cols_RDC_py, get_split_cols_single_RDC_py
from spn.algorithms.SPMNHelper import *
from spn.algorithms.MEU import meu
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.Statistics import get_structure_stats_dict
from spn.io.Graphics import plot_spn
from spn.io.ProgressBar import printProgressBar
from spn.data.simulator import get_env
from spn.algorithms.MEU import best_next_decision
import logging
import numpy as np
import matplotlib.pyplot as plt
from os import path as pth
import sys, os

from spn.algorithms.TransformStructure import Prune


class Anytime_SPMN:

	def __init__(self, dataset, output_path, partial_order, decision_nodes, utility_node, feature_names, feature_labels,
			meta_types, cluster_by_curr_information_set=False, util_to_bin=False):

		self.dataset = dataset
		self.params = SPMNParams(
				partial_order,
				decision_nodes,
				utility_node,
				feature_names,
				feature_labels,
				meta_types,
				util_to_bin
			)
		self.op = 'Any'
		self.cluster_by_curr_information_set = cluster_by_curr_information_set
		self.spmn = None

		self.vars = len(feature_labels)

		self.plot_path = f"{output_path}/{dataset}"


		if not pth.exists(self.plot_path):
			try:
				os.makedirs(self.plot_path)
			except OSError:
				print ("Creation of the directory %s failed" % self.plot_path)
				sys.exit()


	def set_next_operation(self, next_op):
		self.op = next_op

	def get_curr_operation(self):
		return self.op

	def __learn_spmn_structure(self, remaining_vars_data, remaining_vars_scope,
							   curr_information_set_scope, index):

		logging.info(f'start of new recursion in __learn_spmn_structure method of SPMN')
		logging.debug(f'remaining_vars_scope: {remaining_vars_scope}')
		logging.debug(f'curr_information_set_scope: {curr_information_set_scope}')

		# rest set is remaining variables excluding the variables in current information set
		rest_set_scope = [var_scope for var_scope in remaining_vars_scope if
						  var_scope not in curr_information_set_scope]
		logging.debug(f'rest_set_scope: {rest_set_scope}')

		scope_index = sum([len(x) for x in self.params.partial_order[:index]])
		next_scope_index = sum([len(x) for x in self.params.partial_order[:index + 1]])

		if remaining_vars_scope == curr_information_set_scope:
			# this is last information set in partial order. Base case of recursion

			# test if current information set is a decision node
			if self.params.partial_order[index][0] in self.params.decision_nodes:
				raise Exception(f'last information set of partial order either contains random '
								f'and utility variables or just a utility variable. '
								f'This contains decision variable: {self.params.partial_order[index][0]}')

			else:
				# contains just the random and utility variables

				logging.info(f'at last information set of this recursive call: {curr_information_set_scope}')
				ds_context_last_information_set = get_ds_context(remaining_vars_data,
																 remaining_vars_scope, self.params)

				if self.params.util_to_bin:

					last_information_set_spn = learn_parametric_aspmn(remaining_vars_data,
																ds_context_last_information_set,
																n=self.n,
																k_limit=self.limit,
																min_instances_slice=20,
																initial_scope=remaining_vars_scope)

				else:

					last_information_set_spn = learn_mspn_for_aspmn(remaining_vars_data,
																   ds_context_last_information_set,
																   n=self.n,
																   k_limit=self.limit,
																   min_instances_slice=20,
																   initial_scope=remaining_vars_scope)

			logging.info(f'created spn at last information set')
			return last_information_set_spn

		# test for decision node. test if current information set is a decision node
		elif self.params.partial_order[index][0] in self.params.decision_nodes:

			decision_node = self.params.partial_order[index][0]

			logging.info(f'Encountered Decision Node: {decision_node}')

			# cluster the data from remaining variables w.r.t values of decision node
			#clusters_on_next_remaining_vars, dec_vals = anytime_split_on_decision_node(remaining_vars_data, self.d)
			clusters_on_next_remaining_vars, dec_vals = split_on_decision_node(remaining_vars_data)

			decision_node_children_spns = []
			index += 1

			next_information_set_scope = np.array(range(next_scope_index, next_scope_index +
														len(self.params.partial_order[index]))).tolist()

			next_remaining_vars_scope = rest_set_scope
			self.set_next_operation('Any')

			logging.info(f'split clusters based on decision node values')
			for cluster_on_next_remaining_vars in clusters_on_next_remaining_vars:

				decision_node_children_spns.append(self.__learn_spmn_structure(cluster_on_next_remaining_vars,
																			   next_remaining_vars_scope,
																			   next_information_set_scope, index
																			   ))

			decision_node_spn_branch = Max(dec_idx=scope_index, dec_values=dec_vals,
										   children=decision_node_children_spns, feature_name=decision_node)

			assign_ids(decision_node_spn_branch)
			rebuild_scopes_bottom_up(decision_node_spn_branch)
			logging.info(f'created decision node')
			return decision_node_spn_branch

		# testing for independence
		else:

			curr_op = self.get_curr_operation()
			logging.debug(f'curr_op at prod node (independence test): {curr_op}')

			if curr_op != 'Sum':    # fails if correlated variable set found in previous recursive call.
									# Without this condition code keeps looping at this stage

				ds_context = get_ds_context(remaining_vars_data, remaining_vars_scope, self.params)

				#split_cols = get_split_cols_single_RDC_py(rand_gen=None, ohe=False, n_jobs=-1, n=round(self.n))
				split_cols = get_split_cols_distributed_RDC_py1(rand_gen=None, ohe=False, n_jobs=-1, n=round(self.n))
				#split_cols = get_split_cols_RDC_py()
				data_slices_prod = split_cols(remaining_vars_data, ds_context, remaining_vars_scope, rest_set_scope)

				logging.debug(f'{len(data_slices_prod)} slices found at data_slices_prod: ')

				prod_children = []
				next_remaining_vars_scope = []
				independent_vars_scope = []

				for correlated_var_set_cluster, correlated_var_set_scope, weight in data_slices_prod:

					if any(var_scope in correlated_var_set_scope for var_scope in rest_set_scope):

						next_remaining_vars_scope.extend(correlated_var_set_scope)

					else:
						# this variable set of current information set is
						# not correlated to any variable in the rest set

						logging.info(f'independent variable set found: {correlated_var_set_scope}')

						ds_context_prod = get_ds_context(correlated_var_set_cluster,
														 correlated_var_set_scope, self.params)

						if self.params.util_to_bin:

							independent_var_set_prod_child = learn_parametric_aspmn(correlated_var_set_cluster,
																			  ds_context_prod,
																			  n=self.n,
																			  k_limit=self.limit,
																			  min_instances_slice=20,
																			  initial_scope=correlated_var_set_scope)

						else:

							independent_var_set_prod_child = learn_mspn_for_aspmn(correlated_var_set_cluster,
																				 ds_context_prod,
																				 n=self.n,
																				 k_limit=self.limit,
																				 min_instances_slice=20,
																				 initial_scope=correlated_var_set_scope)
						independent_vars_scope.extend(correlated_var_set_scope)
						prod_children.append(independent_var_set_prod_child)

				logging.info(f'correlated variables over entire remaining variables '
							 f'at prod, passed for next recursion: '
							 f'{next_remaining_vars_scope}')
				# check if all variables in current information set are consumed
				if all(var_scope in independent_vars_scope for var_scope in curr_information_set_scope):

					index += 1
					next_information_set_scope = np.array(range(next_scope_index, next_scope_index +
																len(self.params.partial_order[index]))).tolist()

					# since current information set is totally consumed
					next_remaining_vars_scope = rest_set_scope

				else:
					# some variables in current information set still remain
					index = index

					next_information_set_scope = set(curr_information_set_scope) - set(independent_vars_scope)
					next_remaining_vars_scope = next_information_set_scope | set(rest_set_scope)

					# convert unordered sets of scope to sorted lists to keep in sync with partial order
					next_information_set_scope = sorted(list(next_information_set_scope))
					next_remaining_vars_scope = sorted(list(next_remaining_vars_scope))
				self.set_next_operation('Sum')

				next_remaining_vars_data = column_slice_data_by_scope(remaining_vars_data,
																	  remaining_vars_scope,
																	  next_remaining_vars_scope)

				logging.info(
					f'independence test completed for current information set {curr_information_set_scope} '
					f'and rest set {rest_set_scope} ')

				remaining_vars_prod_child = self.__learn_spmn_structure(next_remaining_vars_data,
																		next_remaining_vars_scope,
																		next_information_set_scope,
																		index)

				prod_children.append(remaining_vars_prod_child)

				product_node = Product(children=prod_children)
				assign_ids(product_node)
				rebuild_scopes_bottom_up(product_node)

				logging.info(f'created product node')
				return product_node

			# Cluster the data
			else:

				curr_op = self.get_curr_operation()
				logging.debug(f'curr_op at sum node (cluster test): {curr_op}')

				split_rows = get_split_rows_XMeans(limit=self.limit)    # from SPMNHelper.py
				#split_rows = get_split_rows_KMeans()

				if self.cluster_by_curr_information_set:

					curr_information_set_data = column_slice_data_by_scope(remaining_vars_data,
																		   remaining_vars_scope,
																		   curr_information_set_scope)

					ds_context_sum = get_ds_context(curr_information_set_data, curr_information_set_scope, self.params)
					data_slices_sum, km_model = split_rows(curr_information_set_data, ds_context_sum,
														   curr_information_set_scope)

					logging.info(f'split clusters based on current information set {curr_information_set_scope}')

				else:
					# cluster on whole remaining variables
					ds_context_sum = get_ds_context(remaining_vars_data, remaining_vars_scope, self.params)
					data_slices_sum, km_model = split_rows(remaining_vars_data, ds_context_sum, remaining_vars_scope)

					logging.info(f'split clusters based on whole remaining variables {remaining_vars_scope}')

				sum_node_children = []
				weights = []
				index = index
				logging.debug(f'{len(data_slices_sum)} clusters found at data_slices_sum')



				cluster_num = 0
				labels_array = km_model.labels_
				logging.debug(f'cluster labels of rows: {labels_array} used to cluster data on '
							  f'total remaining variables {remaining_vars_scope}')

				for cluster, scope, weight in data_slices_sum:

					self.set_next_operation("Prod")

					# cluster whole remaining variables based on clusters formed.
					# below methods are useful if clusters were formed on just the current information set

					cluster_indices = get_row_indices_of_cluster(labels_array, cluster_num)
					cluster_on_remaining_vars = row_slice_data_by_indices(remaining_vars_data, cluster_indices)

					# logging.debug(np.array_equal(cluster_on_remaining_vars, cluster ))

					sum_node_children.append(
						self.__learn_spmn_structure(cluster_on_remaining_vars, remaining_vars_scope,
													curr_information_set_scope, index))

					weights.append(weight)

					cluster_num += 1

				sum_node = Sum(weights=weights, children=sum_node_children)

				assign_ids(sum_node)
				rebuild_scopes_bottom_up(sum_node)
				logging.info(f'created sum node')
				return sum_node

	def learn_aspmn(self, train, test, k=None):
		"""
		:param
		:return: learned spmn
		"""
		
		'''
		original_stats = {
			'Export_Textiles': {"ll" : -1.0903135560503194, "meu" : 1922639.5, 'nodes' : 22},
			'Test_Strep': {"ll" : -1.1461735112245122, "meu" : 54.92189449375, 'nodes' : 51},
			'LungCancer_Staging': {"ll" : -1.3292497032277288, "meu" : 3.11376125, 'nodes' : 49},
			'HIV_Screening': {"ll" : -0.5943350928785097, "meu" : 42.60624317138454, 'nodes' : 125},
			'Computer_Diagnostician': {"ll" : -0.8912294493362266, "meu" : 242.863042737567, 'nodes' : 50},
			'Powerplant_Airpollution': {"ll" : -1.8151637099020188, "meu" : -2803562.5, 'nodes' : 45}
		}
		'''
		original_stats = {
			'Export_Textiles': {"ll" : -1.0892559429908522, "meu" : 1922275.95, 'nodes' : 22, 'reward':1721469.45},
			'Test_Strep': {"ll" : -0.9112557170813002, "meu" : 54.93760881256758, 'nodes' : 100, 'reward':54.97011839999944},
			'LungCancer_Staging': {"ll" : -1.1515872880624247, "meu" : 3.1526200852839716, 'nodes' : 260, 'reward':3.1738849999999976},
			'HIV_Screening': {"ll" : -0.6189833438168413, "meu" : 42.63750815337698, 'nodes' : 112, 'reward':42.4838739999994},
			'Computer_Diagnostician': {"ll" : -0.892138328151404, "meu" : 244.94, 'nodes' : 47, 'reward':244.955},
			'Powerplant_Airpollution': {"ll" : -1.081424145432235, "meu" : -2726821.30929344245, 'nodes' : 46, 'reward':-2770200.0}
		}
		
		ll = list()
		meus = list()
		nodes = list()
		avg_rewards = list()
		reward_dev = list()
		past3 = list()
		
		limit = 100000
		n = self.vars

		if k is not None:
			if not pth.exists(f"{self.plot_path}/{k}"):
				try:
					os.makedirs(f"{self.plot_path}/{k}")
				except OSError:
					print ("Creation of the directory %s failed" % f"{self.plot_path}/{k}")
					sys.exit()

		

		index = 0
		
		curr_information_set_scope = np.array(range(len(self.params.partial_order[0]))).tolist()
		remaining_vars_scope = np.array(range(len(self.params.feature_names))).tolist()
		self.set_next_operation('Any')
		self.limit = limit 
		self.n = n  

		print("start")
		spmn = self.__learn_spmn_structure(train, remaining_vars_scope, curr_information_set_scope, index)
		print("done")
		#spmn = Prune(spmn)
		self.spmn = spmn

		nodes.append(get_structure_stats_dict(spmn)["nodes"])

		if k is None:
			plot_spn(spmn, f'{self.plot_path}/spmn.pdf', feature_labels=self.params.feature_labels)
		else:
			plot_spn(spmn, f'{self.plot_path}/{k}/spmn.pdf', feature_labels=self.params.feature_labels)
		
		
		try:
			total_ll = 0
			for j, instance in enumerate(test):
				test_data = np.array(instance).reshape(-1, len(self.params.feature_names))
				total_ll += log_likelihood(spmn, test_data)[0][0]
				printProgressBar(j+1, len(test), prefix = f'Log Likelihood Evaluation :', suffix = 'Complete', length = 50)
			ll.append(total_ll/len(test))
			


			test_data = [[np.nan]*len(self.params.feature_names)]
			m = meu(spmn, test_data)
			meus.append(m[0])

			env = get_env(self.dataset)
			total_reward = 0
			trials = 10000
			batch_size = trials / 10
			batch = list()

			for z in range(trials):
				
				state = env.reset()  #
				while(True):
					output = best_next_decision(spmn, state)
					#output = spmn_topdowntraversal_and_bestdecisions(spmn, test_data)
					action = output[0][0]
					state, reward, done = env.step(action)
					if done:
						total_reward += reward
						break
				if (z+1) % batch_size == 0:
					batch.append(total_reward/batch_size)
					total_reward = 0
				printProgressBar(z+1, trials, prefix = f'Average Reward Evaluation :', suffix = 'Complete', length = 50)

			avg_rewards.append(np.mean(batch))
			reward_dev.append(np.std(batch))
			
			
			
			print("\n\n\n\n\n")
			print(f"X-Means Limit: {limit}, \tVariables for splitting: {round(n)}")
			print("#Nodes: ",nodes[-1])
			print("log_likelihood: ",ll[-1])
			print("MEU: ",meus[-1])
			print("Average rewards: ",avg_rewards[-1])
			print("Deviation: ",reward_dev[-1])
			print(nodes)
			print(meus)
			print("\n\n\n\n\n")

			

			
			f = open(f"{self.plot_path}/stats.txt", "w") if k is None else open(f"{self.plot_path}/{k}/stats.txt", "w")

			f.write(f"\n{self.dataset}")
			f.write(f"\n\tLog Likelihood : {ll}")
			f.write(f"\n\tMEU : {meus}")
			f.write(f"\n\tNodes : {nodes}")
			f.write(f"\n\tAverage Rewards : {avg_rewards}")
			f.write(f"\n\t\tDeviation : {reward_dev}")
			f.close()
		except:
			pass
		
		# Prune(self.spmn)
		return self.spmn



class SPMNParams:

	def __init__(self, partial_order, decision_nodes, utility_nodes, feature_names, feature_labels, meta_types, util_to_bin):

		self.partial_order = partial_order
		self.decision_nodes = decision_nodes
		self.utility_nodes = utility_nodes
		self.feature_names = feature_names
		self.feature_labels = feature_labels
		self.meta_types = meta_types
		self.util_to_bin = util_to_bin
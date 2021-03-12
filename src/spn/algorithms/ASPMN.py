
from spn.structure.Base import Sum, Product, Max
from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up
from spn.algorithms.splitting.RDC import get_split_cols_distributed_RDC_py
from spn.algorithms.splitting.Clustering import get_split_rows_XMeans
from spn.algorithms.LearningWrappers import learn_mspn, learn_parametric
from spn.algorithms.SPMNHelper import *




class Anytime_SPMN:


	def __init__(self, dataset, partial_order , decision_nodes, utility_node, feature_names,
				   util_to_bin = False , output_path):
				   
		self.dataset = dataset
		self.params = SPMN_Params(partial_order, decision_nodes, utility_node, feature_names, util_to_bin )

		self.plot_path = f"{output_path}/{dataset}"
		if not pth.exists(self.plot_path):
			try:
				os.makedirs(self.plot_path)
			except OSError:
				print ("Creation of the directory %s failed" % self.plot_path)
				sys.exit()

	def learn_spmn_structure(self, train_data, index, scope_index, limit=2, n=self.train_data.shape[1]):


		train_data = train_data
		curr_var_set = params.partial_order[index]

		if self.params.partial_order[index][0] in  self.params.decision_nodes:

			decision_node = self.params.partial_order[index][0]
			cl, dec_vals= split_on_decision_node(train_data, curr_var_set)
			spn0 = []
			index= index+1
			set_next_operation("None")

			for c in cl:

				if index < len(self.params.partial_order):

					spn0.append(learn_spmn_structure(c, index, scope_index))
					spn = Max(dec_values=dec_vals, children=spn0, feature_name=decision_node)

				else:
					spn = Max(dec_values=dec_vals, children=None, feature_name=decision_node)

			assign_ids(spn)
			rebuild_scopes_bottom_up(spn)
			return spn



		else:

			curr_train_data_prod, curr_train_data = get_curr_train_data_prod(train_data, curr_var_set)

			split_cols = get_split_cols_distributed_RDC_py(rand_gen=None, ohe=False, n_jobs=-1, n=round(n))
			scope_prod = get_scope_prod(curr_train_data_prod, scope_index, self.params.feature_names)

			ds_context_prod = get_ds_context_prod(curr_train_data_prod, scope_prod, index, scope_index, self.params)

			data_slices_prod = split_cols(curr_train_data_prod, ds_context_prod, scope_prod)
			curr_op = get_next_operation()


			if len(data_slices_prod)>1 or curr_op == "Prod" or index == len(self.params.partial_order) :
				set_next_operation("Sum")

				if self.params.util_to_bin :

					spn0 = learn_parametric(curr_train_data_prod, ds_context_prod, min_instances_slice=20, initial_scope= scope_prod)

				else:

					spn0 = learn_mspn(curr_train_data_prod, ds_context_prod, min_instances_slice=20,
										initial_scope=scope_prod)

				index = index + 1
				scope_index = scope_index +curr_train_data_prod.shape[1]

				if index < len(self.params.partial_order):

					spn1 = learn_spmn_structure(curr_train_data, index, scope_index)
					spn = Product(children=[spn0, spn1])

					assign_ids(spn)
					rebuild_scopes_bottom_up(spn)

				else:
					spn = spn0
					assign_ids(spn)
					rebuild_scopes_bottom_up(spn)

			else:

				split_rows = get_split_rows_XMeans(limit=limit, returnk=False)
				scope_sum = list(range(train_data.shape[1]))

				ds_context_sum = get_ds_context_sum(train_data, scope_sum, index, scope_index, self.params)
				data_slices_sum = split_rows(train_data, ds_context_sum, scope_sum)

				spn0 = []
				weights = []
				index = index

				if index < len(self.params.partial_order):

					for cl, scop, weight in data_slices_sum:

						set_next_operation("Prod")
						spn0.append(learn_spmn_structure(cl, index, scope_index))
						weights.append(weight)

					spn = Sum(weights=weights, children=spn0)
					assign_ids(spn)
					rebuild_scopes_bottom_up(spn)

			assign_ids(spn)
			rebuild_scopes_bottom_up(spn)
			return spn

	def learn_aspmn(self, train_data, test_data):

		"""
		:param scope_vars_val: 'list of all variables in sequence of partial order excluding decison variables' #look metadeta for examples.
		:return: learned spmn
		"""

		#self.train_data = train_data
		#self.test_data = test_data

		meu = list()
		nodes = list()
		past3 = list()
		max_iter = train_data.shape[1]
		
		limit = 2 
		n = int(max_iter**0.5)  
		step = (max_iter - (max_iter**0.5))/20

		i = 0
		while(True):
			index = 0
			scope_index = 0
			set_next_operation("None")

			spmn = learn_spmn_structure(train_data, index, scope_index, limit, n)

			nodes.append(get_structure_stats_dict(spmn)["nodes"])

			print("\n\n\n\n\n")
	        print(f"X-Means Limit: {limit}, \tVariables for splitting: {round(n)}")
	        print("#Nodes: ",nodes[i])
	        print("MEU: ",meu[i])
	        print(nodes)
	        print(meu)
	        print("\n\n\n\n\n")


			i+=1
			limit += 1
			n = min(n+step, max_iter)

		return spmn


class SPMN_Params():

	def __init__(self, partial_order, decision_nodes, utility_node, feature_names, util_to_bin ):

		self.partial_order = partial_order
		self.decision_nodes = decision_nodes
		self.utility_node = utility_node
		self.feature_names = feature_names
		self.util_to_bin = util_to_bin



























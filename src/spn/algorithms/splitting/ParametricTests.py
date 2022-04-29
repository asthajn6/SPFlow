"""
Created on June 21, 2018

@author: Alejandro Molina
"""

from threading import local
import numpy as np
import pandas as pd
import numba
import scipy


from collections import deque

from spn.algorithms.splitting.Base import split_data_by_clusters
from scipy.stats import chi2
import logging
import random

from spn.data.metaData import get_feature_labels
logger = logging.getLogger(__name__)


@numba.njit
def count_nonzero(array):
    nonzero_coords, = np.nonzero(array)
    return len(nonzero_coords)

def totuple(a):
        try:
            return tuple(totuple(i) for i in a)
        except TypeError:
            return a

@numba.jit
def g_test1(feature_id_1, feature_id_2, local_data, feature_vals, scope, g_factor):
    
    col1 = []
    col2 = []

    # creating a contingency matrix over two feature IDs 1 and 2
    col1 = np.array(local_data[:,feature_id_1])
    col2 = np.array(local_data[:,feature_id_2])
    data_crosstab = pd.crosstab(col1,col2)

    # calculating G and P value from chi2 contingency function in scipy package
    # feeding the contingency matrix created above as a parameter to chi2_contingency function
    # lambda = log-likelihood allows to calculate G value
    g_val, p_val, dof, expctd = scipy.stats.chi2_contingency(data_crosstab, lambda_="log-likelihood")

    # comparing the P value to the threshold
    return p_val > g_factor

@numba.jit
def g_test(feature_id_1, feature_id_2, local_data, feature_vals, g_factor):
    """
    Applying a G-test on the two features (represented by ids) on the data
    """
    # logger.info(feature_id_1, feature_id_2, instance_ids)

    #
    # swap to preserve order, is this needed?
    if feature_id_1 > feature_id_2:
        #
        # damn numba this cannot be done lol
        # feature_id_1, feature_id_2 = feature_id_2, feature_id_1
        tmp = feature_id_1
        feature_id_1 = feature_id_2
        feature_id_2 = tmp

    # logger.info(feature_id_1, feature_id_2, instance_ids)
    
    # n_instances = len(instance_ids)
    n_instances = len(local_data)
    feature_size_1 = np.array(feature_vals[feature_id_1], dtype=np.uint32)
    feature_size_2 = np.array(feature_vals[feature_id_2], dtype=np.uint32)
    #
    
    # support vectors for counting the occurrences
    feature_tot_1 = np.zeros(feature_size_1, dtype=np.uint32)
    feature_tot_2 = np.zeros(feature_size_2, dtype=np.uint32)
    
    co_occ_matrix = np.zeros((feature_size_1[-1]+1, feature_size_2[-1]+1), dtype=np.uint32)
    
    #
    # counting for the current instances
    for i in range(n_instances):
        co_occ_matrix[(int)(local_data[i, feature_id_1]), (int)(local_data[i, feature_id_2])] += 1
    
    # logger.info('Co occurrences', co_occ_matrix)
    #
    # getting the sum for each feature
    for i in range(feature_size_1.size):
        for j in range(feature_size_2.size):
            count = co_occ_matrix[i, j]
            feature_tot_1[i] += count
            feature_tot_2[j] += count
    
    # logger.info('Feature tots', feature_tot_1, feature_tot_2)

    #
    # counputing the number of zero total co-occurrences for the degree of
    # freedom
    feature_nonzero_1 = np.count_nonzero(feature_tot_1)
    feature_nonzero_2 = np.count_nonzero(feature_tot_2)
    
    dof = (feature_nonzero_1 - 1) * (feature_nonzero_2 - 1)
    
    g_val = np.float64(0.0)
    
    for i, tot_1 in enumerate(feature_tot_1):
        for j, tot_2 in enumerate(feature_tot_2):
            count = co_occ_matrix[i, j]
            if count != 0:
                exp_count = tot_1 * tot_2 / n_instances
                g_val += count * np.log(count / exp_count)
    
    dep_val = np.float64(2 * dof * g_factor + 0.001)
    return (2 * g_val) < dep_val


@numba.jit
def gtest_greedy_feature_split(local_data, feature_vals, scope, g_factor, rand_gen):
    """
    Implementing the G-test based feature splitting as in
    Gens et al. 'Learning the Structure of Sum-Product Networks' 2013
    """
    # n_features = data_slice.n_features()
    n_features = local_data.shape[1]

    #
    # extracting all features one by one
    for rand_feature_id in range(n_features):

        feature_ids_mask = np.ones(n_features, dtype=bool)
        feature_ids_mask[rand_feature_id] = False

        dependent_features = np.zeros(n_features, dtype=bool)
        dependent_features[rand_feature_id] = True

        # greedy bfs searching
        features_to_process = deque()
        features_to_process.append(rand_feature_id)

        while features_to_process:
            # get one
            current_feature_id = features_to_process.popleft()
            # feature_id_1 = data_slice.feature_ids[current_feature_id]

            # features to remove later
            features_to_remove = np.zeros(n_features, dtype=bool)

            for other_feature_id in feature_ids_mask.nonzero()[0]:
                #
                # logger.info('considering other features', other_feature_id)
                # feature_id_2 = data_slice.feature_ids[other_feature_id]
                #
                # apply a G-test
                if not g_test1(
                    current_feature_id,  # feature_id_1,
                    other_feature_id,  # feature_id_2,
                    local_data,
                    feature_vals,
                    scope,
                    g_factor
                ):

                    #
                    # updating 'sets'
                    features_to_remove[current_feature_id] = True
                    dependent_features[other_feature_id] = True
                    features_to_process.append(other_feature_id)

            # now removing from future considerations
            feature_ids_mask[features_to_remove] = False

        # finding the number of distinct features to cluster together and split the remaining
        values = np.unique(dependent_features)
        
        # proceed to splitting once independence is observed
        if(values.size>1):
            break

    return dependent_features.astype(np.int)


def get_split_cols_GTest(threshold=0.01, rand_gen=None):
    def split_cols_GTest(local_data, ds_context, scope):
        domains = ds_context.get_domains_by_scope(scope)
        clusters = gtest_greedy_feature_split(local_data, domains, scope, threshold, rand_gen)
        return split_data_by_clusters(local_data, clusters, scope, rows=False)

    return split_cols_GTest

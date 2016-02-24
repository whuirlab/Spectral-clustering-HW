# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 14:50:41 2016

@author: Rok
"""

import numpy as np
import process
import spectral
import scipy.spatial.distance as dist
from scipy.sparse.csgraph import minimum_spanning_tree as mst
from sklearn.cluster import KMeans
from sklearn import metrics

filepath = 'D:/Documents/Faks/AD2I/Unsupervised_learning/Project_SC/datasets/Anas_datasets/smaller_samples/r8.txt'
#filepath = 'D:/Documents/Faks/AD2I/Unsupervised_learning/Project_SC/datasets/Anas_datasets/smaller_samples/r52.txt'
#filepath = 'D:/Documents/Faks/AD2I/Unsupervised_learning/Project_SC/datasets/Anas_datasets/smaller_samples/webkb.txt'

words = process.get_words(filepath)
M, labels = process.get_M(filepath, words)
tf_idf_M = process.get_tf_idf_M(M, "raw", "c", norm_samps=True)
n = M.shape[0]
k = len(np.unique(labels))

#define a similarity functions
def gauss_s(x1, x2, d):
    sigma = 1
    return np.exp(-(d(x1, x2))/(2*sigma**2))
    
def cos_s(x1, x2, d):
    return -(dist.cosine(x1, x2) - 1)
    
#Test the spectral clustering for euclidean distance
#c = spectral.spectral(tf_idf_M, labels, gauss_s, dist.euclidean)
#print("Testing for Gaussian similarity, sigma = 1:")
#
##For each simmilarity graph test all three algorithms
#c.full_graph("euclidean")
#print(c.graph)
#for algo in [c.norm_sym_sc, c.norm_rw_sc, c.unnorm_sc]:
#    algo(k)
#    print(c.clustering)
#    c.evaluate()
###################################################
#T = mst(c.W)
#A = T.toarray().astype(float)
#eps = np.min(A[np.nonzero(A)])
#c.eps_graph(eps)
#print(c.graph)
#for algo in [c.norm_sym_sc, c.norm_rw_sc, c.unnorm_sc]:
#    algo(k)
#    print(c.clustering)
#    c.evaluate()
###################################################    
#c.kNN_graph(int(2*(n/np.log(n))), "euclidean", True)
#print(c.graph)
#for algo in [c.norm_sym_sc, c.norm_rw_sc, c.unnorm_sc]:
#    algo(k)    
#    print(c.clustering)
#    c.evaluate()
##################################################
#c.kNN_graph(int(n/np.log(n)), "euclidean", False)
#print(c.graph)
#for algo in [c.norm_sym_sc, c.norm_rw_sc, c.unnorm_sc]:
#    algo(k)
#    print(c.clustering)
#    c.evaluate()
##################################################
#
#
##Test the spectral clustering for cosine distance
#c = spectral.spectral(tf_idf_M, labels, cos_s, dist.cosine)
#print("Testing for cosine similarity")
#
##For each simmilarity graph test all three algorithms
##################################################
#c.full_graph("cosine")
#print(c.graph)
#for algo in [c.norm_sym_sc, c.norm_rw_sc, c.unnorm_sc]:
#    algo(k)
#    print(c.clustering)
#    c.evaluate()
##################################################
#T = mst(c.W)
#A = T.toarray().astype(float)
#eps = np.min(A[np.nonzero(A)])
#c.eps_graph(eps)
#print(c.graph)
#for algo in [c.norm_sym_sc, c.norm_rw_sc, c.unnorm_sc]:
#    algo(k)
#    print(c.clustering)
#    c.evaluate()
#####################################################
#c.kNN_graph(int(3*(n/np.log(n))), "cosine", True)
#print(c.graph)
#for algo in [c.norm_sym_sc, c.norm_rw_sc, c.unnorm_sc]:
#    algo(k)
#    print(c.clustering)
#    c.evaluate()
##################################################
#c.kNN_graph(int(2*n/np.log(n)), "cosine", False)
#print(c.graph)
#for algo in [c.norm_sym_sc, c.norm_rw_sc, c.unnorm_sc]:
#    algo(k)
#    print(c.clustering)
#    c.evaluate()

#kmeans = KMeans(k)
#kmeans_pred = kmeans.fit_predict(tf_idf_M)
#metrics.adjusted_rand_score(c.labels, kmeans_pred)
#metrics.adjusted_mutual_info_score(c.labels, kmeans_pred)
#metrics.normalized_mutual_info_score(c.labels, kmeans_pred)

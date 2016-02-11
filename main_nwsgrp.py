# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:40:35 2016

@author: Rok
"""

import numpy as np
import newsgroups20
import spectral
import scipy.spatial.distance as dist

#rootdir = 'D:/Documents/Faks/AD2I/Unsupervised_learning/Project_SC/datasets/20newsgroups/small_test_set'
#rootdir = 'D:/Documents/Faks/AD2I/Unsupervised_learning/Project_SC/datasets/20newsgroups/bigger_test_set_noncor'
rootdir = 'D:/Documents/Faks/AD2I/Unsupervised_learning/Project_SC/datasets/20newsgroups/bigger_test_set_cor'

words = newsgroups20.get_words(rootdir)
M, labels = newsgroups20.get_M(rootdir, words)
tf_idf_M = newsgroups20.get_tf_idf_M(M, "raw", "c", norm_samps=True)

#tf_idf_M = np.loadtxt("tf_idf_M_2254.txt")
#labels = np.loadtxt("M_35224_labels.txt")

#tf_idf_M = np.loadtxt("tf_idf_M_500.txt")
#labels = np.loadtxt("M_15456_labels.txt")

#define a similarity function
def gauss_s(x1, x2, d):
    sigma = 10
    return np.exp(-(d(x1, x2))/(2*sigma**2))
    
def cos_s(x1, x2, d):
    return -(dist.cosine(x1, x2) - 1)
    
#create a clustering object
#c = spectral.spectral(tf_idf_M, labels, cos_s, dist.cosine)
c = spectral.spectral(tf_idf_M, labels, gauss_s, dist.euclidean)

#********************************************#
#Testing efficiency for different parameters #
#********************************************#

#np.savetxt("prediction.txt", c.pred, fmt="%s")

#for k in np.arange(20, 30, 1):
#    c.kNN_graph(k, "cosine", True)
#    c.norm_sym_sc(2)
#    print(k, "mutual NN graph.")
#    c.evaluate()
    
#for k in np.arange(170, 190, 5):
#    c.kNN_graph(k, "euclidean", False)
#    c.norm_sym_sc(5)
#    print(c.graph)
#    print(c.clustering)
#    c.evaluate()

#for eps in np.arange(-0.3, 1, 0.1):
#    c.eps_graph(eps)
#    c.norm_sym_sc(5)
#    print(eps, "epsilon graph.")
#    c.evaluate()
    
#c.full_graph()
#c.unnorm_sc(5)
#print(c.graph)
#print(c.clustering)
#c.evaluate()
#c.norm_rw_sc(5)
#print(c.graph)
#print(c.clustering)
#c.evaluate()
#c.norm_sym_sc(5)
#print(c.graph)
#print(c.clustering)
#c.evaluate()

#for algo in [c.norm_sym_sc, c.norm_rw_sc, c.unnorm_sc]:
#    c.kNN_graph(300, "euclidean", True)
#    algo(5)
#    print(c.graph)
#    print(c.clustering)
#    c.evaluate()
#    c.kNN_graph(170, "euclidean", False)
#    algo(5)
#    print(c.graph)
#    print(c.clustering)
#    c.evaluate()
##    c.eps_graph(0.2)
##    algo(5)
##    print(c.graph)
##    print(c.clustering)
##    c.evaluate()
#    c.full_graph()
#    algo(5)
#    print(c.graph)
#    print(c.clustering)
#    c.evaluate()

#k = 175
##for k in np.arange(25, 100, 5):
#c.kNN_graph(k, "euclidean", False)
#print(c.graph)
#for algo in [c.norm_sym_sc, c.norm_rw_sc, c.unnorm_sc]:
#    algo(5)
#    print(c.clustering)
#    c.evaluate()
        
        
     
         
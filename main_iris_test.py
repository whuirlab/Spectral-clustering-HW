# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 12:58:38 2016

@author: Rok
"""
import numpy as np
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
import scipy.spatial.distance as dist
import spectral

#import some data to play with
iris = datasets.load_iris()
#X = iris.data[np.arange(0,150,5)]
#Y = iris.target[np.arange(0,150,5)]

X = iris.data
Y = iris.target

#normalizer = Normalizer()
#X = normalizer.fit_transform(X)

#scale to zero mean and unit variance
#scaler = StandardScaler()
#X = scaler.fit_transform(X)


#define a similarity function
def gauss_s(x1, x2, d):
    sigma = 1
    return np.exp(-(d(x1, x2))/(2*sigma**2))

def cos_s(x1, x2, d):
    return -(dist.cosine(x1, x2) - 1)

#similarities = []
#for x1 in X:
#    for x2 in X:
#        similarities.append(gauss_s(x1,x2))

#create a clustering object
c = spectral.spectral(X, Y, gauss_s, dist.euclidean)
#c.kNN_graph(30, "cosine", True)
#c.kNN_graph(10, "euclidean", False)
#c.eps_graph(0.4)
#c.full_graph()
#c.show_sim_g()
#c.norm_rw_sc(3)
#c.norm_sym_sc(3)
#c.show_clust()
#c.show_correct_class()
#c.evaluate()

for algo in [c.norm_sym_sc, c.norm_rw_sc, c.unnorm_sc]:
    c.kNN_graph(30, "euclidean", True)
    algo(3)
    print(c.graph)
    print(c.clustering)
    c.evaluate()
    c.kNN_graph(20, "euclidean", False)
    algo(3)
    print(c.graph)
    print(c.clustering)
    c.evaluate()
    c.eps_graph(0.4)
    algo(3)
    print(c.graph)
    print(c.clustering)
    c.evaluate()
    c.full_graph()
    algo(3)
    print(c.graph)
    print(c.clustering)
    c.evaluate()
    
#testing to see if sklearns spectral clustering works
#clusterer = SpectralClustering(3, affinity='precomputed')
#sklearn_pred = clusterer.fit_predict(c.W)
#print(sklearn_pred) #it indeed works


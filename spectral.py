# -*- coding: utf-8 -*-
"""
Created on Tue Feb  2 11:42:17 2016

@author: Rok
"""

import numpy as np
from scipy.linalg import eigh
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
#from scipy.sparse.linalg import eigsh
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn import metrics
import matplotlib.pyplot as plt 

#A class for testing the efficiency of spectral clustering (in text clustering) for different parameters.
class spectral():
    """X is a real matrix nxm, labels is a vector of integers in range 0:(n-1) s is a simmilaritiy function"""
    def __init__(self, X, labels, s, d):
        self.X = X #rows of X are numerical representations of objets (texts) to be clustered 
        self.labels = labels #labels later needed to measure correctness of constructed clusters
        self.s = s #a similarity function f(x1, x2, d) -> R, maps to real numbers
        self.d = d #a metric used in similarity function
        self.graph = False #a switch that tells if the similartiy graph has allready been constructed
        self.clustering = False #a switch that tells if clustering has allready been done
        self.full_calculated = False #a switch that tells if full graph has been calculated
#        self.latex = []
    
    """Construct an epsilon similarity graph"""        
    def eps_graph(self, eps):
#        self.latex = []
        m = self.X.shape[0]
        self.W = np.zeros((m, m)) #(weighted) adjecancy matrix
        self.D = np.zeros((m, m)) #degree matrix
        if self.full_calculated:
            indices = np.where(self.full_W >= eps)
            self.W[indices] = self.full_W[indices]
            self.D[np.diag_indices(m)] = np.sum(self.W, 1)
        else:
            for i in range(m): #measure the similarty of samples
                for j in range(m):
                    sim = self.s(self.X[i], self.X[j], self.d)
                    if (i != j and sim >= eps): #if similartiy is bigger than some eps
                        self.W[i,j] = sim #put it in adjacency matrix
                        self.D[i,i] += sim #add it to degree of vertex i
        self.graph = "Espilon-graph, epsilon = " + str(eps) #update the switch
    
    """Construct a kNN similarity graph. Make sure the metric is consistent with the choice of d."""    
    def kNN_graph(self, k, metric, mutual=False):
#        self.latex = []
        nn = NearestNeighbors(k, algorithm="brute", metric=metric, n_jobs=-1).fit(self.X)
        UAM = nn.kneighbors_graph(self.X).toarray() #unweighted adjacency matrix
        m = UAM.shape[0]
        self.W = np.zeros((m, m)) #(weighted) adjecancy matrix
        self.D = np.zeros((m, m)) #degree matrix
        if mutual == False:
            if self.full_calculated:
                indices = np.where(UAM == 1)
                self.W[indices] = self.full_W[indices]
                self.D[np.diag_indices(m)] = np.sum(self.W, 1)
            else:
                for i in range(m):
                    for j in range(m):
                        if UAM[i,j] == 1:
                            sim = self.s(self.X[i], self.X[j], self.d)
                            self.W[i,j] = sim
                            self.D[i,i] += sim
        else:
            if self.full_calculated:
                indices = np.where(np.logical_and(UAM == 1, UAM.T == 1).astype(int) == 1)
                self.W[indices] = self.full_W[indices]
                self.D[np.diag_indices(m)] = np.sum(self.W != 0, 1)
            else:
                for i in range(m):
                    for j in range(m):
                        if UAM[i,j] == 1 and UAM[j,i] == 1:
                            sim = self.s(self.X[i], self.X[j], self.d)
                            self.W[i,j] = sim
                            self.D[i,i] += sim
        self.W = np.nan_to_num(self.W)
        self.graph = "kNN graph, k = " + str(k) + ", mutual:" + str(mutual)

    """Construct a fully connected graph"""    
    def full_graph(self, metric):
#        self.latex = []
        sigma = 1
        m = self.X.shape[0]
        self.W = np.zeros((m, m)) #(weighted) adjecancy matrix
        self.D = np.zeros((m, m)) #degree matrix
        dist_M = squareform(pdist(self.X, metric))
        if (metric == "euclidean"):
            self.W = np.exp(-dist_M/(2*sigma**2))
        elif (metric == "cosine"):
            self.W = -(dist_M - 1)
        self.D[np.diag_indices(m)] = np.sum(self.W, 1)
#        for i in range(m):
#            for j in range(m):
#                self.W[i,j] = self.s(self.X[i], self.X[j], self.d)
#                self.D[i,i] += 1
        self.W = np.nan_to_num(self.W)
        self.full_W = np.copy(self.W)
        self.graph = "fully connected graph"
        self.full_calculated = True
    
    """Plot a similarity graph. PCA to 2 dimensions"""
    def show_sim_g(self):
        if self.graph == False: print("Nothing to show! Construct a similarity graph first!")
        else:            
            pca = PCA(n_components = 2)
            X = pca.fit_transform(self.X)
            k = np.unique(self.labels)
            plt.figure()
            for i in k:
                indices = [j for j,x in enumerate(self.labels) if x == i]
                plt.scatter(X[indices,0], X[indices,1], s=30, c= "yellow")
            for i in range(self.W.shape[0]):
                for j in range(self.W.shape[1]):
                    if self.W[i,j] != 0:
                        plt.plot([X[i,0], X[j,0]], [X[i,1], X[j,1]], linewidth=0.2, color = "0.4")
            plt.title("Similarity graph: " + self.graph)
            plt.show()
        
    """Cluster the objects via unnormalized spectral clustering into k clusters"""
    def unnorm_sc(self, k):
        if self.graph == False: print("Construct a similarity graph first!")
        else:
            self.L = self.D - self.W #compute the unnormalized Laplacian
#            if self.graph == "fully connected graph":
            self.w, self.U = eigh(self.L, eigvals=(0,k-1)) #compute the eigenvalues and eigenvectors
#            else: #If L is sparse (sim graph is epsilon graph or kNN graph) use the scipy.sparse methods
#                self.w, self.U = eigsh(self.L, k) #compute the eigenvalues and eigenvectors using method for sparse matrices
            kmeans = KMeans(k)
            self.pred = kmeans.fit_predict(self.U) #use the kmeans to calculate centroids of voronoi
        self.clustering = "unnormalized Laplacian"
        
    """Cluster the objects via normalized spectral clustering (L_rw) into k clusters"""
    def norm_rw_sc(self, k):
        if self.graph == False: print("Construct a similarity graph first!")
        else:
            self.L = self.D - self.W #compute the unnormalized Laplacian
#            if self.graph == "fully connected graph":
            self.w, self.U = eigh(self.L, self.D, eigvals=(0,k-1)) #generalized eigenproblem L u = lambda D u
#            else:
#                self.w, self.U = eigsh(self.L, k, self.D) #using method for sparse matrices
            kmeans = KMeans(k)
            self.pred = kmeans.fit_predict(self.U)
        self.clustering = "normalized (rw) Laplacian"
        
    def norm_sym_sc(self, k):
        if self.graph == False: print("Construct a similarity graph first!")
        else:
            D_sqrt = np.copy(self.D)
            m = self.D.shape[0]
            D_sqrt[np.diag_indices(m)] = np.diag(self.D)**(-1/2)
            D_sqrt = np.nan_to_num(D_sqrt) #not sure if this is a good idea
            self.L = np.identity(m) - np.dot(np.dot(D_sqrt, self.W), D_sqrt) #compute the unnormalized Laplacian
#            if self.graph == "fully connected graph":
            self.w, self.U = eigh(self.L, eigvals=(0,k-1)) #compute the eigenvalues and eigenvectors
#            else:
#                self.w, self.U = eigsh(self.L, k, self.D) #using method for sparse matrices
            self.T = normalize(self.U) #normalize samples to norm 1
            kmeans = KMeans(k)
            self.pred = kmeans.fit_predict(self.T)
        self.clustering = "normalized (sym) Laplacian"
    
    """Visualisation of the clusters constructed. PCA to 2 dimensions"""    
    def show_clust(self):
        if self.graph == False: print("Construct a similarity graph first!")
        else:
            if self.clustering == False: print("Nothing to show. Cluster the data first!")
            else:            
                pca = PCA(n_components = 2)
                X = pca.fit_transform(self.X)
                k = np.unique(self.pred)
                colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
                plt.figure()
                for i in k:
                    indices = [j for j,x in enumerate(self.pred) if x == i]
                    plt.scatter(X[indices,0], X[indices,1], s=30, c=colors[int(i)])
                plt.title("Identifiyed clusters: using " + self.clustering + "\n of " + self.graph)
                plt.show()
    
    """Visualisation of the correct classes of data. PCA to 2 dimensions"""             
    def show_correct_class(self):
        pca = PCA(n_components = 2)
        X = pca.fit_transform(self.X)
        k = np.unique(self.labels)
        colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
        plt.figure()
        for i in k:
            indices = [j for j,x in enumerate(self.labels) if x == i]
            plt.scatter(X[indices,0], X[indices,1], s=30, c=colors[int(i)])
        plt.title("Correct clusters")
        plt.show()

    """Measure the effectiveness of predictions using different metrics"""                        
    def evaluate(self):
        ARI = round(metrics.adjusted_rand_score(self.labels, self.pred), 4)
        AMI = round(metrics.adjusted_mutual_info_score(self.labels, self.pred), 4)
        NMI = round(metrics.normalized_mutual_info_score(self.labels, self.pred), 4)
        print("Adjusted Rand index:", "%.4f" % ARI)
        print("Adjusted Mutual Information:", "%.4f" % AMI)
        print("Normalized Mutual Information:", "%.4f" % NMI)
#        print("ARI: ", "%.4f" % round(metrics.adjusted_rand_score(self.labels, self.pred), 4), "\\\",
#              "AMI: ", "%.4f" % round(metrics.adjusted_mutual_info_score(self.labels, self.pred), 4), "\\\",
#              "NMI: ", "%.4f" % round(metrics.normalized_mutual_info_score(self.labels, self.pred), 4))
#        self.latex.append([ARI, AMI, NMI])
    
#    def print_for_latex(self):
#        self.latex = np.array(self.latex).T
#        print("& ARI: ", "%.4f" % self.latex[0,0], "& ARI: ", "%.4f" % self.latex[0,1], "& ARI: ", "%.4f" % self.latex[0,2], "\\".replace("\\", "\\\\"))
#        print("& AMI: ", "%.4f" % self.latex[1,0], "& AMI: ", "%.4f" % self.latex[1,1], "& AMI: ", "%.4f" % self.latex[1,2], "\\".replace("\\", "\\\\"))
#        print("& NMI: ", "%.4f" % self.latex[2,0], "& NMI: ", "%.4f" % self.latex[2,1], "& NMI: ", "%.4f" % self.latex[2,2], "\\".replace("\\", "\\\\"), "\hline")
#        

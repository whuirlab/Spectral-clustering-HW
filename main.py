# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 14:50:41 2016

@author: Rok
"""

import numpy as np
import process
import spectral
import scipy.spatial.distance as dist

#rootdir = 'D:/Documents/Faks/AD2I/Unsupervised_learning/Project_SC/datasets/20newsgroups/small_test_set'
#rootdir = 'D:/Documents/Faks/AD2I/Unsupervised_learning/Project_SC/datasets/20newsgroups/bigger_test_set_noncor'
#rootdir = 'D:/Documents/Faks/AD2I/Unsupervised_learning/Project_SC/datasets/20newsgroups/bigger_test_set_cor'
filepath = 'D:/Documents/Faks/AD2I/Unsupervised_learning/Project_SC/datasets/Anas_datasets/20ng-test-stemmed.txt'

words = process.get_words(filepath)
M, labels = process.get_M(filepath, words)
tf_idf_M = process.get_tf_idf_M(M, "raw", "c", norm_samps=True)
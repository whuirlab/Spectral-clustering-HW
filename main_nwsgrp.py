# -*- coding: utf-8 -*-
"""
Created on Tue Feb  9 16:40:35 2016

@author: Rok
"""

import numpy as np
import newsgroups20
import spectral
import scipy.spatial.distance as dist

rootdir = 'D:/Documents/Faks/AD2I/Unsupervised_learning/Project_SC/datasets/20newsgroups/Small_test_set'

words = newsgroups20.get_words(rootdir, 1000)
M, labels = newsgroups20.get_M(rootdir, words)
tf_idf_M = newsgroups20.get_tf_idf_M(M, "raw", "c")

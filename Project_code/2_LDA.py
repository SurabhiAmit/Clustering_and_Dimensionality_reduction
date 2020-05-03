import time
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import defaultdict
from helpers import  pairwiseDistCorr,nn_reg,nn_arch,reconstructionErrorN
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from itertools import product
from sklearn.preprocessing import scale
import pickle
from matplotlib import cm
import tables
from matplotlib import offsetbox
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection)
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


out = './LDA/'
cmap = cm.get_cmap('Spectral')


np.random.seed(0)
file1 = open(r'.\bioresponse.pkl', 'rb')
data1 = pickle.load(file1)
file1.close()
X = data1.data
labels = data1.target
X_trainset, X_test, Y_trainset, Y_test = train_test_split(X, labels, test_size=0.3, random_state=42, shuffle=True, stratify=labels)
X_train_unscaled, Y_train, X_val, Y_val = train_test_split(X_trainset, Y_trainset, test_size=0.2, random_state=42, shuffle=True, stratify =Y_trainset)#same as StandardScaler().fit_transform(data1.data)
X_test= scale(X_test)
X_scaled = scale(X_trainset)
X_train = scale(X)
Y_train =labels
instances_bio = X_train
labels_bio = Y_train
clusters =  [2,5,10,15,20,26,30,35,40,45,50]
dims = [1,2,40,80,170,480,670,860,990,1150,1287,1378,1453,1516,1617,1772,1777]

X2 = instances_bio.copy()
X2.flat[::instances_bio.shape[1] + 1] += 0.01  # Make X invertible
t0 = time.time()
X_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(X2, labels_bio)
tmp = defaultdict(dict)
tmp1=defaultdict(dict)
for dim in dims:
    lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=dim)
    lda.fit_transform(X_train,Y_train)
    tmp[dim] = lda.explained_variance_ratio_
    print (dim, lda.explained_variance_ratio_)
tmp2 =pd.DataFrame(tmp[1777]).T
tmp1 = pd.DataFrame(tmp).T
print(tmp1)
print(tmp2)
tmp2.to_csv(out+'bio_variance_ratio.csv')
tmp1.to_csv(out+'sec_bio_variance_ratio.csv')


dim = 200
lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=dim)
madelonX2 = lda.fit_transform(X_train,Y_train)
madelon2 = pd.DataFrame(np.hstack((madelonX2,np.atleast_2d(Y_train).T)))
cols = list(range(madelon2.shape[1]))
cols[-1] = 'Class'
madelon2.columns = cols
print(madelon2)
madelon2.to_hdf(out+'datasets.hdf','bio',complib='blosc',complevel=9)

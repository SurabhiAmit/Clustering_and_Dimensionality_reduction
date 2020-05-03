#%% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import defaultdict
from helpers import   pairwiseDistCorr,nn_reg,nn_arch,reconstructionError
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.random_projection import SparseRandomProjection, GaussianRandomProjection
from itertools import product
from sklearn.preprocessing import scale
import pickle
from matplotlib import cm
import tables
from sklearn.model_selection import train_test_split
out = './RP/'
cmap = cm.get_cmap('Spectral')

np.random.seed(0)
file1 = open(r'.\bioresponse.pkl', 'rb')
data1 = pickle.load(file1)
file1.close()
X = data1.data
labels = data1.target
X_trainset, X_test, Y_trainset, Y_test = train_test_split(X, labels, test_size=0.3, random_state=42, shuffle=True, stratify=labels)
X_train_unscaled, Y_train, X_val, Y_val = train_test_split(X_trainset, Y_trainset, test_size=0.2, random_state=42, shuffle=True, stratify =Y_trainset)
sample_size = 300
X_train = scale(X_train_unscaled)
X_val = scale(X_val)
X_test= scale(X_test)
X_scaled = scale(X_trainset)
X_train = scale(X)
Y_train =labels
instances_bio = X_train #same as StandardScaler().fit_transform(data1.data)
labels_bio = Y_train
clusters =  [2,5,10,15,20,26,30,35,40,45,50]
dims = [2,40,80,170,480,670,860,990,1150,1287,1378,1453,1516,1617,1772]
#raise
#%% data for 1

tmp = defaultdict(dict)
for i,dim in product(range(10),dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    print(rp.fit_transform(instances_bio)[1], labels_bio)
    tmp[dim][i] = pairwiseDistCorr(rp.fit_transform(instances_bio), labels_bio)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'bio_pairwise.csv')

tmp = defaultdict(dict)
for i,dim in product(range(10),dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(instances_bio)
    tmp[dim][i] = reconstructionError(rp, instances_bio)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'bio_recon_error.csv')

dim = 1378
rp = SparseRandomProjection(n_components=dim,random_state=5)
madelonX2 = rp.fit_transform(instances_bio)
madelon2 = pd.DataFrame(np.hstack((madelonX2,np.atleast_2d(labels_bio).T)))
cols = list(range(madelon2.shape[1]))
cols[-1] = 'Class'
madelon2.columns = cols
madelon2.to_csv(out+'vis_bio_DS.csv')
madelon2.to_hdf(out+'datasets.hdf','bio',complib='blosc',complevel=9)



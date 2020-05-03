
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
file1 = open(r'.\d.pkl', 'rb')
data1 = pickle.load(file1)
file1.close()
X = data1.data
labels = data1.target
X_trainset, X_test, Y_trainset, Y_test = train_test_split(X, labels, test_size=0.3, random_state=42, shuffle=True, stratify=labels)
X_train_unscaled, Y_train, X_val, Y_val = train_test_split(X_trainset, Y_trainset, test_size=0.2, random_state=42, shuffle=True, stratify =Y_trainset)
sample_size = 300
X_train = scale(X_train_unscaled)
X_test= scale(X_test)
X_scaled = scale(X_trainset)
X_train = scale(X)
Y_train =labels
instances_letter=X_train
labels_letter = Y_train
clusters =  [2,5,10,15,20,25,30,35,40,45,50]
dims = [1,2,4,5,8,10,12,15,16]
#raise
#%% data for 1

X2 = instances_letter.copy()
X2.flat[::instances_letter.shape[1] + 1] += 0.01  # Make X invertible
t0 = time.time()
X_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(X2, labels_letter)

tmp = defaultdict(dict)
val = defaultdict(dict)
for dim in dims:
    lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=dim)
    lda.fit_transform(X_train, labels_letter)
    tmp[dim] = lda.explained_variance_ratio_
tmp =pd.DataFrame(tmp[16]).T
tmp.to_csv(out+'letter_variance_ratio.csv')

tmp = defaultdict(dict)
val = defaultdict(dict)
for dim in dims:
    lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=dim)
    lda.fit_transform(X_train, labels_letter)
    tmp[dim] = lda.explained_variance_ratio_.cumsum()
tmp =pd.DataFrame(tmp[16]).T
tmp.to_csv(out+'letter_variance_ratio_cumsum.csv')

tmp1 = defaultdict(dict)
for dim in dims:
    lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=dim)
    lda.fit_transform(X_train, labels_letter)
    tmp1[dim] = reconstructionErrorN(lda, X_train)
tmp1 =pd.DataFrame(tmp1[16]).T
tmp1.to_csv(out+'recon_error.csv')


#%% data for 3
# Set this from chart 2 and dump, use clustering script to finish up
dim = 11
lda = discriminant_analysis.LinearDiscriminantAnalysis()
madelonX2 = lda.fit_transform(instances_letter, labels_letter)
madelon2 = pd.DataFrame(np.hstack((madelonX2,np.atleast_2d(labels_letter).T)))
cols = list(range(madelon2.shape[1]))
cols[-1] = 'Class'
madelon2.columns = cols
madelon2.to_hdf(out+'datasets_dim6.hdf','letter',complib='blosc',complevel=9)

dim = 6
lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=6)
madelonX2 = lda.fit_transform(X_scaled,Y_trainset)
madelon2 = pd.DataFrame(np.hstack((madelonX2,np.atleast_2d(Y_trainset).T)))
cols = list(range(madelon2.shape[1]))
cols[-1] = 'Class'
madelon2.columns = cols
madelon2.to_hdf(out+'datasetsNN_dim6.hdf','letter',complib='blosc',complevel=9)
#%% Data for 2

dims=[6,8,11]
nn_reg = [0.1,0.01]
nn_arch = [ (21,), (42,),(18,)]
grid ={'lda__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
# rp = SparseRandomProjection(random_state=5)
lda = discriminant_analysis.LinearDiscriminantAnalysis()
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('lda',lda),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)
gs.fit(X_scaled,Y_trainset)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'LetterGridSearch_CV_dim6.csv')


model = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5, alpha = 0.1, hidden_layer_sizes=(42,))
model.fit(X_scaled,Y_trainset)
score = model.score(X_test, Y_test)
print("TEST SCORE using A1 ANN:", score)

model = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5, alpha = 0.01, hidden_layer_sizes=(17,))
model.fit(X_scaled,Y_trainset)
score = model.score(X_test, Y_test)
print("TEST SCORE :", score)
#%% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from collections import defaultdict
from helpers import   pairwiseDistCorr,reconstructionError
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
from sklearn import metrics
accuracies = []
np.random.seed(0)
file1 = open(r'.\d.pkl', 'rb')
data1 = pickle.load(file1)
file1.close()
X = data1.data
labels = data1.target
X_trainset, X_test, Y_trainset, Y_test = train_test_split(X, labels, test_size=0.3, random_state=42, shuffle=True, stratify=labels)
X_train_unscaled, Y_train, X_val, Y_val = train_test_split(X_trainset, Y_trainset, test_size=0.2, random_state=42, shuffle=True, stratify =Y_trainset)
sample_size = 300
instances_letter = scale(X_train_unscaled) #same as StandardScaler().fit_transform(data1.data)
labels_letter = Y_train
X_test= scale(X_test)
X_scaled = scale(X_trainset)
X_train = scale(X)
Y_train =labels
from sklearn.random_projection import johnson_lindenstrauss_min_dim
print("DIMENSIONS: ",johnson_lindenstrauss_min_dim(20000,eps=0.1))
clusters =  [2,5,10,15,20,25,30,35,40,45,50]
dims = [2,4,5,8,10,12,15,16]
#raise

print(johnson_lindenstrauss_min_dim(1797,eps=0.1))
#accuracies = defaultdict(dict)
tmp = defaultdict(dict)
for i,dim in product(range(10),dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(X_train)
    tmp[dim][i] = reconstructionError(rp, X_train)
    #accuracies[dim][i]=metrics.accuracy_score(rp.predict(X_train),Y_train)
tmp =pd.DataFrame(tmp).T
#accuracies= pd.dataframe(accuracies)
tmp.to_csv(out+'letter_recon_error.csv')
#accuracies.to_csv(out+'letter_acc.csv')
tmp = defaultdict(dict)
for i,dim in product(range(10),dims):
    rp = SparseRandomProjection(random_state=i, n_components=dim)
    rp.fit(X_scaled)
    tmp[dim][i] = reconstructionError(rp, X_scaled)
tmp =pd.DataFrame(tmp).T
tmp.to_csv(out+'NN_letter_recon_error.csv')


dim = 10
rp = SparseRandomProjection(n_components=dim,random_state=5)
madelonX2 = rp.fit_transform(X_train)
madelon2 = pd.DataFrame(np.hstack((madelonX2,np.atleast_2d(Y_train).T)))
cols = list(range(madelon2.shape[1]))
cols[-1] = 'Class'
madelon2.columns = cols
madelon2.to_hdf(out+'datasets.hdf','letter',complib='blosc',complevel=9)

dim = 10
rp = SparseRandomProjection(n_components=dim,random_state=5)
madelonX2 = rp.fit_transform(X_scaled)
madelon2 = pd.DataFrame(np.hstack((madelonX2,np.atleast_2d(Y_trainset).T)))
cols = list(range(madelon2.shape[1]))
cols[-1] = 'Class'
madelon2.columns = cols
madelon2.to_csv(out+'vis_letter_DS.csv')
madelon2.to_hdf(out+'datasetsNN.hdf','letter',complib='blosc',complevel=9)
#%% Data for 2

dims = range(1,17)
nn_reg = [0.1]
nn_arch = [ (42,)]
grid ={'rp__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
rp = SparseRandomProjection(random_state=5)
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('rp',rp),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=10)
gs.fit(X_scaled, Y_trainset)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'LettergridSearch_A1.csv')

model = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5, alpha = 0.1, hidden_layer_sizes=(42,))
model.fit(X_scaled,Y_trainset)
test_score = model.score(X_test, Y_test)
train_score = model.score(X_scaled,Y_trainset)
print("TEST SCORE using A1 ANN:", test_score)





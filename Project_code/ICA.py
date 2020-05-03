import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import nn_arch, nn_reg
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import FastICA
from sklearn.preprocessing import scale
import pickle
from matplotlib import cm
import tables
from sklearn.model_selection import train_test_split
out = './ICA/'
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
labels_letter = Y_train
X_test= scale(X_test)
X_scaled = scale(X_trainset)
X_train = scale(X)
Y_train =labels
clusters =  [2,5,10,15,20,25,30,35,40,45,50]
dims = range(1,17)
#raise
#%% data for 1

ica = FastICA(random_state=5)
kurt = {}
for dim in dims:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(X_train)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt)
kurt.to_csv(out+'training_letter_kurtosis.csv')

ica = FastICA(random_state=5)
kurt = {}
for dim in dims:
    ica.set_params(n_components=dim)
    tmp = ica.fit_transform(X_scaled)
    tmp = pd.DataFrame(tmp)
    tmp = tmp.kurt(axis=0)
    kurt[dim] = tmp.abs().mean()

kurt = pd.Series(kurt)
kurt.to_csv(out+'training_letter_kurtosisNN.csv')

dim = 11
ica = FastICA(n_components=dim,random_state=10)
letterX2 = ica.fit_transform(X_train)
letter2 = pd.DataFrame(np.hstack((letterX2,np.atleast_2d(Y_train).T)))
cols = list(range(letter2.shape[1]))
cols[-1] = 'Class'
letter2.columns = cols
letter2.to_hdf(out+'datasets.hdf','letter',complib='blosc',complevel=9)

dim = 11
ica = FastICA(n_components=dim,random_state=10)
letterX2 = ica.fit_transform(X_scaled)
letter2 = pd.DataFrame(np.hstack((letterX2,np.atleast_2d(Y_trainset).T)))
cols = list(range(letter2.shape[1]))
cols[-1] = 'Class'
letter2.columns = cols
letter2.to_csv(out+'vis_letter_DS.csv')
letter2.to_hdf(out+'datasetsNN.hdf','letter',complib='blosc',complevel=9)


nn_reg = [0.1]
nn_arch = [(42,)]
#%% Data for 2
dims = range(1,17)
grid ={'ica__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
ica = FastICA(random_state=5)
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('ica',ica),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=5)

gs.fit(X_scaled,Y_trainset)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Letters_gridsearchCV.csv')

model = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5, alpha = 0.1, hidden_layer_sizes=(42,))
model.fit(X_scaled,Y_trainset)
score = model.score(X_scaled,Y_trainset)
print("TEST SCORE using A1 ANN:", score)

#%% data for 3
# Set this from chart 2 and dump, use clustering script to finish up
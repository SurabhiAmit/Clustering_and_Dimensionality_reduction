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
file1 = open(r'.\bioresponse.pkl', 'rb')
data1 = pickle.load(file1)
file1.close()
X = data1.data
labels = data1.target
X_trainset, X_test, Y_trainset, Y_test = train_test_split(X, labels, test_size=0.3, random_state=42, shuffle=True, stratify=labels)
X_train_unscaled, Y_train, X_val, Y_val = train_test_split(X_trainset, Y_trainset, test_size=0.2, random_state=42, shuffle=True, stratify =Y_trainset)
sample_size = 300
X_train = scale(X_train_unscaled)
instances_bio = scale(X_train) #same as StandardScaler().fit_transform(data1.data)
labels_bio = Y_train
X_val = scale(X_val)
X_test= scale(X_test)
X_scaled = scale(X_trainset)
X_train = scale(X)
Y_train =labels

clusters =  [2,5,10,15,20,26,30,35,40,45,50]
dims = [2,12,22,62,92,202,302,402,502,700,800,900,1200,1400,1500,1700,1750]
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
print(kurt)
kurt = pd.Series(kurt)
kurt.to_csv(out+'bio_kurtosis.csv')

ica = FastICA(random_state=5)
kurt = {}
for dim in dims:
    ica.set_params(n_components=dim)
    ica.fit(X_scaled)
    tmp1= ica.fit_transform(X_val)
    tmp1 = tmp1.kurt(axis=0)
    kurt[dim] = tmp1.abs().mean()

kurt1 = pd.Series(kurt)
kurt1.to_csv(out+'bio_kurtosis_NN.csv')

dim = 302
ica = FastICA(n_components=dim,random_state=10)

bioX2 = ica.fit_transform(X_train)
bio2 = pd.DataFrame(np.hstack((bioX2,np.atleast_2d(Y_train).T)))
cols = list(range(bio2.shape[1]))
cols[-1] = 'Class'
bio2.columns = cols
bio2.to_csv(out+'vis_bio_DS.csv')
bio2.to_hdf(out+'datasets.hdf','bio',complib='blosc',complevel=9)


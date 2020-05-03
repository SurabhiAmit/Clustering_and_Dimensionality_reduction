#%% Imports
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from matplotlib import cm
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tables
from collections import defaultdict
from helpers import  reconstructionError

out = './PCA/'
cmap = cm.get_cmap('Spectral')

np.random.seed(0)
file1 = open(r'.\d.pkl', 'rb')
data1 = pickle.load(file1)
file1.close()
instances = data1.data
X = data1.data
labels = data1.target
X_trainset, X_test, Y_trainset, Y_test = train_test_split(X, labels, test_size=0.3, random_state=42, shuffle=True, stratify=labels)
X_train_unscaled, Y_train, X_val, Y_val = train_test_split(X_trainset, Y_trainset, test_size=0.2, random_state=42, shuffle=True, stratify =Y_trainset)
n_samples, n_features = instances.shape
n_digits = len(np.unique(data1.target))
sample_size = 300
X_train = scale(X_train_unscaled)
instances_letter = scale(X_trainset) #same as StandardScaler().fit_transform(data1.data)
labels_letter = Y_trainset
#X_val = scale(X_val)
X_test= scale(X_test)
X_scaled = scale(X_trainset)
X_train = scale(X)
Y_train=labels

clusters =  [2,5,10,15,20,26,30,35,40,45,50]
dims = [1,2,5,8,10,12,15,16]
#raise

#%% data for 1
pca = PCA(random_state=5)
pca.fit(X_train)
tmp = pd.Series(data = pca.explained_variance_,index = range(1,17))
tmp.to_csv(out+'letter_variance.csv')
tmp.plot(title='Eigen value distribution')
plt.ylabel('Eigen value')
plt.xlabel('Principal components')
plt.grid(True)
plt.show()
tmp = pd.Series(data = pca.explained_variance_ratio_.cumsum(),index = range(1,17))
tmp.plot(title='Percentage of variance explained')
plt.ylabel('Percentage of variance explained')
plt.xlabel('Number of components')
plt.grid(True)
plt.axhline(y=0.9, color='r', linestyle='dashed')
plt.show()
tmp.to_csv(out+'letter_ratio.csv')
plt.show()
dim = 10
pca = PCA(n_components=dim,random_state=10)
features_letter = pca.fit_transform(X_train)
letterdf = pd.DataFrame(np.hstack((features_letter,np.atleast_2d(Y_train).T)))
cols = list(range(letterdf.shape[1]))
cols[-1] = 'Class'
letterdf.columns = cols
letterdf.to_hdf(out+'datasets.hdf','letter',complib='blosc',complevel=9)
print("Dataset ready for clustering")
tmp = defaultdict(dict)
for i in range(1,17):
    pca = PCA(n_components=i, random_state=10)
    pca.fit(X_scaled)
    tmp[i] = reconstructionError(pca, X_scaled)
print(tmp)
tmp1 =pd.DataFrame(tmp,index=[0]).T
tmp1.to_csv(out+'letter_recon_error.csv')

###############################################################################################################################
pca = PCA(random_state=5)
pca.fit(X_scaled)
tmp = pd.Series(data = pca.explained_variance_,index = range(1,17))
tmp.to_csv(out+'letter_variance_NN.csv')
tmp.plot(title='Eigen value distribution')
plt.ylabel('Eigen value')
plt.xlabel('Principal components')
plt.grid(True)
plt.show()
tmp = pd.Series(data = pca.explained_variance_ratio_.cumsum(),index = range(1,17))
tmp.plot(title='Percentage of variance explained')
plt.ylabel('Percentage of variance explained')
plt.xlabel('Number of components')
plt.grid(True)
plt.axhline(y=0.9, color='r', linestyle='dashed')
plt.show()
tmp.to_csv(out+'letter_ratio_NN.csv')
plt.show()
dim = 10
pca = PCA(n_components=dim,random_state=10)
features_letter = pca.fit_transform(X_train)
letterdf = pd.DataFrame(np.hstack((features_letter,np.atleast_2d(Y_train).T)))
cols = list(range(letterdf.shape[1]))
cols[-1] = 'Class'
letterdf.columns = cols
letterdf.to_csv(out+'vis_letter_DS.csv')
letterdf.to_hdf(out+'datasetsNN.hdf','letter',complib='blosc',complevel=9)
print("Dataset ready for clustering")

###############################################################################################################################
dims = range(1,17)
nn_reg = [0.1]
nn_arch = [(42,)]
#%% Data for 2
grid ={'pca__n_components':dims,'NN__alpha':nn_reg,'NN__hidden_layer_sizes':nn_arch}
pca = PCA(random_state=5)
mlp = MLPClassifier(activation='relu',max_iter=2000,early_stopping=True,random_state=5)
pipe = Pipeline([('pca',pca),('NN',mlp)])
gs = GridSearchCV(pipe,grid,verbose=10,cv=10)
gs.fit(X_scaled,Y_trainset)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out+'Letter_GridSearchCV_A1.csv')

model = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5, alpha = 0.1, hidden_layer_sizes=(42,))
model.fit(X_scaled,Y_trainset)
score = model.score(X_scaled,Y_trainset)
print("TRAIN SCORE using A1 ANN:", score)
###############################################################################################################################


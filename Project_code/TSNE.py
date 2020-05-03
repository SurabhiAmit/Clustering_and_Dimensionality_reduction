import warnings
import sys
from scipy import linalg
from sklearn import mixture
from scipy.spatial.distance import cdist
warnings.simplefilter("ignore")
from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from helpers import cluster_acc, myGMM,nn_arch,nn_reg
from collections import defaultdict
from time import clock
from sklearn.mixture import GaussianMixture as GMM
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.metrics import silhouette_samples, silhouette_score
import matplotlib.cm as cm
import itertools
import matplotlib as mpl
from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import time
import pickle
from sklearn.manifold import TSNE
#from pandas.tools.plotting import scatter_matrix
from mpl_toolkits.mplot3d import Axes3D

out = './{}/'.format("TSNE")
#code reference: https://stackoverflow.com/questions/1047318/easiest-way-to-persist-a-data-structure-to-a-file-in-python
file1 = open(r'.\bioresponse.pkl', 'rb')
data1 = pickle.load(file1)
file1.close()
#print(data1)
np.random.seed(42)
instances = scale(data1.data)
#place to change data
X = data1.data
labels = data1.target
X_trainset, X_test, Y_trainset, Y_test = train_test_split(X, labels, test_size=0.3, random_state=42, shuffle=True, stratify=labels)
X_train_unscaled, X_val_unscaled,Y_train, Y_val = train_test_split(X_trainset, Y_trainset, test_size=0.2, random_state=42, shuffle=True, stratify =Y_trainset)
n_samples, n_features = instances.shape
n_digits = len(np.unique(data1.target))
sample_size = 300
X_train = scale(X_train_unscaled)
X_val = scale(X_val_unscaled)
X_test = scale(X_test)
X_scaled = scale(X_trainset)
X_train = scale(X)
Y_train = labels
print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))


bioresponsesX2D = TSNE(n_components=3,verbose=10, random_state=5).fit_transform(X_train)
biodf = pd.DataFrame(np.hstack((bioresponsesX2D, np.atleast_2d(Y_train).T)), columns=['x', 'y','z', 'target'])
biodf.to_csv(out + 'bioresponses3D.csv')

file1 = open(r'.\d.pkl', 'rb')
data1 = pickle.load(file1)
file1.close()
np.random.seed(42)
instances = scale(data1.data)
X = data1.data
labels = data1.target
X_trainset, X_test, Y_trainset, Y_test = train_test_split(X, labels, test_size=0.3, random_state=42, shuffle=True, stratify=labels)
X_train_unscaled, X_val_unscaled,Y_train, Y_val = train_test_split(X_trainset, Y_trainset, test_size=0.2, random_state=42, shuffle=True, stratify =Y_trainset)
n_samples, n_features = instances.shape
n_digits = len(np.unique(data1.target))
X_val = scale(X_val_unscaled)
X_test = scale(X_test)
X_scaled = scale(X_trainset)
X_train = scale(X)
Y_train = labels

lettersX2D = TSNE(n_components=3,verbose=10, random_state=5).fit_transform(X_train)
letdf = pd.DataFrame(np.hstack((lettersX2D, np.atleast_2d(Y_train).T)), columns=['x', 'y','z', 'target'])
letdf.to_csv(out + 'letters3D.csv')
from mpl_toolkits.mplot3d import Axes3D

biodf = pd.read_csv('./TSNE/bioresponses3D.csv')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmhot = plt.get_cmap("hot")
c=np.abs(biodf['target'])
cax = ax.scatter(biodf['x'], biodf['y'], biodf['z'], c=biodf['target'])
fig.colorbar(cax)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title("t-SNE plot for bioresponse 3-D")
plt.show()
import seaborn as sns
letdf= pd.read_csv('./TSNE/letters3D.csv')
# Unique category labels: 'D', 'F', 'G', ...
color_labels = letdf['target'].unique()
rgb_values = sns.color_palette("Set2", len(color_labels))
color_map = dict(zip(color_labels, rgb_values))
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cax = ax.scatter(letdf['x'], letdf['y'], letdf['z'],  c=letdf['target'].map(color_map), cmap=color_map)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title("t-SNE plot for letter recognition 3-D")
plt.show()
plt.subplots_adjust(top=0.88)
plt.show()


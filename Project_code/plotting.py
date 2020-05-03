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
from helpers import cluster_acc, myGMM, nn_arch, nn_reg
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
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
# from pandas.tools.plotting import scatter_matrix


#RF
letdf= pd.read_csv('./RF/letters3D_dim6.csv')
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
ax.set_title("t-SNE plot for letter recognition 3-D after RF")
plt.show()
plt.subplots_adjust(top=0.88)
plt.show()

biodf = pd.read_csv('./RF/bio_3D.csv')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmhot = plt.get_cmap("hot")
c=np.abs(biodf['target'])
cax = ax.scatter(biodf['x'], biodf['y'], biodf['z'], c=biodf['target'])
fig.colorbar(cax)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title("t-SNE plot for bioresponse 3-D after RF ")
plt.show()

#LDA
letdf= pd.read_csv('./LDA/letters3D.csv')
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
ax.set_title("t-SNE plot for letter recognition 3-D after LDA")
plt.show()
plt.subplots_adjust(top=0.88)
plt.show()

biodf = pd.read_csv('./LDA/bio_3D.csv')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmhot = plt.get_cmap("hot")
c=np.abs(biodf['target'])
cax = ax.scatter(biodf['x'], biodf['y'], biodf['z'], c=biodf['target'])
fig.colorbar(cax)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title("t-SNE plot for bioresponse 3-D after LDA ")
plt.show()

#RP
letdf= pd.read_csv('./RP/letters3D.csv')
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
ax.set_title("t-SNE plot for letter recognition 3-D after RP")
plt.show()
plt.subplots_adjust(top=0.88)
plt.show()

biodf = pd.read_csv('./RP/bio_3D.csv')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmhot = plt.get_cmap("hot")
c=np.abs(biodf['target'])
cax = ax.scatter(biodf['x'], biodf['y'], biodf['z'], c=biodf['target'])
fig.colorbar(cax)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title("t-SNE plot for bioresponse 3-D after RP")
plt.show()

#ICA
letdf= pd.read_csv('./ICA/letters3D.csv')
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
ax.set_title("t-SNE plot for letter recognition 3-D after ICA")
plt.show()
plt.subplots_adjust(top=0.88)
plt.show()

biodf = pd.read_csv('./ICA/bio3D.csv')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmhot = plt.get_cmap("hot")
c=np.abs(biodf['target'])
cax = ax.scatter(biodf['x'], biodf['y'], biodf['z'], c=biodf['target'])
fig.colorbar(cax)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title("t-SNE plot for bioresponse 3-D after ICA ")
plt.show()

#PCA
letdf= pd.read_csv('./PCA/letters3D.csv')
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
ax.set_title("t-SNE plot for letter recognition 3-D after PCA")
plt.show()
plt.subplots_adjust(top=0.88)
plt.show()

biodf = pd.read_csv('./PCA/bio3D.csv')
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
cmhot = plt.get_cmap("hot")
c=np.abs(biodf['target'])
cax = ax.scatter(biodf['x'], biodf['y'], biodf['z'], c=biodf['target'])
fig.colorbar(cax)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title("t-SNE plot for bioresponse 3-D after PCA")
plt.show()


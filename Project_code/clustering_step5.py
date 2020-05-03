import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from time import clock
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from helpers import cluster_acc, myGMM
from sklearn.metrics import adjusted_mutual_info_score as ami
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import sys
import time
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
import warnings
warnings.simplefilter("ignore")
from scipy.spatial.distance import cdist
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score
with warnings.catch_warnings():
    warnings.simplefilter("ignore",category=DeprecationWarning)

out = './{}/'.format(sys.argv[1])

print("DR algorithm used is",sys.argv[1] )
np.random.seed(0)
letters = pd.read_hdf(out + 'datasetsNN.hdf', 'letter')
lettersX = letters.drop('Class', 1).copy().values
lettersY = letters['Class'].copy().values
X_train = scale(lettersX)
Y_train= lettersY
clusters = [2, 5, 10, 15, 20, 25, 30, 35, 40]

sil={}
n_clusters=15

fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)
ax1.set_xlim([-1, 1])
ax1.set_ylim([0, len(X_train) + (n_clusters + 1) * 10])
clusterer = KMeans(n_clusters=n_clusters, random_state=5)
cluster_labels = clusterer.fit_predict(X_train)
model = clusterer.fit(X_train)
silhouette_avg = silhouette_score(X_train, cluster_labels)
print("For n_clusters =", n_clusters,
      "The average training silhouette_score is :", silhouette_avg)
sample_silhouette_values = silhouette_samples(X_train, cluster_labels)
sil[n_clusters] = silhouette_avg
y_lower = 10
for i in range(n_clusters):
    ith_cluster_silhouette_values = \
        sample_silhouette_values[cluster_labels == i]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    color = cm.nipy_spectral(float(i) / n_clusters)
    ax1.fill_betweenx(np.arange(y_lower, y_upper),
                      0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
    y_lower = y_upper + 10  # 10 for the 0 samples
ax1.set_title("The silhouette plot for the various clusters.")
ax1.set_xlabel("The silhouette coefficient values")
ax1.set_ylabel("Cluster label")
# The vertical line for average silhouette score of all the values
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
ax1.set_yticks([])  # Clear the yaxis labels / ticks
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
# 2nd Plot showing the actual clusters formed
colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
ax2.scatter(X_train[:, 0], X_train[:, 1], marker='.', s=30, lw=0, alpha=0.7,
            c=colors, edgecolor='k')
# Labeling the clusters
centers = clusterer.cluster_centers_
# Draw white circles at cluster centers
ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
            c="white", alpha=1, s=200, edgecolor='k')
for i, c in enumerate(centers):
    ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                s=50, edgecolor='k')
ax2.set_title("The visualization of the clustered data.")
ax2.set_xlabel("Feature space for the 1st feature")
ax2.set_ylabel("Feature space for the 2nd feature")
plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
              "with n_clusters = %d" % n_clusters),
             fontsize=14, fontweight='bold')
print("SILHOUETTE VALUES", sil)
plt.show()
plt.figure(1)
res = list()
for n in clusters:
    kmeans = KMeans(n_clusters=n,random_state=5)
    kmeans.fit(X_train)
    res.append(np.average(np.min(cdist(X_train, kmeans.cluster_centers_, 'euclidean'), axis=1)))
plt.grid(True)
plt.plot(clusters, res)
plt.title('Elbow curve for K-Means using Euclidean distance')
plt.xlabel("Number of clusters")
plt.ylabel('Sum of squared distances')



plt.figure(2)
n_components_range = clusters
n_components = np.array(n_components_range)
models = [GMM(n, covariance_type='full', random_state=0).fit(X_train)
          for n in n_components]
plt.plot(n_components, [m.bic(X_train) for m in models], label='BIC')
plt.plot(n_components, [m.aic(X_train) for m in models], label='AIC')

plt.legend(loc='best')
plt.xlabel('n_components')
plt.title('AIC and BIC values for GMM, after PCA')
plt.ylabel("value")
plt.tight_layout()

plt.show()

#apply neural network

# %% NN fit data (2,3)
nn_reg = [0.1]
nn_arch = [(42,)]
clusters=[15,20,25,30,35]
grid = {'km__n_clusters': clusters, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
km = kmeans(random_state=5)
pipe = Pipeline([('km', km), ('NN', mlp)])
gs = GridSearchCV(pipe, grid, verbose=10, cv=5)
gs.fit(X_train, lettersY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out + 'Letters_cluster_Kmeans.csv')

grid = {'gmm__n_components': clusters, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
gmm = myGMM(random_state=5)
pipe = Pipeline([('gmm', gmm), ('NN', mlp)])
gs = GridSearchCV(pipe, grid, verbose=10, cv=5)
gs.fit(X_train, lettersY)
tmp = pd.DataFrame(gs.cv_results_)
tmp.to_csv(out + 'Letters_cluster_GMM.csv')

# %% For chart 4/5
nn_reg = [0.1]
nn_arch = [(42,)]
clusters=[10,15,20,25,30,35,40]
with open(out+'NN_with_clusterfeature_and_dim_red_features_Kmeans.csv', 'a') as f:
    for i in clusters:
        km_model = KMeans(init='k-means++', n_clusters=i, n_init=10)
        km_model.fit(X_train)
        pred = km_model.predict(X_train)[...,None]
        X_train1= np.append(X_train, pred,1)
        mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
        parameters={'alpha': nn_reg, 'hidden_layer_sizes': nn_arch}
        gs = GridSearchCV(mlp, parameters, verbose=10, cv=10)
        gs.fit(X_train1, Y_train)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(f, header=False)

with open(out+'NN_with_clusterfeature_and_dim_red_features_GMM.csv', 'a') as f:
    for i in clusters:
        gmm_model = GMM( n_components=i,covariance_type='full', random_state=0)
        gmm_model.fit(X_train)
        pred2 = gmm_model.predict(X_train)[...,None]
        X_train2= np.append(X_train, pred2,1)
        mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
        parameters={'alpha': nn_reg, 'hidden_layer_sizes': nn_arch}
        gs = GridSearchCV(mlp, parameters, verbose=10, cv=10)
        gs.fit(X_train2,Y_train)
        tmp = pd.DataFrame(gs.cv_results_)
        tmp.to_csv(f, header=False)



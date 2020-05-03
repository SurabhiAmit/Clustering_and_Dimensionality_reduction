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
from helpers import cluster_acc, myGMM
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
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
import sys
from sklearn.pipeline import Pipeline

#Topic: DATA PREPROCESSING
out = './{}/'.format("Output_Bioresponse")
with warnings.catch_warnings():
    warnings.simplefilter("ignore",category=DeprecationWarning)

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
clusters = [2,3,4,5,6,10]
#[2,5,10,15,25,30,40,45]

#TOPIC: GETTING OPTIMAL K
#Elbow method for K_Means using distortion

plt.figure(1)
res = list()
res_val=list()
for n in clusters:
    kmeans = KMeans(n_clusters=n,random_state=5)
    kmeans.fit(X_train)
    res.append(np.average(np.min(cdist(X_train, kmeans.cluster_centers_, 'euclidean'), axis=1)))
    res_val.append(np.average(np.min(cdist(X_val, kmeans.cluster_centers_, 'euclidean'), axis=1)))
plt.grid(True)
plt.plot(clusters, res)
#plt.plot(clusters,res_val)
plt.title('Elbow curve for K-Means using Euclidean distance')
#plt.legend(['training', 'validation'])
plt.xlabel("Number of clusters")
plt.ylabel('Sum of squared distances')
plt.show()

#Elbow curve using score
SSE = defaultdict(dict)
ll = defaultdict(dict)
acc = defaultdict(lambda: defaultdict(dict))
adjMI = defaultdict(lambda: defaultdict(dict))
km_1 = KMeans(random_state=5)
gmm_models1 = GMM(random_state=0)
st = clock()
for k in clusters:
    km_1.set_params(n_clusters=k)
    gmm_models1.set_params(n_components=k)
    km_1.fit(X_train)
    gmm_models1.fit(X_train)
    SSE[k]['Bioresponse'] = km_1.score(X_train)
    ll[k]['Bioresponse'] = gmm_models1.score(X_train)
    acc[k]['Bio_train']['Kmeans'] = cluster_acc(Y_train,km_1.predict(X_train))
    acc[k]['Bio_train']['GMM'] = cluster_acc(Y_train,gmm_models1.predict(X_train))
    adjMI[k]['Bio_train']['Kmeans'] = ami(Y_train,km_1.predict(X_train))
    adjMI[k]['Bio_train']['GMM'] = ami(Y_train,gmm_models1.predict(X_train))
    #SSE[k]['Letter_val'] = km_1.score(X_val)
   # ll[k]['Letter_val'] = gmm_models1.score(X_val)
    #acc[k]['Letter_val']['Kmeans'] = cluster_acc(Y_val, km_1.predict(X_val))
    #acc[k]['Letter_val']['GMM'] = cluster_acc(Y_val, gmm_models1.predict(X_val))
    #adjMI[k]['Letter_val']['Kmeans'] = ami(Y_val, km_1.predict(X_val))
    #adjMI[k]['Letter_val']['GMM'] = ami(Y_val, gmm_models1.predict(X_val))
SSE = (-pd.DataFrame(SSE)).T
plt.figure(2)
plot_SSE = SSE.copy()
plot_SSE.plot(marker = 'o')
#plt.legend(['training','validation'])
plt.xlabel('Number of clusters')
plt.ylabel('Negative score value')
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.grid(True)
plt.title('Elbow curve using negative score value for K-Means')
SSE.rename(columns = lambda x: x+' SSE',inplace=True)
ll = (pd.DataFrame(ll)).T
plot_ll = ll.copy()
plt.figure(3)
plot_ll.plot(marker = 'o')
#plt.legend(['training','validation'])
plt.grid(True)
plt.title('Elbow curve using score value for GMM')
plt.xlabel('Number of clusters')
plt.ylabel('Log-likelihood')
ll.rename(columns = lambda x: x+' log-likelihood',inplace=True)
acc = pd.Panel(acc)
adjMI = pd.Panel(adjMI)
plt.show()

#writing 1.accuracy, 2.Adjusted mutual information values for a range of number of clusters
SSE.to_csv(out+'SSE.csv')
ll.to_csv(out+'loglikelihood.csv')
acc.ix[:,:,'Bio_train'].to_csv(out+'Bio_train_acc1.csv')
adjMI.ix[:,:,'Bio_train'].to_csv(out+'Bio_train_adjMI1.csv')
#acc.ix[:,:,'Letter_val'].to_csv(out+'Letter_val_acc1.csv')
#adjMI.ix[:,:,'Letter_val'].to_csv(out+'Letter_val_adjMI1.csv')

#AIC,BOC for GMM
plt.figure(4)
n_components_range = clusters
n_components = np.array(n_components_range)
#print(x3, n_components)
models = [GMM(n, covariance_type='full', random_state=0).fit(X_train)
          for n in n_components]

plt.plot(n_components, [m.bic(X_train) for m in models], label='BIC')
plt.plot(n_components, [m.aic(X_train) for m in models], label='AIC')
#plt.plot(n_components, [m.bic(X_val) for m in models], label='BIC validation')
#plt.plot(n_components, [m.aic(X_val) for m in models], label='AIC validation')
plt.legend(loc='best')
plt.xlabel('number of clusters')
plt.ylabel("value")
plt.title("BIC and AIC values for GMM")
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()

#Number of samples per component
plt.figure(5)
plt.figure(figsize=(8, 10))
lowest_bic = 0
bic = []
#bic_val=[]
n_components_range = clusters
cv_types = ['spherical', 'tied', 'diag', 'full']
for cv_type in cv_types:
    for n_components in n_components_range:
        gmm = mixture.GaussianMixture(n_components=n_components,
                                      covariance_type=cv_type, random_state=0)
        gmm.fit(X_train)
        bic.append(gmm.bic(X_train))
        #bic_val.append(gmm.bic(X_val))
        if bic[-1] < lowest_bic:
            lowest_bic = bic[-1]
            best_gmm = gmm
bic = np.array(bic)
#bic_val = np.array(bic_val)
color_iter = itertools.cycle(['navy', 'turquoise', 'cornflowerblue','darkorange'])
bars = []
# Plot the BIC scores
for i, (cv_type, color) in enumerate(zip(cv_types, color_iter)):
    xpos = np.array(n_components_range) + .2 * (i - 2)
    bars.append(plt.bar(xpos, bic[i * len(n_components_range):
                                  (i + 1) * len(n_components_range)],
                        width=.2, color=color))
plt.xticks(n_components_range)
plt.ylim([bic.min() * 1.01 - .01 * bic.max(), bic.max()])
plt.title('BIC score per model')
xpos = np.mod(bic.argmin(), len(n_components_range)) + .65 +\
    .2 * np.floor(bic.argmin() / len(n_components_range))
#plt.text(xpos, bic.min() * 0.97 + .03 * bic.max(), '*', fontsize=14)
plt.xlabel('Number of components')
plt.legend([b[0] for b in bars], cv_types)
plt.grid(True)
plt.show()
plt.figure(6)


#silhoutte analysis
#http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
#do till 45 only
range_n_clusters = clusters
for n_clusters in range_n_clusters:
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-1, 1])
    ax1.set_ylim([0, len(X_train) + (n_clusters + 1) * 10])
    clusterer = KMeans(n_clusters=n_clusters, random_state=5)
    cluster_labels = clusterer.fit_predict(X_train)
    model = clusterer.fit(X_train)
    val_labels = model.predict(X_val)
    print(cluster_labels)
    silhouette_avg = silhouette_score(X_train, cluster_labels)
    sil_val = silhouette_score(X_val, val_labels)
    print("For n_clusters =", n_clusters,
          "The average training silhouette_score is :", silhouette_avg)
    #print("For n_clusters =", n_clusters,
         # "The average validation silhouette_score is :", silhouette_avg)
    sample_silhouette_values = silhouette_samples(X_train, cluster_labels)
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
        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        # Compute the new y_lower for next plot
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

plt.show()

#KMEANS - 1. ELBOW METHOD USING SCORE, DISTORTION. 2. SILHOUTTE ANALYSIS
#GMM - 1. ELBOW METHOD USING SCORE 2.AIC and BIC

#Optimal k for K-Means:45
#Optimal k for GMM =40

#EVALUATION and COMPARISON
#1. METRICS (as below)
#2.ACCURACY from helpers.py

#code reference: http://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_digits.html
print ("PRINTING RESULTS TO CONSOLE ....")
print(82 * '_')
print("TRAINING")
print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\t\tAMI\tsilhouette')
def bench_k_means(estimator, name, data):
    t0 = time.time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time.time() - t0), estimator.inertia_,
             metrics.homogeneity_score(Y_train, estimator.labels_),
             metrics.completeness_score(Y_train, estimator.labels_),
             metrics.v_measure_score(Y_train, estimator.labels_),
             metrics.adjusted_rand_score(Y_train, estimator.labels_),
             metrics.adjusted_mutual_info_score(Y_train,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))

bench_k_means(KMeans(init='k-means++', n_clusters=4, n_init=10),
              name="k-means++", data=X_train)
#n-init = number of runs, the best run is considered in terms of inertia
bench_k_means(KMeans(init='random', n_clusters=4, n_init=10),
              name="random", data=X_train)

print("TRAINING")
print('init\t\ttime\thomo\tcompl\tv-meas\tARI\t\tAMI\tsilhouette')
def bench_gmm(estimator, name, data):
    t0 = time.time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time.time() - t0),
             metrics.homogeneity_score(Y_train, estimator.predict(data)),
             metrics.completeness_score(Y_train,estimator.predict(data)),
             metrics.v_measure_score(Y_train, estimator.predict(data)),
             metrics.adjusted_rand_score(Y_train, estimator.predict(data)),
             metrics.adjusted_mutual_info_score(Y_train,  estimator.predict(data)),
             metrics.silhouette_score(data, estimator.predict(data),
                                      metric='euclidean',
                                      sample_size=sample_size)))

bench_gmm(GMM( n_components=2,covariance_type='full', random_state=0,init_params='kmeans'), name="GMM", data=X_train)
bench_gmm(GMM( n_components=2,covariance_type='full', random_state=0,init_params='random'), name="GMM", data=X_train)



import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from time import clock
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans as kmeans
from sklearn.mixture import GaussianMixture as GMM
from collections import defaultdict
from helpers import cluster_acc, myGMM, nn_arch, nn_reg
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
with warnings.catch_warnings():
    warnings.simplefilter("ignore",category=DeprecationWarning)
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score


out = './{}/'.format(sys.argv[1])
print("DR algorithm used is",sys.argv[1] )
algo=sys.argv[1]
np.random.seed(0)
letters = pd.read_hdf(out + 'datasets.hdf', 'bio')
lettersX = letters.drop('Class', 1).copy().values
lettersY = letters['Class'].copy().values
X_trainset, X_test, Y_trainset, Y_test = train_test_split(lettersX,lettersY, test_size=0.2, random_state=42, shuffle=True, stratify=lettersY)
X_train_unscaled, Y_train, X_val, Y_val = train_test_split(X_trainset, Y_trainset, test_size=0.2, random_state=42, shuffle=True, stratify =Y_trainset)
sample_size = 300
X_train = StandardScaler().fit_transform(lettersX)
Y_train=lettersY
lettersX=X_train


clusters = [2,3,4,5,6,7,8,9]
sil={}
n_clusters=5

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
l=i
plt.figure(1)
res = list()
for n in clusters:
    kmeans = KMeans(n_clusters=n,random_state=5)
    kmeans.fit(X_train)
    res.append(np.average(np.min(cdist(X_train, kmeans.cluster_centers_, 'euclidean'), axis=1)))
plt.grid(True)
plt.plot(clusters, res)

plt.title('Elbow curve for K-Means using Euclidean distance after {}'.format(algo))
plt.xlabel("Number of clusters")
plt.ylabel('Sum of squared distances')
plt.show()

SSE = defaultdict(dict)
ll = defaultdict(dict)
acc = defaultdict(lambda: defaultdict(dict))
adjMI = defaultdict(lambda: defaultdict(dict))
km = KMeans(random_state=5)
gmm = GMM(random_state=5)

st = clock()
for k in clusters:
    km.set_params(n_clusters=k)
    gmm.set_params(n_components=k)
    km.fit(lettersX)
    gmm.fit(lettersX)
    SSE[k]['Letters'] = km.score(lettersX)
    ll[k]['Letters'] = gmm.score(lettersX)
    acc[k]['Letters']['Kmeans'] = cluster_acc(lettersY, km.predict(lettersX))
    acc[k]['Letters']['GMM'] = cluster_acc(lettersY, gmm.predict(lettersX))
    adjMI[k]['Letters']['Kmeans'] = ami(lettersY, km.predict(lettersX))
    adjMI[k]['Letters']['GMM'] = ami(lettersY, gmm.predict(lettersX))
    #print(k, clock() - st)

SSE = (-pd.DataFrame(SSE)).T
SSE.rename(columns=lambda x: x + ' SSE (left)', inplace=True)
ll = pd.DataFrame(ll).T
ll.rename(columns=lambda x: x + ' log-likelihood', inplace=True)
acc = pd.Panel(acc)
adjMI = pd.Panel(adjMI)
plot_SSE = SSE.copy()
plot_SSE.plot(marker = 'o',legend=None)
plt.xlabel('Number of clusters')
plt.ylabel('Negative score value')
plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.grid(True)
plt.title('Elbow curve using negative score value for K-Means,after {}'.format(algo))
plot_ll = ll.copy()
plt.figure()
plot_ll.plot(marker = 'o',legend=None)
plt.grid(True)
plt.title('Elbow curve using score value for GMM,  after {}'.format(algo))
plt.xlabel('Number of clusters')
plt.ylabel('Log-likelihood')
SSE.to_csv(out + 'SSE.csv')
ll.to_csv(out + 'loglikelihood.csv')
acc.ix[:, :, 'Letters'].to_csv(out + 'Letters_acc.csv')
adjMI.ix[:, :, 'Letters'].to_csv(out + 'Letters_adjMI.csv')
plt.show()


plt.figure()
n_components_range = clusters
n_components = np.array(n_components_range)
models = [GMM(n, covariance_type='full', random_state=0).fit(X_train)
          for n in n_components]
plt.plot(n_components, [m.bic(X_train) for m in models], label='BIC')
plt.plot(n_components, [m.aic(X_train) for m in models], label='AIC')

plt.legend(loc='best')
plt.xlabel('n_components')
plt.title('AIC and BIC values for GMM, after {}'.format(algo))
plt.ylabel("value")
plt.tight_layout()
plt.show()
#Apply clustering

#sample_size = 30
print ("PRINTING RESULTS TO CONSOLE ....")
print(82 * '_')
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
                                      metric='euclidean')))

bench_k_means(KMeans(init='k-means++', n_clusters=5, n_init=10),
              name="k-means++", data=X_train)
#n-init = number of runs, the best run is considered in terms of inertia
bench_k_means(KMeans(init='random', n_clusters=5, n_init=10),
              name="random", data=X_train)

print(82 * '_')
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

bench_gmm(GMM( n_components=6,covariance_type='full', random_state=0,init_params='kmeans'), name="GMM_kmeans", data=X_train)
bench_gmm(GMM( n_components=6,covariance_type='full', random_state=0,init_params='random'), name="GMM_random", data=X_train)


# %% For chart 4/5
bioX3D = TSNE(n_components=3,verbose=10, random_state=5).fit_transform(X_train)
bio3D = pd.DataFrame(np.hstack((bioX3D, np.atleast_2d(Y_train).T)), columns=['x', 'y', 'z','target'])
bio3D.to_csv(out + 'bio_3D.csv')



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

out = './{}/'.format(sys.argv[1])
print("DR algorithm used is",sys.argv[1] )
algo=sys.argv[1]
np.random.seed(0)
letters = pd.read_hdf(out + 'datasets.hdf', 'letter')
lettersX = letters.drop('Class', 1).copy().values
lettersY = letters['Class'].copy().values
X_trainset, X_test, Y_trainset, Y_test = train_test_split(lettersX,lettersY, test_size=0.2, random_state=42, shuffle=True, stratify=lettersY)
X_train_unscaled, Y_train, X_val, Y_val = train_test_split(X_trainset, Y_trainset, test_size=0.2, random_state=42, shuffle=True, stratify =Y_trainset)
sample_size = 300
X_train = StandardScaler().fit_transform(lettersX)
Y_train=lettersY
lettersX = X_train
X_test= scale(X_test)
X_scaled = scale(X_trainset)

#X_train, Y_train and X_scaled,Y_trainset
clusters = [2, 10, 15, 20, 25, 30,35,40]

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
plt.title('Elbow curve using negative score value for K-Means, after {}'.format(algo))
plot_ll = ll.copy()
plt.figure()
plot_ll.plot(marker = 'o',legend=None)
plt.grid(True)
plt.title('Elbow curve using score value for GMM, after {}'.format(algo))
plt.xlabel('Number of clusters')
plt.ylabel('Log-likelihood')
SSE.to_csv(out + 'SSE.csv')
ll.to_csv(out + 'loglikelihood.csv')
acc.ix[:, :, 'Letters'].to_csv(out + 'Letters_acc.csv')
adjMI.ix[:, :, 'Letters'].to_csv(out + 'Letters_adjMI.csv')
plt.show()

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
sample_size = 300
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
                                      metric='euclidean',
                                      sample_size=sample_size)))

bench_k_means(KMeans(init='k-means++', n_clusters=15, n_init=10),
              name="k-means++", data=X_train)
#n-init = number of runs, the best run is considered in terms of inertia
bench_k_means(KMeans(init='random', n_clusters=15, n_init=10),
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

bench_gmm(GMM( n_components=35,covariance_type='full', random_state=0,init_params='kmeans'), name="GMM_kmeans", data=X_train)
bench_gmm(GMM( n_components=35,covariance_type='full', random_state=0,init_params='random'), name="GMM_random", data=X_train)


X_train3D = TSNE(n_components=3,verbose=10, random_state=5).fit_transform(X_train)
letters3D = pd.DataFrame(np.hstack((X_train3D, np.atleast_2d(lettersY).T)), columns=['x', 'y','z', 'target'])
letters3D.to_csv(out + 'letters3D.csv')



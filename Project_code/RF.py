# %% Imports
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from helpers import nn_arch, nn_reg, ImportanceSelect
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale
import pickle
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import tables
if __name__ == '__main__':
    out = './RF/'

    np.random.seed(0)
    np.random.seed(0)
    file1 = open(r'.\d.pkl', 'rb')
    data1 = pickle.load(file1)
    file1.close()
    instances = data1.data
    X = data1.data
    labels = data1.target
    X_trainset, X_test, Y_trainset, Y_test = train_test_split(X, labels, test_size=0.3, random_state=42, shuffle=True,
                                                              stratify=labels)
    X_train_unscaled, Y_train, X_val, Y_val = train_test_split(X_trainset, Y_trainset, test_size=0.2, random_state=42,
                                                               shuffle=True, stratify=Y_trainset)
    n_samples, n_features = instances.shape
    n_digits = len(np.unique(data1.target))
    madelonX=scale(X_trainset)
    madelonY=Y_trainset

    np.random.seed(0)
    file1 = open(r'.\bioresponse.pkl', 'rb')
    data1 = pickle.load(file1)
    file1.close()
    instances = data1.data
    X = data1.data
    labels = data1.target
    X_trainset, X_test, Y_trainset, Y_test = train_test_split(X, labels, test_size=0.3, random_state=42, shuffle=True,
                                                              stratify=labels)
    n_samples, n_features = instances.shape
    n_digits = len(np.unique(data1.target))
    digitsX=scale(X_trainset)
    digitsY=Y_trainset

    clusters1 = [2, 5, 10, 15, 20, 26, 30, 35, 40, 45, 50]
    dims1 = [1, 2, 5, 8, 10, 12, 15, 16]
    clusters2 = [2, 3, 4, 5, 6, 10]
    dims2 = [2, 40, 80, 170, 480, 670, 860, 990, 1150, 1287, 1378, 1453, 1516, 1617, 1772]

    # %% data for 1

    rfc = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=5, n_jobs=7, criterion = "entropy")
    fs_madelon = rfc.fit(madelonX, madelonY).feature_importances_
    fs_digits = rfc.fit(digitsX, digitsY).feature_importances_

    tmp = pd.Series(np.sort(fs_madelon)[::-1])
    tmp.to_csv(out + 'letter_scree.csv')

    tmp = pd.Series(np.sort(fs_digits)[::-1])
    tmp.to_csv(out + 'bio_scree.csv')

    dim = 8
    filtr = ImportanceSelect(rfc, dim)
    madelonX2 = filtr.fit_transform(madelonX, madelonY)
    madelon2 = pd.DataFrame(np.hstack((madelonX2, np.atleast_2d(madelonY).T)))
    cols = list(range(madelon2.shape[1]))
    cols[-1] = 'Class'
    madelon2.columns = cols
    madelon2.to_csv(out + 'vis_letter_DS.csv')
    madelon2.to_hdf(out + 'datasets.hdf', 'letter', complib='blosc', complevel=9)

    dim = 128
    filtr = ImportanceSelect(rfc, dim)
    digitsX2 = filtr.fit_transform(digitsX, digitsY)
    digits2 = pd.DataFrame(np.hstack((digitsX2, np.atleast_2d(digitsY).T)))
    cols = list(range(digits2.shape[1]))
    cols[-1] = 'Class'
    digits2.columns = cols
    digits2.to_csv(out + 'vis_bio_DS.csv')
    digits2.to_hdf(out + 'datasets.hdf', 'bio', complib='blosc', complevel=9)

    dims = range(1, 17)
    nn_reg = [0.1]
    nn_arch = [(42,)]
    # %% Data for 2
    filtr = ImportanceSelect(rfc)
    grid = {'filter__n': dims, 'NN__alpha': nn_reg, 'NN__hidden_layer_sizes': nn_arch}
    mlp = MLPClassifier(activation='relu', max_iter=2000, early_stopping=True, random_state=5)
    pipe = Pipeline([('filter', filtr), ('NN', mlp)])
    gs = GridSearchCV(pipe, grid, verbose=10, cv=5)
    gs.fit(madelonX, madelonY)
    tmp = pd.DataFrame(gs.cv_results_)
    tmp.to_csv(out + 'letter_gridsearch_A1.csv')


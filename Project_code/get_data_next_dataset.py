import warnings
import sys
from scipy import linalg
warnings.simplefilter("ignore")
from time import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.datasets import fetch_openml
from collections import defaultdict
import pickle


with warnings.catch_warnings():
    warnings.simplefilter("ignore",category=DeprecationWarning)
data1 = fetch_openml(name='Bioresponse')
np.random.seed(42)
instances = scale(data1.data)
X = data1.data
labels = data1.target
n_samples, n_features = instances.shape
n_digits = len(np.unique(data1.target))

print("n_digits: %d, \t n_samples %d, \t n_features %d"
      % (n_digits, n_samples, n_features))

#code reference: https://stackoverflow.com/questions/1047318/easiest-way-to-persist-a-data-structure-to-a-file-in-python
afile = open(r'.\bioresponse.pkl', 'wb')
pickle.dump(data1, afile)
afile.close()




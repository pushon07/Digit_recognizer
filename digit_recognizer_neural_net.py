__author__ = 'ASM Pushon'

"""
This python scrpit will perform the artificial neural network algorithm on the given training dataset of hand-written digits
and will predict the digits from the test dataset.
"""

import numpy as np
import pandas as pd
from nolearn.dbn import DBN
from scipy.ndimage import convolve
import scipy.ndimage as nd
###############################################################################
# Setting up

def nudge_dataset(X, Y):
	"""
	This produces a dataset 5 times bigger than the original one,
	by moving the 8x8 images in X around by 1px to left, right, down, up
	"""
	direction_vectors = [
	[[0, 1, 0],
	[0, 0, 0],
	[0, 0, 0]],[[0, 0, 0],
	[1, 0, 0],
	[0, 0, 0]],[[0, 0, 0],
	[0, 0, 1],
	[0, 0, 0]],[[0, 0, 0],
	[0, 0, 0],
	[0, 1, 0]]]

	shift = lambda x, w: convolve(x.reshape((28, 28)), mode='constant',
	weights=w).ravel()
	X = np.concatenate([X] +
	   [np.apply_along_axis(shift, 1, X, vector)
	   for vector in direction_vectors])
	Y = np.concatenate([Y for _ in range(5)], axis=0)
	return X, Y

def rotate_dataset(X):
	XX = np.zeros(X.shape)
	for index in range(X.shape[0]):
		angle = np.random.randint(-7,7)
		XX[index,:] = nd.rotate(np.reshape(X[index,:],((28,28))),angle,reshape=False).ravel()
	return XX

# Load Data
mnist = pd.read_csv("data_5k.csv")
#mnist = mnist[:50]
y_train = mnist['label'].values
X_train = mnist.loc[:,'pixel0':].values
X_test = pd.read_csv("test.csv").values
X_test = X_test[:2000]
X_train = np.asarray(X_train / 255.0, 'float32')
X_test = np.asarray(X_test / 255.0, 'float32')
#X_train, y_train = nudge_dataset(X_train, y_train)
#X_train = rotate_dataset(X_train)
clf = DBN([X_train.shape[1], 350, 10],\
		learn_rates=0.3,\
		learn_rate_decays=0.95,\
		learn_rates_pretrain=0.005,\
		epochs=120,\
		verbose=1)
clf.fit(X_train, y_train)
subm = pd.read_csv("rf_benchmark.csv")
subm.Label = clf.predict(X_test)
subm.to_csv("result.csv", index_label='ImageId', col=['Label'], index=False)
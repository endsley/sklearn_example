
import numpy as np
from sklearn import preprocessing

def load_data(data_name, prefix='data/'):
	X = np.loadtxt(prefix + data_name + '.csv', delimiter=',', dtype=np.float64)
	Y = np.loadtxt(prefix + data_name + '_label.csv', delimiter=',', dtype=np.int32)			
	X_test = np.loadtxt(prefix + data_name + '_test.csv', delimiter=',', dtype=np.float64)			
	Y_test = np.loadtxt(prefix + data_name + '_label_test.csv', delimiter=',', dtype=np.int32)			
	X = preprocessing.scale(X)
	X_test = preprocessing.scale(X_test)

	return [X,Y,X_test,Y_test]


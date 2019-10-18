#!/usr/bin/env python	

import numpy as np	
import os
	
def gen_10_fold_data(data_name, data_path='./data/'):

	xpath = data_path + data_name

	X = np.loadtxt(xpath + '.csv', delimiter=',', dtype=np.float64)			
	Y = np.loadtxt(xpath + '_label.csv', delimiter=',', dtype=np.int32)			

	fold_path = xpath + '/'
	if not os.path.exists(path): os.mkdir(path)

	count = 1
	kf = KFold(n_splits=10, shuffle=True)
	kf.get_n_splits(X)
	kf.split(X)
	import pdb; pdb.set_trace()

	for train_index, test_index in kf.split(X):
		#print(train_index)
		#print(test_index, '\n')
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]
		

		np.savetxt( fold_path + db['data_name'] + '_' + str(count) + '.csv', X_train, delimiter=',', fmt='%.6f') 
		np.savetxt( fold_path + db['data_name'] + '_' + str(count) + '_label.csv', Y_train, delimiter=',', fmt='%d') 
		np.savetxt( fold_path + db['data_name'] + '_' + str(count) + '_validation.csv', X_test, delimiter=',', fmt='%.6f') 
		np.savetxt( fold_path + db['data_name'] + '_' + str(count) + '_label_validation.csv', Y_test, delimiter=',', fmt='%d') 

		count += 1


if __name__ == "__main__":
	gen_10_fold_data(data_name, data_path='./data/'):

#!/usr/bin/env python

import matplotlib.pyplot as plt
import numpy as np

def label_histograms(Y):
	H = plt.hist(Y, bins=20)
	plt.ylabel('Probability')
	plt.xlabel('value')
	plt.title('Label Histogram')
	plt.savefig('label_histo.png')
	plt.show()
	plt.clf()


data_name = '../data/car'
Y = np.loadtxt(data_name + '_label.csv', delimiter=',', dtype=np.int32)			
label_histograms(Y)

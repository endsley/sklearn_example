#!/usr/bin/env python

import numpy as np
import sys
import sklearn.metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt
from path_tools import *
import pandas as pd
import seaborn as sn
from hsic import *
from scipy.stats import pearsonr



class data_stats():
	def __init__(self, X, Y):
		self.X = X
		self.Y = Y

		self.l = np.unique(Y)
		self.c = len(self.l)
		self.X_list = {}
		self.Y_list = {}
	
		for i in self.l:
			indices = np.where(Y == i)[0]
			self.X_list[i] = {}
			self.Y_list[i] = {}

			self.X_list[i]['X'] = X[indices, :]
			self.Y_list[i]['Y'] = Y[indices]

	def Between_Feature_Scatter_plot(self, path):
		ensure_path_exists(path)
		plt.clf()
		X = self.X

		d = X.shape[1]
		depMatrix = np.zeros((d,d))
		for α in range(d):
			for β in range(d):
				flip_path = path + str(β+1) + '_' + str(α+1) + '_scatter.png'
				if file_exists(flip_path): continue

				plt.plot(X[:,α], X[:,β], '.')
				plt.title('Scatter Plot between Features ' + str(α+1) + ' and ' + str(β+1))

				# add text
				mX = np.min(X[:,α])
				mY = np.max(X[:,β])

				xLoc = mX
				yLoc = mY
	
				Hs = ℍ(X[:,α],X[:,β])	
				ρ, v = pearsonr(X[:,α], X[:,β])
				textstr = 'Correlation ρ: %.3f\n'%ρ
				textstr += 'HSIC: %.3f'%Hs

				props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
				plt.text(xLoc, yLoc, textstr, fontsize=14, verticalalignment='top', bbox=props)


				plt.savefig(path + str(α+1) + '_' + str(β+1) + '_scatter.png')
				plt.clf()


	def get_Correlation_Matrix(self, path):
		ensure_path_exists(path)

		withY = np.hstack((self.X, np.atleast_2d(self.Y).T))
		df = pd.DataFrame(withY)
		corrMatrix = df.corr()

		m_size = corrMatrix.shape[0]
		yticklabels = np.arange(1, m_size+1)
		xticklabels = np.arange(1, m_size+1)
		ax = sn.heatmap(corrMatrix, annot=True, yticklabels=yticklabels, xticklabels=xticklabels)

		ax.set_title('Linear Correlation Matrix')
		plt.savefig(path + 'Linear_Correlation_Matrix.png')
		plt.clf()

	def get_HSIC_Dependence_Matrix(self, path):
		ensure_path_exists(path)

		withY = np.hstack((self.X, np.atleast_2d(self.Y).T))
		d = withY.shape[1]
		depMatrix = np.zeros((d,d))
		for α in range(d):
			for β in range(d):
				depMatrix[α,β] = ℍ(withY[:,α],withY[:,β])

		ax = sn.heatmap(depMatrix, annot=True)
		ax.set_title('HSIC Dependence Matrix')
		plt.savefig(path + 'HSIC_Dependence_Matrix.png')
		plt.clf()

	def get_feature_histograms(self, path):
		ensure_path_exists(path)
		X = self.X
		d = X.shape[1]

		for α in range(d):
			x = X[:,α]
			H = plt.hist(x, bins=20)
			plt.ylabel('Probability')
			plt.xlabel('value')
			plt.title('Feature ' + str(α) + ' Histogram')
			plt.savefig(path + 'Feature_' + str(α) + '.png')
			plt.clf()

		for α in range(d):
			x = X[:,α]
			H = plt.hist(x, bins=20, alpha = 0.5, rwidth=0.1)
			plt.ylabel('Probability')
			plt.xlabel('value')
			plt.title('All Features Superimposed Histogram')

		plt.savefig(path + 'All_Features.png')



	def get_class_info(self):
		D = np.zeros((self.c, self.c))
		for e, i in enumerate(self.l):
			indices = np.where(self.Y == i)[0]
			self.X_list[i]['shape'] = self.X[indices, :].shape
			self.X_list[i]['μ'] = np.mean(self.X[indices, :], axis=0)
			self.X_list[i]['pairwise_distance'] = sklearn.metrics.pairwise.pairwise_distances(self.X_list[i]['X'])
			self.X_list[i]['pairwise_distance_std'] = np.std(self.X_list[i]['pairwise_distance'])
			self.X_list[i]['pairwise_distance_max'] = np.max(self.X_list[i]['pairwise_distance'])


			print('Class %d'%i)
			print('\t data size : ', self.X_list[i]['shape'])
			print('\t Distribution Mean : %s'% str(self.X_list[i]['μ']))
			print('\t distance std : %.3f'% self.X_list[i]['pairwise_distance_std'])
			print('\t distance max : %.3f'% self.X_list[i]['pairwise_distance_max'])
			D[e,e] = self.X_list[i]['pairwise_distance_max']

		for a, i in enumerate(self.l):
			for b, j in enumerate(self.l):
				if i != j:
					indices_i = np.where(Y == i)[0]
					indices_j = np.where(Y == j)[0]
	
					pd = sklearn.metrics.pairwise.pairwise_distances(self.X_list[i]['X'], self.X_list[j]['X'])
					pd_min = np.min(pd)
					D[a,b] = pd_min
		print('Within same class gives furthest distance, between class gives smallest pairwise distance')
		print('\t' + str(D).replace('\n', '\n\t'))


	# labels unsorted into sorted label of same class next to eachother
	def rearrange_sample_to_same_class(self, X,Y):
		l = np.unique(Y)
		newX = np.empty((0, X.shape[1]))
		newY = np.empty((0))
	
		for i in l:
			indices = np.where(Y == i)[0]
			newX = np.vstack((newX, X[indices, :]))
			newY = np.hstack((newY, Y[indices]))
	
		return [newX, newY]

if __name__ == "__main__":
	np.set_printoptions(precision=4)
	np.set_printoptions(threshold=sys.maxsize)
	np.set_printoptions(linewidth=300)
	np.set_printoptions(suppress=True)


	# Use cancer data
	data_name = 'cancer'
	X = np.loadtxt('../data/' + data_name + '.csv', delimiter=',', dtype=np.float64)			
	Y = np.loadtxt('../data/' + data_name + '_label.csv', delimiter=',', dtype=np.int32)			
	X = preprocessing.scale(X)

	# Use 2 Gaussian data	#--------------------------
	#x1 = np.random.randn(40,2)
	#y1 = np.ones((1,40))
	#x2 = np.random.randn(40,2) + 10
	#y2 = np.zeros((1,40))
	#
	#X = np.vstack((x1,x2))
	#Y = np.hstack((y1,y2)).T




	CS = data_stats(X,Y)
	CS.get_Correlation_Matrix('./Dependence_matrices/')
	CS.Between_Feature_Scatter_plot('./Feature_Scatter_Plots/')
	#CS.get_HSIC_Dependence_Matrix('./Dependence_matrices/')
	#CS.get_feature_histograms('./feature_histogram/')
	#CS.get_class_info()

#	np.savetxt(data_name + '.csv', X, delimiter=',', fmt='%.4f') 
#	np.savetxt(data_name + '_label.csv', Y, delimiter=',', fmt='%d') 

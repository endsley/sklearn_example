#!/usr/bin/env python
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import sys
import numpy as np
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score

np.set_printoptions(precision=4)
np.set_printoptions(threshold=30)
np.set_printoptions(linewidth=300)
np.set_printoptions(suppress=True)
np.set_printoptions(threshold=sys.maxsize)

#	models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor

class regression:
	def __init__(self, X,Y, debug=False, model=LinearRegression):
		self.debug = debug
		self.regr = model()
		self.regr.fit(X, Y)
	
	def fit(self, X, Y=None):
		self.Ŷ = self.regr.predict(X)
		if Y is None: return self.Ŷ
		
		mse = mean_squared_error(Y, self.Ŷ)
		if self.debug: print('Mean squared error: %.5f' % mse)
		return self.Ŷ, mse

	def look_at_Y_Ŷ(self, Y):
		YŶ = np.vstack((np.atleast_2d(self.Ŷ[0:20]), np.atleast_2d(Y[0:20])))
		print(YŶ)

# Load the diabetes dataset
X, Y = datasets.load_boston(return_X_y=True)
Reg = regression(X,Y, debug=True, model=GaussianProcessRegressor) # Lasso, LinearRegression, ElasticNet, KernelRidge, GaussianProcessRegressor
Ŷ, mse = Reg.fit(X,Y)
Reg.look_at_Y_Ŷ(Y)




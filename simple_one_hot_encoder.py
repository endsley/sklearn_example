#!/usr/bin/env python


import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

def one_hot_encoding(Y):
	Y = np.reshape(Y,(len(Y),1))
	Yₒ = OneHotEncoder(categories='auto', sparse=False).fit_transform(Y)
	return Yₒ

def one_hot_to_label(Yₒ):
	Y = np.argmax(Yₒ, axis=1)
	return Y


Y = np.array([0,0,1,1,2,2,3,3])
Yₒ = one_hot_encoding(Y)
Y2 = one_hot_to_label(Yₒ)

print(Y)
print(Yₒ)
print(Y2)


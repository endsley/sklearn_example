#!/usr/bin/env python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X = np.vstack((np.random.randn(4,2), np.random.randn(4,2)+5))
Y = np.hstack((np.zeros(4) , np.ones(4)))

clf = LogisticRegression(random_state=0).fit(X, Y)
clf.predict(X)
print(clf.score(X, Y))


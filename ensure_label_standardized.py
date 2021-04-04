#!/usr/bin/env python

import numpy as np
from sklearn.preprocessing import LabelEncoder

Y = np.array([1,1,1,3,3,6,6,9,9])
Y = LabelEncoder().fit_transform(Y)
print(Y)

# Output : [0 0 0 1 1 2 2 3 3]


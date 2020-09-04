#!/usr/bin/env python

import numpy as np

def get_matrix_subset(Φ, subset_id):		# Φ=matrix, subset_id=array
	Φ_rows = np.take(Φ, subset_id, axis=0)
	Φᴼ = np.take(Φ_rows, subset_id, axis=1)
	return Φᴼ

if __name__ == "__main__":
	Φ = np.random.randn(3,3)
	l = np.array([0,2])
	Φᴼ = get_matrix_subset(Φ, l)

	print(Φ,'\n')
	print(Φᴼ,'\n')



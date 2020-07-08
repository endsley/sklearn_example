#!/usr/bin/env python

import torch

dtype = torch.float
device = torch.device("cpu")

v_previous = torch.tensor([[0],[0]], dtype=dtype, requires_grad=True)
v = torch.tensor([[1],[1]], dtype=dtype, requires_grad=True)
A = torch.tensor([[2,1],[1,1]], dtype=dtype, requires_grad=False)
learning_rate = 0.1


def foo():
	p1 = torch.matmul(v.t(), A)
	p2 = torch.matmul(p1, v)
	loss = torch.trace(p2) - 1*torch.sum(torch.abs(v)) 

	return loss

for t in range(200):
	loss = foo()
	loss.backward()
	#learning_rate = learning_rate/(t+1)

	with torch.no_grad():
		v -= learning_rate * v.grad

	print(('%-50s\t%-20s\t%-20s')%(str(v.data.cpu().numpy().T), str(loss.data.cpu().numpy()),  str(v.grad.data.cpu().numpy().T)))
	if torch.norm(v - v_previous) < 0.0001: break
	else: v_previous = v.clone()

	v.grad.zero_()













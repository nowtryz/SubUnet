import torch
from torch.autograd import Variable


def to_var(x):
	if torch.cuda.is_available():
		x = x.cuda()
	return Variable(x)


def getTargetSegmentation(batch):
	# input is 1-channel of values between 0 and 1
	# values are as follows : 0, 0.33333334, 0.6666667 and 0.94117647
	# output is 1 channel of discrete values : 0, 1, 2 and 3
	return (batch * 3).round().long().squeeze()

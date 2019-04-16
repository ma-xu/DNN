import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable

m = nn.AdaptiveAvgPool2d((5,7))
input = autograd.Variable(torch.randn(1, 64, 8, 9))
output = m(input)
print(output.size())

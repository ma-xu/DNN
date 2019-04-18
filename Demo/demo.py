import torch
from torch import nn
from torch import autograd
from torch.autograd import Variable
import os


if 1==2: print(1)
elif 2==2: print (2)
elif 3==2: print (3)
else:
    print(4)


"""
os.system(r"touch {}".format('records/111.txt'))


m = nn.AdaptiveAvgPool2d((4,4))

input = autograd.Variable(torch.randn(1, 3, 2, 2))

#try AdaptiveAvgPool2d
output = m(input)
print(output.size())

#try view
# xx = torch.randn(1, 3, 8, 8).view(1,1,1,192)
# print(xx.size())



x = torch.randn(1, 3, 4, 4)
b, c, _, _ = xx.size()
y = nn.AdaptiveAvgPool2d(xx,1)

y = self.fc(y).view(b, c, 1, 1)
"""
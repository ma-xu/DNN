import torch
import torch.cuda
import torch.optim as optim
from models import *
import os


print("Allocated GPU memory:",torch.cuda.memory_allocated())
net = SEResNet50()
epoch=200
print(epoch%80==0)
optimizer = optim.SGD(net.parameters(),lr=10, momentum=0.9, weight_decay=5e-4)
print(optimizer.param_groups[0]['lr'])


file_path='train_record.txt'
if not os.path.exists(file_path):
    # os.makedirs(file_path)
    os.system(r"touch {}".format(file_path))

f = open(file_path,'a')
f.write('\nHello world.')
f.close()
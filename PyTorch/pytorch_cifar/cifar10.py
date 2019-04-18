'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse
# from torchsummary import summary

from models import *
from utils import progress_bar
from utils import write_record

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r',default=False, action='store_true', help='resume from checkpoint')
parser.add_argument('--netName', default='PreActResNet18', type=str, help='choosing network')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--es', default=300, type=int, help='epoch size')
args = parser.parse_args()


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# Model
print('==> Building model..')
# More models are avaliable in models folder
if args.netName=='VGG16': net = VGG('VGG16')
elif args.netName=='ResNet18': net = VGG('VGG16')
elif args.netName=='PreActResNet18': net = PreActResNet18()
elif args.netName=='GoogLeNet': net = GoogLeNet()
elif args.netName=='DenseNet121': net = DenseNet121()
elif args.netName=='ResNeXt29_2x64d': net = ResNeXt29_2x64d()
elif args.netName=='MobileNet': net = MobileNet()
elif args.netName=='MobileNetV2': net = MobileNetV2()
elif args.netName=='DPN92': net = DPN92()
elif args.netName=='SENet18': net = SENet18()
elif args.netName=='SEResNet18': net = SEResNet18()
elif args.netName=='SEResNet34': net = SEResNet34()
elif args.netName=='ShuffleNetV2(1)': net = ShuffleNetV2(1)
else:
    args.netName = PreActResNet18
    net = PreActResNet18()
    print("\n=====NOTICING:=======\n")
    print("=====Not a valid netName, using default PreActResNet18=====\n\n")

net = net.to(device)

# Get parameters number in a model
pytorch_total_params = sum(p.numel() for p in net.parameters())
print(pytorch_total_params)
# Try torchsummary library
# summary(net, (3, 32, 32))

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint_path = './checkpoint/ckpt_'+args.netName+'.t7'
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    print("BEST_ACCURACY: "+str(best_acc))
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)



def adjust_learning_rate(optimizer, epoch, lr):
    """Sets the learning rate to the initial LR decayed by 10 every 80 epochs"""
    lr = lr * (0.1 ** (epoch // 70))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# Training
def train(epoch):
    adjust_learning_rate(optimizer, epoch, args.lr)
    print('\nEpoch: %d   Learning rate: %f' % (epoch, optimizer.param_groups[0]['lr']))
    print("\nAllocated GPU memory:", torch.cuda.memory_allocated())
    net.train()
    train_loss = 0
    correct = 0
    total = 0



    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))

    file_path='records/'+args.netName+'_train.txt'
    record_str=str(epoch)+'\t'+"%.3f"%(train_loss/(batch_idx+1))+'\t'+"%.3f"%(100.*correct/total)+'\n'
    write_record(file_path,record_str)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))

    file_path = 'records/'+args.netName + '_test.txt'
    record_str = str(epoch) + '\t' + "%.3f" % (test_loss / (batch_idx + 1)) + '\t' + "%.3f" % (
                100. * correct / total) + '\n'
    write_record(file_path, record_str)

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        save_path = './checkpoint/ckpt_'+args.netName+'.t7'
        torch.save(state, save_path)
        best_acc = acc


for epoch in range(start_epoch, start_epoch+args.es):
    train(epoch)
    test(epoch)

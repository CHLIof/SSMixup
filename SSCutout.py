# -*- coding: utf-8 -*-
# @Author  : ly
# @Time    : 2021/9/17 16:44

'''Train CIFAR10 with PyTorch.'''
import argparse
import os
import random

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

import torch

from cutout.cutout import SSCutout


from utils import progress_bar, setup_seed
import torch.nn as nn
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

parser = argparse.ArgumentParser(description='SSCutout Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--weight_path','-w',default='weights/cifar-10/main_cifarX_crosscutmix/ckpt_expc04.pth', type=str,
                    help='resume or test model weight path')
parser.add_argument('--testonly', '-t', default=False, action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--dataset','-d', default='cifar-100', type=str,
                    help='dataset,[cifar-10,cifar-100]')
parser.add_argument('--model','-m', default='resnet-18', type=str,
                    help='model,[resnet-18,resnet-50]')
parser.add_argument('--exp', default='0001', type=str,
                    help='exp num')
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=8,
                    help='length of the holes')
parser.add_argument('--alpha', type=float, default=0.5,
                    help='alpha')
parser.add_argument('--epoch', default=200, type=int,
                    help='epoch')
parser.add_argument('--seed', default=25695100, type=int,
                    help='seed')
args = parser.parse_args()

# device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cuda'
best_acc = 0  # best test accuracy
best_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch
train_acc_record = []
val_acc_record = []
train_loss_record = []
val_loss_record = []

setup_seed(args.seed)


print('==> Preparing data..')

root = r'./data'
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])
transform_train.transforms.append(SSCutout(n_holes=args.n_holes, length=args.length,alpha=args.alpha))

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset == 'cifar-10':
    trainset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform_test)
    num_classes = 10
elif args.dataset == 'cifar-100':
    trainset = torchvision.datasets.CIFAR100(
        root=root, train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR100(
        root=root, train=False, download=True, transform=transform_test)
    num_classes = 100

trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)

testloader = torch.utils.data.DataLoader(
    testset, batch_size=100, shuffle=False, num_workers=2)


# Model
print('==> Building model..')

if args.model == 'resnet-18':
    net = models.resnet18(pretrained=False,progress=False,num_classes=num_classes)
elif args.model == 'resnet-50':
    net = models.resnet50(pretrained=False,progress=False,num_classes=num_classes)

net = net.to(device)

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


def get_file_name():
    filename = os.path.abspath(__file__)
    filename = os.path.basename(filename)

    filename = os.path.splitext(filename)[0]
    return filename


if args.resume or args.testonly:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')

    checkpoint = torch.load(args.weight_path)
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']



criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

def train(epoch):
    global train_acc_record
    global train_loss_record
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    f_loss = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)

        # r = np.random.rand(1)
        # if r < args.prob:
        #     inputs = cutout(inputs,args.n_holes,args.length)

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
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        f_loss = train_loss / (batch_idx + 1)

    train_acc_record.append(correct / total)
    train_loss_record.append(f_loss)


def test(epoch):
    global best_acc
    global best_epoch
    global val_acc_record
    global val_loss_record
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    f_loss = 0
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
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))

            f_loss = test_loss / (batch_idx + 1)

        val_acc_record.append(correct / total)
        val_loss_record.append(f_loss)

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'acc': acc,
            'epoch': epoch,
        }
        checkpoint_name = get_file_name()
        if not os.path.exists('./weights'):
            os.mkdir('./weights')
        if not os.path.exists('./weights/' + args.dataset):
            os.mkdir('./weights/' + args.dataset)
        checkpoint_name = os.path.join('./weights/' + args.dataset, checkpoint_name)
        if not os.path.isdir(checkpoint_name):
            os.mkdir(checkpoint_name)
        # torch.save(state, checkpoint_name + '/ckpt.pth')
        torch.save(state, checkpoint_name + '/ckpt_exp' + args.exp + '.pth')
        best_acc = acc
        best_epoch = epoch


if args.testonly:
    test(0)

    exit(0)

for epoch in range(start_epoch, start_epoch + args.epoch):
    train(epoch)
    test(epoch)
    scheduler.step()

print('best_epoch:', best_epoch)
print('best_acc:', best_acc)





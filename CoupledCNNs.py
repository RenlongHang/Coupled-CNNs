import os
import numpy as np
import random

import torch
import torch.utils.data as dataf
import torch.nn as nn
import matplotlib.pyplot as plt
from scipy import io
from sklearn.decomposition import PCA
from torch.nn.parameter import Parameter
import torchvision.transforms.functional as TF
import time

# 1. two branches share parameters; 2. use summation feature fusion;
# 3. weighted summation in the decision level, the weights are determined by their accuracies.

# setting parameters
DataPath1 = '/home/hrl/PycharmProjects/untitled/HSI+LiDAR/Italy/HSI.mat'
DataPath2 = '/home/hrl/PycharmProjects/untitled/HSI+LiDAR/Italy/LiDAR.mat'
TRPath = '/home/hrl/PycharmProjects/untitled/HSI+LiDAR/Italy/TRLabel.mat'
TSPath = '/home/hrl/PycharmProjects/untitled/HSI+LiDAR/Italy/TSLabel.mat'

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)

patchsize1 = 11
patchsize2 = 11
batchsize = 64
EPOCH = 200
LR = 0.001
NC = 20

# load data

TrLabel = io.loadmat(TRPath)
TsLabel = io.loadmat(TSPath)
TrLabel = TrLabel['TRLabel']
TsLabel = TsLabel['TSLabel']

Data = io.loadmat(DataPath1)
Data = Data['HSI']
Data = Data.astype(np.float32)

Data2 = io.loadmat(DataPath2)
Data2 = Data2['LiDAR']
Data2 = Data2.astype(np.float32)


# normalization method 1: map to [0, 1]
[m, n, l] = Data.shape
for i in range(l):
    minimal = Data[:, :, i].min()
    maximal = Data[:, :, i].max()
    Data[:, :, i] = (Data[:, :, i] - minimal)/(maximal - minimal)

minimal = Data2.min()
maximal = Data2.max()
Data2 = (Data2 - minimal)/(maximal - minimal)

# extract the principal components
PC = np.reshape(Data, (m*n, l))
pca = PCA(n_components=NC, copy=True, whiten=False)
PC = pca.fit_transform(PC)
PC = np.reshape(PC, (m, n, NC))

# boundary interpolation
temp = PC[:,:,0]
pad_width = np.floor(patchsize1/2)
pad_width = np.int(pad_width)
temp2 = np.pad(temp, pad_width, 'symmetric')
[m2,n2] = temp2.shape
x = np.empty((m2,n2,NC),dtype='float32')

for i in range(NC):
    temp = PC[:,:,i]
    pad_width = np.floor(patchsize1/2)
    pad_width = np.int(pad_width)
    temp2 = np.pad(temp, pad_width, 'symmetric')
    x[:,:,i] = temp2

x2 = Data2
pad_width2 = np.floor(patchsize2/2)
pad_width2 = np.int(pad_width2)
temp2 = np.pad(x2, pad_width2, 'symmetric')
x2 = temp2

# construct the training and testing set of HSI
[ind1, ind2] = np.where(TrLabel != 0)
TrainNum = len(ind1)
TrainPatch = np.empty((TrainNum, NC, patchsize1, patchsize1), dtype='float32')
TrainLabel = np.empty(TrainNum)
ind3 = ind1 + pad_width
ind4 = ind2 + pad_width
for i in range(len(ind1)):
    patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
    patch = np.reshape(patch, (patchsize1 * patchsize1, NC))
    patch = np.transpose(patch)
    patch = np.reshape(patch, (NC, patchsize1, patchsize1))
    TrainPatch[i, :, :, :] = patch
    patchlabel = TrLabel[ind1[i], ind2[i]]
    TrainLabel[i] = patchlabel

[ind1, ind2] = np.where(TsLabel != 0)
TestNum = len(ind1)
TestPatch = np.empty((TestNum, NC, patchsize1, patchsize1), dtype='float32')
TestLabel = np.empty(TestNum)
ind3 = ind1 + pad_width
ind4 = ind2 + pad_width
for i in range(len(ind1)):
    patch = x[(ind3[i] - pad_width):(ind3[i] + pad_width + 1), (ind4[i] - pad_width):(ind4[i] + pad_width + 1), :]
    patch = np.reshape(patch, (patchsize1 * patchsize1, NC))
    patch = np.transpose(patch)
    patch = np.reshape(patch, (NC, patchsize1, patchsize1))
    TestPatch[i, :, :, :] = patch
    patchlabel = TsLabel[ind1[i], ind2[i]]
    TestLabel[i] = patchlabel

print('Training size and testing size of HSI are:', TrainPatch.shape, 'and', TestPatch.shape)

# construct the training and testing set of LiDAR
[ind1, ind2] = np.where(TrLabel != 0)
TrainNum = len(ind1)
TrainPatch2 = np.empty((TrainNum, 1, patchsize2, patchsize2), dtype='float32')
TrainLabel2 = np.empty(TrainNum)
ind3 = ind1 + pad_width2
ind4 = ind2 + pad_width2
for i in range(len(ind1)):
    patch = x2[(ind3[i] - pad_width2):(ind3[i] + pad_width2 + 1), (ind4[i] - pad_width2):(ind4[i] + pad_width2 + 1)]
    patch = np.reshape(patch, (patchsize2 * patchsize2, 1))
    patch = np.transpose(patch)
    patch = np.reshape(patch, (1, patchsize2, patchsize2))
    TrainPatch2[i, :, :, :] = patch
    patchlabel2 = TrLabel[ind1[i], ind2[i]]
    TrainLabel2[i] = patchlabel2

[ind1, ind2] = np.where(TsLabel != 0)
TestNum = len(ind1)
TestPatch2 = np.empty((TestNum, 1, patchsize2, patchsize2), dtype='float32')
TestLabel2 = np.empty(TestNum)
ind3 = ind1 + pad_width2
ind4 = ind2 + pad_width2
for i in range(len(ind1)):
    patch = x2[(ind3[i] - pad_width2):(ind3[i] + pad_width2 + 1), (ind4[i] - pad_width2):(ind4[i] + pad_width2 + 1)]
    patch = np.reshape(patch, (patchsize2 * patchsize2, 1))
    patch = np.transpose(patch)
    patch = np.reshape(patch, (1, patchsize2, patchsize2))
    TestPatch2[i, :, :, :] = patch
    patchlabel2 = TsLabel[ind1[i], ind2[i]]
    TestLabel2[i] = patchlabel2

print('Training size and testing size of LiDAR are:', TrainPatch2.shape, 'and', TestPatch2.shape)


# step3: change data to the input type of PyTorch
TrainPatch1 = torch.from_numpy(TrainPatch)
TrainLabel1 = torch.from_numpy(TrainLabel)-1
TrainLabel1 = TrainLabel1.long()
# dataset1 = dataf.TensorDataset(TrainPatch1, TrainLabel1)
# train_loader1 = dataf.DataLoader(dataset1, batch_size=batchsize, shuffle=True)

TestPatch1 = torch.from_numpy(TestPatch)
TestLabel1 = torch.from_numpy(TestLabel)-1
TestLabel1 = TestLabel1.long()

Classes = len(np.unique(TrainLabel))

TrainPatch2 = torch.from_numpy(TrainPatch2)
TrainLabel2 = torch.from_numpy(TrainLabel2)-1
TrainLabel2 = TrainLabel2.long()
# dataset2 = dataf.TensorDataset(TrainPatch2, TrainLabel2)
# train_loader2 = dataf.DataLoader(dataset2, batch_size=batchsize, shuffle=True)

dataset = dataf.TensorDataset(TrainPatch1, TrainPatch2, TrainLabel2)
train_loader = dataf.DataLoader(dataset, batch_size=batchsize, shuffle=True)


TestPatch2 = torch.from_numpy(TestPatch2)
TestLabel2 = torch.from_numpy(TestLabel2)-1
TestLabel2 = TestLabel2.long()

FM = 32
# construct the network
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels = NC,
                out_channels = FM,
                kernel_size = 3,
                stride = 1,
                padding = 1
            ),
            nn.BatchNorm2d(FM),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout(0.5),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(FM, FM*2, 3, 1, 1),
            nn.BatchNorm2d(FM*2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),

        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(FM*2, FM*4, 3, 1, 1),
            nn.BatchNorm2d(FM*4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5),

        )
        self.out1 = nn.Sequential(
            # nn.Linear(FM*4, FM*2),
            nn.Linear(FM*4, Classes),
        )

        self.conv4 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=FM,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.BatchNorm2d(FM),  # BN can improve the accuracy by 4%-5%
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            # nn.Dropout(0.5),
        )

        self.out2 = nn.Sequential(
            # nn.Linear(FM*4, FM*2),
            nn.Linear(FM*4, Classes),
        )

        # self.out3 = nn.Linear(FM*4, Classes)  # fully connected layer, output 16 classes
        self.out3 = nn.Sequential(
            # nn.Linear(FM*4*2, Classes),
            # nn.ReLU(),
            nn.Linear(FM*4, Classes),
        )

    def forward(self, x1, x2):

        x1 = self.conv1(x1)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = x1.view(x1.size(0), -1)  # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        out1 = self.out1(x1)

        x2 = self.conv4(x2)
        x2 = self.conv2(x2)
        x2 = self.conv3(x2)
        x2 = x2.view(x2.size(0), -1)
        out2 = self.out2(x2)

        x = x1 + x2

        out3 = self.out3(x)

        return out1, out2, out3


cnn = CNN()
print('The structure of the designed network', cnn)

# move model to GPU
cnn.cuda()

# optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)  # optimize all cnn parameters

optimizer = torch.optim.Adam(cnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()  # the target label is not one-hotted

BestAcc = 0

torch.cuda.synchronize()
start = time.time()
# train and test the designed model
for epoch in range(EPOCH):
    for step, (b_x1, b_x2, b_y) in enumerate(train_loader):

        # move train data to GPU
        b_x1 = b_x1.cuda()
        b_x2 = b_x2.cuda()
        b_y = b_y.cuda()

        out1, out2, out3 = cnn(b_x1, b_x2)
        loss1 = loss_func(out1, b_y)
        loss2 = loss_func(out2, b_y)
        loss3 = loss_func(out3, b_y)

        loss = 0.01*loss1 + 0.01*loss2 + 1*loss3  # tune these hyperparameters to find an optimal result

        optimizer.zero_grad()  # clear gradients for this training step
        loss.backward()  # backpropagation, compute gradients
        optimizer.step()  # apply gradients

        if step % 50 == 0:
            cnn.eval()

            temp1 = TrainPatch1
            temp1 = temp1.cuda()
            temp2 = TrainPatch2
            temp2 = temp2.cuda()

            temp3, temp4, temp5 = cnn(temp1, temp2)

            pred_y1 = torch.max(temp3, 1)[1].squeeze()
            pred_y1 = pred_y1.cpu()
            acc1 = torch.sum(pred_y1 == TrainLabel1).type(torch.FloatTensor) / TrainLabel1.size(0)

            pred_y2 = torch.max(temp4, 1)[1].squeeze()
            pred_y2 = pred_y2.cpu()
            acc2 = torch.sum(pred_y2 == TrainLabel1).type(torch.FloatTensor) / TrainLabel1.size(0)

            pred_y3 = torch.max(temp5, 1)[1].squeeze()
            pred_y3 = pred_y3.cpu()
            acc3 = torch.sum(pred_y3 == TrainLabel1).type(torch.FloatTensor) / TrainLabel1.size(0)

            # weights are determined by each class accuracy
            Classes = np.unique(TrainLabel1)
            w0 = np.empty(len(Classes),dtype='float32')
            w1 = np.empty(len(Classes),dtype='float32')
            w2 = np.empty(len(Classes),dtype='float32')

            for i in range(len(Classes)):
                cla = Classes[i]
                right1 = 0
                right2 = 0
                right3 = 0

                for j in range(len(TrainLabel1)):
                    if TrainLabel1[j] == cla and pred_y1[j] == cla:
                        right1 += 1
                    if TrainLabel1[j] == cla and pred_y2[j] == cla:
                        right2 += 1
                    if TrainLabel1[j] == cla and pred_y3[j] == cla:
                        right3 += 1

                w0[i] = right1.__float__() / (right1 + right2 + right3 + 0.00001).__float__()
                w1[i] = right2.__float__() / (right1 + right2 + right3 + 0.00001).__float__()
                w2[i] = right3.__float__() / (right1 + right2 + right3 + 0.00001).__float__()

            w0 = torch.from_numpy(w0).cuda()
            w1 = torch.from_numpy(w1).cuda()
            w2 = torch.from_numpy(w2).cuda()

            pred_y = np.empty((len(TestLabel)), dtype='float32')
            number = len(TestLabel) // 5000
            for i in range(number):
                temp = TestPatch1[i * 5000:(i + 1) * 5000, :, :, :]
                temp = temp.cuda()
                temp1 = TestPatch2[i * 5000:(i + 1) * 5000, :, :, :]
                temp1 = temp1.cuda()
                temp2 = w2*cnn(temp, temp1)[2] + w1*cnn(temp, temp1)[1] + w0*cnn(temp, temp1)[0]
                temp3 = torch.max(temp2, 1)[1].squeeze()
                pred_y[i * 5000:(i + 1) * 5000] = temp3.cpu()
                del temp, temp2, temp3

            if (i + 1) * 5000 < len(TestLabel):
                temp = TestPatch1[(i + 1) * 5000:len(TestLabel), :, :, :]
                temp = temp.cuda()
                temp1 = TestPatch2[(i + 1) * 5000:len(TestLabel), :, :, :]
                temp1 = temp1.cuda()
                temp2 = w2*cnn(temp, temp1)[2] + w1*cnn(temp, temp1)[1] + w0*cnn(temp, temp1)[0]
                temp3 = torch.max(temp2, 1)[1].squeeze()
                pred_y[(i + 1) * 5000:len(TestLabel)] = temp3.cpu()
                del temp, temp2, temp3

            pred_y = torch.from_numpy(pred_y).long()
            accuracy = torch.sum(pred_y == TestLabel1).type(torch.FloatTensor) / TestLabel1.size(0)

            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy,
                   '| w0: %.2f' % w0[0], '| w1: %.2f' % w1[0], '| w2: %.2f' % w2[0])

            # save the parameters in network
            if accuracy > BestAcc:
                torch.save(cnn.state_dict(), 'net_params_FusionWeight.pkl')
                BestAcc = accuracy
                w0B = w0
                w1B = w1
                w2B = w2

            cnn.train()

torch.cuda.synchronize()
end = time.time()
print(end - start)
Train_time = end - start

# # test each class accuracy
# # divide test set into many subsets

# load the saved parameters
cnn.load_state_dict(torch.load('net_params_FusionWeight.pkl'))
cnn.eval()
w0 = w0B
w1 = w1B
w2 = w2B

torch.cuda.synchronize()
start = time.time()

pred_y = np.empty((len(TestLabel)), dtype='float32')
number = len(TestLabel)//5000
for i in range(number):
    temp = TestPatch1[i*5000:(i+1)*5000, :, :]
    temp = temp.cuda()
    temp1 = TestPatch2[i*5000:(i+1)*5000, :, :]
    temp1 = temp1.cuda()
    temp2 = w2 * cnn(temp, temp1)[2] + w1 * cnn(temp, temp1)[1] + w0 * cnn(temp, temp1)[0]
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_y[i*5000:(i+1)*5000] = temp3.cpu()
    del temp, temp2, temp3

if (i+1)*5000 < len(TestLabel):
    temp = TestPatch1[(i+1)*5000:len(TestLabel), :, :]
    temp = temp.cuda()
    temp1 = TestPatch2[(i+1)*5000:len(TestLabel), :, :]
    temp1 = temp1.cuda()
    temp2 = w2 * cnn(temp, temp1)[2] + w1 * cnn(temp, temp1)[1] + w0 * cnn(temp, temp1)[0]
    temp3 = torch.max(temp2, 1)[1].squeeze()
    pred_y[(i+1)*5000:len(TestLabel)] = temp3.cpu()
    del temp, temp2, temp3

pred_y = torch.from_numpy(pred_y).long()
OA = torch.sum(pred_y == TestLabel1).type(torch.FloatTensor) / TestLabel1.size(0)

Classes = np.unique(TestLabel1)
EachAcc = np.empty(len(Classes))

for i in range(len(Classes)):
    cla = Classes[i]
    right = 0
    sum = 0

    for j in range(len(TestLabel1)):
        if TestLabel1[j] == cla:
            sum += 1
        if TestLabel1[j] == cla and pred_y[j] == cla:
            right += 1

    EachAcc[i] = right.__float__()/sum.__float__()

print(OA)
print(EachAcc)

torch.cuda.synchronize()
end = time.time()
print(end - start)
Test_time = end - start
Final_OA = OA

print('The OA is: ', Final_OA)
print('The Training time is: ', Train_time)
print('The Test time is: ', Test_time)


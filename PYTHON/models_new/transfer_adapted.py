#!/usr/bin/env python
# coding: utf-8

# # Transfer Learning Network
# This network has been taken directly from **"Transfer learning for radio galaxy classification
# "**:
# https://arxiv.org/abs/1903.11921

import torch
import torch.nn as nn
import torch.nn.functional as F


class transfer_adapted(nn.Module):
    def __init__(self):
        super(transfer_adapted,self).__init__()
        # Order matters for feature extraction, as then the ordered dictionary can be used to make a manual pass (grad-cam ~line 30-35)
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=(11,11),padding=5,stride=1)
        self.relu1 = nn.ReLU()
        self.bnorm1 = nn.BatchNorm2d(6)
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=(5,5),padding=2,stride=1)
        self.relu2 = nn.ReLU()
        self.bnorm2 = nn.BatchNorm2d(16)
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=24,kernel_size=(3,3),padding=1,stride=1)
        self.relu3 = nn.ReLU()
        self.bnorm3 = nn.BatchNorm2d(24)
        
        self.conv4 = nn.Conv2d(in_channels=24,out_channels=24,kernel_size=(3,3),padding=1,stride=1)
        self.relu4 = nn.ReLU()
        self.bnorm4 = nn.BatchNorm2d(24)
        
        self.conv5 = nn.Conv2d(in_channels=24,out_channels=16,kernel_size=(3,3),padding=1,stride=1)
        self.relu5 = nn.ReLU()
        self.bnorm5 = nn.BatchNorm2d(16)
        self.mpool3 = nn.MaxPool2d(kernel_size=5, stride=5)
        
        self.flatten = nn.Flatten(1)
        self.fc1 = nn.Linear(400,256) #channel_size * width * height
        self.relu6 = nn.ReLU()
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(256,256)
        self.relu7 = nn.ReLU()
        self.dropout1 = nn.Dropout()
        self.fc3 = nn.Linear(256,256)
        self.relu8 = nn.ReLU()
        self.dropout2 = nn.Dropout()
        self.fc4 = nn.Linear(256,2)
        
        # This doesnt work, as it looks for weights to fill it with when it loads in ...
        #self.classifier = nn.Sequential(self.fc1,self.dropout,self.fc2,self.dropout,self.fc2,self.dropout,self.fc3)
        
    def forward(self,inputs):
        conv1 = self.conv1(inputs)
        re1 = self.relu1(conv1)
        bn1 = self.bnorm1(re1)
        mp1 = self.mpool1(bn1)
        conv2 = self.conv2(mp1)
        re2 = self.relu2(conv2)
        bn2 = self.bnorm2(re2)
        mp2 = self.mpool2(bn2)
        
        conv3 = self.conv3(mp2)
        re3 = self.relu3(conv3)
        bn3 = self.bnorm3(re3)
        
        conv4 = self.conv4(bn3)
        re4 = self.relu4(conv4)
        bn4 = self.bnorm4(re4)
        
        conv5 = self.conv5(bn4)
        re5 = self.relu5(conv5)
        bn5 = self.bnorm5(re5)
        mp3 = self.mpool3(bn5)
        
        flat = self.flatten(mp3)
        fc1 = self.fc1(flat)
        re6 = self.relu6(fc1)
        do = self.dropout(re6)
        fc2 = self.fc2(do)
        re7 = self.relu7(fc2)
        do1 = self.dropout1(re7)
        fc3 = self.fc3(do1)
        re8 = self.relu8(fc3)
        do2 = self.dropout2(re8)
        fc4 = self.fc4(do2)
        
        return F.log_softmax(fc4,dim=1)


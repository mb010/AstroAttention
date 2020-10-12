#!/usr/bin/env python
# coding: utf-8

# # Transfer Learning Network
# This network has been taken directly from **"Transfer learning for radio galaxy classification
# "**:
# https://arxiv.org/abs/1903.11921

import torch
import torch.nn as nn
import torch.nn.functional as F

def sequence():
        return nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=6,kernel_size=(11,11),padding=5,stride=1),
            nn.ReLU(), nn.BatchNorm2d(6), nn.MaxPool2d(kernel_size=2, stride=2),
            \
            nn.Conv2d(in_channels=6,out_channels=16,kernel_size=(5,5),padding=2,stride=1),
            nn.ReLU(), nn.BatchNorm2d(16), nn.MaxPool2d(kernel_size=3, stride=3),
            \
            nn.Conv2d(in_channels=16,out_channels=24,kernel_size=(3,3),padding=1,stride=1),
            nn.ReLU(),nn.BatchNorm2d(24),
            \
            nn.Conv2d(in_channels=24,out_channels=24,kernel_size=(3,3),padding=1,stride=1),
            nn.ReLU(), nn.BatchNorm2d(24),
            \
            nn.Conv2d(in_channels=24,out_channels=16,kernel_size=(3,3),padding=1,stride=1),
            nn.ReLU(), nn.BatchNorm2d(16), nn.MaxPool2d(kernel_size=5, stride=5)
        )

class transfer_original(nn.Module):
    def __init__(self):
        super(transfer_original,self).__init__()
        self.features = sequence()
        self.conv1 = nn.Conv2d(in_channels=1,out_channels=6,kernel_size=(11,11),padding=5,stride=1)
        self.conv2 = nn.Conv2d(in_channels=6,out_channels=16,kernel_size=(5,5),padding=2,stride=1)
        self.conv3 = nn.Conv2d(in_channels=16,out_channels=24,kernel_size=(3,3),padding=1,stride=1)
        self.conv4 = nn.Conv2d(in_channels=24,out_channels=24,kernel_size=(3,3),padding=1,stride=1)
        self.conv5 = nn.Conv2d(in_channels=24,out_channels=16,kernel_size=(3,3),padding=1,stride=1)
        
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.mpool3 = nn.MaxPool2d(kernel_size=5, stride=5)
        
        self.bnorm1 = nn.BatchNorm2d(6)
        self.bnorm2 = nn.BatchNorm2d(16)
        self.bnorm3 = nn.BatchNorm2d(24)
        self.bnorm4 = nn.BatchNorm2d(24)
        self.bnorm5 = nn.BatchNorm2d(16)
        
        self.flatten = nn.Flatten(1)
        
        self.fc1 = nn.Linear(400,256) #channel_size * width * height
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,2)
        
        self.dropout = nn.Dropout()
    
    
        
    def forward(self,inputs):
        x = self.features(inputs)
        
        flatten = self.flatten(x)
        fc1     = F.relu(self.fc1(flatten))
        do      = self.dropout(fc1)
        fc2     = F.relu(self.fc2(do))
        do      = self.dropout(fc2)
        fc2     = F.relu(self.fc2(do))
        do      = self.dropout(fc2)
        fc3     = self.fc3(do)
        
        return F.log_softmax(fc3,dim=1)


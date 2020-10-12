#!/usr/bin/env python
# coding: utf-8

# # Sononet Grid Attention Network
# 
# This network has been simplified (for legibility) from the "./networks/sononet_grid_attention.py" file to be found under:https://github.com/ozan-oktay/Attention-Gated-Networks

# In[3]:


#basics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks_other import init_weights


# In[4]:


############################################
# This is the one we are acctually using. We call it in sononet_grid_attention.py as AttentionBlock2D. Input Changes:
# 'dimension = 2' and 'sub_sample_factor = (1,1)'
class _GridAttentionBlock2D_TORR(nn.Module): #Cleaned up from _GridAttentionBlockND_TORR(...)
    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=2, mode='concatenation_softmax',
                 sub_sample_factor=(1,1), bn_layer=True, use_W=False, use_phi=True, use_theta=True, use_psi=True, nonlinearity1='relu'):
        super(_GridAttentionBlock2D_TORR, self).__init__()

        assert dimension in [2] #for 3 dimensional functionality, use original implementation of functions.
        assert mode in ['concatenation_softmax'] #Removed all other options for legibility.

        # Default parameter set
        self.mode = mode
        self.dimension = dimension
        self.sub_sample_factor = sub_sample_factor if isinstance(sub_sample_factor, tuple) else tuple([sub_sample_factor])*dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None: # This is the value we use.
            self.inter_channels = in_channels // 2 #Either half of in_channels (in_channels > 1) or = 1 (in_channels = 1).
            if self.inter_channels == 0:
                self.inter_channels = 1 #We go down to one channel because of this!

        if dimension == 2:
            conv_nd = nn.Conv2d
            bn = nn.BatchNorm2d
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplemented

        # initialise id functions
        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.W = lambda x: x        # These are essentially base functions for if any of the conditions (below) aren't met,
        self.theta = lambda x: x    # in which case each of these methods returns the data without any alterations / augmentations. ie. x -> x
        self.psi = lambda x: x
        self.phi = lambda x: x
        self.nl1 = lambda x: x
        
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=(1,1), stride=(1,1), padding=0, bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels, kernel_size=(1,1), stride=(1,1), padding=0, bias=False)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        self.nl1 = lambda x: F.relu(x, inplace=True)

        ### Initialise weights using their package (see imports) ###
        for m in self.children():
            init_weights(m, init_type='kaiming')
        #if use_psi and self.mode == 'concatenation_softmax':
        nn.init.constant(self.psi.bias.data, 10.0) # Initialises the tensor self.psi.bias.data with values of 10.0 (Because bias=True in initialisation)

    def forward(self, x, g):
        # As we assert that mode must contain concatenation, this holds for all passes where we pass the initial assertion
        #(this was a seperate method, called _concatenation - see sononet_grid_attention.py).
        '''
        :param x: (b, c, t, h, w),
        ie. (batch dim, channel dim, thickness, height, width), in our case we omit thickness as we are working with 2D data.
        :param g: (b, g_d)
        :return:
        '''
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)
        
        # Compute compatibility score: psi_f
        #print(x.size())
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()
        
        # nl(theta.x + phi.g + bias) -> f = (b, i_c, t, h/s2, w/s3)
        phi_g = F.upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        f = theta_x + phi_g
        f = self.nl1(f)

        psi_f = self.psi(f)

        # Calculate Attention map (sigm_psi_f) and weighted output (W_y)
        # This block was conditional with: 'self.mode == concatenation_softmax'. Other options are listed in grid_attention_layer.py
        # normalisation & scale to x.size()[2:]
        # psi^T . f -> (b, 1, t/s1, h/s2, w/s3)
        #sigm_psi_f = F.softmax(psi_f.view(batch_size, 1, -1), dim=2)
        # TRY THIS !!!
        sigm_psi_f = torch.sigmoid(psi_f)
        sigm_psi_f = sigm_psi_f.view(batch_size, 1, *theta_x_size[2:])

        # sigm_psi_f is attention map! upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode) ### mode = bilinear in 2D, ipnut_size is the input WxH of the data.
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y) #As W = False, W_y = y

        return W_y, sigm_psi_f


# In[5]:


class AGSononet(nn.Module):
    def __init__(self):
        super(AGSononet,self).__init__()
        self.conv1a = nn.Conv2d(in_channels=1,out_channels=16,kernel_size=(3,3),padding=1,stride=1)
        self.conv1b = nn.Conv2d(in_channels=16,out_channels=16,kernel_size=(3,3),padding=1,stride=1)
        self.conv2a = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=(3,3),padding=1,stride=1)
        self.conv2b = nn.Conv2d(in_channels=32,out_channels=32,kernel_size=(3,3),padding=1,stride=1)
        self.conv3a = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=(3,3),padding=1,stride=1)
        self.conv3b = nn.Conv2d(in_channels=64,out_channels=64,kernel_size=(3,3),padding=1,stride=1)
        self.conv4a = nn.Conv2d(in_channels=64,out_channels=128,kernel_size=(3,3),padding=1,stride=1)
        self.conv4b = nn.Conv2d(in_channels=128,out_channels=128,kernel_size=(3,3),padding=1,stride=1)
        
        self.mpool = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        
        self.bnorm1 = nn.BatchNorm2d(16)
        self.bnorm2 = nn.BatchNorm2d(32)
        self.bnorm3 = nn.BatchNorm2d(64)
        self.bnorm4 = nn.BatchNorm2d(128)
        
        self.flatten = nn.Flatten(1)
        
        self.fc1 = nn.Linear(128*9*9,2) #channel_size * width * height
        self.dropout = nn.Dropout()
        
        # These are effectively the same calls as in sononet_grid_attention.py due to us redefining the standard selections in _GridAttentionBlock2D(..)
        filters = [16,32,64,128]
        self.filters = filters
        self.compatibility_score1 = _GridAttentionBlock2D_TORR(in_channels=64 , gating_channels=128, inter_channels=128, use_W=False) ### Why did they choose use_W = False ?
        self.compatibility_score2 = _GridAttentionBlock2D_TORR(in_channels=128, gating_channels=128, inter_channels=128, use_W=False)
        
        # Primary Aggregation selection: (simplest format): ie. >>> if 'concat':
        self.classifier = nn.Linear(filters[2]+filters[3]+filters[3], 2)
        self.aggregate = self.aggregation_concat
        
    def aggregation_concat(self, *attended_maps):
        return self.classifier(torch.cat(attended_maps, dim=1))
        
    def forward(self,inputs):
        batch_size = inputs.shape[0]
        
        # Layer building
        # unetConv2(3) call ([-1,1,150,150]->[-1,16,150,150]) followed by an mpool
        conv1 = F.relu(self.bnorm1(self.conv1a(inputs)))
        conv1 = F.relu(self.bnorm1(self.conv1b(conv1)))
        conv1 = F.relu(self.bnorm1(self.conv1b(conv1)))
        mpool = self.mpool(conv1)
        
        # unetConv2(3) call ([-1,16,75,75]->[-1,32,75,75]) followed by an mpool
        conv2 = F.relu(self.bnorm2(self.conv2a(mpool)))
        conv2 = F.relu(self.bnorm2(self.conv2b(conv2)))
        conv2 = F.relu(self.bnorm2(self.conv2b(conv2)))
        mpool = self.mpool(conv2)
        
        # unetConv2(3) call ([-1,32,37,37]->[-1,64,37,37]) followed by an mpool
        conv3 = F.relu(self.bnorm3(self.conv3a(mpool)))
        conv3 = F.relu(self.bnorm3(self.conv3b(conv3)))
        conv3 = F.relu(self.bnorm3(self.conv3b(conv3)))
        mpool = self.mpool(conv3)
        
        # unetConv2(2) call ([-1,64,18,18]->[-1,128,18,18]) followed by an mpool
        conv4 = F.relu(self.bnorm4(self.conv4a(mpool)))
        conv4 = F.relu(self.bnorm4(self.conv4b(conv4)))
        mpool = self.mpool(conv4)
        
        # unetConv2(2) call ([-1,128,9,9]->[-1,128,9,9])
        conv5 = F.relu(self.bnorm4(self.conv4b(mpool)))
        conv5 = F.relu(self.bnorm4(self.conv4b(conv5)))
        
        # Adaptive average pooling pools by kernels which may overlap but will output to a specific size, here 1x1.
        pooled = F.adaptive_avg_pool2d(conv5, (1, 1)).view(batch_size, -1)

        # Applied Attention , Attention map
        #We use conv5 as the global attention as this is the most advanced stage (upsampled to conv3/conv4 to make it work).
        #(inputs are before maxpool layer in sononet_grid_attention.py)
        #print(conv3.size(),conv4.size(),conv5.size())
        attendedConv3 , atten3 = self.compatibility_score1(conv3, conv5)
        attendedConv4 , atten4 = self.compatibility_score2(conv4, conv5)
        
        # Aggregation
        g1 = torch.sum(attendedConv3.view(batch_size, self.filters[2], -1), dim=-1)
        g2 = torch.sum(attendedConv4.view(batch_size, self.filters[3], -1), dim=-1)
        
        out = self.aggregate(g1,g2,pooled)

        return F.softmax(out,dim=1)
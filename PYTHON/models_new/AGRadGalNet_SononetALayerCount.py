# # Custom Network with Sononet Grid Attention Gates included (version 0)
# 
# For the original files on the attention gates please see the **"./networks/sononet_grid_attention.py"** file to be found under:https://github.com/ozan-oktay/Attention-Gated-Networks
# 
# This network has been adapted from **"Transfer learning for radio galaxy classification"**:
# https://arxiv.org/abs/1903.11921

#basics
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.networks_other import init_weights

# Defining the Network
class AGRadGalNet(nn.Module):
    def __init__(self,aggregation_mode='concat',n_classes=2):
        super(AGRadGalNet,self).__init__()
        filters = [6,16,32,64,128]
        ksizes = [3,3,3,3,3,3] # Must all be odd for calculation of padding.        
        self.filters = filters
        assert aggregation_mode in ['concat','mean','deep_sup','ft'], "aggregation_mode not valid selection, must be any of: ['concat','mean','deep_sup','ft']"
        self.aggregation_mode = aggregation_mode
        
        self.conv1a = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, padding=1, stride=1); self.relu1a = nn.ReLU(); self.bnorm1a= nn.BatchNorm2d(6)
        self.conv1b = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding=1, stride=1); self.relu1b = nn.ReLU(); self.bnorm1b= nn.BatchNorm2d(6)
        self.conv1c = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=3, padding=1, stride=1); self.relu1c = nn.ReLU(); self.bnorm1c= nn.BatchNorm2d(6)
        self.mpool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        
        self.conv2a = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, padding=1, stride=1); self.relu2a = nn.ReLU(); self.bnorm2a= nn.BatchNorm2d(16)
        self.conv2b = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1); self.relu2b = nn.ReLU(); self.bnorm2b= nn.BatchNorm2d(16)
        self.conv2c = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1, stride=1); self.relu2c = nn.ReLU(); self.bnorm2c= nn.BatchNorm2d(16)
        self.mpool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        
        self.conv3a = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1); self.relu3a = nn.ReLU(); self.bnorm3a= nn.BatchNorm2d(32)
        self.conv3b = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1); self.relu3b = nn.ReLU(); self.bnorm3b= nn.BatchNorm2d(32)
        self.conv3c = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1); self.relu3c = nn.ReLU(); self.bnorm3c= nn.BatchNorm2d(32)
        self.mpool3 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        
        self.conv4a = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1); self.relu4a = nn.ReLU(); self.bnorm4a= nn.BatchNorm2d(64)
        self.conv4b = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1); self.relu4b = nn.ReLU(); self.bnorm4b= nn.BatchNorm2d(64)
        
        
        self.flatten = nn.Flatten(1)
        
        self.fc1 = nn.Linear(16*5*5,256) #channel_size * width * height
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,2)
        
        self.dropout = nn.Dropout()
    
        # These are effectively the same calls as in sononet_grid_attention.py due to us redefining the standard selections in _GridAttentionBlock2D(..)        
        self.compatibility_score1 = GridAttentionBlock2D(in_channels=16, gating_channels=64, inter_channels=64, use_W=False, input_size=[150//2,150//2]) #inter_channels=gating_channels in sononet implementation.
        self.compatibility_score2 = GridAttentionBlock2D(in_channels=32, gating_channels=64, inter_channels=64, use_W=False, input_size=[150//4,150//4])
        
        self.module_order = ['conv1a', 'relu1a', 'bnorm1a', #1->6
                             'conv1b', 'relu1b', 'bnorm1b', #6->6
                             'conv1c', 'relu1c', 'bnorm1c', #6->6
                             'mpool1',
                             'conv2a', 'relu2a', 'bnorm2a', #6->16
                             'conv2b', 'relu2b', 'bnorm2b', #16->16
                             'conv2c', 'relu2c', 'bnorm2c', #16->16
                             'mpool2',
                             'conv3a', 'relu3a', 'bnorm3a', #16->32
                             'conv3b', 'relu3b', 'bnorm3b', #32->32
                             'conv3c', 'relu3c', 'bnorm3c', #32->32
                             'mpool3',
                             'conv4a', 'relu4a', 'bnorm4a', #32->64
                             'conv4b', 'relu4b', 'bnorm4b', #64->64

                             'compatibility_score1', 
                             'compatibility_score2']
        
        """ 
        # Previous model that worked well(ish)
        self.module_order = ['conv1', 'relu', 'bnorm1', #1->6
                             'mpool1',
                             'conv2', 'relu', 'bnorm2', #6->16
                             'mpool2',
                             'conv3', 'relu', 'bnorm3', #16->16
                             'conv3', 'relu', 'bnorm3', #16->16
                             'conv3', 'relu', 'bnorm3', #16->16
                             #'mpoolN',
                             'conv4', 'relu', 'bnorm4', #32->64
                             'conv5', 'relu', 'bnorm5', #32->64
                             'conv6', 'relu', 'bnorm6', #64->64
                             'mpool3',
                             'compatibility_score1', 
                             'compatibility_score2']
        """
        
        
        #########################
        # Aggreagation Strategies
        self.attention_filter_sizes = [16, 32]
        if aggregation_mode == 'concat':
            self.classifier = nn.Linear(16 + 32 + 64, n_classes)
            self.aggregate = self.aggregation_concat
        else:
            self.classifier1 = nn.Linear(16, n_classes)
            self.classifier2 = nn.Linear(32, n_classes)
            self.classifier3 = nn.Linear(64, n_classes)
            self.classifiers = [self.classifier1, self.classifier2, self.classifier3]
            if aggregation_mode == 'mean':
                self.aggregate = self.aggregation_sep
                #self.aggregate = torch.mean(torch.stack(self.aggregation_sep),dim=[0,1]) #This will output a single vector for classification instead of a list of classifications.
            elif aggregation_mode == 'deep_sup':
                self.classifier = nn.Linear(16 + 32 + 64, n_classes)
                self.aggregate = self.aggregation_ds
            elif aggregation_mode == 'ft':
                self.classifier = nn.Linear(n_classes*3, n_classes)
                self.aggregate = self.aggregation_ft
            else:
                raise NotImplementedError

        ####################
        # initialise weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_weights(m, init_type='kaiming')
            elif isinstance(m, nn.BatchNorm2d):
                init_weights(m, init_type='kaiming')
    
    # Define Aggregation Methods
    def aggregation_sep(self, *attended_maps):
        return [ clf(att) for clf, att in zip(self.classifiers, attended_maps) ]

    def aggregation_ft(self, *attended_maps):
        preds =  self.aggregation_sep(*attended_maps)
        return self.classifier(torch.cat(preds, dim=1))

    def aggregation_ds(self, *attended_maps):
        preds_sep =  self.aggregation_sep(*attended_maps)
        pred = self.aggregation_concat(*attended_maps)
        return [pred] + preds_sep

    def aggregation_concat(self, *attended_maps):
        return self.classifier(torch.cat(attended_maps, dim=1))
    
    # Activation function
    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)
        return log_p
    
    
    #######################################
    # Define forward pass:
    def forward(self,inputs):
        conv1a = self.bnorm1a(self.relu1a(self.conv1a(inputs))) #1->6
        conv1b = self.bnorm1b(self.relu1b(self.conv1b(conv1a))) #6->6
        conv1c = self.bnorm1c(self.relu1c(self.conv1c(conv1b))) #6->6
        mpool1 = self.mpool1(conv1c)
        
        conv2a = self.bnorm2a(self.relu2a(self.conv2a(mpool1))) #6->16
        conv2b = self.bnorm2b(self.relu2b(self.conv2b(conv2a))) #16->16
        conv2c = self.bnorm2c(self.relu2c(self.conv2c(conv2b))) #16->16
        mpool2 = self.mpool2(conv2c)
        
        conv3a = self.bnorm3a(self.relu3a(self.conv3a(mpool2))) #16->32
        conv3b = self.bnorm3b(self.relu3b(self.conv3b(conv3a))) #32->32
        conv3c = self.bnorm3c(self.relu3c(self.conv3c(conv3b))) #32->32
        mpool3 = self.mpool3(conv3c)
        
        conv4a = self.bnorm4a(self.relu4a(self.conv4a(mpool3))) #32->64
        conv4b = self.bnorm4b(self.relu4b(self.conv4b(conv4a))) #64->64

        # Applied Attention , Attention map
        #We use conv5 as the global attention as this is the most advanced stage (upsampled to conv3/conv4 to make it work).
        #(inputs are before maxpool layer in sononet_grid_attention.py)
        attendedConv2 , atten2 = self.compatibility_score1(conv2c, conv4b)
        attendedConv3 , atten3 = self.compatibility_score2(conv3c, conv4b)
        
        # Aggregation
        batch_size = inputs.shape[0]
        pooled = F.adaptive_avg_pool2d(conv4b,(1,1)).view(batch_size,-1)        
        g1 = torch.sum(attendedConv2.view(batch_size, 16, -1), dim=-1)
        g2 = torch.sum(attendedConv3.view(batch_size, 32, -1), dim=-1)
        
        #print('TOTAL SIZE OF LINAR FUNCTION SHOULD BE: ',filters[4]+2*filters[5])
        #print('G1 , G2 and POOLED shape: ',g1.shape,g2.shape,pooled.shape)
        out = self.aggregate(g1,g2,pooled)
        
        
        #return F.log_softmax(out,dim=1)
        if type(out)==list:
            if self.aggregation_mode == 'mean':
                out = torch.mean(torch.stack(out),dim=[0]) #This will output a single vector for classification instead of a list of classifications.
                out = F.softmax(out,dim=1)
            elif self.aggregation_mode == 'deep_sup':
                out = torch.mean(torch.stack(out),dim=[0])
                out = F.softmax(out,dim=1) #Can be added for a second attempt.
            elif self.aggregation_mode == 'ft': # Might not be a problem
                out = F.softmax(out,dim=1)
            else:
                raise NotImplementedError
            return out
        else:
            return F.softmax(out,dim=1)

        
        
############################################
# This is the one we are acctually using. We call it in sononet_grid_attention.py as AttentionBlock2D. Input Changes:
# 'dimension = 2' and 'sub_sample_factor = (1,1)'
class GridAttentionBlock2D(nn.Module): #Cleaned up from _GridAttentionBlockND_TORR(...)
    def __init__(self, in_channels, gating_channels, input_size=[150,150], inter_channels=None, dimension=2, mode='concatenation_sigmoid',
                 sub_sample_factor=(1,1), bn_layer=True, use_W=False, use_phi=True, use_theta=True, use_psi=True, nonlinearity1='relu'):
        super(GridAttentionBlock2D, self).__init__()

        assert dimension in [2] #for 3 dimensional functionality, use original implementation of functions.
        assert mode in ['concatenation_softmax','concatenation_sigmoid'] #Removed all other options for legibility.

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
            raise NotImplementedError

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
        self.upsample = nn.Upsample(size=input_size, mode=self.upsample_mode)

        ### Initialise weights using their package (see imports) ###
        for m in self.children():
            init_weights(m, init_type='kaiming')
        #if use_psi and self.mode == 'concatenation_softmax':
        nn.init.constant_(self.psi.bias.data, 10.0) # Initialises the tensor self.psi.bias.data with values of 10.0 (Because bias=True in initialisation)

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
        #phi_g = nn.Upsample(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode)
        phi_g = self.upsample(self.phi(g))
        f = theta_x + phi_g
        f = self.nl1(f)

        psi_f = self.psi(f)

        # Calculate Attention map (sigm_psi_f) and weighted output (W_y)
        # Normalisation & scale to x.size()[2:]
        # psi^T . f -> (b, 1, t/s1, h/s2, w/s3)
        if self.mode == 'concatenation_softmax':
            sigm_psi_f = F.softmax(psi_f.view(batch_size, 1, -1), dim=2)
        elif self.mode == 'concatenation_sigmoid':
            sigm_psi_f = torch.sigmoid(psi_f)
        else:
            raise NotImplementedError
        
        sigm_psi_f = sigm_psi_f.view(batch_size, 1, *theta_x_size[2:])

        # sigm_psi_f is attention map! upsample the attentions and multiply
        sigm_psi_f = self.upsample(sigm_psi_f) ### mode = bilinear in 2D, ipnut_size is the input WxH of the data.
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y) #As W = False, W_y = y

        return W_y, sigm_psi_f
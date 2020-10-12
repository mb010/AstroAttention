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
class playgroundv2(nn.Module):
    def __init__(self,aggregation_mode='concat',n_classes=2):
        super(playgroundv2,self).__init__()
        filters = [6,16,16,32,64,128]
        #ksizes = [11,5,5,3,3,3] # Must all be odd for calculation of padding.
        ksizes = [3,3,3,3,3,3] # Must all be odd for calculation of padding.        
        self.filters = filters
        assert aggregation_mode in ['concat','mean','deep_sup','ft'], "aggregation_mode not valid selection, must be any of: ['concat','mean','deep_sup','ft']"
        self.aggregation_mode = aggregation_mode
        
        self.conv1 = nn.Conv2d(in_channels=1,         out_channels=filters[0],
                               kernel_size=ksizes[0],padding=ksizes[0]//2,stride=1)
        
        self.conv2 = nn.Conv2d(in_channels=filters[0],out_channels=filters[1],
                               kernel_size=ksizes[1],padding=ksizes[1]//2,stride=1)
        
        self.conv3 = nn.Conv2d(in_channels=filters[1],out_channels=filters[2],
                               kernel_size=ksizes[2],padding=ksizes[2]//2,stride=1)
        
        self.conv4 = nn.Conv2d(in_channels=filters[2],out_channels=filters[3],
                               kernel_size=ksizes[3],padding=ksizes[3]//2,stride=1)
        
        self.conv5 = nn.Conv2d(in_channels=filters[3],out_channels=filters[4],
                               kernel_size=ksizes[4],padding=ksizes[4]//2,stride=1)
        
        self.conv6 = nn.Conv2d(in_channels=filters[4],out_channels=filters[5],
                               kernel_size=ksizes[5],padding=ksizes[5]//2,stride=1)
        
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        self.mpool3 = nn.MaxPool2d(kernel_size=5, stride=5)
        
        self.bnorm1 = nn.BatchNorm2d(filters[0])
        self.bnorm2 = nn.BatchNorm2d(filters[1])
        self.bnorm3 = nn.BatchNorm2d(filters[2])
        self.bnorm4 = nn.BatchNorm2d(filters[3])
        self.bnorm5 = nn.BatchNorm2d(filters[4])
        self.bnorm6 = nn.BatchNorm2d(filters[5])
        
        self.flatten = nn.Flatten(1)
        
        self.fc1 = nn.Linear(16*5*5,256) #channel_size * width * height
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,2)
        
        self.dropout = nn.Dropout()
    
        ########## ### Why did they choose use_W = False ?
        # These are effectively the same calls as in sononet_grid_attention.py due to us redefining the standard selections in _GridAttentionBlock2D(..)        
        self.compatibility_score1 = _GridAttentionBlock2D_TORR(in_channels=filters[3] , gating_channels=filters[5], inter_channels=filters[2], use_W=False)        
        self.compatibility_score2 = _GridAttentionBlock2D_TORR(in_channels=filters[4], gating_channels=filters[5], inter_channels=filters[2], use_W=False)
            
        #########################
        # Aggreagation Strategies
        self.attention_filter_sizes = [filters[2], filters[3]]

        if aggregation_mode == 'concat':
            self.classifier = nn.Linear(filters[3]+filters[4]+filters[5], n_classes)
            self.aggregate = self.aggregation_concat

        else:
            self.classifier1 = nn.Linear(filters[3], n_classes)
            self.classifier2 = nn.Linear(filters[4], n_classes)
            self.classifier3 = nn.Linear(filters[5], n_classes)
            self.classifiers = [self.classifier1, self.classifier2, self.classifier3]

            if aggregation_mode == 'mean':
                self.aggregate = self.aggregation_sep
                #self.aggregate = torch.mean(torch.stack(self.aggregation_sep),dim=[0,1]) #This will output a single vector for classification instead of a list of classifications.

            elif aggregation_mode == 'deep_sup':
                self.classifier = nn.Linear(filters[3] + filters[4] + filters[5], n_classes)
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


    def aggregation_sep(self, *attended_maps):
        return [ clf(att) for clf, att in zip(self.classifiers, attended_maps) ]

    def aggregation_ft(self, *attended_maps):
        preds =  self.aggregation_sep(*attended_maps)
        return self.classifier(torch.cat(preds, dim=1))

    def aggregation_ds(self, *attended_maps):
        preds_sep =  self.aggregation_sep(*attended_maps)
        pred = self.aggregation_concat(*attended_maps)
        #print([pred]+preds_sep,[pred],preds_sep)
        return [pred] + preds_sep

    def aggregation_concat(self, *attended_maps):
        #print(f'aggregation_mode = concat: 2+3+3: {self.attention_filter_sizes}')
        #print(attended_maps[0].size(),attended_maps[1].size(),attended_maps[2].size())
        return self.classifier(torch.cat(attended_maps, dim=1))

    
    # Activation function
    @staticmethod
    def apply_argmax_softmax(pred):
        log_p = F.softmax(pred, dim=1)
        return log_p
    
    
    
    def forward(self,inputs):
        conv1  = F.relu(self.conv1(inputs))
        bnorm1 = self.bnorm1(conv1)
        mpool1 = self.mpool1(bnorm1)
        
        conv2  = F.relu(self.conv2(mpool1))
        bnorm2 = self.bnorm2(conv2)
        mpool2 = self.mpool2(bnorm2)
        
        ###
        # Arbitrarily added to extend size of network (goal: hopefully better classification - seemed to not be 'learning')
        conv3  = F.relu(self.conv3(mpool2))
        bnorm3 = self.bnorm3(conv3)
        #mpool2 = bnorm2
        
        conv3  = F.relu(self.conv3(bnorm3))
        bnorm3 = self.bnorm3(conv3)
        
        conv3  = F.relu(self.conv3(bnorm3))
        bnorm3 = self.bnorm3(conv3)
        mpool3 = bnorm2
        ###
        
        conv4  = F.relu(self.conv4(mpool3))
        bnorm4 = self.bnorm4(conv4)
        
        conv5  = F.relu(self.conv5(bnorm4))
        bnorm5 = self.bnorm5(conv5)
        
        conv6  = F.relu(self.conv6(bnorm5))
        bnorm6 = self.bnorm6(conv6)
        mpool3 = self.mpool3(bnorm6)

        # Applied Attention , Attention map
        #We use conv5 as the global attention as this is the most advanced stage (upsampled to conv3/conv4 to make it work).
        #(inputs are before maxpool layer in sononet_grid_attention.py)
        attendedConv3 , atten3 = self.compatibility_score1(conv4, conv6)
        attendedConv4 , atten4 = self.compatibility_score2(conv5, conv6)
        
        filters = self.filters
        batch_size = inputs.shape[0]
        
        # Aggregation
        pooled = F.adaptive_avg_pool2d(conv6,(1,1)).view(batch_size,-1)
        g1 = torch.sum(attendedConv3.view(batch_size, filters[3], -1), dim=-1)
        g2 = torch.sum(attendedConv4.view(batch_size, filters[4], -1), dim=-1)
        
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
        sigm_psi_f = F.softmax(psi_f.view(batch_size, 1, -1), dim=2)
        sigm_psi_f = sigm_psi_f.view(batch_size, 1, *theta_x_size[2:])

        # sigm_psi_f is attention map! upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode) ### mode = bilinear in 2D, ipnut_size is the input WxH of the data.
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y) #As W = False, W_y = y

        return W_y, sigm_psi_f
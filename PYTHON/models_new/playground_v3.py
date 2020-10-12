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
class playgroundv3(nn.Module):
    def __init__(self,aggregation_mode='concat',n_classes=2):
        super(playgroundv3,self).__init__()
        filters = [6,16,16,32,64,128]
        #ksizes = [11,5,5,3,3,3] # Must all be odd for calculation of padding.
        ksizes = [3,3,3,3,3,3] # Must all be odd for calculation of padding.        
        self.filters = filters
        assert aggregation_mode in ['concat','mean','deep_sup','ft'], "aggregation_mode not valid selection, must be any of: ['concat','mean','deep_sup','ft']"
        self.aggregation_mode = aggregation_mode
        
        # 1->6
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=filters[0],kernel_size=3, padding=3//2, stride=1)
        self.relu1 = nn.ReLU()
        self.bnorm1 = nn.BatchNorm2d(filters[0])
        self.mpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 6->16
        self.conv2 = nn.Conv2d(in_channels=filters[0], out_channels=filters[1],kernel_size=3, padding=3//2, stride=1)
        self.relu2 = nn.ReLU()
        self.bnorm2 = nn.BatchNorm2d(filters[1])
        
        # 16->16
        self.conv4 = nn.Conv2d(in_channels=filters[2], out_channels=filters[3],kernel_size=3, padding=3//2, stride=1)
        self.relu4 = nn.ReLU()
        self.bnorm4 = nn.BatchNorm2d(filters[3])
        
        # 16->64
        self.conv5 = nn.Conv2d(in_channels=filters[3], out_channels=filters[4],kernel_size=3, padding=3//2, stride=1)
        self.relu5 = nn.ReLU()
        self.bnorm5 = nn.BatchNorm2d(filters[4])
        
        # 64->128
        self.conv6 = nn.Conv2d(in_channels=filters[4], out_channels=filters[5],kernel_size=3, padding=3//2, stride=1)
        self.relu6 = nn.ReLU()
        self.bnorm6 = nn.BatchNorm2d(filters[5])
        self.mpool3 = nn.MaxPool2d(kernel_size=5, stride=5)
        
        # These are effectively the same calls as in sononet_grid_attention.py due to us redefining the standard selections in _GridAttentionBlock2D(..)        
        self.compatibility_score1 = _GridAttentionBlock2D_TORR(in_channels=filters[3] , gating_channels=filters[5], 
                                                               inter_channels=filters[2], use_W=False)        
        self.compatibility_score2 = _GridAttentionBlock2D_TORR(in_channels=filters[4], gating_channels=filters[5], 
                                                               inter_channels=filters[2], use_W=False)
        
        self.conv3 = nn.Conv2d(in_channels=filters[1],out_channels=filters[2],kernel_size=ksizes[2],padding=ksizes[2]//2,stride=1)
        self.bnorm3 = nn.BatchNorm2d(filters[2])
        self.mpool2 = nn.MaxPool2d(kernel_size=3, stride=3)
        
        self.flatten = nn.Flatten(1)
        
        self.fc1 = nn.Linear(16*5*5,256) #channel_size * width * height
        self.fc2 = nn.Linear(256,256)
        self.fc3 = nn.Linear(256,2)
        self.dropout = nn.Dropout()
        
        self.module_order = ['conv1','relu1','bnorm1','mpool1',
                             'conv2','relu2','bnorm2',
                             'conv4','relu4','bnorm4',
                             'conv5','relu5','bnorm5',
                             'conv6','relu6','bnorm6',
                             'compatibility_score1',
                             'compatibility_score2']
        self.module_order += ['g1sum','g2sum','avg_pool']
            
        #########################
        # Aggreagation Strategies
        self.attention_filter_sizes = [filters[2], filters[3]]

        if aggregation_mode == 'concat':
            self.module_order += ['cat','classifier','softmax']
            self.classifier = nn.Linear(filters[3]+filters[4]+filters[5], n_classes)
            self.aggregate = self.aggregation_concat

        else:
            self.classifier1 = nn.Linear(filters[3], n_classes)
            self.classifier2 = nn.Linear(filters[4], n_classes)
            self.classifier3 = nn.Linear(filters[5], n_classes)
            self.classifiers = [self.classifier1, self.classifier2, self.classifier3]

            if aggregation_mode == 'mean':
                self.module_order += ['classifier1','classifier2','classifier3']
                self.module_order += ['mean','softmax']
                self.aggregate = self.aggregation_sep

            elif aggregation_mode == 'deep_sup':
                self.module_order += ['classifier1','classifier2','classifier3','cat']
                self.module_order += ['classifier','mean_stack_ds','softmax']
                self.classifier = nn.Linear(filters[3] + filters[4] + filters[5], n_classes)
                self.aggregate = self.aggregation_ds

            elif aggregation_mode == 'ft':
                self.module_order += ['classifier1','classifier2','classifier3','cat_classifications']
                self.module_order += ['classifier','softmax']
                self.classifier = nn.Linear(n_classes*3, n_classes)
                self.aggregate = self.aggregation_ft
            else:
                raise NotImplementedError

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
        return [pred] + preds_sep

    def aggregation_concat(self, *attended_maps):
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
        
        conv4  = F.relu(self.conv4(bnorm2))
        bnorm4 = self.bnorm4(conv4)
        
        conv5  = F.relu(self.conv5(bnorm4))
        bnorm5 = self.bnorm5(conv5)
        
        conv6  = F.relu(self.conv6(bnorm5))
        bnorm6 = self.bnorm6(conv6)
        #mpool3 = self.mpool3(bnorm6)

        # Applied Attention , Attention map
        attendedConv4 , atten3 = self.compatibility_score1(conv4, conv6)
        attendedConv5 , atten4 = self.compatibility_score2(conv5, conv6)
        
        # Aggregation
        batch_size = inputs.shape[0]
        pooled = F.adaptive_avg_pool2d(conv6,(1,1)).view(batch_size,-1)
        
        g1 = torch.sum(attendedConv4.view(batch_size, self.filters[3], -1), dim=-1)
        g2 = torch.sum(attendedConv5.view(batch_size, self.filters[4], -1), dim=-1)
        out = self.aggregate(g1,g2,pooled)
        
        
        #return F.log_softmax(out,dim=1)
        if type(out)==list:
            if self.aggregation_mode == 'mean':
                out = torch.mean(torch.stack(out),dim=[0])
                out = F.softmax(out,dim=1)
            elif self.aggregation_mode == 'deep_sup':
                out = torch.mean(torch.stack(out),dim=[0])
                out = F.softmax(out,dim=1) #Can be added for a second attempt.
            elif self.aggregation_mode == 'ft':
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
        if use_W: # Option taken as the operation in line 53 of grid_attention_layer.py
            self.W = nn.Sequential(
                nn.Conv2d(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d())
        
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=(1,1), stride=(1,1), padding=0, bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels, kernel_size=(1,1), stride=(1,1), padding=0, bias=False)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=True)
        self.nl1 = lambda x: nn.ReLU(x, inplace=True)
        self.instance_norm = nn.InstanceNorm2d(1)

        ### Initialise weights using their package (see imports) ###
        for m in self.children():
            init_weights(m, init_type='kaiming')
        #if use_psi and self.mode == 'concatenation_softmax':
        nn.init.constant_(self.psi.bias.data, 10.0) # Initialises the tensor self.psi.bias.data with values of 10.0 (Because bias=True in initialisation)
    
    # Custom normalisation. Sigmoid not learning. Vanishing gradient problem (I think)
    def norm(self, x, dim=2):
        norm = F.softmax(torch.log(x), dim=dim)
        return norm
    def normC(self, x):
        return self.instance_norm(x)

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
        f = F.relu(theta_x + phi_g)

        psi_f = self.psi(f)

        # Calculate Attention map (sigm_psi_f) and weighted output (W_y)
        # This block was conditional with: 'self.mode == concatenation_softmax'. Other options are listed in grid_attention_layer.py
        # normalisation & scale to x.size()[2:]
        # psi^T . f -> (b, 1, t/s1, h/s2, w/s3)
        
        # Test A
        #sigm_psi_f = self.norm(psi_f.view(batch_size, 1, -1), dim=2) #Nan as training validation

        # Test B
        #sigm_psi_f = F.log_softmax(psi_f.view(batch_size, 1, -1), dim=2) # Vanishing Gradients
        
        # Test C
        #sigm_psi_f = self.normC(psi_f)
        
        # Test D
        #sigm_psi_f_ = torch.sigmoid(psi_f)
        
        # Test E
        sigm_psi_f = F.normalize(psi_f, dim=2)
        
        #sigm_psi_f = F.softmax(psi_f.view(batch_size, 1, -1), dim=2) #Original from playground_v2.py !!!
        sigm_psi_f = sigm_psi_f.view(batch_size, 1, *theta_x_size[2:]) # Fixes dimensions

        # sigm_psi_f is attention map! upsample the attentions and multiply
        sigm_psi_f = F.upsample(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y) #As W = False, W_y = y

        return W_y, sigm_psi_f
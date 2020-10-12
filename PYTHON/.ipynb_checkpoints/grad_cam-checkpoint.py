# From: https://github.com/jacobgil/pytorch-grad-cam > grad-cam.py
# From his most recent update. May7th
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Function
from torchvision import models
from PIL import Image
import os
from utils import *

import argparse


class FeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from targetted intermediate layers """

    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []

    def save_gradient(self, grad):
        self.gradients.append(grad)

    def __call__(self, x):
        outputs = []
        self.gradients = []
        if bool(self.model._modules):
            for name, module in self.model._modules.items():
                x = module(x)
                if name in self.target_layers:
                    x.register_hook(self.save_gradient)
                    outputs += [x]
        else:
            x = self.model(x)
            x.register_hook(self.save_gradient)
            outputs += [x]
        
        return outputs, x



class AttentionFeatureExtractor():
    """ Class for extracting activations and 
    registering gradients from target attention layers """
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.gradients = []
    
    def save_gradient(self, grad):
        self.gradients.append(grad)
    
    def __call__(self,x,g):
        outputs = []
        self.gradients = []
        if bool(self.model._modules):
            # Build attention output:
            theta = self.model._modules['theta'](x)
            phi = self.model._modules['phi'](g)
            psi = self.model._modules['psi'](nn.ReLU()(theta+phi))
            # Register values here, as softmax tends cause outputs to be 0
            psi.register_hook(self.save_gradient)
            outputs += [psi]
            psi = nn.Softmax(dim=2)(psi.view(*psi.shape[:2], -1)).view_as(psi)
            x = psi.expand_as(x)*x
            #x.register_hook(self.save_gradient)
            #outputs += [x]
        else:
            x = self.model(x)
            x.register_hook(self.save_gradient)
            outputs += [x]
        
        return outputs, x

#attendedConv3 , atten3 = self.compatibility_score1(conv4, conv6)
#attendedConv4 , atten4 = self.compatibility_score2(conv5, conv6)
class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, feature_module, target_layers):
        self.model = model
        self.feature_module = feature_module
        self.target_layers = target_layers
        
        self.attention_network = False
        if 'compatibility_score1' in self.model._modules.keys():
            self.attention_network = True
            
        
        self.attn_feature_extractor = AttentionFeatureExtractor(self.feature_module, target_layers)
        self.feature_extractor = FeatureExtractor(self.feature_module, target_layers)

    def get_gradients(self):
        if self.attention_network:
            return self.attn_feature_extractor.gradients
        else:
            return self.feature_extractor.gradients
    
    def gsum(self, x):
        return torch.sum(x.view(*x.shape[:2], -1), dim = -1)

    def avg_pool(self,x):
        return F.adaptive_avg_pool2d(x,(1,1)).view(x.shape[0],-1)
    
    def __call__(self, x):
        # Current goal is to generate an grad-CAM at all... Final goal might be to include each of the three imputs to the aggregation?
        target_activations = []
        # For attention based networks (i.e. playgroundv2)
        if self.attention_network:
            for name in self.model.module_order:
                
                # If the module is found in the odict of modules, call module (and feature extract if required)
                if name in self.model._modules.keys():
                    module = self.model._modules[name]
                    if name in self.target_layers:
                        #print(f'Placing hook for {name}')
                        if name == 'conv6':
                            target_activations, x = self.attn_feature_extractor(x,0)
                            conv6 = x
                        if name == 'compatibility_score1':
                            target_activations, x1 = self.attn_feature_extractor(conv4, conv6) 
                        if name == 'compatibility_score2':
                            target_activations, x2 = self.attn_feature_extractor(conv5, conv6)
                    elif name == 'compatibility_score1': x1, att_map = module(conv4, conv6)
                    elif name == 'compatibility_score2': x2, att_map = module(conv5, conv6)
                    elif name == 'classifier1': c1 = module(g1)
                    elif name == 'classifier2': c2 = module(g2)
                    elif name == 'classifier3': c3 = module(pooled)
                    else:
                        x = module(x)
                        if name == 'conv4': conv4 = x
                        if name == 'conv5': conv5 = x
                        if name == 'conv6': conv6 = x
                
                # Call for custom modules (attention based existance)
                elif name == 'softmax': pass #x = F.softmax(x, dim=1) # Softmax can cause the guided backpropogation values to go to 0.
                # fine tuning (ft)
                elif name == 'g1sum': g1 = self.gsum(x1)
                elif name == 'g2sum': g2 = self.gsum(x2)
                elif name == 'avg_pool': pooled = self.avg_pool(conv6)
                elif name == 'cat_classifications': x = torch.cat([c1,c2,c3], dim=1)
                # deep supervised (deep_sup)
                elif name == 'cat': x = torch.cat([g1,g2,pooled],dim=1)
                elif name == 'mean_stack_ds': x = torch.mean(torch.stack([x,c1,c2,c3]),dim=[0])
                # mean
                elif name == 'mean': x = torch.mean(torch.stack([c1,c2,c3]),dim=0)
                
                # Module called from self.model.module_order is unknown (will pass, and try to continue)
                else:
                    print(f'---->\tUnknown custom module: {name} (can cause error!)')
                
        else: # For non-attention networks (i.e. transfer_original)
            for name, module in self.model._modules.items():
                if module == self.feature_module:
                    print(f'Extracting for {name}')
                    target_activations, x = self.feature_extractor(x)
                elif "avgpool" in name.lower():
                    x = module(x)
                    x = x.view(x.size(0),-1)
                else:
                    x = module(x)
        
        return target_activations, x


def preprocess_image(img):
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]
    for i in range(img.shape[-1]):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    
    temp = torch.from_numpy(np.ascontiguousarray([preprocessed_img[:,:,0]]))
    if preprocessed_img.shape[-1]==3:
        preprocessed_img = \
            np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))
        preprocessed_img = torch.from_numpy(preprocessed_img)
    elif preprocessed_img.shape[-1]==1:
        preprocessed_img = np.ascontiguousarray(np.expand_dims(preprocessed_img[:,:,0],axis=0))
        preprocessed_img = torch.from_numpy(preprocessed_img)
    else:
        print("This isn't right at all ...")
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)
    return input


def show_cam_on_image(img, mask):
    cm = plt.get_cmap('gist_rainbow')
    heatmap = cm(mask)
    heatmap = np.float32(heatmap)[:,:,:3]
    cam = heatmap+np.float32(img)
    # I do not know why, but this is not how images are usually normalised.
    cam = cam / np.max(cam)
    image = Image.fromarray((cam[:, :, :3] * 255).astype(np.uint8))
    return image


class GradCam:
    def __init__(self, model, feature_module, target_layer_names, use_cuda, input_size):
        self.model = model
        self.feature_module = feature_module
        self.model.eval()
        self.cuda = use_cuda
        self.input_size = input_size
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, self.feature_module, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            features, output = self.extractor(input.cuda())
        else:
            features, output = self.extractor(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.feature_module.zero_grad()
        self.model.zero_grad()
        one_hot.backward(retain_graph=True)

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = np.asarray(Image.fromarray(cam).resize(self.input_size,Image.BILINEAR))
        cam = cam - np.min(cam)
        if np.max(cam)!=0:
            cam = cam / np.max(cam)
        return cam


class GuidedBackpropReLU(Function):

    @staticmethod
    def forward(self, input):
        positive_mask = (input > 0).type_as(input)
        output = torch.addcmul(torch.zeros(input.size()).type_as(input), input, positive_mask)
        self.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(self, grad_output):
        input, output = self.saved_tensors
        grad_input = None

        positive_mask_1 = (input > 0).type_as(grad_output)
        positive_mask_2 = (grad_output > 0).type_as(grad_output)
        grad_input = torch.addcmul(torch.zeros(input.size()).type_as(input),
                                   torch.addcmul(torch.zeros(input.size()).type_as(input), grad_output,
                                                 positive_mask_1), positive_mask_2)

        return grad_input


class GuidedBackpropReLUModel:
    def __init__(self, model, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == 'ReLU':
                    module_top._modules[idx] = GuidedBackpropReLU.apply
                
        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None):
        if self.cuda:
            output = self.forward(input.cuda())
        else:
            output = self.forward(input)

        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        # self.model.features.zero_grad()
        # self.model.classifier.zero_grad()
        one_hot.backward(retain_graph=True)

        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")

    return args

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    if img.shape[-1]==1:
        temp = img[:,:,0]
        img = np.stack([temp,temp,temp],axis=2)
    return np.uint8(img*255)


def grad_cam_main(image_path, use_cuda=True, model_path='resnet50', target_layer_names=['2'], save_to='test/', target_index=None, save=True):
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """
    available_networks =['playground','playgroundv1',
                         'playgroundv2_concat','playgroundv2_mean',
                         'playgroundv2_deep_sup','playgroundv2_ft',
                         'transfer_original','transfer_adapted','AGSononet','AGTransfer',
                         'vgg19','resnet50']
    assert os.path.isdir(save_to), f"Entered file path does not lead to valid path: {save_to}"
    model_available = False
    for n in available_networks:
        if n in model_path:
            model_available=True
    assert model_available, 'Model entry not valid'
    
    
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device('cpu' if not use_cuda else 'cuda')
    # Can work with any model, but it assumes that the model has a ########## NO LONGER TRUE
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    if model_path == 'resnet50':
        os.environ['TORCH_HOME'] = '/raid/scratch/mbowles'
        model = models.resnet50(pretrained=True)
        input_size = (224,224)
        grad_cam = GradCam(model=model, feature_module=model.layer4,
                           target_layer_names=target_layer_names,
                           use_cuda=use_cuda,input_size=input_size
                          )
        channel_no = 3
    elif model_path == 'vgg19':
        os.environ['TORCH_HOME'] = '/raid/scratch/mbowles'
        model = models.vgg19(pretrained=True)
    elif 'transfer' in model_path:
        input_size = (150,150)
        model = load_net(path_to_model(model_path),device=device)
        grad_cam = GradCam(model=model, feature_module=model.conv5,
                           target_layer_names=target_layer_names,
                           use_cuda=use_cuda,input_size=input_size
                          )
        channel_no = 1
    else:
        input_size = (150,150)
        model = load_net(path_to_model(model_path),device=device)
        feature_module = model._modules[target_layer_names[0]]
        grad_cam = GradCam(model=model, feature_module=feature_module,
                           target_layer_names=target_layer_names,
                           use_cuda=use_cuda,input_size=input_size
                          )
        channel_no = 1
    
    if type(image_path) == str:
        img = Image.open(image_path).resize(input_size,Image.BILINEAR)
        img = np.float32(np.asarray(img))[:,:,:3]/255
    elif type(image_path) == type(np.asarray([])):
        if len(image_path.shape)<3:
            image_path = np.stack([image_path,image_path,image_path],axis=2)
        #image_path = Image.fromarray(image_path).resize(input_size,Image.BILINEAR)
        img = np.float32(np.asarray(image_path))[:,:,:3]
    else:
        print('reached the ELSE!')
        img = image_path.resize(input_size,Image.BILINEAR)
        img = np.float32(np.asarray(img))[:,:,:3]/255
    input = preprocess_image(img[:,:,:channel_no])#no 2 in description
    
    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    mask = grad_cam(input, target_index)
    
    cam_image = show_cam_on_image(img, mask)
    
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=use_cuda)
    gb = gb_model(input, index=target_index).transpose(1,2,0)
    cam_mask = np.stack([mask, mask, mask],axis=2)
    cam_gb = deprocess_image(cam_mask*gb)
    gb = deprocess_image(gb)
    
    if save:
        cam_image.save(save_to+'cam.jpg')
        Image.fromarray(gb).save(save_to+'gb.jpg')
        Image.fromarray(cam_gb).save(save_to+'cam_gb.jpg')
    return cam_image, mask , gb, cam_gb
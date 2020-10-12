# From: https://github.com/jacobgil/pytorch-grad-cam > grad-cam.py
import argparse
#import cv2 #Have to remove the cv2 module - only use nupmy
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Function
from torchvision import models
from PIL import Image
import os
from utils import *


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
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.target_layers:
                x.register_hook(self.save_gradient)
                #x.register_hook(lambda grad:print(1.5))
                outputs += [x]
        return outputs, x


class ModelOutputs():
    """ Class for making a forward pass, and getting:
    1. The network output.
    2. Activations from intermeddiate targetted layers.
    3. Gradients from intermeddiate targetted layers. """

    def __init__(self, model, target_layers):
        self.model = model
        self.feature_extractor = FeatureExtractor(self.model.features, target_layers)

    def get_gradients(self):
        return self.feature_extractor.gradients

    def __call__(self, x):
        target_activations, output = self.feature_extractor(x)
        output = output.view(output.size(0), -1)
        output = self.model.classifier(output)
        return target_activations, output


def preprocess_image(img): ### Might have to add dimension checker, as this currently expects a coloured (3d) image.
    """Return adapted torch tensor of image with gradients enabled"""
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1] #Reverses the order of the 3rd dim. See: extended slicing
    for i in range(3): #subtract mean and divide by standard deviation?
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - means[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / stds[i]
    preprocessed_img = \
        np.ascontiguousarray(np.transpose(preprocessed_img, (2, 0, 1)))#move third dim and resort data in storage (contiguous: C-like?)
    preprocessed_img = torch.from_numpy(preprocessed_img)
    preprocessed_img.unsqueeze_(0)
    input = preprocessed_img.requires_grad_(True)#Enable gradient computation for torch tensor.
    return input


def show_cam_on_image(img, mask):
    """Gives mask a colormap format and overlays onto img."""
    cm = plt.get_cmap('gist_rainbow')
    heatmap = cm(mask)
    heatmap = np.float32(heatmap)[:,:,:3] #Removed alpha dimension #Removed scaling down by 255 as range is already between 0 and 1.
    ### Temporary ###
    #plt.imshow(heatmap)
    #plt.show()
    cam = heatmap+np.float32(img)
    cam = cam / np.max(cam)/2+1
    Image.fromarray((cam[:, :, :3] * 255).astype(np.uint8)).save('test/cam.jpg')


class GradCam:
    def __init__(self, model, target_layer_names, use_cuda):
        self.model = model
        self.model.eval()
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()

        self.extractor = ModelOutputs(self.model, target_layer_names)

    def forward(self, input):
        return self.model(input)

    def __call__(self, input, index=None,input_size=(224,224)):
        if self.cuda:
            features, output = self.extractor(input.cuda()) #features is a list of outputs at target layers, output is the classification of the data.
        else:
            features, output = self.extractor(input)
        if index == None:
            index = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][index] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True)
        plt.imshow(features[0].data.cpu().numpy().squeeze().mean(axis=0))
        plt.show()
        
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)

        self.model.features.zero_grad()
        self.model.classifier.zero_grad()
        
        print(one_hot,index,output.shape)
        print(f'From grad-cam4 {input.grad}')
        one_hot.backward(retain_graph=True) ###################################################### THIS IS WHERE MY FORWARD HOOK IS ACCTUALLY BEING EXECUTED!!!!
        print(f'From grad-cam5 {input.grad.shape}')

        grads_val = self.extractor.get_gradients()[-1].cpu().data.numpy()

        target = features[-1]
        target = target.cpu().data.numpy()[0, :]

        weights = np.mean(grads_val, axis=(2, 3))[0, :]
        cam = np.zeros(target.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * target[i, :, :]

        cam = np.maximum(cam, 0)
        cam = np.asarray(Image.fromarray(cam).resize(input_size,Image.BILINEAR))
        cam = cam - np.min(cam)
        cam = cam / np.max(cam)
        
        return cam

# ==============================================================================================================
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

        # replace ReLU with GuidedBackpropReLU
        for idx, module in self.model.features._modules.items():
            if module.__class__.__name__ == 'ReLU':
                self.model.features._modules[idx] = GuidedBackpropReLU.apply

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
        print(f'One hot class prediction: {one_hot.shape}\t(I.e. FR{index+1})')
        if self.cuda:
            one_hot = torch.sum(one_hot.cuda() * output)
        else:
            one_hot = torch.sum(one_hot * output)
        print(input.grad.cpu().data.numpy().max())
        one_hot.backward(retain_graph=True)
        print(input.grad.cpu().data.numpy().max())
        print(f'One hot class prediction: {one_hot}\t(I.e. FR{index+1})')


        output = input.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output

def deprocess_image(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img*255)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=False,
                        help='Use NVIDIA GPU acceleration')
    # Reads in the images path - ie. I need images in a path to read from (as input).
    parser.add_argument('--image-path', type=str, default='./examples/both.png',
                        help='Input image path')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        print("Using GPU for acceleration")
    else:
        print("Using CPU for computation")
    return args


def grad_cam_main(image_path, use_cuda=True,model='vgg19'):
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """
    available_networks =['playground','playgroundv1',
                         'playgroundv2_concat','playgroundv2_mean',
                         'playgroundv2_deep_sup','playgroundv2_ft',
                         'transfer_original','AGSononet','AGTransfer',
                         'vgg19']
    model_available = False
    for n in available_networks:
        if n in model:
            model_available=True
    assert model_available, 'Model entry not valid'
    
    
    use_cuda = use_cuda and torch.cuda.is_available()
    device = torch.device('cpu' if not use_cuda else 'cuda')
    print(device)
    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    if model == 'vgg19':
        os.environ['TORCH_HOME'] = '/raid/scratch/mbowles'
        model = models.vgg19(pretrained=True)
        print(f'vgg19 model: \n\n{model}')
        print(f'Layer 35 usually extracted: \n{model.features._modules["35"]}')
        input_size = (224,224)
    else:
        input_size = (150,150)
        model = load_net(path_to_model(model),device=device)
        
    
    grad_cam = GradCam(model=model,
                       target_layer_names=["35"],
                       use_cuda=use_cuda
                      )
    
    img = Image.open(image_path,).resize(input_size,Image.BILINEAR)
    img = np.float32(np.asarray(img))[:,:,:3]/255
    print(img.shape)
    print(img.max(),img.min())
    plt.imshow(img[:,:,:3])
    plt.show()
    #img = np.float32(cv2.resize(img, (224, 224))) / 255
    input = preprocess_image(img)#no 2 in description
    print(f'From main0 {input.grad}')

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    print(f'From main1 {input.grad}')
    mask = grad_cam(input, target_index)
    print(f'From main2 {input.grad}')
    show_cam_on_image(img, mask)
    
    
    gb_model = GuidedBackpropReLUModel(model=model, use_cuda=use_cuda)
    gb = gb_model(input, index=target_index).transpose(1,2,0)
    #gb = gb.transpose((1, 2, 0)) #Applied because in cv2, its BGR not RGB ??? <-Cant be true
    cam_mask = np.stack([mask, mask, mask],axis=2)
    cam_gb = deprocess_image(cam_mask*gb)
    gb = deprocess_image(gb)
    Image.fromarray(gb).save('test/gb.jpg')
    Image.fromarray(cam_gb).save('test/cam_gb.jpg')

    
# ===========================================================
if __name__ == '__main__':
    """ python grad_cam.py <path_to_image>
    1. Loads an image with opencv.
    2. Preprocesses it for VGG19 and converts to a pytorch variable.
    3. Makes a forward pass to find the category index with the highest score,
    and computes intermediate activations.
    Makes the visualization. """

    args = get_args()
    os.environ['TORCH_HOME'] = '/raid/scratch/mbowles'
    
    # Can work with any model, but it assumes that the model has a
    # feature method, and a classifier method,
    # as in the VGG models in torchvision.
    grad_cam = GradCam(model=models.vgg19(pretrained=True), \
                       target_layer_names=["35"], use_cuda=args.use_cuda)

    img = cv2.imread(args.image_path, 1)#no 1 in description
    img = np.float32(cv2.resize(img, (224, 224))) / 255
    input = preprocess_image(img)#no 2 in description

    # If None, returns the map for the highest scoring category.
    # Otherwise, targets the requested index.
    target_index = None
    mask = grad_cam(input, target_index)

    show_cam_on_image(img, mask)

    gb_model = GuidedBackpropReLUModel(model=models.vgg19(pretrained=True), use_cuda=args.use_cuda)
    gb = gb_model(input, index=target_index)
    #gb = gb.transpose((1, 2, 0)) #Applied because in cv2, its BGR not RGB ??? <-Cant be true
    cam_mask = np.stack([mask, mask, mask],axis=2)
    cam_gb = deprocess_image(cam_mask*gb)
    gb = deprocess_image(gb)
    Image.fromarray(gb).save('test/gb.jpg')
    Image.fromarray(cam_gb).save('test/cam_gb.jpg')
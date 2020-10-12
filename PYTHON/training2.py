# Basics
import numpy as np
import matplotlib.pyplot as plt
import os
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import SubsetRandomSampler
# Torchvision
import torchvision
from torchvision import transforms, datasets
# Other
import PIL
from torchsummary import summary
from FRDEEP import FRDEEPF
from MiraBest import MiraBest_full
from models_new import *
from models.networks_other import init_weights
from skimage.transform import resize
# Special Plot Functions
import matplotlib.cm as cm

# Function to train a model
def train(date,
          Epoch,
          dataset,
          datasubset,
          net_name,
          optimizer_name,
          learning_rate='automatic',
          validation_epoch=360,
          num_batches=1,
          normalisation=None,
          aggregation_mode='',
          No_AttentionGates = 3
         ):
    """
    Args:
        date: MMDD
        Epoch: number of epochs to train over
        dataset: in ['MiraBest','FRDEEP']
        datasubset: in ['NOUNC','NOHYBRID','HYBRID']
        net_name in available_networks
        validatopm_epoch: number of epochs to calc loss for validation
    Out:
        saves loss data and trained models
    """
    available_networks =['playground', 'playgroundv1',
                         'playgroundv2_concat', 'playgroundv2_mean',
                         'playgroundv2_deep_sup', 'playgroundv2_ft',
                         'playgroundv3',
                         'playgroundv3_concat', 'playgroundv3_mean',
                         'playgroundv3_deep_sup', 'playgroundv3_ft',
                         'playgroundv4',
                         'AGRadGalNet',
                         'transfer_original', 'transfer_adapted', 'AGSononet', 'AGTransfer']
    optim_names =['SGD', 'Adagrad', 'Adam', 'Adadelta']
    
    # Input Assertions:
    assert dataset in ['MiraBest', 'FRDEEP']
    assert datasubset in ['NOUNC', 'NOHYBRID', 'HYBRID', '']
    assert net_name in available_networks
    assert optimizer_name in optim_names
    
    # Other conditions
    batch_size = 16
    valid_size = 0.2
    varying_optimizer_sononet = False
    loss_function = nn.CrossEntropyLoss()
    optim_label=optimizer_name
    if type(learning_rate) == float:
        learning_rate=learning_rate
    else:
        if optimizer_name =='Adagrad':
            learning_rate=1.0e-3
        if optimizer_name == 'Adam':
            learning_rate = 1.0e-6
        else:
            learning_rate = 1.0e-4 #Adagrad: 1.0e-(5,6,7) didnt work #Adam: 1.0e-4 didnt work
    

    # Selection of data subset
    all_labels = [*range(10)]
    if datasubset == 'NOUNC':
        allowed_labels = all_labels[0:3]+all_labels[5:7]
    elif datasubset == 'NOHYBRID':
        allowed_labels = all_labels[0:5]+all_labels[5:8]
    elif datasubset == 'HYBRID':
        allowed_labels = all_labels
    else:
        allowed_labels = all_labels

    date_label = f'{date}-{normalisation}-{dataset}{datasubset}{optim_label}' #format: mmdd
    ckpt_name = f"{net_name}-{date_label}-{learning_rate}_{Epoch}Epochs.pt"
    folder_name = f"TrainedNetworks/{date_label}-{net_name+aggregation_mode}-{learning_rate}"

    print(f"Final Model Saved Under {folder_name}")
    if os.path.isdir(folder_name):
        print(f'Directory to save model already exists!')
    else:
        os.mkdir(folder_name)

    
    #############
    # Import Data
    test_transform = transforms.Compose([#transforms.RandomRotation([0,360],resample=PIL.Image.BILINEAR),
                                          transforms.RandomAffine(degrees=180, translate=(2/150,2/150), scale=(0.9,1.1), resample=PIL.Image.BILINEAR),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5],[0.5])
                                         ])
    out_transform = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize([0.5],[0.5])
                                       ])
    train_transform = transforms.Compose([#transforms.RandomRotation([0,360],resample=PIL.Image.BILINEAR),
                                          #transforms.RandomResizedCrop(size=(150,150), scale=(0.7,1.3),ratio=1),
                                          transforms.RandomAffine(degrees=180, translate=(2/150,2/150), scale=(0.9,1.1), resample=PIL.Image.BILINEAR),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5],[0.5])
                                         ])
    ##################
    # Sampler Function
    def Sampler(trainset, valid_size = 0., allowed_labels=[0,1,2,3,4,5,6,7,8,9]):
        # Obtain training indices that will be used for validation
        indices = []
        # Filter out all unwanted data before batching
        for batch_idx, (data,label) in enumerate(trainset):
            if label in allowed_labels:
                indices.append(batch_idx)

        np.random.shuffle(indices)
        split = int(np.floor(valid_size * len(indices)))
        train_idx, valid_idx = indices[split:], indices[:split]
        print(f"Samples retained: {len(indices)} of {len(trainset)} with allowed_labels = {allowed_labels}")

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        return (train_sampler, valid_sampler)

    #################
    # Reading in Data
    if dataset == 'MiraBest':
        class_splitting_index = 5
        traindata = MiraBest_full(root='./FIRST_data', train=True, download=True, transform=train_transform)
        testdata = MiraBest_full(root='./FIRST_data', train=False, download=True, transform=test_transform)
        outdata = MiraBest_full(root='./FIRST_data', train=False, download=True, transform=out_transform)
    elif dataset == 'FRDEEP':
        class_splitting_index = 1
        traindata = FRDEEPF(root='./FIRST_data', train=True, download=True, transform=train_transform)
        testdata = FRDEEPF(root='./FIRST_data', train=False, download=True, transform=test_transform)
        outdata = FRDEEPF(root='./FIRST_data', train=False, download=True, transform=out_transform)

    ##############################
    # Sample and select data which is within 'allowed_labels'
    train_sampler,valid_sampler = Sampler(traindata, valid_size=valid_size, allowed_labels=allowed_labels)

    outset = torch.utils.data.DataLoader(outdata, batch_size=batch_size)
    trainset = torch.utils.data.DataLoader(traindata, batch_size=batch_size, sampler=train_sampler)
    validset = torch.utils.data.DataLoader(traindata, batch_size=batch_size, sampler=valid_sampler)
    testset = torch.utils.data.DataLoader(testdata, batch_size=batch_size, shuffle=True)

    ##################
    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"We will be putting our network and data on >> {device} <<")

    ##################
    # Network Selection
    if net_name=='playground': net=playground()
    if net_name=='playgroundv1': net=playgroundv1b()
    if net_name=='playgroundv2_concat': net=playgroundv2(aggregation_mode='concat')
    if net_name=='playgroundv2_mean': net = playgroundv2(aggregation_mode='mean')
    if net_name=='playgroundv2_deep_sup': net = playgroundv2(aggregation_mode='deep_sup')
    if net_name=='playgroundv2_ft': net = playgroundv2(aggregation_mode='ft')
    if net_name=='playgroundv3': net = playgroundv3(aggregation_mode='ft')
    if net_name=='playgroundv3_concat': net=playgroundv3(aggregation_mode='concat')
    if net_name=='playgroundv3_mean': net = playgroundv3(aggregation_mode='mean')
    if net_name=='playgroundv3_deep_sup': net = playgroundv3(aggregation_mode='deep_sup')
    if net_name=='playgroundv3_ft': net = playgroundv3(aggregation_mode='ft')
    if net_name=='playgroundv4': net = playgroundv4()
    if net_name=='AGRadGalNet': net = AGRadGalNet(aggregation_mode=aggregation_mode,normalisation=normalisation,AG=No_AttentionGates)
    if net_name=='transfer_original': net = transfer_original()
    if net_name=='transfer_adapted': net = transfer_adapted()
    if net_name=='AGSononet': net = AGSononet()
    if net_name=='AGTransfer': net = AGTransfer()
    net = net.to(device)

    # Selection of Optimizer
    if optimizer_name == 'SGD': 
        optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9,weight_decay=1e-6)#, nesterov=True)
        varying_optimizer_sononet = False
        optim_label='SGD' #lr=1.0e-4 bestso far
    elif optimizer_name == 'Adagrad':
        optimizer = optim.Adagrad(net.parameters(), lr=learning_rate) #Best lr: 1.0e-3 (so far)
    elif optimizer_name == 'Adadelta':
        optimizer = optim.Adadelta(net.parameters()) # lr = 1.0 standard <-- https://pytorch.org/docs/stable/optim.html
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(net.parameters(),lr=learning_rate) # lr = 0.001 standard
        #Best lr: 1.0e-5 (so far - doesnt go past 0.5 for 360 epochs)
    
    #######
    # Train
    train_net = True #remnant of laziness
    if train_net:
        # Variable selections
        valid_loss_min = np.Inf
        num_samples = 1 #No. of samples per dataset.
        train_loss_plot=[]
        valid_loss_plot=[]
        min_v_loss_plot=[]

        for epoch_count in range(Epoch):

            # Model Training 
            # Test repititions (ie 360 reps for loss calc before backpropagation) ??? ### !!!
            train_loss = 0.
            valid_loss = 0.
            net.train() #Set network to train mode.    
            if 'binary_labels' in locals():
                del binary_labels
            if 'outputs' in locals():
                del outputs
            
            # Repeat to account for data augmentation
            for i in range(num_batches):
                for batch_idx , (data, labels) in enumerate(trainset): #Iterates through each batch.        
                    data = data.to(device)
                    labels = labels.to(device)

                    # Create binary labels to remove morphological subclassifications (for MiraBest)
                    binary_labels = np.zeros(labels.size(),dtype=int)
                    binary_labels = np.where(labels.cpu().numpy()<class_splitting_index, binary_labels, binary_labels+1)
                    binary_labels = torch.from_numpy(binary_labels).to(device)

                    pred = net.forward(data)
                    optimizer.zero_grad()
                    loss = loss_function(pred,binary_labels)
                    loss.backward(retain_graph=True)
                    optimizer.step()
                    train_loss += (loss.item()*data.size(0)) / num_samples


            ### Model Validation ###
            net.eval()
            for epoch_valid in range(validation_epoch):
                for batch_idx, (data, labels) in enumerate(validset):
                    data = data.to(device)
                    labels = labels.to(device)

                    # Create binary labels to remove morphological subclassifications
                    binary_labels = np.zeros(labels.size(), dtype=int)
                    binary_labels = np.where(labels.cpu().numpy()<class_splitting_index, binary_labels, binary_labels+1)
                    binary_labels = torch.from_numpy(binary_labels).to(device)

                    outputs = net.forward(data)
                    loss = loss_function(outputs, binary_labels)
                    valid_loss += (loss.item()*data.size(0)) / num_samples

            # Average losses (scaled according to validation dataset size)
            train_loss = train_loss/(len(trainset.dataset)*(1-valid_size)*num_batches)
            valid_loss = valid_loss/(len(validset.dataset)*valid_size*validation_epoch)

            # Print
            print(f"Epoch:{epoch_count:3}\tTraining Loss: {train_loss:8.6f}\t\tValidation Loss: {valid_loss:8.6f}")

            # Save model if validation loss decreased (ie. best model with least overfitting)
            if valid_loss <= valid_loss_min:
                print(f"\tValidation Loss Down: \t({valid_loss_min:8.6f}-->{valid_loss:8.6f}) ... Updating saved model.")
                #torch.save(net.state_dict(), f'TrainedNetworks/{ckpt_name}')
                ckpt_name_temp = f'{epoch_count}of{Epoch}-vloss{valid_loss:.2f}'
                torch.save(net.state_dict(), f'{folder_name}/{epoch_count}.pt')
                valid_loss_min = valid_loss

            # Save training loss / validation loss for plotting
            train_loss_plot.append(train_loss)
            valid_loss_plot.append(valid_loss)
            min_v_loss_plot.append(valid_loss_min)

        print(f"\nFinished training.\nMinimum Validation Loss: {valid_loss_min:8.6}\n")
        
        # Save final model, no matter the loss
        torch.save(net.state_dict(), f'TrainedNetworks/temp.pt')

        #######
        # Plotting and saving loss changes
        plt.figure(figsize=(8,8))

        plt.subplot(211)
        plt.plot(train_loss_plot)
        plt.plot(valid_loss_plot,':')
        plt.plot(min_v_loss_plot,'g')
        
        plt.title(f'{net_name} Loss (lr: {learning_rate})')
        plt.ylabel('Loss')
        plt.ylim(0,1)
        plt.grid()
        plt.legend(['Training loss','Validation loss','Minimal Validation Loss'])

        plt.subplot(212)

        plt.plot(train_loss_plot)
        plt.plot(valid_loss_plot)
        plt.plot(min_v_loss_plot,':g')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid()
        plt.legend(['Training loss','Validation loss','Minimal Validation Loss'])

        plt.savefig(f'TrainingLosses/{date_label}-{net_name}-{learning_rate}_Losses.png')
        np.savez(f'TrainingLosses/{date_label}-{net_name}-{learning_rate}_Losses',
                 train_loss_plot=train_loss_plot,
                 valid_loss_plot=valid_loss_plot,
                 min_v_loss_plot=min_v_loss_plot)
        plt.show()

if __name__ == '__main__':
    print(f"Import this file and run 'train' function to train a model of your choice.")
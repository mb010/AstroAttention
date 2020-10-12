# ==========================================================
# Imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
from itertools import product

import torchvision
from torchvision import transforms, datasets
from FRDEEP import FRDEEPF
from MiraBest import MiraBest_full
from models_new import *
from PIL import Image
import PIL
from torchsummary import summary
from models.networks_other import init_weights
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, recall_score, f1_score, precision_score, auc#, plot_confusion_matrix
from skimage.transform import resize
from mpl_toolkits.axes_grid1 import ImageGrid

# ==========================================================
def functions_list():
    print("""Utils Functions:

Data Manipulation:
\tpath_to_model(file_name)
\tdata_call(dataset_name)
\tdetermine_dataset(dataset,model_name) ... dataset in ['automatic','FRDEEP-F','MiraBest']

Model Manipulation:
\tload_net(model_name,device)
\ttraining_validation(PATH,xlims=[None,None],save=False,full_path=False) ... PATH is a local title of a folder or file (within ./TrainedNetworks)
\tprediction(dataset, net, class_groups,(device='cuda',reps='360'))
\tevaluate(file_name,dataset='automatic')

Evaluation Plots:
\tplot_conf_mat(conf_matrix,normalised=True,n_classes=2,format_input=None,title='Confusion Matrix')
\tplot_roc_curve(fpr,tpr,title='ROC Curve (AUC=\{auc:.3f\})')
\tout_print(out)

Attention Maps:
\tattentions_func(batch_of_images, net, mean=True, device=torch.device('cpu'))
\tattention_analysis(source, source_only=True, attention_maps=None, GradCAM=None)
\tAttentionImagesByEpoch(sources, folder_name, net,epoch=1500, device=torch.device('cpu'))
\tattention_epoch_plot(source_images,folder_name, logged=False, width=3, device=torch.device('cpu'))

GradCAM:
\tTo be completed.

Other:
\tmask_on_image(img, mask)
\tSortedDataSamples(data_name, transformed=True,  rotations=1, subset='NOHYBRID')
\tnet_name_extraction(PATH)

Incomplete:
\t- Loading from Pickled dicts
\t- GradCAM Call for a given image
""")

# ==========================================================
# Import testing and output Data
def data_call(dataset_name,dataloader=True):
    assert dataset_name in ['MiraBest','FRDEEP-F'], f"Called dataset ({name}) not valid. Must be either 'MiraBest' or 'FRDEEP-F'"
    batch_size = 16
    # Define transformations required for analysis (ie. func evaluate(f))
    out_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.5],[0.5])
                                        ])
    test_transform = transforms.Compose([#transforms.RandomRotation([0,360],resample=PIL.Image.BILINEAR),
                                          transforms.RandomAffine(degrees=180, translate=(2/150,2/150), scale=(0.9,1.1), resample=PIL.Image.BILINEAR),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.5],[0.5])
                                         ])
    if dataset_name == 'MiraBest':
        testdata = MiraBest_full(root='./FIRST_data', train=False, download=True, transform=test_transform)
        outdata = MiraBest_full(root='./FIRST_data', train=False, download=True, transform=out_transform)
    elif dataset_name == 'FRDEEP-F':
        testdata = FRDEEPF(root='./FIRST_data', train=False, download=True, transform=test_transform)
        outdata = FRDEEPF(root='./FIRST_data', train=False, download=True, transform=out_transform)
    
    if dataloader:
        outset = torch.utils.data.DataLoader(outdata, batch_size=batch_size)
        testset = torch.utils.data.DataLoader(testdata, batch_size=batch_size)
    else:
        outset, testset = outdata, testdata
    return testset, outset


# ==========================================================
# PATH -> model_name (ie. full path)
def path_to_model(PATH):
    if os.path.isfile(PATH):
        model_name = PATH
    elif os.path.isfile('TrainedNetworks/'+PATH):
        model_name = 'TrainedNetworks/'+PATH
    elif os.path.isdir(PATH):
        folder = PATH
        # Select model with lowest loss
        local_files = os.listdir(folder)
        b=0
        for i in local_files:
            a=i.split('.')[0]
            if a.isnumeric() and int(a)>b:
                b=int(a)
        model_name = folder+f'/{b}.pt'
    elif os.path.isdir('TrainedNetworks/'+PATH):
        folder = 'TrainedNetworks/'+PATH
        # Select model with lowest loss
        local_files = os.listdir(folder)
        b=0
        for i in local_files:
            a=i.split('.')[0]
            if a.isnumeric() and int(a)>b:
                b=int(a)
        model_name = f'{folder}/{b}.pt'
    else:
        model_name = None
    return model_name

# ==========================================================
# Selecting dataset to be loaded and setting class_groups
def determine_dataset(dataset,model):
    assert dataset in ['FRDEEP-F','MiraBest','MiraBestNOHYBRID','MiraBestNOUNC','automatic'], f"Dataset {dataset} is not applicable."  
    if dataset == 'automatic':
        if 'MiraBest' in model:
            dataset = 'MiraBest'
        else:
            dataset = 'FRDEEP-F'
    if dataset == 'MiraBest':
        if 'NOHYBRID' in model:
            class_groups = [[0,1,2,3,4],[5,6,7]]
            data_name=dataset+'NOHYBRID'
        elif 'NOUNC' in model:
            class_groups = [[0,1,2],[5,6]]
            data_name=dataset+'NOUNC'
        else:
            class_groups = [[0,1,2,3,4],[5,6,7,8,9]]
            data_name=dataset+'HYBRID'
    elif dataset == 'FRDEEP-F':
        class_groups = [[0],[1]]
        data_name=dataset
    elif 'NOHYBRID' in dataset:
        dataset = 'MiraBest'
        data_name=dataset+'NOHYBRID'
        class_groups = [[0,1,2,3,4],[5,6,7]]
    elif 'NOUNC' in dataset:
        dataset= 'MiraBest'
        data_name=dataset+'NOUNC'
        class_groups = [[0,1,2],[5,6]]
    elif 'HYBRID' in MiraBest:
        dataset= 'MiraBest'
        data_name=dataset+'HYBRID'
        class_groups = [[0,1,2,3,4],[5,6,7,8,9]]
    return dataset, data_name, class_groups

# ==========================================================
# Load in the correct network according to the file path
def load_net(model, device):
    assert device in [torch.device('cpu'),torch.device('cuda')], f"Device {device} must be either 'cuda' or 'cpu'."
    net_name = net_name_extraction(model)
    # Select network, put on device and load in model
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if net_name=='playground': net=playground()
    if net_name=='playgroundv1': net=playgroundv1b()
    if net_name=='playgroundv2_concat': net=playgroundv2(aggregation_mode='concat')
    if net_name=='playgroundv2_mean': net = playgroundv2(aggregation_mode='mean')
    if net_name=='playgroundv2_deep_sup': net = playgroundv2(aggregation_mode='deep_sup')
    if net_name=='playgroundv2_ft': net = playgroundv2(aggregation_mode='ft')
    if net_name=='playgroundv3': net = playgroundv3(aggregation_mode='ft')
    if net_name=='playgroundv3_concat': net=playgroundv2(aggregation_mode='concat')
    if net_name=='playgroundv3_mean': net = playgroundv2(aggregation_mode='mean')
    if net_name=='playgroundv3_deep_sup': net = playgroundv2(aggregation_mode='deep_sup')
    if net_name=='playgroundv3_ft': net = playgroundv2(aggregation_mode='ft')
    if net_name=='playgroundv4': net = playgroundv4()
    
    if net_name=='AGRadGalNet':
        # Normalisation
        norm = model.split('-')[1]
        # Attention gate no:
        date_code = model.split('-')[0]
        ag_no = 3
        if date_code[-5:-1] == 'Exp3':
            ag_no = int(date_code[-1])
        print(date_code[-9:], date_code[-5:-1], date_code[-1])
        # Aggregation Mode
        for tag in model.split('-'):
            if 'AGRadGalNet' in tag:
                agg_mode = tag[11:]
        # Call respective network:
        net = AGRadGalNet(
            aggregation_mode = agg_mode,
            normalisation = norm, 
            AG = ag_no
        )
    #if net_name=='AGRadGalNet': net = AGRadGalNet(aggregation_mode='concat')
    #if net_name=='AGRadGalNet': net = AGRadGalNet(aggregation_mode='deep_sup')
    #if net_name=='AGRadGalNet': net = AGRadGalNet(aggregation_mode='ft')
    
    if net_name=='transfer_original': net = transfer_original()
    if net_name=='transfer_adapted': net = transfer_adapted()
    if net_name=='AGSononet': net = AGSononet()
    if net_name=='AGTransfer': net = AGTransfer()
    net.to(device)
    net.load_state_dict(torch.load(model,map_location=torch.device(device)))
    net.eval()
    return net
    
# ==========================================================
# Validation / Training plot functions

# Training Validation Plot
def training_validation(PATH, xlims=[None,None], save=False, full_path=False, publication=False): # Full path required for training pre 0429(MMDD)
    if not full_path:
        files = os.listdir('TrainingLosses')
        for i in files:
            if (PATH in i) and (i[-3:]=='npz'):
                PATH = 'TrainingLosses/'+i
    f = np.load(PATH)
    train_loss_plot = f['train_loss_plot']
    valid_loss_plot = f['valid_loss_plot']
    min_v_loss_plot = f['min_v_loss_plot']
    

    net_name = net_name_extraction(PATH)
    if publication:
        # Set up plot
        plt.figure(figsize=(8,4.5))

        #plt.subplot(211)
        #plt.plot(train_loss_plot)
        #plt.plot(valid_loss_plot,':')
        #plt.plot(min_v_loss_plot,'g')    
        #plt.title(f'{net_name} Loss')
        #plt.ylabel('Loss')
        #plt.ylim(0,1)
        #plt.xlim(xlims[0],xlims[1])
        #plt.grid()
        #plt.legend(['Training loss','Validation loss','Minimal Validation Loss'])

        #plt.subplot(212)
        plt.plot(train_loss_plot)
        plt.plot(valid_loss_plot,':')
        plt.plot(min_v_loss_plot,'g')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xlim(xlims[0],xlims[1])
        plt.grid()
        plt.legend(['Training loss','Validation loss','Minimal Validation Loss'])

        if save:
            plt.savefig(f'TrainingLosses/{ckpt_name}_Losses.png')        
        plt.show()

    else:
        # Set up plot
        plt.figure(figsize=(8,8))

        plt.subplot(211)
        plt.plot(train_loss_plot)
        plt.plot(valid_loss_plot,':')
        plt.plot(min_v_loss_plot,'g')    
        plt.title(f'{net_name} Loss')
        plt.ylabel('Loss')
        plt.ylim(0,1)
        plt.xlim(xlims[0],xlims[1])
        plt.grid()
        plt.legend(['Training loss','Validation loss','Minimal Validation Loss'])

        plt.subplot(212)
        plt.plot(train_loss_plot)
        plt.plot(valid_loss_plot)
        plt.plot(min_v_loss_plot,':g')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.xlim(xlims[0],xlims[1])
        plt.grid()
        plt.legend(['Training loss','Validation loss','Minimal Validation Loss'])

        if save:
            plt.savefig(f'TrainingLosses/{ckpt_name}_Losses.png')        
        plt.show()
    
# ==========================================================
# Make predictions:
def prediction(dataset, net, class_groups,
               device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
               reps=360,
               dropout=False
              ):
    """
    Args:
        net   loaded and trained network to be evaluated
        class_groups   [[],[]] List of lists to transform multiclass problem into a two class problem.
        device   device onto which the model has been loaded (if not provided, prefers cuda over cpu)
        reps   How many random transformations of the data are tested.
    Out:
        predicted   The predicted (binary) class
        predicted_probabilities   The probabilities of the predicted (binary) class (ie. number in [0,1])
        labels   original labels of test data
        output   raw output of the model
        
    """
    assert device in [torch.device('cpu'),torch.device('cuda')], f"Device {device} must be either torch.device('cuda' or 'cpu')."
    # Predictions of Testing Data
    labels, temp, predicted = [], [], []
    outputs = []
    raw_outputs = np.zeros((1,2))
    if dropout:
        net.train()
    else:
        net.eval()
    for counter in range(reps):
        testset, outset = data_call(dataset)
        with torch.no_grad():
            for data, label in testset:
                data = data.to(device)
                output = net.forward(data)
                for i in range(output.cpu().numpy().shape[0]):
                    raw_outputs = np.append(raw_outputs, np.expand_dims(output.cpu().numpy()[i],0), 1)
                    if label.cpu().numpy()[i] in class_groups[0]:
                        predicted.append(np.argmax(output.cpu().numpy()[i]))
                        labels.append(0)
                        outputs.append(output.cpu().numpy()[i])
                    elif label.cpu().numpy()[i] in class_groups[1]:
                        predicted.append(np.argmax(output.cpu().numpy()[i]))
                        labels.append(1)
                        outputs.append(output.cpu().numpy()[i])
                    else: # Do not consider labels outside of class_groups
                        pass
    outputs_torch = torch.as_tensor(np.asarray(outputs)).to(device)
    predicted_probabilities = F.softmax(outputs_torch,dim=1).cpu().numpy() # Makes probabilities
    
    return predicted, predicted_probabilities[:,1], labels#, raw_outputs[1:,:]

# ==========================================================
# RMSE
def RMS(x):
    mean = np.mean(x)
    rms = 0
    for i in x:
        rms += (mean-i)**2
    rms = rms**0.5
    rms = rms/len(x)
    return mean, rms

# ==========================================================
# Evaluation
def evaluate(f, dataset='automatic', error=False, dictionary=False):
    """
    Goal is to evaluate the model in TrainedNetworks/f directory/file.
    Args:
        f             folder or file to be evaluated.
        dataset       dataset to be tested on. Valid selections include: ['Automatic','FRDEEP-F','MiraBest']
        error         should the predictions be made with dropout, repeated 10 times and an uncertainty estimated.
    Returns:
        [
        data_name,net_name,date,lr,epoch,
        auc,
        confusion[0,0],confusion[1,0],confusion[0,1],confusion[1,1],
        recall[0],recall[1],
        precision[0],precision[1],
        f1[0],f1[1],
        fpr,tpr,thresholds
        ]
    """
    PATH = 'TrainedNetworks/'+f
    exemptions = ['playground-0128-0.1_500Epochs.pt',
                  '0303-MiraBest-playgroundv1']#,'0224-log-playgroundv1']
    assert os.path.isdir(PATH) or os.path.isfile(PATH), f"Entered file path does not lead to valid path: {PATH}"
    assert dataset in ['FRDEEP-F','MiraBest','MiraBestNOHYBRID','MiraBestNOUNC','automatic'], f"Dataset {dataset} is not applicable."  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # For files which are outdated or tests (exemptions):
    if ('CIFAR' in f) or (f in exemptions):
        print(f'NOT ABLE TO EVALUATE: {f}')
        return [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]
    
    # Extract model path and details of model
    if os.path.isfile(PATH):
        model = PATH
        variables = model.replace('/','-').split('-')
        date = variables[2][:4]
        
        lr_helper = variables[-1].split('_')[0]
        if lr_helper[-3:] == '.pt':
            lr_helper = lr_helper[:-3]
        lr = float(lr_helper)
        if variables[-2][-1]=='e':
            lr = 1*10**-lr
        try:
            epoch = int(variables[-1].split('_')[-1][:-9])
        except:
            epoch=999
    elif os.path.isdir(PATH):
        folder = PATH
        local_files = os.listdir(folder)
        # Select model with lowest loss
        b=0
        for i in local_files:
            a=i.split('.')[0]
            if a.isnumeric() and int(a)>b:
                b=int(a)
        model = folder+f'/{b}.pt'
        # Extract info. from file / folder name:
        variables = model.replace('/','-').split('-')        
        date = variables[1]
        epoch = int(variables[-1][:-3])
        lr = np.nan
        if variables[-2].isnumeric():
            lr = float(variables[-2])
            if variables[-3][-1]=='e':
                lr = 1*10**-lr
    
    # Use helper functions for network loading and test set predictions
    net_name = net_name_extraction(PATH)
    net = load_net(model,device) # Load in correct network based on model name
    dataset, temp1, class_groups = determine_dataset(dataset,model) # Select class groups and which dataset the model should be evaluated on.
    temp2, data_name, temp3 = determine_dataset('automatic',model) # Extract data_name of data which 'model' was trained on.
    
    # Create empty lists of evaluation metrics:
    auc_, recall_, precision_, f1_, accuracy_, accfr1_, accfr2_ = [], [], [], [], [], [], []
    iterations = 1
    if error: iterations = 10
    for i in range(iterations):
        # Make predictions
        predicted, predicted_prob, labels = prediction(dataset, net, class_groups, device=device, reps=360, dropout=error) # Predict on testset
        # Call Evaluation Metrics
        auc = roc_auc_score(labels, predicted_prob)
        confusion = confusion_matrix(labels, predicted)
        fpr, tpr, thresholds = roc_curve(labels, predicted_prob)
        recall = recall_score(labels,predicted, average=None)
        precision = precision_score(labels, predicted, average=None)
        f1 = f1_score(labels, predicted, average=None)
        
        # Append to lists for RMS calculation
        if error:
            auc_.append(auc)
            recall_.append(recall[1])
            precision_.append(precision[1])
            f1_.append(f1[1])
            accuracy_.append(np.trace(confusion)/confusion.sum())
            accfr1_.append(confusion[0,0]/np.sum(confusion[0]))
            accfr2_.append(confusion[1,1]/np.sum(confusion[1]))
    
    if error:
        # Calculate RMS:
        recall = RMS(recall_)
        precision = RMS(precision_)
        f1 = RMS(f1_)
        auc = RMS(auc_)
        accuracy = RMS(accuracy_)
        fr1_acc = RMS(accfr1_)
        fr2_acc = RMS(accfr2_)
        
        # Return dicts with values
        out = {
            'data_name':data_name,
            'net_name':net_name,
            'date':date,
            'lr':lr,
            'epoch':epoch,
            'auc':auc,
            'confusion_matrix':confusion,
            'recall':recall,
            'precission':precision,
            'f1':f1,
            'fpr':fpr,
            'tpr':tpr,
            'thresholds':thresholds,
            'accuracy':accuracy,
            'FR1_accuracy':fr1_acc,
            'FR2_accuracy':fr2_acc
        }
    elif dictionary:
        out = {
            'data_name':data_name,
            'net_name':net_name,
            'date':date,
            'lr':lr,
            'epoch':epoch,
            'auc':auc,
            'confusion_matrix':confusion,
            'recall':recall,
            'precission':precision,
            'f1':f1,
            'fpr':fpr,
            'tpr':tpr,
            'thresholds':thresholds
        }
    else:
        # Output evaluations as list:
        out = [
            data_name, net_name, date, lr, epoch,#[0:5]
            auc,#[5]
            confusion[0,0], confusion[1,0], confusion[0,1], confusion[1,1],#[6:10]
            recall[0], recall[1],#[10:12]
            precision[0], precision[1],#[12:14]
            f1[0], f1[1],#[14:16]
            fpr, tpr, thresholds#[16:]
            ]
    return out



def out_print(out):
    recall = [out[10],out[11]]
    precision = [out[12],out[13]]
    f1 = [out[14],out[15]]
    confusion = np.asarray([out[6],out[8],out[7],out[9]]).reshape(2,2)
    accuracy = np.trace(confusion)/confusion.sum()*100
    auc = out[5]
    print(f"""Table 3 HM Transfer Learning Equivalent Results:

\t\tFRI \tFRII
Recall \t\t{recall[0]:.3f} \t{recall[1]:.3f}
Precision \t{precision[0]:.3f}\t{precision[1]:.3f}
F1 Score \t{f1[0]:.3f}\t{f1[1]:.3f}

Avg. Accuracy \t{accuracy:.1f}%
AUC \t\t{auc:.3f}
""")
#Accuracies \t{confusion[0,0]/np.sum(confusion[0])*100:.1f}%\t{confusion[1,1]/np.sum(confusion[1])*100:.1f}%


# ==========================================================
### Evaluation Metric Plots ###

# ROC Curve
def plot_roc_curve(fpr, tpr, title=None):
    AUC = auc(fpr,tpr)
    plt.figure(figsize=(8,8))
    plt.plot(fpr,tpr,linewidth='2')
    
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    if title==None:
        plt.title(f'ROC Curve with AUC={AUC:.3f}')
    else:
        plt.title(title)
    plt.grid('large')
    plt.xlim(None,1)
    plt.ylim(0,None)
    plt.show()

# Binary Confusion Matrix
def plot_conf_mat(conf_matrix, normalised=True, n_classes=2, format_input=None, title='Confusion Matrix', publication=False):
    # Following along the lines of (from the github on 29.04.2020)
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    plt.rcParams.update({'font.size': 14})

    classes = ['FRI','FRII']
    xticks_rotation='horizontal'
    matrix = conf_matrix.copy() #Otherwise can change matrix inplace, which is undesirable for potential further processing.
    temp = np.asarray(matrix)
    values_format = '.4g'
    if normalised==True:
        values_format = '.1%'
        for i in range(matrix.shape[0]):
            matrix = matrix.astype('float64') 
            if publication:
                matrix[i] = matrix[i]/matrix[i].sum()
            else:
                matrix[i] = matrix[i]/matrix[i].sum()
            
    if type(format_input) == str:
        values_format = format_input
    
    # Initialise figure
    fig, ax = plt.subplots(figsize=(8,8))
    img = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    fig.colorbar(img,ax=ax)
    cmap_min, cmap_max = img.cmap(0), img.cmap(256)
    
    # print text with appropriate color depending on background
    text = np.empty_like(matrix, dtype=object)
    thresh = (matrix.max() + matrix.min()) / 2.0
    for i, j in product(range(n_classes), range(n_classes)):
        color = cmap_max if matrix[i, j] < thresh else cmap_min
        text[i, j] = ax.text(j, i, format(matrix[i, j], values_format),
                             ha="center", va="center",
                             color=color)
    ax.set(xticks=np.arange(n_classes),
           yticks=np.arange(n_classes),
           xticklabels=classes,
           yticklabels=classes,
           ylabel="True label",
           xlabel="Predicted label")
    ax.set_ylim((n_classes - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=xticks_rotation)
    plt.title(title)
    plt.show()


# ==========================================================
# Attention Map Call
def attentions_func(batch_of_images, 
                    net, 
                    mean=True, 
                    device=torch.device('cpu'),
                    layer_name_base='compatibility_score',
                    layer_no=2
                   ):
    """
    Args: 
        batch_of_images: Images with type==torch.tensor, of dimension (-1,1,150,150)
        net: model being loaded in.
    Calls on: HookBasedFeatureExtractor to call out designed attention maps.
    Output: Upscaled Attention Maps, Attention Maps
    """
    assert device in [torch.device('cpu'), torch.device('cuda')], f"Device needs to be in: [torch.device('cpu'), torch.device('cuda')]"
    assert len(batch_of_images.shape)==4, f'Batch input expected to be of dimensions: BxCxWxH (4 dims)'
    
    if type(batch_of_images) == type(np.array([])):
        batch_of_images = torch.tensor(batch_of_images)
    
    images = batch_of_images
    AMap_originals = []
    for iteration in range(batch_of_images.shape[0]): # Should be able to do this without iterating through the batches.
        for i in range(layer_no):
            feature_extractor = HookBasedFeatureExtractor(net, f'{layer_name_base}{i+1}', upscale=False)
            imap, fmap = feature_extractor.forward(images[iteration:iteration+1].to(device))
            
            if not fmap: #Will pass if fmap is none or empty etc. 
                continue #(ie. Skips iteration if compatibility_score{i} does not compute with the network.)
            
            attention = fmap[1].cpu().numpy().squeeze()
            attention = np.expand_dims(resize(attention, (150, 150), mode='constant', preserve_range=True), axis=2)
            
            if (i == 0):
                attentions_temp = attention
            else:
                attentions_temp = np.append(attentions_temp, attention, axis=2)
            AMap_originals.append(np.expand_dims(fmap[1].cpu().numpy().squeeze(), axis=2))
            
        if iteration == 0: 
            attentions = np.expand_dims(attentions_temp, 3)
        else:
            attentions = np.append(attentions, np.expand_dims(attentions_temp, 3), axis=3)
    
    # Channel dimension is compatibility_score1 / compatibility_score2 respectively (take mean for total attention):
    attentions_out = np.reshape(attentions.transpose(3, 2, 0, 1),(-1, layer_no, 150,150))
    if mean:
        # Take the mean over all attention maps
        attentions_out = np.mean(attentions_out, axis=1)
        
    return attentions_out , AMap_originals

# Attention Full Analysis Plots
def attention_analysis(sources, model, source_only=True, attention_maps=None, GradCAM=None):
    """ Take a source image and plot a comparitive selection of GradCAM and attention maps """
    assert type(sources)==torch.tensor, f'sources must be tensor'
    if source_only:
        plt.imshow(sources.squeeze())
        pass
    else:
        pass
        
    #plt.subplot(241) # Source Image
    #plt.subplot(242) # Attention CompScore1
    #plt.subplot(243) # Attention CompScore2
    #plt.subplot(244) # Attention Mean (Overlay)
    #plt.subplot(245) # GradCAM GB-CAM Conv6
    #plt.subplot(246) # GradCAM GB-CAM CompScore1
    #plt.subplot(247) # GradCAM GB-CAM CompScore2
    #plt.subplot(248) # GradCAM GB-CAM Mean (Overlay)
    return source


# Attention Epoch Plot
def AttentionImagesByEpoch(sources, 
                           folder_name, 
                           net,
                           epoch=1500, 
                           device=torch.device('cpu'),
                           layer_name_base='compatibility_score',
                           layer_no=2
                          ):
    """
    Args:
        sources: list of Images with type==torch.tensor, of dimension (-1,1,150,150)
        folder_name: directory of pickled .pt parameters to load into our network.
    dependancies:
        attentions_func()
        HookedBasedFeatureExtraction() (from within attention_func)
    out:
        attention_maps_temp: list of arrays of all attention maps according to the epoch they were generated.
        epoch_updates: list of epoch numbers for the attention map generations.
    """
    assert device in [torch.device('cpu'), torch.device('cuda')], f"Device needs to be in: [torch.device('cpu'), torch.device('cuda')]"
    assert os.path.exists(folder_name), f"Folder input {folder_name} is not a valid folder path."
    
    attention_maps = []
    original_attention_maps = []
    epoch_updates = []

    # Load in models in improving order based on the folder name
    for epoch_temp in range(epoch):
        PATH = f'{folder_name}/{epoch_temp}.pt'
        if os.path.exists(PATH):
            net.load_state_dict(torch.load(PATH,map_location=torch.device(device)))
            net.eval()
            # Generate attention maps with attentions_func and save appropriately.
            attentions , original_attentions = attentions_func(np.asarray(sources), net, mean=True, device=device,
                                                               layer_name_base=layer_name_base, layer_no=layer_no
                                                              )
            #print('amap shape epoch call: ', attentions.shape)
            for i in range(attentions.shape[0]):
                attention_maps.append(attentions[i]) # Averaged attention maps of the images selected in the cell above.
                original_attention_maps.append(original_attentions[i]) # Averaged but unsampled attention maps.
                epoch_updates.append(epoch_temp) #List of when the validation loss / attention maps were updated.

    return attention_maps, original_attention_maps, epoch_updates

# Plot for attention by epoch (calls AttentionImagesByEpoch)
def attention_epoch_plot(source_images,
                         folder_name,
                         logged=False,
                         width=3,
                         device=torch.device('cpu'),
                         layer_name_base='compatibility_score',
                         layer_no=2,
                         cmap_name='magma'
                        ):
    """
    Function for plotting clean grid of attention maps as they develop throughout the learning stages.
    Args:
        The attention map data, 
        original images of sources
        number of unique sources, 
        if you want your image logged,
        number of output attentions desired (sampled evenly accross available space)
        epoch labels of when the images were extracted
    Out:
        plt of images concatenated in correct fashion
    """    
    # Load in network based on folder_name
    net = load_net(path_to_model(folder_name), device)
    
    # Generate attention maps for each available Epoch
    attention_maps_temp, og_attention_maps, epoch_labels = AttentionImagesByEpoch(source_images, 
                                                                                  'TrainedNetworks/'+folder_name, 
                                                                                  net, 
                                                                                  epoch=1500, 
                                                                                  device=device,
                                                                                  layer_name_base=layer_name_base,
                                                                                  layer_no=layer_no
                                                                                 )
    sample_number = source_images.shape[0]
    no_saved_attentions_epochs = np.asarray(attention_maps_temp).shape[0]//sample_number
    attentions = np.asarray(attention_maps_temp)
    imgs=[]
    labels=[]
    width_array = range(no_saved_attentions_epochs)
    
    if width <= no_saved_attentions_epochs:
        width_array = np.linspace(0, no_saved_attentions_epochs-1, num=width, dtype=np.int32)
    else: 
        width = no_saved_attentions_epochs
    #print(f"""
    #width_array: {len(width_array)}\t{width_array}
    #no_saved_attentions_epochs: {no_saved_attentions_epochs}
    #epoch_labels: {len(epoch_labels)}\t{epoch_labels}
    #""")

    # Prepare the selection of images in the correct order as to be plotted reasonably (and prepare epoch labels)
    for j in range(sample_number):
        if logged:
            imgs.append(np.exp(source_images[j].squeeze()))
        else:
            imgs.append(source_images[j].squeeze())
        for i in width_array:
            #print(sample_number,i,j)
            imgs.append(attention_maps_temp[sample_number*i+j])
            try:
                labels[width-1]
            except:
                labels.append(epoch_labels[sample_number*i])
    
    # Define the plot of the grid of images
    fig = plt.figure(figsize=(100, 100))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(sample_number,width+1),#Sets size of array of images
                     axes_pad=0.02,  # pad between axes in inch.
                     )
    for ax, im in zip(grid, imgs):
        # Iterating over the grid returns the Axes.
        if logged:
            ax.imshow(np.log(im), cmap=cmap_name)
        else:
            ax.imshow(im, cmap=cmap_name)
        ax.axis('off')
    print(f'Source images followed by their respective averaged attention maps at epochs:\n{labels}')
    plt.show()

# ==========================================================
### OTHER FUNCTIONS ###

# Place a cmap mask ontop of an image:
def mask_on_image(img, mask, cmap_name='gnuplot'):
    assert (img.shape[-1]==150) | (img.shape[-1]==3), f'Input image is not correct size. Must either be greyscale [:,:,3], or [150,150]'
    assert (img.shape[-1]==150) | (img.shape[-1]==3), f'Input mask is not correct size. Must either be greyscale [:,:,3], or [150,150]'
    
    # Convert to Numpy if required:
    if type(img) == type(torch.tensor(0)):
        img = img.detach().cpu().numpy()
    if type(mask)== type(torch.tensor(0)):
        mask = mask.detach().cpu().numpy()
    
    # Remove unnecessary dimensions
    img = img.squeeze()
    mask= mask.squeeze()
        
    # Image to greyscale if not already
    if img.shape[-1]!=3:
        img = (img-img.min())/(img.max()-img.min())
        img = np.stack([img, img, img]).transpose(1,2,0)
    # Normalise to [0-1]
    mask = (mask-mask.min())/(mask.max()-mask.min())
    
    cm = matplotlib.cm.get_cmap(cmap_name)
    heatmap = cm(mask.reshape(150,150,-1)[:,:,0])
    heatmap = np.float32(heatmap)[:,:,:3]

    out = heatmap+np.float32(img)
    out = out / np.max(out)
    out_image = Image.fromarray((out[:, :, :3] * 255).astype(np.uint8))
    return out_image

# Extract testdata into (sorted) loopable dataset
def SortedDataSamples(data_name, transformed = True,  rotations = 1, subset = 'NOHYBRID'):
    transformation_number = rotations
    images = []
    labels = []
    data = ['MiraBest','FRDEEP-F']
    assert data_name in data, f'{data_name} not a valid selection, must be either MiraBest or FRDEEP-F.'
    assert subset in [None,'NOHYBRID'], f'Subset selection not implemented yet.'
    for j in range(transformation_number):
        testdata, outdata = data_call(data_name, dataloader=False)
        
        if transformed:
            dataset = testdata
        else:
            dataset = outdata
        labels += dataset.targets
        for i in range(dataset.data.shape[0]):
            images.append(np.asarray(dataset[i][0]).squeeze())

    images = np.asarray(images)

    if (data_name == 'MiraBest') & (subset == 'NOHYBRID'):
        fri = images[(0 <= np.asarray(labels)) & (np.asarray(labels)<5)]
        frii = images[(5 <= np.asarray(labels)) & (np.asarray(labels)<8)]
        hybrid = images[(8 <= np.asarray(labels))]
    elif data_name == 'FRDEEP-F':
        fri = images[np.asarray(labels)==0]
        frii = images[np.asarray(labels)==1]
        hybrid = np.zeros_like(fri[:1])
    
    return fri, frii, hybrid, images, labels


# Extract net_name
def net_name_extraction(PATH):
    "Extract net_name from the local or global path (local prefered - less risk)"
    available_networks = ['playground',
                          'playgroundv1',
                          'playgroundv2_concat',
                          'playgroundv2_mean',
                          'playgroundv2_deep_sup',
                          'playgroundv2_ft',
                          'playgroundv3',
                          'playgroundv4',
                          'AGRadGalNet',
                          'transfer_original',
                          'transfer_adapted',
                          'AGSononet',
                          'AGTransfer']
    print(f'PATH: {PATH}')
    for i in available_networks:
        if i in PATH:
            net_name = i
    return(net_name)

# Hook Based Extractor for Attention Map Calls
class HookBasedFeatureExtractor(nn.Module):
    def __init__(self, submodule, layername, upscale=False):
        """ Extracts 'attention maps' from network submodule at layer layername.
        Args:
            submodule: loaded model
            layername: one of: ['compatibility_score1','compatibility_score2']
            upscale: is completed after call if required
        """
        super(HookBasedFeatureExtractor, self).__init__()

        self.submodule = submodule
        self.submodule.eval()
        self.layername = layername
        self.outputs_size = None
        self.outputs = None
        self.inputs = None
        self.inputs_size = None
        self.upscale = upscale

    def get_input_array(self, m, i, o):
        if isinstance(i, tuple):
            self.inputs = [i[index].data.clone() for index in range(len(i))]
            self.inputs_size = [input.size() for input in self.inputs]
        else:
            self.inputs = i.data.clone()
            self.inputs_size = self.input.size()
        #print('Input Array Size: ', self.inputs_size)

    def get_output_array(self, m, i, o):
        if isinstance(o, tuple):
            self.outputs = [o[index].data.clone() for index in range(len(o))]
            self.outputs_size = [output.size() for output in self.outputs]
        else:
            self.outputs = o.data.clone()
            self.outputs_size = self.outputs.size()
        #print('Output Array Size: ', self.outputs_size)

    def rescale_output_array(self, newsize):
        us = nn.Upsample(size=newsize[2:], mode='bilinear')
        if isinstance(self.outputs, list):
            for index in range(len(self.outputs)): self.outputs[index] = us(self.outputs[index]).data()
        else:
            self.outputs = us(self.outputs).data()

    def forward(self, x):
        target_layer = self.submodule._modules.get(self.layername)

        # Collect the output tensor
        h_inp = target_layer.register_forward_hook(self.get_input_array)
        h_out = target_layer.register_forward_hook(self.get_output_array)
        self.submodule(x)
        h_inp.remove()
        h_out.remove()
        # Rescale the feature-map if it's required
        if self.upscale: self.rescale_output_array(x.size())

        return self.inputs, self.outputs

# ==========================================================
if __name__ == '__main__':
    PATH = input('What TrainedNetwork folder should be evaluated?')
    assert os.path.isdir(f'TrainedNetworks/{PATH}'),f'{PATH} could not be found.'
    
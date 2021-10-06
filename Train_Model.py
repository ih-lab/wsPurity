import os
import time
import random
import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
from barbar import Bar
from PIL import Image
import cv2
import imgaug
from imgaug import augmenters as iaa
from collections import Counter

import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import *

subsetsize = 120 # Number of tiles per subset

def seed_everything(seed=1234):
    imgaug.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Parse arguments from command line --------------------------------------------
parser = argparse.ArgumentParser(description='Training Model')
parser.add_argument('--seedint'  , type = int, default = '', help = 'Seed Set')
parser.add_argument('--path', type=str, default='', help='Path to Dataset')
parser.add_argument('--mod', type=str, default='', help='Type of Pretrained Model')
parser.add_argument('--fold_ACC', type=str, default='', help='Folder of Dataset ACC')
parser.add_argument('--fold_ACC_Norm', type=str, default='', help='Folder of Dataset ACC Norm')
parser.add_argument('--fold_BRCA', type=str, default='', help='Folder of Dataset BRCA')
parser.add_argument('--fold_BRCA_Norm', type=str, default='', help='Folder of Dataset BRCA Norm')
parser.add_argument('--fold_HNSC', type=str, default='', help='Folder of Dataset HNSC')
parser.add_argument('--fold_HNSC_Norm', type=str, default='', help='Folder of Dataset HNSC Norm')
parser.add_argument('--fold_LUAD_LUSC', type=str, default='', help='Folder of Dataset LUAD & LUSC')
parser.add_argument('--fold_LUAD_LUSC_Norm', type=str, default='', help='Folder of Dataset LUAD & LUSC Norm')
parser.add_argument('--fold_PRAD', type=str, default='', help='Folder of Dataset PRAD')
parser.add_argument('--fold_PRAD_Norm', type=str, default='', help='Folder of Dataset PRAD Norm')
parser.add_argument('--fold_OV', type=str, default='', help='Folder of Dataset OV')
parser.add_argument('--fold_OV_Norm', type=str, default='', help='Folder of Dataset OV Norm')
parser.add_argument('--fold_BLCA', type=str, default='', help='Folder of Dataset BLCA')
parser.add_argument('--fold_BLCA_Norm', type=str, default='', help='Folder of Dataset BLCA Norm')
parser.add_argument('--imsize', type=int, default='', help='Size fo Input Images')
parser.add_argument('--Drop', type=str, default='', help='Use Dropout')
parser.add_argument('--batch_size', type=int, default='', help='Batch Size')
parser.add_argument('--Epochs', type=int, default='', help='Number of Epochs')
parser.add_argument('--scheduler', type=str, default='', help='Use Scheduler')
parser.add_argument('--checkpoint', type=str, default='', help='Use Checkpoint Model')
parser.add_argument('--checkfile', type=str, default='', help='Checkpoint File')
parser.add_argument('--numwork', type=int, default='', help='Number of Workers')
parser.add_argument('--savepath', type=str, default='', help='Save Path')
parser.add_argument('--csvfile'  , type = str, default = '', help = 'CSV with Data Split')

args = parser.parse_args()
seed_everything(args.seedint)
# ------------------------------------------------------------------------------

# Store path, folder names, image size, batch size, number of epochs, number ---
# of workers, save path, and csv filename --------------------------------------
path = args.path

fold_ACC = args.fold_ACC
fold_ACC_Norm = args.fold_ACC_Norm
fold_BRCA = args.fold_BRCA
fold_BRCA_Norm = args.fold_BRCA_Norm
fold_HNSC = args.fold_HNSC
fold_HNSC_Norm = args.fold_HNSC_Norm
fold_LUAD_LUSC = args.fold_LUAD_LUSC
fold_LUAD_LUSC_Norm = args.fold_LUAD_LUSC_Norm
fold_PRAD = args.fold_PRAD
fold_PRAD_Norm = args.fold_PRAD_Norm
fold_OV = args.fold_OV
fold_OV_Norm = args.fold_OV_Norm
fold_BLCA = args.fold_BLCA
fold_BLCA_Norm = args.fold_BLCA_Norm

imsize = args.imsize
batchsize = args.batch_size
epochs = args.Epochs
numwork = args.numwork
savepath = args.savepath
csvfile = args.csvfile
# ------------------------------------------------------------------------------

# Record dropout, scheduler and checkpoint requests ----------------------------

# Using dropout
if  args.Drop == 'true':
    Dropout = True
    drop = 'drop'
else:
    Dropout = False
    drop = 'nodrop'

# Using scheduler
if  args.scheduler == 'true':
    Scheduler = True
    sched = 'sched'
else:
    Scheduler = False
    sched = 'nosched'

# Using checkpoint
if args.checkpoint == 'true':
    Checkpoint = True
    checkfile = args.checkfile
else:
    Checkpoint = False
    checkfile = ''

# Find the corresponding directories per slide ---------------------------------
directories = {} # dict to store the directories per slide
direc_label = {} # dict to store the cancer ID per slide

for x in os.listdir(path+fold_ACC+'/'):
    directories[x] = path+fold_ACC+'/'
    direc_label[x] = 0 #torch.Tensor(0)

for x in os.listdir(path+fold_ACC_Norm+'/'):
    directories[x] = path+fold_ACC_Norm+'/'
    direc_label[x] = 0 #torch.Tensor(0)

for x in os.listdir(path+fold_BRCA+'/'):
    directories[x] = path+fold_BRCA+'/'
    direc_label[x] = 1 #torch.Tensor(1)

for x in os.listdir(path+fold_BRCA_Norm+'/'):
    directories[x] = path+fold_BRCA_Norm+'/'
    direc_label[x] = 1 #torch.Tensor(1)

for x in os.listdir(path+fold_HNSC+'/'):
    directories[x] = path+fold_HNSC+'/'
    direc_label[x] = 2 #torch.Tensor(2)

for x in os.listdir(path+fold_HNSC_Norm+'/'):
    directories[x] = path+fold_HNSC_Norm+'/'
    direc_label[x] = 2 #torch.Tensor(2)

for x in os.listdir(path+fold_LUAD_LUSC+'/'):
    directories[x] = path+fold_LUAD_LUSC+'/'
    direc_label[x] = 3 #torch.Tensor(3)

for x in os.listdir(path+fold_LUAD_LUSC_Norm+'/'):
    directories[x] = path+fold_LUAD_LUSC_Norm+'/'
    direc_label[x] = 3 #torch.Tensor(3)

for x in os.listdir(path+fold_PRAD+'/'):
    directories[x] = path+fold_PRAD+'/'
    direc_label[x] = 4 # torch.Tensor(4)

for x in os.listdir(path+fold_PRAD_Norm+'/'):
    directories[x] = path+fold_PRAD_Norm+'/'
    direc_label[x] = 4 #torch.Tensor(4)

for x in os.listdir(path+fold_OV+'/'):
    directories[x] = path+fold_OV+'/'
    direc_label[x] = 5 #torch.Tensor(5)

for x in os.listdir(path+fold_OV_Norm+'/'):
    directories[x] = path+fold_OV_Norm+'/'
    direc_label[x] = 5 #torch.Tensor(5)

for x in os.listdir(path+fold_BLCA+'/'):
    directories[x] = path+fold_BLCA+'/'
    direc_label[x] = 6 #torch.Tensor(6)

for x in os.listdir(path+fold_BLCA_Norm+'/'):
    directories[x] = path+fold_BLCA_Norm+'/'
    direc_label[x] = 6 #torch.Tensor(6)

#print('Directories:', directories)
# ------------------------------------------------------------------------------

# Data distribution ------------------------------------------------------------
cancfreq_dict = Counter(direc_label.values()).most_common() # frequency of cancer IDs
#print('Frequency of cancer IDs:', cancfreq_dict)

cancids = [] # list of cancer IDs
cancfreqs = [] # list of frequencies

for cancid in cancfreq_dict:
    cancids.append(cancid[0])
    cancfreqs.append(cancid[1])
#print('Cancer IDs and frequencies:', cancids, cancfreqs)

cancids = sorted(range(len(cancids)), key = lambda k: cancids[k]) # sort indices by cancer ID
#print('Indices sorted by cancer ID:', cancids)

cancfreqs = np.array(cancfreqs)[cancids] # sort frequencies by cancer ID
#print('Frequencies sorted by cancer ID:', cancfreqs)

cancweight = torch.Tensor((1/cancfreqs)/np.sum(1/cancfreqs))
#print('Weights per cancer ID:', cancweight)
print(cancweight)
distribution = {0:0.02036, 1:0.34718, 2: 0.27337, 3:0.15767, 4:0.09319, 5:0.04357, 6:0.02419, 7:0.02036, 8:0.02011}  #cb4 full
# ------------------------------------------------------------------------------

# Distribute slides into individual patches and subsets ------------------------
class Clusters:

    def __init__(self, tile, cluster, subset):
        self.tile = tile # tile name
        self.cluster = int(cluster) # individual patch ID
        self.subset = int(subset) # subset ID

data_clusters = pd.read_csv(f'{path}TileClusterSubset_{subsetsize}.csv', header=0, index_col=0) # csv with cluster IDs and subset IDs per tile

clusters = {} # dict to store clusters per slide

#print('Number of tiles:', len(data_clusters.index))
for idx, row in tqdm(data_clusters.iterrows()):

    slide = row['Slide']
    tile = row['Tile']
    cluster = row['Cluster']
    subset = row['Subset']

    if slide not in clusters:
        clusters[slide] = []

    clusters[slide].append(Clusters(tile, cluster, subset))
# ------------------------------------------------------------------------------

class PurDataset(object):

    def __init__(self, root, csvfile, mode, aug, trans):

        self.root = root # directory path
        self.csvfile = csvfile # csv filename
        self.mode = mode # dataset

        self.datacomplete = pd.read_csv(f'{root}{csvfile}', header = 0, index_col = 0) # csv with complete data split
        self.datamode = self.datacomplete[self.datacomplete['Dataset'] == mode].copy() # csv with mode data split

        # Remove slides with insufficient number of tiles ----------------------
        idxs = [] # list of indices to remove

        for i in range(0, self.datamode['Slide_SVS_Name'].count()):

            slide = self.datamode.iloc[i, 0]
            idx = self.datamode.index.values[i]

            if slide in clusters:

                tiles = clusters[slide]
                if len(tiles) < subsetsize:
                    idxs.append(idx)

            else:
                idxs.append(idx)

        for i in idxs:
            self.datamode = self.datamode.drop([i])
        # ----------------------------------------------------------------------

        bins = np.array([0.09, 0.29, 0.39, 0.49, 0.59, 0.69, 0.79, 0.89]) # list of bins for predictions
        #print('Bins:', bins)

        subsets = [] # list of slides in the mode dataset
        scores = [] # list of scores in the mode dataset

        for i in range(self.datamode['Slide_SVS_Name'].count()):

            slide = self.datamode.iloc[i, 0]
            score = self.datamode.iloc[i, 2]

            tiles = clusters[slide]

            for j in range(tiles[len(tiles)-1].subset+1): # use all subsets in slide
            #for j in range(1): # use a single subset in slide
                subsets.append(np.asarray([slide, score, j])) # add slide name, score, and subset ID
                scores.append(float(score))

        self.datamode = pd.DataFrame(subsets)
        self.datamode['Bins'] = np.digitize(scores, bins, right = True)

        #print(f'{mode} dataset:', self.datamode)

        self.aug = aug # augmentation
        self.trans = trans # transformation

    def __getitem__(self, idx):

        batch = [] # list of images included in current batch

        slide = self.datamode.iloc[idx, 0]
        subset = int(self.datamode.iloc[idx, 2])
        cancid = torch.tensor(direc_label[slide])
        tiles = clusters[slide]

        # Collect 'subsetsize' number of tiles ---------------------------------
        cnt = 0 # counter for number of tiles collected

        while cnt < subsetsize:

            for tile in tiles: # loop through all tiles in slide

                if tile.subset < subset:
                    continue # skip tiles from other subsets

                if tile.subset > subset:
                    break # end loop if higher subset is reached

                img = cv2.imread(f'{directories[slide]}/{slide}/{tile.tile}')

                try:
                    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                except:
                    continue # skip tile if img assertion failed

                if self.aug is not None:
                    img = np.array([self.aug.augment_image(img)])[0] # augmentation

                img = Image.fromarray(img, mode = 'RGB')

                if self.trans is not None:
                    img = self.trans(img) # transformation

                batch.append(img) # if successful, add image to current batch
                cnt += 1 # if successful, increment tile counter

                if cnt == subsetsize:
                    break # end loop if counter has reached subsetsize
        # ----------------------------------------------------------------------

        batch = torch.stack(batch, dim=0)
        label = self.datamode.iloc[idx, 3] # true bin label
        labels = torch.tensor([1]*label + [0]*(8-label), dtype=torch.float32) # tensor of bin probabilities
        weight = torch.from_numpy(np.asarray(distribution[label])).float() # weight of slide label
        weight.unsqueeze(0)

        return batch, labels, cancid, slide, weight # batch of images, bin probabilties, cancer ID, slide name, slide label weight

    def __len__(self):
        return self.datamode.shape[0] # number of subsets across all slides

# Gated attention weights ------------------------------------------------------
class GatedAttention(nn.Module):

    def __init__(self):

        super(GatedAttention, self).__init__()

        self.L = 512
        self.D = 128
        self.K = 1
        self.otherdim = 'k'

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights1 = nn.Linear(self.D, self.K)
        self.attention_weights2 = nn.Linear(self.D, self.K)

    def forward(self, x):

        A_V = self.attention_V(x) # NxD
        A_U = self.attention_U(x) # NxD
        A1 = self.attention_weights1(A_V * A_U)[...,0] # element wise multiplication # NxK
        A2 = self.attention_weights2(A_V * A_U)[...,0]
        A1 = F.softmax(A1, dim=-1) # softmax over N
        A2 = F.softmax(A2, dim=-1)
        M1 = torch.bmm(torch.transpose(x,1,2), A1.unsqueeze(-1)).squeeze()
        M2 = torch.bmm(torch.transpose(x,1,2), A2.unsqueeze(-1)).squeeze()

        return M1,M2, A1,A2
# ------------------------------------------------------------------------------

# Model mechanism --------------------------------------------------------------
class Net(nn.Module):

    def __init__(self, res, subset_size, num_classes):

        super(Net, self).__init__()

        self.resnet = res
        self.subset_size = subset_size
        self.att = GatedAttention()
        self.num_classes = num_classes
        self.Linear = nn.Sequential(nn.Dropout(0.5),nn.Linear(512,256), nn.ReLU(), nn.Dropout(0.5),nn.Linear(256,1, bias = False))
        self.LinearClass = nn.Sequential(nn.Dropout(0.5),nn.Linear(512,256), nn.ReLU(), nn.Dropout(p=0.5),nn.Linear(256,7))
        self.linear_1_bias = nn.Parameter(torch.zeros(self.num_classes-1).float())

    def forward(self, x):

        x1 = self.resnet(x)
        x3 = self.att(x1.view(-1,self.subset_size,512))
        x4 = self.Linear(x3[0])
        x7 = x4 + self.linear_1_bias
        probas = torch.sigmoid(x7)
        x8 = self.LinearClass(x3[1])

        return x1, x7, x8, probas, x3[2], x3[3]
# ------------------------------------------------------------------------------

# Loss Function ----------------------------------------------------------------
def loss_fn(logits, levels, imp):

    val = (-torch.sum((F.logsigmoid(logits)*levels +
                   (F.logsigmoid(logits) - logits) *
                   (1-levels)) * imp, dim = 1))

    return torch.mean(val)
# ------------------------------------------------------------------------------

# Dataset ----------------------------------------------------------------------
if 'Normalized' in csvfile:
    normalize = transforms.Normalize(mean=[0.765544824, 0.783040825, 0.580101844], std=[0.081656048, 0.073158708, 0.138637243])
    mean = np.asarray([0.765544824, 0.783040825, 0.580101844])
    std = np.asarray([0.081656048, 0.073158708, 0.138637243])

else:
    normalize = transforms.Normalize(mean=[0.697759863, 0.68592817, 0.582152582], std=[0.095469999, 0.102921345, 0.139130713])
    mean = np.asarray([0.697759863, 0.68592817, 0.582152582])
    std = np.asarray([0.095469999, 0.102921345, 0.139130713])

## Data augmentation -----------------------------------------------------------
iaa_state = 123

augmentation = iaa.Sequential([iaa.size.Resize(224), iaa.SomeOf((3,7), [
    iaa.Fliplr(0.5, random_state=iaa_state),
    iaa.CoarseDropout(0.1, size_percent=0.2, random_state=iaa_state),
    iaa.Flipud(0.5, random_state=iaa_state),
    iaa.OneOf([iaa.Affine(rotate=90, random_state=iaa_state),
                iaa.Affine(rotate=180, random_state=iaa_state),
                iaa.Affine(rotate=270, random_state=iaa_state)], random_state=iaa_state),
    iaa.Multiply((0.8, 1.5), random_state=iaa_state),
    iaa.AdditiveGaussianNoise(scale=(0,0.2*255), per_channel=True, random_state=iaa_state),
    iaa.MultiplyHue((0.94,1.04), random_state=iaa_state),
    iaa.AddToBrightness((-20,64), random_state=iaa_state),
    iaa.MultiplySaturation((0.75,1.25), random_state=iaa_state),
    iaa.LinearContrast((0.25,1.75), random_state=iaa_state)])], random_order=True)

augmentation_valtest = iaa.Sequential([iaa.size.Resize(224)])

## Data transformation ---------------------------------------------------------
transform = transforms.ToTensor()

data_transforms = {
    'train': transforms.Compose([transform, normalize]),
    'validation': transforms.Compose([transform, normalize]),
    'test': transforms.Compose([transform, normalize])
}

image_datasets = {
    'train':
        PurDataset(path, csvfile, 'train', augmentation, data_transforms['train']),
    'validation':
        PurDataset(path, csvfile, 'validation', augmentation_valtest, data_transforms['validation']),
    'test':
        PurDataset(path, csvfile, 'test', augmentation_valtest, data_transforms['test'])
}

## Data loaders ----------------------------------------------------------------
dataloaders = {
    'train':
        torch.utils.data.DataLoader(
            image_datasets['train'],
            batch_size = batchsize,
            shuffle = True,
            num_workers = numwork, drop_last = True, pin_memory = True),
    'validation':
        torch.utils.data.DataLoader(
            image_datasets['validation'],
            batch_size = batchsize,
            shuffle = False,
            num_workers = numwork, drop_last = True, pin_memory = True),
    'test':
        torch.utils.data.DataLoader(
            image_datasets['test'],
            batch_size = batchsize,
            shuffle = False,
            num_workers = numwork, drop_last = True, pin_memory = True)
}
# ------------------------------------------------------------------------------

# Run model --------------------------------------------------------------------
def run_model(params):

    LR, Optimizer, Momentum, WD = params['LR'], params['Optimizer'], params['Momentum'], params['WD']

    start_time = time.time()

    # Training set and its hyperparameters -------------------------------------
    trainset = 'Train_edit'

    if not os.path.isdir(f'{savepath}{trainset}-{start_time}/'):
        os.mkdir(f'{savepath}{trainset}-{start_time}/')
        savepath_model = f'{savepath}{trainset}-{start_time}/'
    # --------------------------------------------------------------------------

    # Select GPU and build model -----------------------------------------------
    cuda_pick = 1
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_pick)
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

    temp_model = torch.hub.load('XingangPan/IBN-Net', 'resnet34_ibn_b', pretrained=True)
    temp_model.avgpool = nn.AdaptiveAvgPool2d(output_size = (1, 1))
    temp_model.fc = nn.Identity()

    model = Net(temp_model, subsetsize, num_classes = 9)
    model.to(device)

    #print('CUDA current device:', torch.cuda.current_device())
    # --------------------------------------------------------------------------

    # Initiate optimizer -------------------------------------------------------
    optimizer = optim.SGD(model.parameters(), lr = LR, momentum = Momentum, weight_decay = WD, nesterov = True)
    # --------------------------------------------------------------------------

    start_epoch = 0 # starting epoch number

    # Check for a checkpoint model ---------------------------------------------
    if Checkpoint:
        modelpath = f'{savepath}{checkfile}/model-{checkfile}.trained.pth'
        checkpointmodel = torch.load(modelpath, device)
        model.load_state_dict(checkpointmodel['model_state_dict'])
        optimizer.load_state_dict(checkpointmodel['optimizer_state_dict'])

        for state in optimizer.state.values():
            for key, val in state.items():
                torch.cuda.set_device(device)
                state[key] = val.cuda()

        start_epoch = checkpointmodel['epoch']+1 # synch starting epoch number
        loss = checkpointmodel['loss'] # synch loss from checkpoint model
    # --------------------------------------------------------------------------

    # Initiate scheduler -------------------------------------------------------
    if Scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.5)
    # --------------------------------------------------------------------------

    # Train and validate the model ---------------------------------------------
    runningloss = [] # list of running losses per subset for train
    runningout = [] # list of running outputs per subset for train
    avgloss = [] # list of average losses per epoch for train
    avgloss_score = [] # list of average tumor purity score losses per epoch for train
    avgloss_class = [] # list of average tissue type prediction losses per epoch for train

    runninglossval = [] # list of running losses per subset for validation
    runningoutval = [] # list of running outputs per subset for validation
    avglossval = [] # list of average losses per epoch for validation
    avglossval_score = [] # list of average tumor purity score losses per epoch for validation
    avglossval_class = [] # list of average tissue type prediction losses per epoch for validation

    bestloss = np.inf # store best loss across all epochs

    print('')

    class_loss_fn = torch.nn.CrossEntropyLoss(weight = cancweight.to(device, non_blocking=True))

    for epoch in range(epochs):

        print("Epoch:", start_epoch+epoch) # current epoch

        # Train the model ------------------------------------------------------
        comb_loss = [] # list of losses based on (1*loss_score) + (0.125*loss_class)
        score_loss = [] # list of losses based on loss_score
        class_loss = [] # list of losses based on loss_class

        model = model.train()

        for data in Bar(dataloaders['train']):

            optimizer.zero_grad()

            batch, labels, cancid, slides, weight = (data[0].to(device, non_blocking=True),
                                        data[1].to(device, non_blocking=True),
                                        data[2].to(device, non_blocking=True),
                                        data[3],
                                        data[4].to(device, non_blocking=True))

            inputs = batch.view(-1, 3, 224, 224)

            features, output_score, output_class, prob_score, att_purity, att_tissue = model(inputs)

            loss_score = loss_fn(output_score, labels, torch.unsqueeze(weight, 1))
            loss_class = class_loss_fn(output_class, cancid)
            prob_class = F.softmax(output_class, dim = 1)

            loss = (1*loss_score) + (0.125*loss_class)

            loss.backward()
            optimizer.step()

            for p in range(batch.size(0)):

                temp_slide = slides[p]

                temp_prob_score = prob_score[p].detach().cpu().numpy()
                temp_prob_score = [float(prob) for prob in temp_prob_score]

                temp_labels = labels[p].detach().cpu().numpy()
                temp_labels = [int(label) for label in temp_labels]

                temp_prob_class = prob_class[p].detach().cpu().numpy()
                temp_prob_class = [float(prob) for prob in temp_prob_class]

                temp_cancid = cancid[p].detach().cpu().numpy()
                temp_cancid = int(temp_cancid)

                runningout.append(np.asarray([temp_slide,
                                temp_prob_score,
                                temp_labels,
                                temp_prob_class,
                                temp_cancid,
                                epoch], dtype = object).reshape((1,6)))

            comb_loss.append(loss.detach().cpu().numpy())
            score_loss.append(loss_score.detach().cpu().numpy())
            class_loss.append(loss_class.detach().cpu().numpy())

            runningloss.append(loss.detach().cpu().numpy())

        print('')

        avgloss.append(np.mean(np.asarray(comb_loss)))
        avgloss_score.append(np.mean(np.asarray(score_loss)))
        avgloss_class.append(np.mean(np.asarray(class_loss)))

        print(f'The loss for the training data for epoch {start_epoch+epoch} is: {np.mean(np.asarray(comb_loss))} with score loss: {np.mean(np.asarray(score_loss))} and class loss: {np.mean(np.asarray(class_loss))}')
        # ----------------------------------------------------------------------

        # Validate the model ---------------------------------------------------
        comb_lossval = [] # list of losses based on (1*loss_scoreval) + (0.125*loss_classval)
        score_lossval = [] # list of losses based on loss_scoreval
        class_lossval = [] # list of losses based on loss_classval

        model = model.eval()

        with torch.no_grad():

            for dataval in Bar(dataloaders['validation']):

                batchval, labelsval, cancidval, slidesval, weightval = (dataval[0].to(device, non_blocking=True),
                                            dataval[1].to(device, non_blocking=True),
                                            dataval[2].to(device, non_blocking=True),
                                            dataval[3],
                                            dataval[4].to(device, non_blocking=True))

                inputsval = batchval.view(-1, 3, 224, 224)

                featuresval, output_scoreval, output_classval, prob_scoreval, attsval_purity, attsval_tissue = model(inputsval)

                loss_scoreval = loss_fn(output_scoreval, labelsval, torch.unsqueeze(weightval, 1))
                loss_classval = class_loss_fn(output_classval, cancidval)
                prob_classval = F.softmax(output_classval, dim = 1)

                lossval = (1*loss_scoreval) + (0.125*loss_classval)

                for p in range(batchval.size(0)):

                    temp_slideval = slidesval[p]

                    temp_prob_scoreval = prob_scoreval[p].detach().cpu().numpy()
                    temp_prob_scoreval = [float(prob) for prob in temp_prob_scoreval]

                    temp_labelsval = labelsval[p].detach().cpu().numpy()
                    temp_labelsval = [int(label) for label in temp_labelsval]

                    temp_prob_classval = prob_classval[p].detach().cpu().numpy()
                    temp_prob_classval = [float(prob) for prob in temp_prob_classval]

                    temp_cancidval = cancidval[p].detach().cpu().numpy()
                    temp_cancidval = int(temp_cancidval)

                    runningoutval.append(np.asarray([temp_slideval,
                                    temp_prob_scoreval,
                                    temp_labelsval,
                                    temp_prob_classval,
                                    temp_cancidval,
                                    epoch], dtype = object).reshape((1,6)))

                comb_lossval.append(lossval.detach().cpu().numpy())
                score_lossval.append(loss_scoreval.detach().cpu().numpy())
                class_lossval.append(loss_classval.detach().cpu().numpy())

                runninglossval.append(lossval.detach().cpu().numpy())

        print('')

        avglossval.append(np.mean(np.asarray(comb_lossval)))
        avglossval_score.append(np.mean(np.asarray(score_lossval)))
       	avglossval_class.append(np.mean(np.asarray(class_lossval)))

        print(f'The loss for the validation data for epoch {start_epoch+epoch} is: {np.mean(np.asarray(comb_lossval))} with score loss: {np.mean(np.asarray(score_lossval))} and class loss: {np.mean(np.asarray(class_lossval))}')
        # ----------------------------------------------------------------------

        # Save results into csv files ------------------------------------------
        torch.save({ # record current epoch
            'epoch': start_epoch+epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': np.mean(np.asarray(comb_lossval))
        },  f'{savepath_model}model-{start_epoch+epoch}-{trainset}-{str(start_time)}.trained.pth')

        if np.mean(np.asarray(comb_lossval)) < bestloss: # record best performing epoch

            bestepoch = epoch
            bestloss = np.mean(np.asarray(comb_lossval))

            torch.save({
                'epoch': start_epoch+epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': bestloss
            },  f'{savepath_model}model-{trainset}-{str(start_time)}.trained.pth')

        atl = pd.DataFrame(avgloss) # Average training loss per epoch
        atl.to_csv(f'{savepath_model}model-{trainset}-{start_time}_avgtrainloss.csv')

        astl = pd.DataFrame(avgloss_score) # Average tumor purity score training loss per epoch
        astl.to_csv(f'{savepath_model}model-{trainset}-{start_time}_avgscoretrainloss.csv')

        actl = pd.DataFrame(avgloss_class) # Average tissue type prediction training loss per epoch
        actl.to_csv(f'{savepath_model}model-{trainset}-{start_time}_avgclasstrainloss.csv')

        avl = pd.DataFrame(avglossval) # Average validation loss per epoch
        avl.to_csv(f'{savepath_model}model-{trainset}-{start_time}_avgvalloss.csv')

        asvl = pd.DataFrame(avglossval_score) # Average tumor purity score validation loss per epoch
       	asvl.to_csv(f'{savepath_model}model-{trainset}-{start_time}_avgscorevalloss.csv')

        acvl = pd.DataFrame(avglossval_class) # Average tissue type prediction validation loss per epoch
       	acvl.to_csv(f'{savepath_model}model-{trainset}-{start_time}_avgclassvalloss.csv')

        rtl = pd.DataFrame(runningloss) # Running training loss per subset
        rtl.to_csv(f'{savepath_model}model-{trainset}-{start_time}_runningtrainloss.csv')

        rvl = pd.DataFrame(runninglossval) # Running validation loss per subset
        rvl.to_csv(f'{savepath_model}model-{trainset}-{start_time}_runningvalloss.csv')

        rto = pd.DataFrame(np.concatenate(runningout, axis = 0)) # Running training outputs per subset
        rto.to_csv(f'{savepath_model}model-{trainset}-{start_time}_runningtrainoutput.csv')

        rvo = pd.DataFrame(np.concatenate(runningoutval, axis = 0)) # Running validation outputs per subset
        rvo.to_csv(f'{savepath_model}model-{trainset}-{start_time}_runningvaloutput.csv')
        # ----------------------------------------------------------------------

        # Update scheduler -----------------------------------------------------
        if Scheduler:
            scheduler.step()
            for param_group in optimizer.param_groups:
                print('The current learning rate is:', str(param_group['lr']))
        # ----------------------------------------------------------------------

        print('')
    #---------------------------------------------------------------------------

    print('Completed training!')

    return bestloss, bestepoch

# Initialize hyperparameters and run model -------------------------------------
hyperparam = {
    'LR': 0.005,
    'Optimizer': 'SGD',
    'Momentum': 0.9,
    'WD': 0.0001
}

bestloss, bestepoch = run_model(hyperparam)
print('Best loss and best epoch:', str(bestloss), str(bestepoch))
# ------------------------------------------------------------------------------

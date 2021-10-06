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
parser.add_argument('--fold_PRAD_OV_BRCA', type=str, default='', help='Folder of Dataset PRAD/OV/BRCA')
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

fold_ACC = args.fold_ACC # TCGA
fold_ACC_Norm = args.fold_ACC_Norm # TCGA
fold_BRCA = args.fold_BRCA # TCGA
fold_BRCA_Norm = args.fold_BRCA_Norm # TCGA
fold_HNSC = args.fold_HNSC # TCGA
fold_HNSC_Norm = args.fold_HNSC_Norm # TCGA
fold_LUAD_LUSC = args.fold_LUAD_LUSC # TCGA
fold_LUAD_LUSC_Norm = args.fold_LUAD_LUSC_Norm # TCGA
fold_PRAD = args.fold_PRAD # TCGA
fold_PRAD_Norm = args.fold_PRAD_Norm # TCGA
fold_OV = args.fold_OV # TCGA
fold_OV_Norm = args.fold_OV_Norm # TCGA
fold_BLCA = args.fold_BLCA # TCGA
fold_BLCA_Norm = args.fold_BLCA_Norm # TCGA
fold_PRAD_OV_BRCA = args.fold_PRAD_OV_BRCA # EIPM

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

data_cancs = pd.read_csv('TissueTypes_EIPM.csv', header=None, index_col=False) # csv with cancer types per slide in EIPM dataset
canc_types = {} # dict to store cancer IDs per slide in EIPM dataset

for idx, row in tqdm(data_cancs.iterrows()): # assigning corresponding cancer IDs

    slide = row[0] # slide name
    canc_type = row[1] # cancer type

    if canc_type == 'BRCA':
        canc_types[slide] = 1 # corresponding cancer ID

    if canc_type == 'PRAD':
        canc_types[slide] = 4 # corresponding cancer ID

    if canc_type == 'OV':
        canc_types[slide] = 5 # corresponding cancer ID

for x in os.listdir(path+fold_PRAD_OV_BRCA+'/'):

    if x not in canc_types:
        continue # skip unassigned slides

    directories[x] = path+fold_PRAD_OV_BRCA+'/'
    direc_label[x] = canc_types[x] #torch.Tensor(canc_types[x])

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

distribution = {0:0.02036, 1:0.34718, 2: 0.27337, 3:0.15767, 4:0.09319, 5:0.04357, 6:0.02419, 7:0.02036, 8:0.02011}  #cb4 full
# ------------------------------------------------------------------------------

# Distribute slides into individual patches and subsets ------------------------
class Clusters:

    def __init__(self, tile, cluster, subset):
        self.tile = tile # tile name
        self.cluster = int(cluster) # individual patch ID
        self.subset = int(subset) # subset ID

data_clusters = pd.read_csv(f'{path}TileClusterSubset_{subsetsize}_v1.csv', header=0, index_col=0) # csv with cluster IDs and subset IDs per tile

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
        return self.datamode.shape[0]

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

        #self.attention_weights = nn.Linear(self.D, self.K)
        self.attention_weights1 = nn.Linear(self.D, self.K)
        self.attention_weights2 = nn.Linear(self.D, self.K)

    def forward(self, x):

        A_V = self.attention_V(x) # NxD
        A_U = self.attention_U(x) # NxD
        A1 = self.attention_weights1(A_V * A_U)[...,0] # element wise multiplication

        A2 = self.attention_weights2(A_V * A_U)[...,0]
        A1 = F.softmax(A1, dim=-1) # softmax over N
        A2 = F.softmax(A2, dim=-1)
        M1 = torch.bmm(torch.transpose(x,1,2), A1.unsqueeze(-1)).squeeze()
        M2 = torch.bmm(torch.transpose(x,1,2), A2.unsqueeze(-1)).squeeze()

        return M1,M2, A1,A2

#---------------------------------------------------------------------------------

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

        purity = x3[0]
        puritylist = list(self.Linear.modules())

        for layer in puritylist[1:4]:
            purity = layer(purity) # extracting tumor purity layer

        tissue = x3[1]
        tissuelist = list(self.LinearClass.modules())

        for layer in tissuelist[1:4]:
            tissue = layer(tissue) # extracting tissue type layer

        return x7, x8, probas, x3[2], x3[3], [purity, tissue]
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

augmentation_val = iaa.Sequential([iaa.size.Resize(224)])

## Data transformation ---------------------------------------------------------
transform = transforms.ToTensor()

data_transforms = {
    'train': transforms.Compose([transform, normalize]),
    'validation': transforms.Compose([transform, normalize]),
    'test': transforms.Compose([transform, normalize])
}

image_datasets = {

    'validation':
        PurDataset(path, csvfile, 'validation', augmentation_val, data_transforms['validation']),
    'test':
        PurDataset(path, csvfile, 'test', augmentation_val, data_transforms['test'])
}

## Data loaders ----------------------------------------------------------------
dataloaders = {

    'validation':
        torch.utils.data.DataLoader(
            image_datasets['validation'],
            batch_size = batchsize,
            shuffle = False,
            num_workers = numwork, drop_last = False, pin_memory = True),
    'test':
        torch.utils.data.DataLoader(
            image_datasets['test'],
            batch_size = batchsize,
            shuffle = False,
            num_workers = numwork, drop_last = False, pin_memory = True)
}
# ------------------------------------------------------------------------------

# Run model --------------------------------------------------------------------
def run_model(params):

    LR, Optimizer, Momentum, WD = params['LR'], params['Optimizer'], params['Momentum'], params['WD']

    start_time = time.time()

    # Validation/test set and its hyperparameters ------------------------------
    #valtestset = 'Validation' # use for validation
    valtestset = 'Test' # use for test

    if not os.path.isdir(f'{savepath}{valtestset}-{start_time}/'):
        os.mkdir(f'{savepath}{valtestset}-{start_time}/')
        savepath_model = f'{savepath}{valtestset}-{start_time}/'
    # --------------------------------------------------------------------------

    # Select GPU and build model -----------------------------------------------
    cuda_pick = 7
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
            for k, v in state.items():
                torch.cuda.set_device(device)
                state[k] = v.cuda()

        start_epoch = checkpointmodel['epoch']+1 # synch starting epoch number
        loss = checkpointmodel['loss'] # synch loss from checkpoint model
    # ------------------------------------------------------------------------

    # Initiate scheduler -------------------------------------------------------
    if Scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 20, gamma = 0.5)
    # --------------------------------------------------------------------------

    # Validate/test the model --------------------------------------------------
    runningloss = [] # list of running losses per subset
    runningout = [] # list of running outputs per subset
    avgloss = [] # list of average losses per epoch

    bestloss = np.inf # store best loss across all epochs

    print('')

    class_loss_fn = torch.nn.CrossEntropyLoss(weight = cancweight.to(device, non_blocking=True))

    for epoch in range(epochs):

        print("Epoch:", start_epoch+epoch) # current epoch

        comb_loss = [] # list of losses based on (1*loss_score) + (0.125*loss_class)
        score_loss = [] # list of losses based on loss_score
        class_loss = [] # list of losses based on loss_class

        preds = [] # list of predictions per tile
        layers = [] # list of model layers per subset

        prev = '' # track previous slide name
        subset = -1 # subset counter per slide

        model = model.eval()

        with torch.no_grad():

            #for data in Bar(dataloaders['validation']): # use for validation
            for data in Bar(dataloaders['test']): # use for test

                try:
                    batch, labels, cancid, slides, weight = (data[0].to(device, non_blocking=True),
                                                data[1].to(device, non_blocking=True),
                                                data[2].to(device, non_blocking=True),
                                                data[3],
                                                data[4].to(device, non_blocking=True))

                    inputs = batch.view(-1, 3, 224, 224)

                    output_score, output_class, prob_score, att_purity,att_tissue, [layer_score, layer_class] = model(inputs)

                    loss_score = loss_fn(output_score, labels, torch.unsqueeze(weight, 1))
                    loss_class = class_loss_fn(output_class, cancid)
                    prob_class = F.softmax(output_class, dim = 1)

                    loss = (1*loss_score) + (0.125*loss_class)

                    # Tile names in subset -------------------------------------
                    tilenames = []

                    for slide in slides:

                        if slide != prev: # if a new slide name occurs
                            subset = -1 # reinitialize subset counter

                        subset += 1

                        tiles = clusters[slide]

                        if len(tiles) < subsetsize:
                            continue

                        # Collect 'subsetsize' number of tiles -----------------
                        cnt = 0 # counter for number of tiles collected

                        while cnt < subsetsize:

                            for tile in tiles:  # loop through all tiles in slide

                                if tile.subset < subset:
                                    continue # skip tiles from other subsets

                                if tile.subset > subset:
                                    break # end loop if higher subset is reached

                                img = cv2.imread(f'{directories[slide]}/{slide}/{tile.tile}')

                                try:
                                    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                                except:
                                    continue # skip tile if img assertion failed

                                tilenames.append(tile.tile) # if successful, add tile to tilenames
                                cnt += 1 # if successful, increment tile counter

                                if cnt == subsetsize:
                                    break # end loop if counter has reached subsetsize
                        # ------------------------------------------------------

                        prev = slide # track previous slide name
                    # ----------------------------------------------------------

                    # Tumor purity score labels --------------------------------
                    trues_score = [] # list of score labels

                    for p in labels:
                        for q in range(subsetsize):
                            temp_label = [float(label) for label in p]
                            trues_score.append(temp_label)
                    # ----------------------------------------------------------

                    # Tissue type labels ---------------------------------------
                    trues_class = [] # list of class labels

                    for p in cancid:
                        for q in range(subsetsize):
                            trues_class.append(int(p))
                    # ----------------------------------------------------------

                    # Predictions and weigths ----------------------------------
                    preds_score = [] # list of score predictions per tile

                    for p in prob_score:
                        for q in range(subsetsize):

                            temp_pred_score = p.detach().cpu().numpy()
                            temp_pred_score = [float(pred) for pred in temp_pred_score]

                            preds_score.append(temp_pred_score)

                    preds_class = [] # list of class predictions per tile

                    for p in prob_class:
                        for q in range(subsetsize):

                            temp_pred_class = p.detach().cpu().numpy()
                            temp_pred_class = [float(pred) for pred in temp_pred_class]

                            preds_class.append(temp_pred_class)

                    atts_pur_list = [] # list of attentions per tile
                    atts_tiss_list = [] # list of attentions per tile
                    for p in range(att_purity.size(0)):
                        for q in range(att_purity.size(1)):
                            atts_pur_list.append(float(att_purity[p][q]))
                            atts_tiss_list.append(float(att_tissue[p][q]))
                    # ----------------------------------------------------------

                    # Store model layers per subset ----------------------------
                    for p in range(layer_score.size(0)):

                        temp_slide = slides[p]

                        temp_layer_score = layer_score[p].detach().cpu().numpy()
                        temp_layer_score = [float(feature) for feature in temp_layer_score]

                        temp_layer_class = layer_class[p].detach().cpu().numpy()
                        temp_layer_class = [float(feature) for feature in temp_layer_class]

                        layers.append(np.asarray([temp_slide,
                          temp_layer_score,
                          temp_layer_class]).reshape((1,3)))
                    # ----------------------------------------------------------

                    # Store predictions per tile -------------------------------
                    for p in range(len(tilenames)):

                        rep = False # track repeating tiles

                        for q in range(p):
                            if tilenames[q] == tilenames[p]:
                                rep = True # record repeating tile
                                break

                        if rep == True:
                            continue # skip repeating tile

                        preds.append(np.asarray([tilenames[p],
                            preds_score[p],
                            trues_score[p],
                            preds_class[p],
                            trues_class[p],
                            atts_pur_list[p],
                            atts_tiss_list[p]]).reshape((1,7)))
                    # ----------------------------------------------------------

                    # Store predictions per subset -----------------------------
                    for p in range(batch.size(0)):

                        temp_slide = slides[p]

                        temp_pred_score = prob_score[p].detach().cpu().numpy()
                        temp_pred_score = [float(pred) for pred in temp_pred_score]

                        temp_label = labels[p].detach().cpu().numpy()
                        temp_label = [int(label) for label in temp_label]

                        temp_pred_class = prob_class[p].detach().cpu().numpy()
                        temp_pred_class = [float(pred) for pred in temp_pred_class]

                        temp_cancid = cancid[p].detach().cpu().numpy()
                        temp_cancid = int(temp_cancid)

                        runningout.append(np.asarray([temp_slide,
                            temp_pred_score,
                            temp_label,
                            temp_pred_class,
                            temp_cancid]).reshape((1,5)))
                    # ----------------------------------------------------------

                    comb_loss.append(loss.detach().cpu().numpy())
                    score_loss.append(loss_score.detach().cpu().numpy())
                    class_loss.append(loss_class.detach().cpu().numpy())

                    runningloss.append(loss.detach().cpu().numpy())

                except:
                    print('Unsuccessful batch!')

                    print(slide)
                    print(tile.subset)
                    print(subset)
                    print(cnt)
                    print(tile.tile)

                    #import traceback
                    #traceback.print_exc()
                    #contiue

        print('')

        avgloss.append(np.mean(np.asarray(comb_loss)))

        #print(f'The loss for the validation data for epoch {start_epoch+epoch} is: {np.mean(np.asarray(comb_loss))} with score loss: {np.mean(np.asarray(score_loss))} and class loss: {np.mean(np.asarray(class_loss))}') # use for validation
        print(f'The loss for the test data for epoch {start_epoch+epoch} is: {np.mean(np.asarray(comb_loss))} with score loss: {np.mean(np.asarray(score_loss))} and class loss: {np.mean(np.asarray(class_loss))}') # use for test
        # ----------------------------------------------------------------------

        # Save results into csv files ------------------------------------------
        if np.mean(np.asarray(comb_loss)) < bestloss: # record best performing epoch

            bestepoch = epoch
            bestloss = np.mean(np.asarray(comb_loss))

        avtl = pd.DataFrame(avgloss) # Average validation/test loss per epoch
        avtl.to_csv(f'{savepath_model}model-{valtestset}-{start_time}_avgloss.csv')

        rvtl = pd.DataFrame(runningloss) # Running validation/test loss per subset
        rvtl.to_csv(f'{savepath_model}model-{valtestset}-{start_time}_runningloss.csv')

        rvto = pd.DataFrame(np.concatenate(runningout, axis = 0)) # Running validation/test outputs per subset
        rvto.to_csv(f'{savepath_model}model-{valtestset}-{start_time}_runningoutput.csv')

        dvto = pd.DataFrame(np.concatenate(preds, axis = 0)) # Detailed validation/test outputs per tile
        dvto.to_csv(f'{savepath_model}model-{valtestset}-{start_time}_detailedoutput.csv')

        lvto = pd.DataFrame(np.concatenate(layers, axis = 0)) # Model layers per subset
        lvto.to_csv(f'{savepath_model}model-{valtestset}-{start_time}_layers.csv')
        # ----------------------------------------------------------------------

        # Update scheduler -----------------------------------------------------
        if Scheduler:
            scheduler.step()
            for param_group in optimizer.param_groups:
                print('The current learning rate is:', str(param_group['lr']))
        # ----------------------------------------------------------------------

        print('')
    #---------------------------------------------------------------------------

    #print('Completed validation!') # use for validation
    print('Completed testing!') # use for test

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

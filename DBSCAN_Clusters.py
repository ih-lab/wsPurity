import os
import numpy as np
import pandas as pd
import argparse
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

subsetsize = 120

# Parse arguments from command line --------------------------------------------
parser = argparse.ArgumentParser(description='DBSCAN Clusters')

parser.add_argument('--path', type=str, default='', help='Path to Dataset')
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

args = parser.parse_args()
# ------------------------------------------------------------------------------

# Store path and folder names --------------------------------------------------
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
# ------------------------------------------------------------------------------

cancs = {} # dict to store the tile files in each slide folder

cancs[f'{fold_ACC}/'] = os.listdir(f'{path}{fold_ACC}/')
cancs[f'{fold_ACC_Norm}/'] = os.listdir(f'{path}{fold_ACC_Norm}/')
cancs[f'{fold_BRCA}/'] = os.listdir(f'{path}{fold_BRCA}/')
cancs[f'{fold_BRCA_Norm}/'] = os.listdir(f'{path}{fold_BRCA_Norm}/')
cancs[f'{fold_HNSC}/'] = os.listdir(f'{path}{fold_HNSC}/')
cancs[f'{fold_HNSC_Norm}/'] = os.listdir(f'{path}{fold_HNSC_Norm}/')
cancs[f'{fold_LUAD_LUSC}/'] = os.listdir(f'{path}{fold_LUAD_LUSC}/')
cancs[f'{fold_LUAD_LUSC_Norm}/'] = os.listdir(f'{path}{fold_LUAD_LUSC_Norm}/')
cancs[f'{fold_PRAD}/'] = os.listdir(f'{path}{fold_PRAD}/')
cancs[f'{fold_PRAD_Norm}/'] = os.listdir(f'{path}{fold_PRAD_Norm}/')
cancs[f'{fold_OV}/'] = os.listdir(f'{path}{fold_OV}/')
cancs[f'{fold_OV_Norm}/'] = os.listdir(f'{path}{fold_OV_Norm}/')
cancs[f'{fold_BLCA}/'] = os.listdir(f'{path}{fold_BLCA}/')
cancs[f'{fold_BLCA_Norm}/'] = os.listdir(f'{path}{fold_BLCA_Norm}/')
cancs[f'{fold_PRAD_OV_BRCA}/'] = os.listdir(f'{path}{fold_PRAD_OV_BRCA}/')

class Coordinates:

    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return (self.x, self.y) == (other.x, other.y)

class Clusters:

    def __init__(self, x, y, cluster, subset):
        self.x = int(x)
        self.y = int(y)
        self.cluster = int(cluster)
        self.subset = int(subset)

tiledetails = {} # dict to store the Clusters details per slide
tilenames = {} # dict to store the tile names per slide

for cancer in tqdm(cancs):

    slides = os.listdir(f'{path}{cancer}')

    for slide in slides:

        tiledetails[slide] = [] # list to store Clusters labels in the slide
        tilenames[slide] = {} # dict to store tile name per coordinate in the slide

        tiles = os.listdir(f'{path}{cancer}{slide}/')

        if len(tiles) < subsetsize:
            continue # skip slides with insufficient number of tiles

        binimg = [] # list with binary representation of the slide

        for tile in tiles:

            x = int(tile.split('_')[1]) # get the x coordinate
            y = int(tile.split('_')[2][:-4]) # get the y coordinate

            tilenames[slide][Coordinates(x, y)] = tile
            binimg.append([x, y])

        binimg = np.array(binimg)

        X = StandardScaler().fit_transform(binimg)
        db = DBSCAN(eps=0.3, min_samples=5).fit(X)

        labels = db.labels_

        for i in range(len(binimg)):

            if labels[i] == -1:
                continue # exclude noisy tiles

            tiledetails[slide].append(Clusters(binimg[i][0], binimg[i][1], labels[i], -1)) # x, y, cluster, subset

        tiledetails[slide] = sorted(tiledetails[slide], key = lambda k: [k.cluster, k.x, k.y]) # sort by cluster, x, and y

clusters = {} # dict with clusters details per slide

for cancer in cancs:

    print(cancer)

    slides = os.listdir(f'{path}{cancer}')

    for slide in tqdm(slides):

        clusters[slide] = [] # list with cluster details per tile in the slide

        subset = 0 # subset number
        cnt = 0 # tiles counter

        for i in range(len(tiledetails[slide])):

            if i > 0 and tiledetails[slide][i-1].cluster < tiledetails[slide][i].cluster: # new cluster

                temp = cnt

                while cnt < subsetsize: # perform padding until subsetsize is reached

                    for j in range(i-temp, i):

                        if cnt == subsetsize:
                            break # end loop if counter has reached subsetsize

                        clusters[slide].append(Clusters(tiledetails[slide][j].x,
                            tiledetails[slide][j].y,
                            tiledetails[slide][j].cluster,
                            subset))

                        cnt += 1

            if cnt == subsetsize: # subsetsize is reached
                subset += 1 # increment subset number
                cnt = 0 # restart tiles counter

            clusters[slide].append(Clusters(tiledetails[slide][i].x,
                tiledetails[slide][i].y,
                tiledetails[slide][i].cluster,
                subset))

            cnt += 1

        if len(tiledetails[slide]) == 0:
            continue

        temp = cnt

        while cnt < subsetsize: # perform padding until subsetsize is reached

            for j in range(i+1-temp, i+1):

                if cnt == subsetsize:
                    break # end loop if counter has reached subsetsize

                clusters[slide].append(Clusters(tiledetails[slide][j].x,
                    tiledetails[slide][j].y,
                    tiledetails[slide][j].cluster,
                    subset))

                cnt += 1

data_clusters = [] # store clusters details in a csv

for slide in tqdm(clusters):
    for tile in clusters[slide]:
        data_clusters.append(np.asarray([slide,
            tilenames[slide][Coordinates(tile.x, tile.y)],
            tile.cluster,
            tile.subset]).reshape((1,4)))

df = pd.DataFrame(np.concatenate(data_clusters, axis = 0), columns = ['Slide', 'Tile', 'Cluster', 'Subset'])
df.to_csv(f'TileClusterSubset_{subsetsize}.csv')

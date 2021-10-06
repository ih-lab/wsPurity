import os
import numpy as np
import pandas as pd

cancs = {'ACC': 0, 'BRCA': 1, 'HNSC': 2, 'LUAD_LUSC': 3, 'PRAD': 4, 'OV': 5, 'BLCA': 6} # cancer type IDs

data_tissues = pd.read_csv('Data/TissueTypes.csv', header = 0, index_col = 0) # csv with true cancer IDs per slide

slides = {} # dict to store true cancer IDs per slide

for idx, row in data_tissues.iterrows():

    slide = row[0]
    tissue = row[1]

    slides[slide] = tissue

data_splits = pd.read_csv('Data/DataSplits.csv', header = 0, index_col = 0) # csv with data split

distribution = {} # dict to track data distribution per dataset

for idx, row in data_splits.iterrows():

    slide = row[0]
    score = float(row[2])
    dataset = row[3]

    tissue = slides[slide]
    canctype = slide.split('-')[3] # tissue type ID

    if tissue == 'BLCA':
        continue # skip BLCA slides

    if dataset not in distribution:
        distribution[dataset] = {'ACC': {}, 'BRCA': {}, 'HNSC': {}, 'LUAD_LUSC': {}, 'OV': {}, 'PRAD': {}} # dict to track data distribution per cancer type per dataset

    if len(distribution[dataset][tissue]) == 0:
        distribution[dataset][tissue]['Cancer'] = 0 # track data distribution of cancer slides per cancer type per dataset
        distribution[dataset][tissue]['Normal'] = 0 # track data distribution of normal slides per cancer type per dataset

    if canctype != '11A':
        distribution[dataset][tissue]['Cancer'] += 1
    else:
        distribution[dataset][tissue]['Normal'] += 1

print('Train:', distribution['train'])
print('Validate:', distribution['validation'])
print('Test:', distribution['test'])

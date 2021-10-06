import os
import numpy as np
import pandas as pd

#dataset = 'Test' # use for test predictions
dataset = 'Val' # use for validation predictions

cancs = {'ACC': 0, 'BRCA': 1, 'HNSC': 2, 'LUAD_LUSC': 3, 'PRAD': 4, 'OV': 5, 'BLCA': 6} # cancer type IDs

data = pd.read_csv(f'Data/{dataset}TilesPred.csv', header = 0, index_col = 0) # csv with predictions per tile

preds = {} # dict to store number of votes per cancer ID per slide based on predictions
trues = {} # dict to store true cancer IDs per slide

for idx, row in data.iterrows():

    slide = row[0].split('_')[0]

    pred = row[3]
    pred = pred[pred.find('[')+1:pred.find(']')]
    pred = [float(p) for p in pred.split(', ')]
    pred = pred.index(max(pred)) # get the predicted cancer ID

    true = int(row[4])

    if slide not in preds:
        preds[slide] = {}
        trues[slide] = true

    if pred not in preds[slide]:
        preds[slide][pred] = 0

    preds[slide][pred] += 1 # increment number of votes for predicted cancer ID for current slide

for cancer in cancs:

    if cancer == 'BLCA':
        continue # skip BLCA cancer type

    TP = 0 # true positives
    TN = 0 # true negatives
    FP = 0 # false positives
    FN = 0 # false negatives

    for slide in preds:

        maxvotes = max(preds[slide], key=preds[slide].get) # get cancer ID with maximum number of votes

        if trues[slide] == cancs[cancer] and maxvotes == cancs[cancer]:
            TP += 1
        elif trues[slide] != cancs[cancer] and maxvotes != cancs[cancer]:
            TN += 1
        elif trues[slide] != cancs[cancer] and maxvotes == cancs[cancer]:
            FP += 1
        elif trues[slide] == cancs[cancer] and maxvotes != cancs[cancer]:
            FN += 1

    print('Tissue type:', cancer)
    print(f'Accuracy: {round((TP+TN)/(TP+TN+FP+FN), 2)}')
    print(f'Precision: {round((TP)/(TP+FP), 2)}')
    print(f'Recall: {round((TP)/(TP+FN), 2)}')

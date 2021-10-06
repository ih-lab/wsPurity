import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics

cancs = {'ACC': 0, 'BRCA': 1, 'HNSC': 2, 'LUAD_LUSC': 3, 'PRAD': 4, 'OV': 5, 'BLCA': 6} # cancer type IDs

data = pd.read_csv(f'Data/TestTilesPred.csv', header = 0, index_col = 0) # csv with predictions per tile

for cancer in cancs: # create ROC curves per cancer type

    if cancer == 'BLCA':
        continue # skip BLCA cancer type

    if not os.path.isdir(f'Results/ROC/ROC_{cancer}'):
        os.makedirs(f'Results/ROC/ROC_{cancer}')

    path = f'Results/ROC/ROC_{cancer}/'

    preds = {} # dict to store list of predictions per slide
    trues = {} # dict to store list of true scores per slide
    cnt = {} # dict to store number of tiles per slide

    for idx, row in data.iterrows():

        if row[4] != cancs[cancer]:
            continue

        slide = row[0].split('_')[0]

        pred = row[1]
        pred = pred[pred.find('[')+1:pred.find(']')]
        pred = [float(p) for p in pred.split(', ')]

        true = row[2]
        true = true[true.find('[')+1:true.find(']')]
        true = [int(t) for t in true.split(', ')]

        if slide not in preds:
            preds[slide] = pred
            trues[slide] = true
            cnt[slide] = 1
        else:
            for i in range(len(preds[slide])):
                preds[slide][i] += pred[i] # add onto the list of predictions
                trues[slide][i] += true[i] # add onto the list of true scores
            cnt[slide] += 1 # increment number of slides

    for slide in preds:
        for i in range(len(preds[slide])):
            preds[slide][i] /= cnt[slide] # calculate an average list of predictions per slide
            trues[slide][i] /= cnt[slide] # calculate an average list of true scores per slide

    preds = np.asarray([preds[p] for p in preds])
    trues = np.asarray([trues[t] for t in trues])

    #for i in range(preds.shape[1])
    for i in [0, 4, 5]: # selected threshold corresponding to scores 0.09, 0.69, 0.79

        predcol = preds[:,i]
        truecol = trues[:,i]

        fpr, tpr, threshold = metrics.roc_curve(truecol, predcol)

        roc_auc = metrics.auc(fpr, tpr)

        plt.plot(fpr, tpr)
        plt.title(f'AUC - {round(roc_auc, 4)}')
        plt.savefig(f'{path}RocBin{i}.tif', dpi = 300)
        plt.close()

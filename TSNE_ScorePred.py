import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from MulticoreTSNE import MulticoreTSNE as TSNE

perp = 80 # perplexity for tsne
iter = 2000 # number of iterations for tsne

cancs = {'ACC': 0, 'BRCA': 1, 'HNSC': 2, 'LUAD_LUSC': 3, 'PRAD': 4, 'OV': 5, 'BLCA': 6} # cancer type IDs

model = 'Test-1632167988.6723921' # model to be used
data_layers = pd.read_csv(f'{model}/model-{model}_layers.csv', header = 0, index_col = 0) # csv with layers per subset from model

if not os.path.isdir('Results/TSNE/'):
    os.makedirs('Results/TSNE/')

features = [] # list of features for tsne
groups = [] # list of labels for tsne

for idx, row in data_layers.iterrows():

    slide = row[0]

    layer = row[1]
    layer = layer[layer.find('[')+1:layer.find(']')].split(', ')
    layer = [float(feature) for feature in layer]

    cancer = int(row[3])

    if cancer == cancs['BLCA']:
        continue # skip BLCA slides

    features.append(layer)

features = np.vstack(features)

data_preds = pd.read_csv(f'{model}/model-{model}_runningoutput.csv', header = 0, index_col = 0) # csv with predictions per subset from model

for idx, row in data_preds.iterrows():

    slide = row[0]

    pred = row[1]
    pred = pred[pred.find('[')+1:pred.find(']')]
    pred = [float(p) for p in pred.split(', ')]
    pred = len([p for p in pred if p > 0.5]) # use a 0.5 cutoff to get prediction bin

    true = row[2]
    true = true[true.find('[')+1:true.find(']')]
    true = [int(t) for t in true.split(', ')]
    true = true.count(1) # count number of 1's to get true score bin

    cancer = int(row[4])

    if cancer == cancs['BLCA']:
        continue # skip BLCA slides

    #groups.append(pred) # use for tsne based on predicted scores
    groups.append(true) # use for tsne based on true scores

# Create the tsne plot
tsne = TSNE(n_components = 2, random_state = 0, perplexity = perp, n_iter = iter, verbose = 1, n_jobs = 4)

tsne_obj = tsne.fit_transform(features)
#tsne_df = pd.DataFrame({'X': tsne_obj[:,0], 'Y': tsne_obj[:,1], 'Predicted Score': groups}) # use for tsne based on predicted scores
tsne_df = pd.DataFrame({'X': tsne_obj[:,0], 'Y': tsne_obj[:,1], 'True Score': groups}) # use for tsne based on true scores

#tsne_df.to_csv('Results/TSNE/TsnePredScore.csv') # use for tsne based on predicted scores
tsne_df.to_csv('Results/TSNE/TsneTrueScore.csv') # use for tsne based on true scores
print('Tsne dataframe saved')

#sns.scatterplot(x = 'X', y = 'Y', hue = 'Predicted Score', legend = 'full', data = tsne_df, palette = 'Paired', s = 8) # use for tsne based on predicted scores
sns.scatterplot(x = 'X', y = 'Y', hue = 'True Score', legend = 'full', data = tsne_df, palette = 'Paired', s = 8) # use for tsne based on true scores
plt.legend(bbox_to_anchor = (1, 1), loc = 'upper left', ncol = 1)

#plt.savefig('Results/TSNE/TsnePredScore.tif', bbox_inches='tight') # use for tsne based on predicted scores
plt.savefig('Results/TSNE/TsneTrueScore.tif', bbox_inches='tight') # use for tsne based on true scores
print('Tsne plot created')

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

temp_features = [] # temporary list of features for tsne

for idx, row in data_layers.iterrows():

    slide = row[0]

    layer = row[1]
    layer = layer[layer.find('[')+1:layer.find(']')].split(', ')
    layer = [float(feature) for feature in layer]

    cancer = int(row[3])

    if cancer == cancs['BLCA']:
        continue # skip BLCA slides

    temp_features.append(layer)

data_preds = pd.read_csv(f'{model}/model-{model}_runningoutput.csv', header = 0, index_col = 0) # csv with predictions per subset from model

preds = [] # list with predicted cancer types
trues = [] # list with true cancer types

for idx, row in data_preds.iterrows():

    slide = row[0]

    pred = row[3]
    pred = pred[pred.find('[')+1:pred.find(']')]
    pred = [float(p) for p in pred.split(', ')]
    pred = pred.index(max(pred)) # get the predicted cancer type ID

    true = int(row[4])

    if true == cancs['BLCA']:
        continue # skip BLCA slides

    preds.append(pred)
    trues.append(true)

features = [] # list of featurs for tsne
groups = [] # list of labels for tsne

for i in range(len(temp_features)):

    if preds[i] == cancs['BLCA']:
        continue # skip BLCA predictions

    features.append(temp_features[i])
    groups.append(preds[i]) # use for tsne based on predicted types
    #groups.append(trues[i]) # use for tsne based on true types

features = np.vstack(features)

# Create the tsne plot
tsne = TSNE(n_components = 2, random_state = 0, perplexity = perp, n_iter = iter, verbose = 1, n_jobs = 4)

tsne_obj = tsne.fit_transform(features)
tsne_df = pd.DataFrame({'X': tsne_obj[:,0], 'Y': tsne_obj[:,1], 'Predicted Tissue': groups}) # use for tsne based on predicted types
#tsne_df = pd.DataFrame({'X': tsne_obj[:,0], 'Y': tsne_obj[:,1], 'True Tissue': groups}) # use for tsne based on true types

tsne_df.to_csv('Results/TSNE/TsnePredTissue.csv') # use for tsne based on predicted types
#tsne_df.to_csv('Results/TSNE/TsneTrueTissue.csv') # use for tsne based on true types
print('Tsne dataframe saved')

sns.scatterplot(x = 'X', y = 'Y', hue = 'Predicted Tissue', legend = 'full', data = tsne_df, palette = 'Paired', s = 8) # use for tsne based on predicted types
#sns.scatterplot(x = 'X', y = 'Y', hue = 'True Tissue', legend = 'full', data = tsne_df, palette = 'Paired', s = 8) # use for tsne based on true types

L = plt.legend(bbox_to_anchor = (1, 1), loc = 'upper left', ncol = 1)

cancs_list = ['ACC', 'BRCA', 'HNSC', 'LUAD_LUSC', 'PRAD', 'OV'] # list with cancer types sorted by cancer ID

for i in range(len(cancs_list)):
    L.get_texts()[i+1].set_text(cancs_list[i])

plt.savefig('Results/TSNE/TsnePredTissue.tif', bbox_inches='tight') # use for tsne based on predicted types
#plt.savefig('Results/TSNE/TsneTrueTissue.tif', bbox_inches='tight') # use for tsne based on true types
print('Tsne plot created')

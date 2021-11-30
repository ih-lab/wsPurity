# wsPurity

Weakly-Supervised Tumor Purity Prediction From Frozen H&E Stained Slides

## Data preparation:

1) Frozen H&E stained slides are collected from the TCGA database: TCGA-ACC, TCGA-BRCA, TCGA-HNSC, etc., and stored into folders by tissue type (e.g. ACC_Cases/, BRCA_Cases/, etc.). Normal tissue slides are collected in the same manner and stored into folders (e.g. ACC_Norm_Cases/, BRCA_Norm_Cases/, etc.).



2) Slides are labeled based on pathologist derived consensus purity scores and split into 70% | 15% | 15% for train | validation | test sets (Data/DataSplits.csv). A metadata file consisting of slide | tissue type columns is created (Data/TissueTypes.csv).

3) The slides are tiled into 512 x 512px patches at 20x resolution and stored into subfolders (e.g. ACC_Cases/Slide_1/, ACC_Cases/Slide_2/, etc.). Severely out of focus tiles are removed based on an empirically derived threshold for blur detection (createTiles.py, haar.py).

4) The individual tissue slices (or clusters) in each slide are identified using DBSCAN, and the tiles in each cluster are split into subsets of *120* tiles each using the following command (DBSCAN_Clusters.py):

```
python DBSCAN_Clusters.py --path '' --fold_ACC 'ACC_Cases' --fold_ACC_Norm 'ACC_Norm_Cases' --fold_BRCA 'BRCA_Cases' --fold_BRCA_Norm 'BRCA_Norm_Cases' --fold_HNSC 'HNSC_Cases' --fold_HNSC_Norm 'HNSC_Norm_Cases' --fold_LUAD_LUSC 'LUAD_LUSC_Cases' --fold_LUAD_LUSC_Norm 'LUAD_LUSC_Norm_Cases_' --fold_PRAD 'PRAD_Cases' --fold_PRAD_Norm 'PRAD_Norm_Cases' --fold_OV 'OV_Cases' --fold_OV_Norm 'OV_Norm_Cases' --fold_BLCA 'BLCA_Cases' --fold_BLCA_Norm 'BLCA_Norm_Cases'
```

### Arguments description:

```
--fold_ACC = 'Folder of Dataset ACC'
--fold_ACC_Norm = 'Folder of Dataset ACC Norm'
--fold_BRCA = 'Folder of Dataset BRCA'
--fold_BRCA_Norm = 'Folder of Dataset BRCA Norm'
--fold_HNSC = 'Folder of Dataset HNSC'
--fold_HNSC_Norm = 'Folder of Dataset HNSC Norm'
--fold_LUAD_LUSC = 'Folder of Dataset LUAD & LUSC'
--fold_LUAD_LUSC_Norm = 'Folder of Dataset LUAD & LUSC Norm'
--fold_PRAD = 'Folder of Dataset PRAD'
--fold_PRAD_Norm = 'Folder of Dataset PRAD Norm'
--fold_OV = 'Folder of Dataset OV'
--fold_OV_Norm = 'Folder of Dataset OV Norm'
--fold_BLCA = 'Folder of Dataset BLCA'
--fold_BLCA_Norm = 'Folder of Dataset BLCA Norm'
```

*Note: The subset size can be changed in the code.*

## Training the model:

5) The model is trained using the following command (Train_Model.py):

```
python Train_Model.py --seedint 1234 --path '' --fold_ACC 'ACC_Cases' --fold_ACC_Norm 'ACC_Norm_Cases' --fold_BRCA 'BRCA_Cases' --fold_BRCA_Norm 'BRCA_Norm_Cases' --fold_HNSC 'HNSC_Cases' --fold_HNSC_Norm 'HNSC_Norm_Cases' --fold_LUAD_LUSC 'LUAD_LUSC_Cases' --fold_LUAD_LUSC_Norm 'LUAD_LUSC_Norm_Cases_' --fold_PRAD 'PRAD_Cases' --fold_PRAD_Norm 'PRAD_Norm_Cases' --fold_OV 'OV_Cases' --fold_OV_Norm 'OV_Norm_Cases' --fold_BLCA 'BLCA_Cases' --fold_BLCA_Norm 'BLCA_Norm_Cases' --imsize 512 --Drop 'false' --batch_size 2 --Epochs 20 --scheduler 'true' --checkpoint 'false' --checkfile '' --numwork 0 --savepath '' --csvfile 'Data/DataSplits.csv'
```

### Arguments description:

```
--seedint = 'Seed Set'
--path = 'Path to Dataset'
--imsize = 'Size fo Input Images'
--Drop = 'Use Dropout'
--batch_size = 'Batch Size'
--Epochs = 'Number of Epochs'
--scheduler = 'Use Scheduler'
--checkpoint = 'Use Checkpoint Model'
--checkfile = 'Checkpoint File'
--numwork = 'Number of Workers'
--savepath = 'Save Path'
--csvfile = 'CSV with Data Split'
```

*Note: The subset size can be changed in the code.*

## Evaluating the model:

6) The model is validated/tested using the following command (Validate_Test_Model.py):

```
python Validate_Test_Model.py --seedint 1234 --path '' --fold_ACC 'ACC_Cases' --fold_ACC_Norm 'ACC_Norm_Cases' --fold_BRCA 'BRCA_Cases' --fold_BRCA_Norm 'BRCA_Norm_Cases' --fold_HNSC 'HNSC_Cases' --fold_HNSC_Norm 'HNSC_Norm_Cases' --fold_LUAD_LUSC 'LUAD_LUSC_Cases' --fold_LUAD_LUSC_Norm 'LUAD_LUSC_Norm_Cases_' --fold_PRAD 'PRAD_Cases' --fold_PRAD_Norm 'PRAD_Norm_Cases' --fold_OV 'OV_Cases' --fold_OV_Norm 'OV_Norm_Cases' --fold_BLCA 'BLCA_Cases' --fold_BLCA_Norm 'BLCA_Norm_Cases' --imsize 512 --Drop 'false' --batch_size 2 --Epochs 1 --scheduler 'true' --checkpoint 'true' --checkfile 'Train-1619581630.8990996' --numwork 0 --savepath '' --csvfile 'Data/DataSplits.csv'
```

*Note: The subset size can be changed in the code. To switch between validating and testing, follow the instructions in the code comments (Validate_Test_Model.py).*

## Data analysis:

7) Data distribution analysis is conducted (Distribution.py).

8) ROC curve plots are created based on tumor purity score predictions (ROC_Curves.py).

*Note: To change the model to be used for the predictions, edit `model` in the code (ROC_Curves.py).*

9) Performance report is created based on tissue type predictions (Report_TissuePred.py).

*Note: To change the model to be used for the predictions, edit `model` in the code (Report_TissuePred.py).*

10) TSNE plots are created based on tumor purity score predictions (TSNE_ScorePred.py).

*Note: To change the model to be used for the predictions, edit `model` in the code (TSNE_ScorePred.py).*

11) TSNE plots are created based on tissue type predictions (TSNE_TissuePred.py).

*Note: To change the model to be used for the predictions, edit `model` in the code (TSNE_TissuePred.py).*

12) The tiles are stitched back into slides, and heatmaps and threshold images are created based on tile-level predictions using the following command (Stitch_Tiles.py):

```
python Stitch_Tiles.py --path '' --fold_ACC 'ACC_Cases' --fold_ACC_Norm 'ACC_Norm_Cases' --fold_BRCA 'BRCA_Cases' --fold_BRCA_Norm 'BRCA_Norm_Cases' --fold_HNSC 'HNSC_Cases' --fold_HNSC_Norm 'HNSC_Norm_Cases' --fold_LUAD_LUSC 'LUAD_LUSC_Cases' --fold_LUAD_LUSC_Norm 'LUAD_LUSC_Norm_Cases_' --fold_PRAD 'PRAD_Cases' --fold_PRAD_Norm 'PRAD_Norm_Cases' --fold_OV 'OV_Cases' --fold_OV_Norm 'OV_Norm_Cases' --fold_BLCA 'BLCA_Cases' --fold_BLCA_Norm 'BLCA_Norm_Cases'
```

*Note: To change the model to be used for the predictions, edit `model` in the code (Stitch_Tiles.py).*

13) The model's performance is compared to the performance of a model from previous literature (Comparison.py)

*Note: To change the model to be used for the predictions, edit `model` in the code (Comparison.py).*

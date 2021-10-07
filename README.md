# wsPurity

Weakly-Supervised Tumor Purity Prediction From Frozen H&E Stained Slides

1) Collect frozen H&E stained slides from the TCGA database: TCGA-ACC, TCGA-BRCA, TCGA-HNSC, etc., and store slides into folder by tissue type (e.g. ACC_Cases/, BRCA_Cases/, HNSC_Cases/, etc.). The slides are currently unavailable for download from the GDC data portal.

2) Label slides based on pathologist derived consensus purity scores and split slides into 70% | 15% | 15% for train | validation | test sets (Data/DataSplits.csv). Create a metadata file consisting of slide | tissue type columns (Data/TissueTypes.csv).

3) Tile slides into 512 x 512 px tiles at 20x resolution and store tiles into subfolders (e.g. ACC_Cases/Slide_1/, ACC_Cases/Slide_2/, etc.).

4) Identify induvidual tissue slices in each slide using DBSCAN and split the tiles in each cluster into subsets (DBSCAN_Clusters.py).

5) Train model (Train_Model.py):

```bash
python Train_Model.py --seedint 1234 --path '' --fold_ACC 'ACC_Cases' --fold_ACC_Norm 'ACC_Norm_Cases' --fold_BRCA 'BRCA_Cases' --fold_BRCA_Norm 'BRCA_Norm_Cases' --fold_HNSC 'HNSC_Cases' --fold_HNSC_Norm 'HNSC_Norm_Cases' --fold_LUAD_LUSC 'LUAD_LUSC_Cases' --fold_LUAD_LUSC_Norm 'LUAD_LUSC_Norm_Cases_' --fold_PRAD 'PRAD_Cases' --fold_PRAD_Norm 'PRAD_Norm_Cases' --fold_OV 'OV_Cases' --fold_OV_Norm 'OV_Norm_Cases' --fold_BLCA 'BLCA_Cases' --fold_BLCA_Norm 'BLCA_Norm_Cases' --imsize 512 --Drop 'false' --batch_size 2 --Epochs 20 --scheduler 'true' --checkpoint 'false' --checkfile '' --numwork 0 --savepath '' --csvfile 'DataSplits.csv'
```

Legend:
```bash
--seedint = 'Seed Set'
--path = 'Path to Dataset'
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

6) Validate/test model (Validate_Test_Model.py):

```bash
python Validate_Test_Model.py --seedint 1234 --path '' --fold_ACC 'ACC_Cases' --fold_ACC_Norm 'ACC_Norm_Cases' --fold_BRCA 'BRCA_Cases' --fold_BRCA_Norm 'BRCA_Norm_Cases' --fold_HNSC 'HNSC_Cases' --fold_HNSC_Norm 'HNSC_Norm_Cases' --fold_LUAD_LUSC 'LUAD_LUSC_Cases' --fold_LUAD_LUSC_Norm 'LUAD_LUSC_Norm_Cases_' --fold_PRAD 'PRAD_Cases' --fold_PRAD_Norm 'PRAD_Norm_Cases' --fold_OV 'OV_Cases' --fold_OV_Norm 'OV_Norm_Cases' --fold_BLCA 'BLCA_Cases' --fold_BLCA_Norm 'BLCA_Norm_Cases' --imsize 512 --Drop 'false' --batch_size 2 --Epochs 1 --scheduler 'true' --checkpoint 'true' --checkfile 'Train-1619581630.8990996' --numwork 0 --savepath '' --csvfile 'DataSplits.csv'
```

Data analysis:
- Performance comparison (Comparison.py)
- Data distribution plots (Distribution.py)
- ROC curves (ROC_Curves.py)
- Tissue type prediction performance report (Report_TissuePred.py)
- Heatmaps and threshold images (Stitch_Tiles.py)
- TSNE plots based on score predictions (TSNE_ScorePred.py)
- TSNE plots based on tissue type predictions (TSNE_TissuePred.py)

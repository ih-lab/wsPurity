import os
import cv2
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

# Parse arguments from command line --------------------------------------------
parser = argparse.ArgumentParser(description='Stitch Tiles')

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

fold_ACC = args.fold_ACC + '/' # TCGA
fold_ACC_Norm = args.fold_ACC_Norm + '/' # TCGA
fold_BRCA = args.fold_BRCA + '/' # TCGA
fold_BRCA_Norm = args.fold_BRCA_Norm + '/' # TCGA
fold_HNSC = args.fold_HNSC + '/' # TCGA
fold_HNSC_Norm = args.fold_HNSC_Norm + '/' # TCGA
fold_LUAD_LUSC = args.fold_LUAD_LUSC + '/' # TCGA
fold_LUAD_LUSC_Norm = args.fold_LUAD_LUSC_Norm + '/' # TCGA
fold_PRAD = args.fold_PRAD + '/' # TCGA
fold_PRAD_Norm = args.fold_PRAD_Norm + '/' # TCGA
fold_OV = args.fold_OV + '/' # TCGA
fold_OV_Norm = args.fold_OV_Norm + '/' # TCGA
fold_BLCA = args.fold_BLCA + '/' # TCGA
fold_BLCA_Norm = args.fold_BLCA_Norm + '/' # TCGA
fold_PRAD_OV_BRCA = args.fold_PRAD_OV_BRCA + '/' # EIPM
# ------------------------------------------------------------------------------

save_path = 'Stitched/' # resulting stitched slides path

slides = {} # dict with slide folder names per cancer type

slides[fold_ACC] = os.listdir(f'{path}{fold_ACC}')
slides[fold_ACC_Norm] = os.listdir(f'{path}{fold_ACC_Norm}')
slides[fold_BRCA] = os.listdir(f'{path}{fold_BRCA}')
slides[fold_BRCA_Norm] = os.listdir(f'{path}{fold_BRCA_Norm}')
slides[fold_HNSC] = os.listdir(f'{path}{fold_HNSC}')
slides[fold_HNSC_Norm] = os.listdir(f'{path}{fold_HNSC_Norm}')
slides[fold_LUAD_LUSC] = os.listdir(f'{path}{fold_LUAD_LUSC}')
slides[fold_LUAD_LUSC_Norm] = os.listdir(f'{path}{fold_LUAD_LUSC_Norm}')
slides[fold_PRAD] = os.listdir(f'{path}{fold_PRAD}')
slides[fold_PRAD_Norm] = os.listdir(f'{path}{fold_PRAD_Norm}')
slides[fold_OV] = os.listdir(f'{path}{fold_OV}')
slides[fold_OV_Norm] = os.listdir(f'{path}{fold_OV_Norm}')
slides[fold_BLCA] = os.listdir(f'{path}{fold_BLCA}')
slides[fold_BLCA_Norm] = os.listdir(f'{path}{fold_BLCA_Norm}')
slides[fold_PRAD_OV_BRCA] = os.listdir(f'{path}{fold_PRAD_OV_BRCA}')

bins = [0.09, 0.29, 0.39, 0.49, 0.59, 0.69, 0.79, 0.89] # list of bins for tumor purity scores

tile_size = 512
subset_size = 120

model = 'Test-1632167988.6723921' # model to be used for heatmap and threshold images
pred_file = pd.read_csv(f'{model}/model-{model}_detailedoutput.csv', header = 0, index_col = 0)
subset = {} # dict with subset numbers per tile

pred = {} # dict with predicted scores per tile
true = {} # dict with true scores per tile

weight1 = {} # dict with attention weights #1 per tile
weight2 = {} # dict with attention weights #2 per tile

prev = '' # previous tile name - helper
cnt = -1 # subset counter - helper

for i, row in tqdm(pred_file.iterrows()):

    cnt += 1

    # row[0] - slide name
    # row[1] - predicted score
    # row[2] - true score
    # row[3] - predicted class
    # row[4] - true class
    # row[5] - attention weight #1
    # row[6] - attention weight #2

    # if current tile is from the same slide as previous tile
    # and belongs to the same subset, give same subset number
    if row[0].split('_')[0] == prev.split('_')[0] and cnt%subset_size != 0:
        subset[row[0][:-4]] = subset[prev[:-4]]

    # if current tile is not from the same slide as previous tile,
    # reinitialize counter and give subset number of 0
    elif row[0].split('_')[0] != prev.split('_')[0]:
        cnt = 0
        subset[row[0][:-4]] = 0

    # if current tile is from the same slide as previous tile
    # and belongs to a new subset, give a new subset number
    elif cnt % subset_size == 0:
        subset[row[0][:-4]] = int(cnt/subset_size)

    prev = row[0] # update previous tile name

    prob_pred = row[1][row[1].find('[')+1:row[1].find(']')].split(', ')
    prob_pred = len(list(filter(lambda x: float(x) > 0.5, prob_pred)))
    pred[row[0][:-4]] = bins[prob_pred-1]

    weight1[row[0][:-4]] = float(row[5])
    weight2[row[0][:-4]] = float(row[6])

#print(subset)

data_trues = pd.read_csv('Data/DataSplits.csv', header = 0, index_col = 0) # csv with true purity scores per slide

for i, row in tqdm(data_trues.iterrows()):

    # row[0] - slide name
    # row[2] - true score

    true[row[0]] = float(row[2])

# minimum and maximum attention weights #1 per subset

min_weight_subset1 = {}
max_weight_subset1 = {}

prev = ''

for tile1 in tqdm(weight1):

    if (tile1.split('_')[0] == prev.split('_')[0] and
        subset[tile1] == subset[prev]):

        min_weight_subset1[tile1] = min_weight_subset1[prev]
        max_weight_subset1[tile1] = max_weight_subset1[prev]

        continue

    weights = []

    for tile2 in weight1:

        if (tile2.split('_')[0] == tile1.split('_')[0] and
            subset[tile2] == subset[tile1]):

            weights.append(weight1[tile2])

        elif len(weights) > 0:
            break

    min_weight_subset1[tile1] = min(weights)
    max_weight_subset1[tile1] = max(weights)

    prev = tile1

#print(min_weight_subset1, max_weight_subset1)

# minimum and maximum attention weights #1 per slide

min_weight_slide1 = {}
max_weight_slide1 = {}

prev = ''

for tile1 in tqdm(weight1):

    if tile1.split('_')[0] == prev.split('_')[0]:
        continue

    weights = []

    for tile2 in weight1:

        if tile2.split('_')[0] == tile1.split('_')[0]:
            weights.append(weight1[tile2])

        elif len(weights) > 0:
            break

    min_weight_slide1[tile1.split('_')[0]] = min(weights)
    max_weight_slide1[tile1.split('_')[0]] = max(weights)

    prev = tile1

#print(min_weight_slide1, max_weight_slide1)

# minimum and maximum attention weights #2 per subset

min_weight_subset2 = {}
max_weight_subset2 = {}

prev = ''

for tile1 in tqdm(weight2):

    if (tile1.split('_')[0] == prev.split('_')[0] and
        subset[tile1] == subset[prev]):

        min_weight_subset2[tile1] = min_weight_subset2[prev]
        max_weight_subset2[tile1] = max_weight_subset2[prev]

        continue

    weights = []

    for tile2 in weight2:

        if (tile2.split('_')[0] == tile1.split('_')[0] and
            subset[tile2] == subset[tile1]):

            weights.append(weight2[tile2])

        elif len(weights) > 0:
            break

    min_weight_subset2[tile1] = min(weights)
    max_weight_subset2[tile1] = max(weights)

    prev = tile1

#print(min_weight_subset2, max_weight_subset2)

# minimum and maximum attention weights #2 per slide

min_weight_slide2 = {}
max_weight_slide2 = {}

prev = ''

for tile1 in tqdm(weight2):

    if tile1.split('_')[0] == prev.split('_')[0]:
        continue

    weights = []

    for tile2 in weight2:

        if tile2.split('_')[0] == tile1.split('_')[0]:
            weights.append(weight2[tile2])

        elif len(weights) > 0:
            break

    min_weight_slide2[tile1.split('_')[0]] = min(weights)
    max_weight_slide2[tile1.split('_')[0]] = max(weights)

    prev = tile1

#print(min_weight_slide2, max_weight_slide2)

print("Dictionaries created.")

for cancer_type in slides:
    for slide in slides[cancer_type]:

        if slide not in true:
            print(slide, 'not found.')
            continue

        is_stitched = 0

        for image in os.listdir(save_path):
            if slide in image:
                is_stitched = 1
                break

        if is_stitched == 1:
            print(slide, 'already stitched.')
            continue

        try:

            tiles = os.listdir(f'{cancer_type}{slide}/')

            xmax = 0
            ymax = 0

            for tile in tiles:

                x, y = tile[:-4].split('_')[1:]

                if int(x) > xmax:
                    xmax = int(x)
                if int(y) > ymax:
                    ymax = int(y)

            startstitch_original = (np.zeros((tile_size*ymax+tile_size, tile_size*xmax+tile_size, 3), dtype = 'uint8'))+255

            startstitch_heatmap_score = (np.zeros((tile_size*ymax+tile_size, tile_size*xmax+tile_size), dtype = 'uint8'))

            startstitch_heatmap_weighted_subset1 = (np.zeros((tile_size*ymax+tile_size, tile_size*xmax+tile_size), dtype = 'uint8'))
            startstitch_heatmap_weighted_slide1 = (np.zeros((tile_size*ymax+tile_size, tile_size*xmax+tile_size), dtype = 'uint8'))

            startstitch_heatmap_weighted_subset2 = (np.zeros((tile_size*ymax+tile_size, tile_size*xmax+tile_size), dtype = 'uint8'))
            startstitch_heatmap_weighted_slide2 = (np.zeros((tile_size*ymax+tile_size, tile_size*xmax+tile_size), dtype = 'uint8'))

            startstitch_threshold_weighted_subset1 = (np.zeros((tile_size*ymax+tile_size, tile_size*xmax+tile_size, 3), dtype = 'uint8'))+255
            startstitch_threshold_weighted_slide1 = (np.zeros((tile_size*ymax+tile_size, tile_size*xmax+tile_size, 3), dtype = 'uint8'))+255

            startstitch_threshold_weighted_subset2 = (np.zeros((tile_size*ymax+tile_size, tile_size*xmax+tile_size, 3), dtype = 'uint8'))+255
            startstitch_threshold_weighted_slide2 = (np.zeros((tile_size*ymax+tile_size, tile_size*xmax+tile_size, 3), dtype = 'uint8'))+255

            total_pred = 0
            total_cnt = 0

            for tile in tqdm(tiles):

                tile_image = cv2.imread(f'{cancer_type}{slide}/{tile}', cv2.IMREAD_COLOR)

                lower_tissue = (122,30,0)
                upper_tissue = (179,255,255)

                tile_image_hsv = cv2.cvtColor(tile_image, cv2.COLOR_BGR2HSV)
                tile_image_hsv = cv2.inRange(tile_image_hsv, lower_tissue, upper_tissue)

                tile_pred = pred[tile[:-4]]
                total_pred += tile_pred
                total_cnt += 1

                weight_subset1 =  weight1[tile[:-4]]
                weight_slide1 =  weight1[tile[:-4]]

                weight_subset2 = weight2[tile[:-4]]
                weight_slide2 = weight2[tile[:-4]]

                try:
                    tile_weighted_subset1 = ((weight_subset1-min_weight_subset1[tile[:-4]])/(max_weight_subset1[tile[:-4]]-min_weight_subset1[tile[:-4]]))*tile_pred
                except:
                    tile_weighted_subset1 = 0

                try:
                    tile_weighted_slide1 = ((weight_slide1-min_weight_slide1[slide])/(max_weight_slide1[slide]-min_weight_slide1[slide]))*tile_pred
                except:
                    tile_weighted_slide1 = 0

                try:
                    tile_weighted_subset2 = ((weight_subset2-min_weight_subset2[tile[:-4]])/(max_weight_subset2[tile[:-4]]-min_weight_subset2[tile[:-4]]))
                except:
                    tile_weighted_subset2 = 0

                try:
                    tile_weighted_slide2 = ((weight_slide2-min_weight_slide2[slide])/(max_weight_slide2[slide]-min_weight_slide2[slide]))
                except:
                    tile_weighted_slide2 = 0

                # heatmap images
                heatmap_score_image = tile_image_hsv * (float(tile_pred)+0.1)

                heatmap_weighted_subset1_image = tile_image_hsv * (max(tile_weighted_subset1, 0.1))
                heatmap_weighted_slide1_image = tile_image_hsv * (max(tile_weighted_slide1, 0.1))

                heatmap_weighted_subset2_image = tile_image_hsv * (max(tile_weighted_subset2, 0.1))
                heatmap_weighted_slide2_image = tile_image_hsv * (max(tile_weighted_slide2, 0.1))

                # threshold images
                threshold_weighted_subset1_image = tile_image * tile_weighted_subset1

                if weight_subset1 < (max_weight_subset1[tile[:-4]] * 0.8):
                    threshold_weighted_subset1_image = tile_image * 0

                threshold_weighted_slide1_image = tile_image * tile_weighted_slide1

                if weight_slide1 < (max_weight_slide1[slide] * 0.8):
                    threshold_weighted_slide1_image = tile_image * 0

                threshold_weighted_subset2_image = tile_image * tile_weighted_subset2

                if weight_subset2 < (max_weight_subset2[tile[:-4]] * 0.8):
                    threshold_weighted_subset2_image = tile_image * 0

                threshold_weighted_slide2_image = tile_image * tile_weighted_slide2

                if weight_slide2 < (max_weight_slide2[slide] * 0.8):
                    threshold_weighted_slide2_image = tile_image * 0

                # stitching tile image into slide image
                x, y = tile[:-4].split('_')[1:]

                xstart = int(x)*tile_size
                xend = int(x)*tile_size + tile_size
                ystart = int(y)*tile_size
                yend = int(y)*tile_size + tile_size

                startstitch_original[ystart:ystart+tile_image.shape[0], xstart:xstart+tile_image.shape[1],:] = tile_image

                startstitch_heatmap_score[ystart:ystart+tile_image.shape[0], xstart:xstart+tile_image.shape[1]] = heatmap_score_image

                startstitch_heatmap_weighted_subset1[ystart:ystart+tile_image.shape[0], xstart:xstart+tile_image.shape[1]] = heatmap_weighted_subset1_image
                startstitch_heatmap_weighted_slide1[ystart:ystart+tile_image.shape[0], xstart:xstart+tile_image.shape[1]] = heatmap_weighted_slide1_image

                startstitch_heatmap_weighted_subset2[ystart:ystart+tile_image.shape[0], xstart:xstart+tile_image.shape[1]] = heatmap_weighted_subset2_image
                startstitch_heatmap_weighted_slide2[ystart:ystart+tile_image.shape[0], xstart:xstart+tile_image.shape[1]] = heatmap_weighted_slide2_image

                startstitch_threshold_weighted_subset1[ystart:ystart+tile_image.shape[0], xstart:xstart+tile_image.shape[1]] = threshold_weighted_subset1_image
                startstitch_threshold_weighted_slide1[ystart:ystart+tile_image.shape[0], xstart:xstart+tile_image.shape[1]] = threshold_weighted_slide1_image

                startstitch_threshold_weighted_subset2[ystart:ystart+tile_image.shape[0], xstart:xstart+tile_image.shape[1]] = threshold_weighted_subset2_image
                startstitch_threshold_weighted_slide2[ystart:ystart+tile_image.shape[0], xstart:xstart+tile_image.shape[1]] = threshold_weighted_slide2_image

            # applying color maps to heatmaps
            startstitch_heatmap_score = cv2.applyColorMap(startstitch_heatmap_score, cv2.COLORMAP_JET)
            startstitch_heatmap_weighted_subset1 = cv2.applyColorMap(startstitch_heatmap_weighted_subset1, cv2.COLORMAP_MAGMA)
            startstitch_heatmap_weighted_slide1 = cv2.applyColorMap(startstitch_heatmap_weighted_slide1, cv2.COLORMAP_MAGMA)
            startstitch_heatmap_weighted_subset2 = cv2.applyColorMap(startstitch_heatmap_weighted_subset2, cv2.COLORMAP_MAGMA)
            startstitch_heatmap_weighted_slide2 = cv2.applyColorMap(startstitch_heatmap_weighted_slide2, cv2.COLORMAP_MAGMA)

            factor = 10

            # creating output images
            original_output = cv2.resize(startstitch_original, (int((tile_size*xmax+tile_size)/factor), int((tile_size*ymax+tile_size)/factor)))

            heatmap_score_output = cv2.resize(startstitch_heatmap_score, (int((tile_size*xmax+tile_size)/factor), int((tile_size*ymax+tile_size)/factor)))

            heatmap_weighted_subset1_output = cv2.resize(startstitch_heatmap_weighted_subset1, (int((tile_size*xmax+tile_size)/factor), int((tile_size*ymax+tile_size)/factor)))
            heatmap_weighted_slide1_output = cv2.resize(startstitch_heatmap_weighted_slide1, (int((tile_size*xmax+tile_size)/factor), int((tile_size*ymax+tile_size)/factor)))

            heatmap_weighted_subset2_output = cv2.resize(startstitch_heatmap_weighted_subset2, (int((tile_size*xmax+tile_size)/factor), int((tile_size*ymax+tile_size)/factor)))
            heatmap_weighted_slide2_output = cv2.resize(startstitch_heatmap_weighted_slide2, (int((tile_size*xmax+tile_size)/factor), int((tile_size*ymax+tile_size)/factor)))

            threshold_weighted_subset1_output = cv2.resize(startstitch_threshold_weighted_subset1, (int((tile_size*xmax+tile_size)/factor), int((tile_size*ymax+tile_size)/factor)))
            threshold_weighted_slide1_output = cv2.resize(startstitch_threshold_weighted_slide1, (int((tile_size*xmax+tile_size)/factor), int((tile_size*ymax+tile_size)/factor)))

            threshold_weighted_subset2_output = cv2.resize(startstitch_threshold_weighted_subset2, (int((tile_size*xmax+tile_size)/factor), int((tile_size*ymax+tile_size)/factor)))
            threshold_weighted_slide2_output = cv2.resize(startstitch_threshold_weighted_slide2, (int((tile_size*xmax+tile_size)/factor), int((tile_size*ymax+tile_size)/factor)))

            cv2.imwrite(f'{save_path}{slide}_original_pred_{round(total_pred/total_cnt,2)}_true_{round(true[slide],2)}.jpg', original_output)

            cv2.imwrite(f'{save_path}{slide}_heatmap_score_pred_{round(total_pred/total_cnt,2)}_true_{round(true[slide],2)}.jpg', heatmap_score_output)

            cv2.imwrite(f'{save_path}{slide}_heatmap_weighted_subset1_pred_{round(total_pred/total_cnt,2)}_true_{round(true[slide],2)}.jpg', heatmap_weighted_subset1_output)
            cv2.imwrite(f'{save_path}{slide}_heatmap_weighted_slide1_pred_{round(total_pred/total_cnt,2)}_true_{round(true[slide],2)}.jpg', heatmap_weighted_slide1_output)

            cv2.imwrite(f'{save_path}{slide}_heatmap_weighted_subset2_pred_{round(total_pred/total_cnt,2)}_true_{round(true[slide],2)}.jpg', heatmap_weighted_subset2_output)
            cv2.imwrite(f'{save_path}{slide}_heatmap_weighted_slide2_pred_{round(total_pred/total_cnt,2)}_true_{round(true[slide],2)}.jpg', heatmap_weighted_slide2_output)

            cv2.imwrite(f'{save_path}{slide}_threshold_weighted_subset1_pred_{round(total_pred/total_cnt,2)}_true_{round(true[slide],2)}.jpg', threshold_weighted_subset1_output)
            cv2.imwrite(f'{save_path}{slide}_threshold_weighted_slide1_pred_{round(total_pred/total_cnt,2)}_true_{round(true[slide],2)}.jpg', threshold_weighted_slide1_output)

            cv2.imwrite(f'{save_path}{slide}_threshold_weighted_subset2_pred_{round(total_pred/total_cnt,2)}_true_{round(true[slide],2)}.jpg', threshold_weighted_subset2_output)
            cv2.imwrite(f'{save_path}{slide}_threshold_weighted_slide2_pred_{round(total_pred/total_cnt,2)}_true_{round(true[slide],2)}.jpg', threshold_weighted_slide2_output)

            print(slide, 'was stitched successfully.')

        except Exception as e:

            print(e)
            print(slide, 'could not be stitched.')
            continue

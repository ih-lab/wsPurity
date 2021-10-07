'''
Code for tiling images from digitally scanned slides. There can be adjustments made to tile_size
Matt Brendel - Rotation Project January 2019
'''

import os
os.environ['PATH'] = "C:\\path_to_openslide\\openslide-win64-20171122\\bin" + ";" + os.environ['PATH'] 

import openslide
import numpy as np
from openslide.deepzoom import *
import cv2
import PIL
import time
from tqdm import tqdm
import pandas as pd
import haar

start_time = time.time()

def main():
    save_path = ''
    cancers = [   'Cancer1', 'Cancer2', 'Cancer3' ]

    for w in cancers:
    #TF_CPP_MIN_LOG_LEVEL=2
        fold = '/folder/of/interest/'
        dir_path = f'{fold}{w} Cases\\'
        file_names = os.listdir(dir_path)

        if not os.path.isdir( f'{fold}{w} Cases Tiles Pass\\'):
            os.mkdir(f'{fold}{w} Cases Tiles Pass\\')
            os.mkdir(f'{fold}{w} Cases Tiles Fail\\')
        save_path = f'{fold}{w} Cases Tiles Pass\\'
        bad_path = f'{fold}{w} Cases Tiles Fail\\'

        for i in tqdm(file_names[220:]):
            try: 
                slide = openslide.OpenSlide(f'{dir_path}{i}')
                end_tile = 512
            
                mag = int(slide.properties[openslide.PROPERTY_NAME_OBJECTIVE_POWER])

                conv = float(slide.properties[openslide.PROPERTY_NAME_MPP_X])
                mag_conv = conv/0.497
                    
                tile_size = int(end_tile/mag_conv)
                #print(conv, mag_conv, tile_size)
                slide_zoom = DeepZoomGenerator(slide, tile_size = tile_size, overlap = 0)
                
                lev = slide_zoom.level_tiles[-1]
                lev_val = slide_zoom.level_count

                #print(lev, lev_val)

                for j in range(lev[0]):
                    for k in range(lev[1]):
                    
                        im = slide_zoom.get_tile(lev_val-1, (j,k))				
                        np_im = np.array(im, dtype  = float)
                        bgr = np_im[...,::-1]
                        dim = bgr.shape

                        if dim[0] < tile_size:
                            row_white = np.zeros((1,dim[1],3))+255
                            row_add = np.broadcast_to(row_white, (tile_size-dim[0],dim[1],3))
                            bgr = np.concatenate((bgr, row_add) , axis = 0)

                        dimnew = bgr.shape

                        if dim[1] < tile_size:
                            col_white = np.zeros((dimnew[0],1,3))+255
                            col_add = np.broadcast_to(col_white, (dimnew[0],tile_size-dimnew[1],3))
                            bgr = np.concatenate((bgr, col_add), axis = 1)
                        
                        bgr = bgr.astype('uint8')

                        image_area = np.size(bgr[:,:,0])
                        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

                        lower_tissue = (122,30,0)
                        upper_tissue = (179,255,255)
                        tissue_area = np.count_nonzero(cv2.inRange(hsv, lower_tissue, upper_tissue))
                        #print(tissue_area)

                        if tissue_area > (0.40*image_area):	
                            final_im = cv2.resize(bgr, (end_tile,end_tile))

                            b = haar.blur_detect(final_im,35)
                            #print(b[1])
                
                            if b[1] >= 35:
                                cv2.imwrite(f'{bad_path}{i[:-4]}_{str(j)}_{str(k)}.jpg', final_im)	
                            else:
                                cv2.imwrite(f'{save_path}{i[:-4]}_{str(j)}_{str(k)}.jpg', final_im)	

                            
            except:
                if 'j' in locals():
                    print(i,j,k, lev)	
                else:
                    print(i)
if __name__ == "__main__":
    main()

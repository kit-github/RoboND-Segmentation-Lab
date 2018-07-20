

import os
import glob

import ipdb
import random

from utils import scoring_utils
from utils import data_iterator
from utils import plotting_tools
from utils import model_tools

main_dir = '/home/robond/RoboND-Segmentation-Lab/data/'
subset_name = 'runs/run5'

pred_folder = '{}/{}'.format(main_dir, subset_name)
gt_folder = '{}/{}'.format(main_dir, 'validation')

def get_files(folder, type='*.png'):
    return sorted(glob.glob('{}/{}'.format(folder, type)))


def get_img_mask(pred_name, gt_folder):
    """replace this to img name and mask name"""
    '6_run1cam1_00143_prediction.png'
    corr_name = pred_name.replace('_prediction','')
    im_name = corr_name.replace('.png','.jpeg')
    mask_name = corr_name.replace('cam1','_mask')
    im_name = '{}/images/{}'.format(gt_folder, im_name)
    mask_name = '{}/masks/{}'.format(gt_folder, mask_name)
    return im_name, mask_name


def shuffle(x):
    x = list(x)
    random.shuffle(x)
    return x

im_files0 = get_files(pred_folder)

im_files = shuffle(im_files0)

ipdb.set_trace()

# I don't need to go with any of these things.

validation_path
output_path = pred_folder
scoring_utils.score_run(gt_folder, output_path)

for i in range(30):
    pred_name = im_files[i]
    base_pred_name = os.path.basename(pred_name)
    im_name, mask_name = get_img_mask(base_pred_name, gt_folder)
    if not os.path.exists(im_name):
        print('{} does not exist'.format(im_name))

    if not os.path.exists(mask_name):
        print('{} does not exist'.format(mask_name))

    new_im_files = (im_name, mask_name, pred_name)
    im_tuple = plotting_tools.load_images(new_im_files)
    plotting_tools.show_images(im_tuple, fig_id = 3)
    ipdb.set_trace()

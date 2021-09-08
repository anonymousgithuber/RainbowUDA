import scipy
from scipy import ndimage
import numpy as np
from multiprocessing import Pool
from PIL import Image, ImageFile
import torch
import torch.nn as nn #
from torch.utils import data, model_zoo
from dataset.cityscapes_dataset import cityscapesDataSet
from dataset.gta5_dataset import gta5DataSet
from dataset.synthia_dataset import synthiaDataSet #
from utils.compute_iou_train import compute_mIoU
from os.path import join
import time


gt_dir = '/home/user/Dataset/cityscapes/gtFine/train'
result_dir = '/home/user/Code/RainbowUDA/train_deeplabv2/result/v2_resnet_rf_251_thr_0_train'
gt_list = './dataset/cityscapes_list'
mIoUs = compute_mIoU(gt_dir, result_dir, 'train', gt_list, 'gta5')
    

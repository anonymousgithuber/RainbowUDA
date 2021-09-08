import argparse
import numpy as np
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
from PIL import Image, ImageFile
from torch.utils import data, model_zoo
import yaml
import time
import os
from dataset.cityscapes_dataset_extract import cityscapesExtraction
from utils.fusion_methods import channel_aggregation, certainty_aggregation, majority_aggregation, softlabel_aggregation
from tqdm import tqdm
from tqdm.contrib import tzip

#### Model Settings  ####
SOURCE_DOMAIN = 'gta5'
TARGET_DOMAIN = 'cityscapes'
IGNORE_LABEL = 255
NUM_CLASSES = 19
# NUM_CLASSES = 16
BATCH_SIZE = 1
GPU = 0
INPUT_SIZE_TARGET = '2048,1024'
GT_SIZE = '2048,1024'
FUSION_MODE = 'channel'
THRESHOLD = 70
RF = '35,35'
####  Path Settings  ####
DATA_DIRECTORY_TARGET = '/home/user/Dataset/cityscapes' # (user) data root
DATA_LIST_PATH_TARGET = './dataset/cityscapes_list/train.txt'
RESULT_DIR = '/home/user/Dataset/cityscapes/pseudo_label/GTA5/channel_distilled'
EXTRACTION_LIST_DIR = '/home/user/Code/RainbowUDA/label_fusion/extraction_list.npy'
DELTA_LIST_DIR = ''

NUM_TEACHERS = 7
TEACHERS_LIST = ['CBST-GTA5-DT', 'MRKLD-GTA5-DT', 'R-MRNet-GTA5-DT', 'CAG-GTA5-DT', 'SAC-GTA5-DT', 'DACS-GTA5-DT', 'ProDA-GTA5-DT']

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]
zero_pad = 256 * 3 - len(palette)

def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    #### Model Settings  ####
    parser.add_argument("--fusion-mode", type=str, default=FUSION_MODE, choices=['majority', 'certainty', 'channel', 'soft_label'],
                        help="available options : majority, certainty, channel")
    parser.add_argument("--source-domain", type=str, default=SOURCE_DOMAIN, choices=['gta5', 'synthia'],
                        help="available options : gta5, synthia")
    parser.add_argument("--target-domain", type=str, default=TARGET_DOMAIN, choices=['cityscapes'],
                        help="available options : cityscapes")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict.")
    parser.add_argument("--num-teachers", type=int, default=NUM_TEACHERS,
                        help="Number of teacher models.")
    parser.add_argument("--gpu-id", type=int, default=GPU,
                        help = 'choose gpus')
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="choose gpu device.")
    parser.add_argument("--threshold", type=int, default=THRESHOLD,
                        help="the threshold to filter some noises out.")
    parser.add_argument("--rf", type=str, default=RF,
                        help="The kappa.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--gt-size", type=str, default=GT_SIZE,
                        help="Comma-separated string with height and width of gt images.")

    # (user) add teacher list
    parser.add_argument("--teachers-list", type=str, nargs='+', default=TEACHERS_LIST, 
                        help="Teacher lists.")
    
    ####  Path Settings  ####
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--result-dir", type=str, default=RESULT_DIR,
                        help="Path to the results. (prediction)")
    parser.add_argument("--extraction-list-dir", type=str, default=EXTRACTION_LIST_DIR,
                        help="Path to load extration list for channel fusion.")
    parser.add_argument("--delta-list-dir", type=str, default=DELTA_LIST_DIR,
                        help="Path to load delta list for soft_label fusion.")
    return parser.parse_args()
def main():
    print("Testing...")
    
    # args parsing
    args = get_arguments()
    
    print(args.teachers_list)
    print('Mode: ', args.fusion_mode)
    
    w, h = map(int, args.input_size_target.split(','))
    args.input_size_target = (w, h)
    crop_size = (h, w)
    w, h = map(int, args.gt_size.split(','))
    args.gt_size = (w, h)
    
    rf1, rf2 = map(int, args.rf.split(','))
    args.rf = (rf1, rf2)
    print('RF: ', args.rf)

    # create result dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    # Target loader
    total = 2975
    # [TODO] use for loop to build data loaders.
   
    targetloader_list = [ 
        data.DataLoader(
        cityscapesExtraction(args.data_dir_target, args.data_list_target, set=teacher, 
                max_iters=total/args.batch_size, source_domain=args.source_domain),
    batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True, drop_last=False) 
        for teacher in args.teachers_list ]
    
        
    sm = torch.nn.Softmax(dim = 1)
    log_sm = torch.nn.LogSoftmax(dim = 1)
    kl_distance = nn.KLDivLoss( reduction = 'none')
    interp = nn.Upsample(size=crop_size, mode='bilinear', align_corners=True)

    # [TODO] extraction list
    extraction_list = [
            [0, 2],
            [1, 6],
            [2, 6],
            [3, 3],
            [4, 6],
            [5, 5],
            [6, 6],
            [7, 6],
            [8, 6],
            [9, 6],
            [10, 6],
            [11, 6],
            [12, 6],
            [13, 6],
            [14, 6],
            [15, 6],
            [16, 5],
            [17, 6],
            [18, 6]
    ]
    # best from cheating (selecting from evaluation)
    extraction_list = [
            [0, 5],
            [1, 6],
            [2, 5],
            [3, 6],
            [4, 6],
            [5, 6],
            [6, 6],
            [7, 5],
            [8, 6],
            [9, 6],
            [10, 5],
            [11, 6],
            [12, 6],
            [13, 5],
            [14, 5],
            [15, 6],
            [16, 3],
            [17, 6],
            [18, 6]
    ]
    # load from npy generate by policy_selection.py
    if args.fusion_mode == 'channel':
        extraction_list = np.load(args.extraction_list_dir)
        print('Extraction List: ', extraction_list)
    
    if args.fusion_mode == 'soft_label':
        delta = np.load(args.delta_list_dir)
        delta = torch.from_numpy(delta)
        print('Delta List: ', delta)
    
    # [TODO]: ratio 
    ratio = 0.0 

    # evaluate
    for img_data in tzip(*targetloader_list):
        # print(str(index*args.batch_size)+" / 2975")
        label_list = []
        certainty_list = []
        for batch_t in img_data:
            label, certainty_arr, name = batch_t
            label = label.cuda(args.gpu_id)
            certainty_arr = certainty_arr.cuda(args.gpu_id)
            label_list.append(label)
            certainty_list.append(certainty_arr)
       
        num_classes = args.num_classes
        num_teachers = args.num_teachers
        
        with torch.no_grad():
            if args.fusion_mode == 'channel':
                # pseudo_label = channel_aggregation(num_classes, num_teachers, extraction_list, label_list, certainty_list, args.threshold, args.gpu_id)
                pseudo_label, count_map = channel_aggregation(num_classes, num_teachers, extraction_list, label_list, certainty_list, args.threshold, args.gpu_id, args.rf)
            if args.fusion_mode == 'certainty':
                pseudo_label, certainty_avg = certainty_aggregation(num_classes, num_teachers, label_list, certainty_list, args.threshold, args.gpu_id)
            if args.fusion_mode == 'majority':
                pseudo_label, certainty_avg = majority_aggregation(num_classes, num_teachers, label_list, certainty_list, args.gpu_id)
            if args.fusion_mode == 'soft_label':
                pseudo_label, certainty_avg = softlabel_aggregation(num_classes, num_teachers, label_list, certainty_list, delta, args.threshold, args.gpu_id)
        
        if args.fusion_mode in ['certainty', 'majority', 'soft_label']:
            pseudo_label = pseudo_label.cpu().data.numpy()
            pseudo_label = np.asarray(pseudo_label, dtype=np.uint8)

            for i in range(args.batch_size):
                # PL
                name[i] = '%s/%s' % (args.result_dir, name[i])
                
                output_col = colorize_mask(pseudo_label[i,:,:])
                output_col.save('%s_color.png' % (name[i]))

                output = Image.fromarray(pseudo_label[i,:,:])
                output.save('%s.png' % (name[i]))
                
                # Certainty Map
                certainty_avg_batch = certainty_avg[i]
                certainty_avg_batch = certainty_avg_batch.cpu().data.numpy()
                certainty_avg_batch = certainty_avg_batch.squeeze()
                certainty_avg_batch = np.asarray(certainty_avg_batch, dtype=np.uint8)
                np.savez_compressed('%s.npz' % (name[i]), certainty_avg_batch)    
            
            del pseudo_label, output_col, output

        if args.fusion_mode == 'channel':
            pseudo_label = pseudo_label.cpu().data.numpy()
            pseudo_label = np.asarray(pseudo_label, dtype=np.uint8)
            
            for i in range(args.batch_size):
                # PL
                name[i] = '%s/%s' % (args.result_dir, name[i])
                
                output_col = colorize_mask(pseudo_label[i,:,:])
                output_col.save('%s_color.png' % (name[i]))

                output = Image.fromarray(pseudo_label[i,:,:])
                output.save('%s.png' % (name[i]))

                # [TODO]: calculate ratio       
                count_map_overlap = torch.zeros((count_map.shape[1], count_map.shape[2]), dtype=torch.uint8) #(B, H, W)
                count_map_overlap[count_map[i] > 1] = 1
                total_overlap = torch.sum(count_map_overlap).float()
                total_pt = (count_map.shape[1] * count_map.shape[2])
                
                ratio = ratio + total_overlap / total_pt  

    # [user] ratio
    print(ratio / total)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

if __name__ == '__main__':
    main()

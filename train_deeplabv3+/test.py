import argparse
import numpy as np
import sys
import torch
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
import yaml
import time
import os

from modeling.deeplabv3 import Deeplabv3
from utils.evaluate import evaluate
from utils.compute_class_certainty import evaluate_class_certainty
from utils.generate_pl import generate_pl_certainty

#### Model Settings  ####
SOURCE_DOMAIN = 'gta5'
TARGET_DOMAIN = 'cityscapes'
MODEL = 'Deeplab'
BACKBONE = 'drn' 
IGNORE_LABEL = 255
NUM_CLASSES = 19
BATCH_SIZE = 10
RESTORE_FROM = './snapshots/model_majority_best.pth'
GPU = 0
INPUT_SIZE = '2048,1024' # no resize now
GT_SIZE = '2048,1024'
####  Path Settings  ####
DATA_DIRECTORY = '/home/user/Dataset/cityscapes' # (user) data root
DATA_LIST_PATH = './dataset/cityscapes_list/val.txt' #[TODO:PL]
GT_DIR = '/home/user/Dataset/cityscapes/gtFine/val' #[TODO:PL]
GT_LIST_PATH = './dataset/cityscapes_list'
RESULT_DIR = './result/DUMMY' 

def get_arguments():
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    #### Model Settings  ####
    parser.add_argument("--model", type=str, default=MODEL, choices=['Deeplab'],
                        help="Model Choice Deeplab.")
    parser.add_argument("--source-domain", type=str, default=SOURCE_DOMAIN, choices=['gta5', 'synthia'],
                        help="available options : gta5, synthia")
    parser.add_argument("--target-domain", type=str, default=TARGET_DOMAIN, choices=['cityscapes'],
                        help="available options : cityscapes")
    parser.add_argument("--backbone", type=str, default=BACKBONE, choices=['resnet', 'mobilenet', 'drn'],
                        help="available options : resnet, mobilenet, drn")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu-id", type=int, default=GPU,
                        help = 'choose gpus')
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="choose gpu device.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--gt-size", type=str, default=GT_SIZE,
                        help="Comma-separated string with height and width of gt images.")
    ## add utils ##
    parser.add_argument("--utils", type=str, default=SOURCE_DOMAIN, choices=['test', 'train_pl', 'class_certainty'],
                        help="available options : test, train_pl, class_certainty")
    
    ####  Path Settings  ####
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--gt-dir", type=str, default=GT_DIR,
                        help="Path to the directory containing the target ground truth dataset.")
    parser.add_argument("--gt-list", type=str, default=GT_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--result-dir", type=str, default=RESULT_DIR,
                        help="Path to the results. (prediction)")

    return parser.parse_args()

def main():
    print("Testing...")
    # args parsing

    args = get_arguments()
    print("Loading :", args.restore_from)
    w, h = map(int, args.input_size.split(','))
    args.input_size = (w, h)
    w, h = map(int, args.gt_size.split(','))
    args.gt_size = (w, h)

    # result dir path combine set
    # args.result_dir = os.path.join(args.result_dir, args.set)

    # create result dir
    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    print("Saveing result in: ", args.result_dir)
    model = Deeplabv3(num_classes=args.num_classes, output_stride=16, backbone=args.backbone)
    saved_state_dict = torch.load(args.restore_from)
    model.load_state_dict(saved_state_dict)
    model.cuda(args.gpu_id)

    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print("Number of parameters: " + str(pytorch_total_params))
    
    # evaluate
    tt = time.time()
    if args.utils == 'test': # evaluate mIoU
        _, avg_time = evaluate(args, args.gt_dir, args.gt_list, args.result_dir, model)
    elif args.utils == 'train_pl': # PL, certainty    
        _, avg_time = generate_pl_certainty(args, args.gt_dir, args.gt_list, args.result_dir, model)
    elif args.utils == 'class_certainty': # test certainty for policy selection
        _, avg_time = evaluate_class_certainty(args, args.gt_dir, args.gt_list, args.result_dir, model)
    else:
        print("ERR: not specify utils")

    print('Time used: {} sec'.format(time.time()-tt))
    print(avg_time)

if __name__ == '__main__':
    main()
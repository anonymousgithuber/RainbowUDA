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
from utils.compute_iou import compute_mIoU
from os.path import join
import time

palette = [128, 64, 128, 244, 35, 232, 70, 70, 70, 102, 102, 156, 190, 153, 153, 153, 153, 153, 250, 170, 30,
           220, 220, 0, 107, 142, 35, 152, 251, 152, 70, 130, 180, 220, 20, 60, 255, 0, 0, 0, 0, 142, 0, 0, 70,
           0, 60, 100, 0, 80, 100, 0, 0, 230, 119, 11, 32]

zero_pad = 256 * 3 - len(palette)

for i in range(zero_pad):
    palette.append(0)

def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def save(output_name):
    output, name = output_name
    output_col = colorize_mask(output)
    output_col.save('%s_color.png' % (name.split('.jpg')[0]))

    output = Image.fromarray(output)
    output.save('%s' % (name))
    return

def evaluate_class_certainty(args, gt_dir, gt_list, result_dir, model):
    
    # set model's mode
    model.eval()

    # set image size
    w, h = args.gt_size
    gt_size = (h, w)
    image_path_list = join(gt_list, 'val.txt')

    # Target loader
    TARGET_IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)
    total = 500
    targetloader = data.DataLoader(
        cityscapesDataSet(args.data_dir, image_path_list,
                max_iters=total/args.batch_size,
                resize_size=args.input_size,
                crop_size=(args.input_size[1],args.input_size[0]),
                set='val', scale=False, mirror=False, mean=TARGET_IMG_MEAN, autoaug = False),
    batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False)

    interp = nn.Upsample(size=gt_size, mode='bilinear', align_corners=True)
    sm = torch.nn.Softmax(dim = 1)
    log_sm = torch.nn.LogSoftmax(dim = 1)
    kl_distance = nn.KLDivLoss( reduction = 'none')
    acc_time = 0

    certainty_class = np.zeros(args.num_classes, dtype=np.float64) # for certainty per class 
    certainty_class_count = np.zeros(args.num_classes, dtype=np.float64) # for number per class

    for index, img_data in enumerate(targetloader):
        batch = img_data
        image, _, _, name = batch

        inputs = image.cuda()

        print('\r>>>>Extracting feature...%03d/%03d'%(index*args.batch_size, total), end='')

        with torch.no_grad():
            tt = time.time()
            output = model(inputs)
            acc_time = acc_time + (time.time()-tt)
            # output_batch = interp(sm(output))
            output_batch = sm(interp(output))

            del output, inputs

            output_batch = output_batch.cpu().data.numpy()

        output_batch = output_batch.transpose(0,2,3,1) # b, h, w, 19
        output_batch_max = np.max(output_batch, axis=3) # for certatiny # b, h, w
        output_batch = np.asarray(np.argmax(output_batch, axis=3), dtype=np.uint8) # b, h, w
        output_iterator = []
        
        # for i in range(output_batch.shape[0]):
        #     output_iterator.append(output_batch[i,:,:])
        #     name_tmp = name[i].split('/')[-1]
        #     name[i] = '%s/%s' % (result_dir, name_tmp)

        # with Pool(4) as p:
        #     p.map(save, zip(output_iterator, name) )
        
        # [TODO] user calculate per class certainty 
        for i in range(output_batch.shape[0]):
            for h in range(output_batch.shape[1]):
                for w in range(output_batch.shape[2]):
                    certainty_class[output_batch[i, h, w]] += output_batch_max[i, h, w]
                    # certainty_class[output_batch[i, h, w]] += 0.5340574 # for debug
                    certainty_class_count[output_batch[i, h, w]] += 1
                 
        print('certainty_class', certainty_class)
        print('certainty_class_count', certainty_class_count)
        print(certainty_class / certainty_class_count)
        del output_batch
    
    certainty_per_class = certainty_class / certainty_class_count
    print('certainty_per_class: ', certainty_class / certainty_class_count)
    save_name = join(result_dir, 'certainty_per_class.npy')
    np.save(save_name, certainty_per_class)
    model.train()
    
    # mIoUs = compute_mIoU(gt_dir, result_dir, 'val', gt_list, args.source_domain)
    mIoUs = 0 # dummy value
    return mIoUs, acc_time/500.0
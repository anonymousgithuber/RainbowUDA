import numpy as np
import torch
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn

def rank(input, gpu_id):
    ars = torch.argsort(input, dim=-1)
    idx = torch.tile(torch.arange(0, input.shape[-1]), list(input.shape[:-1]) + [1]).cuda(gpu_id)
    ret = torch.zeros_like(input, dtype=torch.int64).cuda(gpu_id)
    ret.scatter_(-1, ars, idx)
    return ret

def softlabel_aggregation(num_classes, num_teachers, label_list, certainty_list, delta, threshold, gpu_id):
    ###
    # img_size: (B, H, W) -> (4, 1024, 2048) for GTA5 to Cityecapes
    # tensor_size: (B, H, W, C) -> (4, 1024, 2048, 19) for GTA5 to Cityecapes
    ###
    img_size = label_list[0].shape # (B, H, W)
    tensor_size = certainty_list[0].shape # (B, H, W, C)

    # # average certainty ranking
    delta = torch.tile(delta, list(tensor_size[:-1]) + [1]).cuda(gpu_id) # (B, H, W, C)
    # print('delta', delta.shape)
    # # Calculate delta
    # for teacher_certainty in certainty_list: # num_teachers
    #     # teacher_certainty (B, 1024, 2048, 19)
    #     # print('teacher certainty', teacher_certainty[0, 0, 0])
    #     teacher_certainty_sort, _ = torch.sort(teacher_certainty, dim=3) # (B, 1024, 2048, 19) 
    #     # print('teacher certainty sorted', teacher_certainty_sort[0, 0, 0])

    #     delta = delta + teacher_certainty_sort
    # # Average Certainty
    # delta = delta / 7 

    # Convert
    # teacher_certainty_delta = torch.zeros([num_teachers, tensor_size[0], tensor_size[1], tensor_size[2], tensor_size[3]]).cuda(gpu_id) # (T, B, H, W)
    teacher_certainty_delta = torch.zeros(tensor_size).cuda(gpu_id) # (B, H, W, c)
    certainty_sum = torch.zeros(tensor_size).cuda(gpu_id) # (B, H, W, c)
    for teacher_certainty in certainty_list: # num_teachers
        '''
        convert each pixel prediction to delta.
            ex. 10, 20, 70 => 20, 30, 50
        '''
        # print('teacher certainty', teacher_certainty[0, 0, 0])
        # teacher_certainty_rank = torch.argsort(teacher_certainty, dim=3) # (B, 1024, 2048, 19) 
        teacher_certainty_rank = rank(teacher_certainty, gpu_id=gpu_id)
        # print('rank',teacher_certainty_rank[0, 0, 0])
        # print('delta', delta[0, 0, 0])
        # teacher_certainty_delta = delta[teacher_certainty_rank]
        teacher_certainty_delta = torch.gather(delta, 3, teacher_certainty_rank)
        # print('teacher delta',teacher_certainty_delta[0, 0, 0])
        certainty_sum = certainty_sum + teacher_certainty_delta
    # Average all teacher's predictions
    certainty_sum = certainty_sum / num_teachers

    # PL 
    pseudo_label = 255*torch.ones(img_size, dtype=torch.uint8).cuda(gpu_id)
    pseudo_label_mask = torch.zeros(img_size, dtype=torch.uint8).cuda(gpu_id)
    # mask those pixels with certainty < threshold
    pseudo_label_value, pseudo_label_idx = torch.max(certainty_sum, dim=3)
    pseudo_label_idx.type(torch.uint8)

    # Take the threshold
    pseudo_label_mask[pseudo_label_value >= threshold] = 1
    pseudo_label = pseudo_label_idx*pseudo_label_mask + pseudo_label*(1-pseudo_label_mask)

    
    return pseudo_label, certainty_sum


def certainty_aggregation(num_classes, num_teachers, label_list, certainty_list, threshold, gpu_id):
    ###
    # img_size: (B, H, W) -> (4, 1024, 2048) for GTA5 to Cityecapes
    # tensor_size: (B, H, W, C) -> (4, 1024, 2048, 19) for GTA5 to Cityecapes
    ###
    img_size = label_list[0].shape 
    tensor_size = certainty_list[0].shape 

    # Generate the pseudo labels
    pseudo_label = 255*torch.ones(img_size, dtype=torch.uint8).cuda(gpu_id)
    pseudo_label_mask = torch.zeros(img_size, dtype=torch.uint8).cuda(gpu_id)
    certainty_sum = torch.zeros(tensor_size).cuda(gpu_id)

    # Sum up the certainty tensors
    for l in range(num_teachers):
        certainty_sum = certainty_sum + certainty_list[l]
    certainty_sum = certainty_sum / num_teachers

    # [TODO] Save certainty maps
    # certainty_sum -> Output (1024, 2048, 19) -> .npz
    # ----
    
    # mask those pixels with certainty < threshold
    pseudo_label_value, pseudo_label_idx = torch.max(certainty_sum, dim=3)
    pseudo_label_idx.type(torch.uint8)

    # Take the threshold
    pseudo_label_mask[pseudo_label_value >= threshold] = 1
    pseudo_label = pseudo_label_idx*pseudo_label_mask + pseudo_label*(1-pseudo_label_mask)

    del pseudo_label_mask
    
    return pseudo_label, certainty_sum

def majority_aggregation(num_classes, num_teachers, label_list, certainty_list, gpu_id):
    ###
    # img_size: (B, H, W) -> (4, 1024, 2048) for GTA5 to Cityecapes
    # tensor_size: (B, H, W, C) -> (4, 1024, 2048, 19) for GTA5 to Cityecapes
    ###
    img_size = label_list[0].shape 
    tensor_size = certainty_list[0].shape 
    
    pseudo_label_count = torch.zeros(tensor_size).cuda(gpu_id)
    for i in range(num_classes):
        for l in range(num_teachers):
            # Build the one-hot tensor 
            class_map = torch.zeros(img_size).cuda(gpu_id)
            class_map[label_list[l] == i] = 1
            # Sum up the votes
            pseudo_label_count[:,:,:,i] += class_map
    
    pseudo_label_value, pseudo_label_idx = torch.max(pseudo_label_count, dim=3)
    
    # (B, H, W, C) certainty
    pseudo_label_count = pseudo_label_count / num_teachers * 100
    return pseudo_label_idx, pseudo_label_count


def channel_aggregation(num_classes, num_teachers, extraction_list, label_list, certainty_list, threshold, gpu_id, RF=(35, 35)):
    ###
    # img_size: (B, H, W) -> (4, 1024, 2048) for GTA5 to Cityecapes
    # tensor_size: (B, H, W, C) -> (4, 1024, 2048, 19) for GTA5 to Cityecapes
    # extraction_list: (C, 2) ex. [[0, 6], [1, 5], ...]
    # label_list: (T, B, H, W)
    # certainty_list: (T, B, H, W, C)
    ###
    img_size = label_list[0].shape # (B, H, W)
    tensor_size = certainty_list[0].shape # (B, H, W, C)

    # Generate the pseudo labels
    pseudo_label = torch.zeros(img_size, dtype=torch.uint8).cuda(gpu_id) # (B, H, W)
    pseudo_label_cubic = torch.zeros((img_size[0],num_classes,img_size[1],img_size[2])).cuda(gpu_id) # (B, C, H, W)
    pseudo_label_cubic_certainty = torch.zeros((img_size[0],num_classes,img_size[1],img_size[2])).cuda(gpu_id) # (B, C, H, W)

    count_map = torch.zeros(img_size, dtype=torch.uint8).cuda(gpu_id) # (B, H, W)
    
    # Extract the tensor
    for p in range(num_classes):
        # Extract pseudo label
        i, m = extraction_list[p][0], extraction_list[p][1]  # i: class, m: teacher
        pseudo_label[label_list[m] == i] = i
        count_map[label_list[m] == i] += 1

        # Build the cubic
        class_map = torch.zeros(img_size).cuda(gpu_id)
        class_map[label_list[m] == i] = 1
        pseudo_label_cubic[:,i,:,:] = class_map

        # Build certainty cubic
        pseudo_label_cubic_certainty[:,i,:,:] = certainty_list[m][:,:,:,i]
    
    # Generate the mask M
    M = torch.zeros(img_size, dtype=torch.uint8).cuda(gpu_id)
    M[count_map == 1] = 1

    # Generate the tensors for disagreement
    avgPooling = nn.AvgPool2d(RF, stride=(1, 1), padding=(int(RF[0]/2), int(RF[1]/2))).cuda(gpu_id)
    # Pseudo label
    pseudo_label_cubic = avgPooling(pseudo_label_cubic)
    _, pseudo_label_disagree = torch.max(pseudo_label_cubic, dim=1)
    pseudo_label_disagree = pseudo_label_disagree.type(torch.ByteTensor).cuda(gpu_id)
    pseudo_label = pseudo_label*M + pseudo_label_disagree*(1-M)
    # Certainty tensor
    certainty_disagree = avgPooling(pseudo_label_cubic_certainty)
    # print(certainty_disagree.shape)
    # print(pseudo_label.unsqueeze(0).type(torch.LongTensor).cuda(gpu_id).shape)
    certainty_disagree = torch.gather(certainty_disagree, 1, pseudo_label.unsqueeze(1).type(torch.LongTensor).cuda(gpu_id)).squeeze(1)
    certainty_agree = torch.gather(pseudo_label_cubic_certainty, 1, pseudo_label.unsqueeze(1).type(torch.LongTensor).cuda(gpu_id)).squeeze(1)
    certainty_tensor = certainty_agree*M + certainty_disagree*(1-M)

    mask_threshold = torch.zeros(img_size, dtype=torch.uint8).cuda(gpu_id)
    mask_threshold[certainty_tensor >= threshold] = 1

    unlabeled = 255*torch.ones(img_size, dtype=torch.uint8).cuda(gpu_id)
    pseudo_label = pseudo_label*mask_threshold + unlabeled*(1-mask_threshold)
    
    # [user] return ratio

    # return pseudo_label
    return pseudo_label, count_map

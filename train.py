import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
np.set_printoptions(precision=3)
import time
import os
import pandas as pd
import copy

from dataloader.action_genome import AG, cuda_collate_fn
from lib.object_detector import detector
from lib.config import Config
from lib.evaluation_recall import BasicSceneGraphEvaluator
from lib.AdamW import AdamW

from lib.models.tr2_model import TR2
from lib.models.utils import *

def train():
    # dataset
    AG_dataset_train = AG(mode="train", 
        datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
        filter_small_box=False if conf.mode == 'predcls' else True)
    dataloader_train = torch.utils.data.DataLoader(AG_dataset_train, shuffle=True,
        num_workers=0,collate_fn=cuda_collate_fn, pin_memory=True)
    AG_dataset_test = AG(mode="test", 
        datasize=conf.datasize, data_path=conf.data_path, filter_nonperson_box_frame=True,
        filter_small_box=False if conf.mode == 'predcls' else True)
    dataloader_test = torch.utils.data.DataLoader(AG_dataset_test, shuffle=False, 
        num_workers=0,collate_fn=cuda_collate_fn, pin_memory=True)

    # freeze the detection backbone
    gpu_device = torch.device("cuda:0")
    object_detector = detector(train=True, object_classes=AG_dataset_train.object_classes, \
        use_SUPPLY=True, mode=conf.mode).to(device=gpu_device)
    object_detector.eval()

    # model
    model = TR2(mode=conf.mode,
        attention_class_num=len(AG_dataset_train.attention_relationships),
        spatial_class_num=len(AG_dataset_train.spatial_relationships),
        contact_class_num=len(AG_dataset_train.contacting_relationships),
        obj_classes=AG_dataset_train.object_classes, rel_classes=AG_dataset_train.relationship_classes,
        enc_layer_num=conf.enc_layer, dec_layer_num=conf.dec_layer,
        pre_path=conf.pre_path).to(device=gpu_device)

    # parameters
    for pname, pvalue in model.named_parameters():
        if pname[:4]=='clip' or pname.split('.')[1][:4]=='clip':
            pvalue.requires_grad=False
    params=filter(lambda p:p.requires_grad, model.parameters())

    # evaluator
    evaluator = BasicSceneGraphEvaluator(mode=conf.mode,
        AG_object_classes=AG_dataset_train.object_classes,
        AG_all_predicates=AG_dataset_train.relationship_classes,
        AG_attention_predicates=AG_dataset_train.attention_relationships,
        AG_spatial_predicates=AG_dataset_train.spatial_relationships,
        AG_contacting_predicates=AG_dataset_train.contacting_relationships,
        iou_threshold=0.5,constraint='with')

    # loss function
    focaln_spa=[4643,58176,254476,46368, 69810, 12921]
    focaln_con=[4008, 4076, 4377, 3214, 314, 156897, 11506, 3395, 105067,
                8743, 40545, 7606, 52165, 86, 6761, 772, 1102]
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    focal_loss_spa=FocalLoss(focaln_spa,conf.mode)
    focal_loss_con=FocalLoss(focaln_con,conf.mode)

    # optimizer
    if conf.optimizer == 'adamw':
        optimizer = AdamW(filter(lambda p:p.requires_grad, model.parameters()),\
            lr=conf.lr, weight_decay=0.01)
    elif conf.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=conf.lr)
    elif conf.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=conf.lr, momentum=0.9, weight_decay=0.01)

    scheduler = ReduceLROnPlateau(optimizer, "max", patience=1, factor=0.5, verbose=True,\
        threshold=1e-4, threshold_mode="abs", min_lr=1e-6)

    # train
    tr = []
    for epoch in range(conf.nepoch):
        print("*" * 50, 'train begin')
        model.train()
        object_detector.is_train = True
        start = time.time()
        train_iter = iter(dataloader_train)
        test_iter = iter(dataloader_test)

        for b in range(len(dataloader_train)):
            data = next(train_iter)
            im_data = copy.deepcopy(data[0].cuda(0)) # n_frame*3*h*w
            im_info = copy.deepcopy(data[1].cuda(0)) # n_frame*3
            gt_boxes = copy.deepcopy(data[2].cuda(0))
            num_boxes = copy.deepcopy(data[3].cuda(0))
            gt_annotation = AG_dataset_train.gt_annotations[data[4]] 
            with torch.no_grad(): # prevent gradients to FasterRCNN
                entry = object_detector(im_data, im_info, gt_boxes, num_boxes,gt_annotation)
            
            if conf.mode=='sgcls':
                entry['origin_ims'] = data[6]
            entry['im_info'] = im_info
            entry['video_name'] = gt_annotation[0][1]['metadata']['tag'][:5]

            pred, diff_v, diff_t = model(entry)

            att_distribution = pred["attention_distribution"]
            spa_distribution = pred["spatial_distribution"]
            con_distribution = pred["contacting_distribution"]
            
            # prepare labels
            att_label = torch.tensor(pred["attention_gt"], dtype=torch.long,device=gpu_device).squeeze()
            spa_label = torch.zeros([len(pred["spatial_gt"]), 6],dtype=torch.float32,device=gpu_device)
            con_label=torch.zeros([len(pred["contacting_gt"]),17],dtype=torch.float32,device=gpu_device)
            for i in range(len(pred["spatial_gt"])):
                spa_label[i, pred["spatial_gt"][i]] = 1
                con_label[i, pred["contacting_gt"][i]] = 1

            # calculate loss
            losses = {}
            losses['align_a'] = mse_loss(diff_v[0], diff_t[0])  # kd loss
            losses['align_s'] = mse_loss(diff_v[1], diff_t[1])
            losses['align_c'] = mse_loss(diff_v[2], diff_t[2])

            if conf.mode == 'sgcls':
                losses['object_loss'] = ce_loss(pred['distribution'], pred['labels']) * conf.object_loss
            losses["att"] = ce_loss(att_distribution, att_label)
            losses["spa_focal"] = focal_loss_spa(spa_distribution, spa_label,pred["spatial_gt"])
            losses["con_focal"] = focal_loss_con(con_distribution, con_label,pred["contacting_gt"])
            
            # optimize
            optimizer.zero_grad()
            loss = sum(losses.values())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5, norm_type=2)
            optimizer.step()

            # print
            tr.append(pd.Series({x: y.item() for x, y in losses.items()}))
            if b % 50 == 0 and b >= 50:
                time_per_batch = (time.time() - start) / 50
                print("epoch{:2d}  batch{:5d}/{:5d}  {:.3f}s/batch, {:.1f}m/epoch".format(epoch, b,
                    len(dataloader_train),time_per_batch, len(dataloader_train) * time_per_batch / 60))
                mn = pd.concat(tr[-50:], axis=1).mean(1)
                for loss_k, loss_v in mn.items():
                    print('%s:%.4f'%(loss_k,loss_v),end=' ')
                print("")
                start = time.time()

        print("*" * 50, 'test begin')
        model.eval()
        object_detector.is_train = False
        with torch.no_grad():
            for b in range(len(dataloader_test)):
                data = next(test_iter)
                im_data = copy.deepcopy(data[0].cuda(0))
                im_info = copy.deepcopy(data[1].cuda(0))
                gt_boxes = copy.deepcopy(data[2].cuda(0))
                num_boxes = copy.deepcopy(data[3].cuda(0))
                gt_annotation = AG_dataset_test.gt_annotations[data[4]]

                entry = object_detector(im_data, im_info, gt_boxes, num_boxes,gt_annotation)
                entry['origin_ims'] = data[6]
                entry['im_info'] = im_info
                entry['video_name'] = gt_annotation[0][1]['metadata']['tag'][:5]
                pred, _, _ = model(entry, ifTest=True)
                evaluator.evaluate_scene_graph(gt_annotation, pred)
                
        score = np.mean(evaluator.result_dict[conf.mode + "_recall"][20])
        evaluator.print_stats()
        evaluator.reset_result()
        torch.save({"state_dict":model.state_dict()},conf.output_path+"model_%d_%.3f.tar"%(epoch,score))
        print("save the checkpoint model_%d_%.3f.tar"%(epoch,score))

        scheduler.step(score)

if __name__ == "__main__":
    conf = Config()
    if not os.path.exists(conf.output_path):
        os.mkdir(conf.output_path)
    for i in conf.args:
        print(i,':', conf.args[i])
    """-----------------------------------------------------------------------------------------"""
    train()    

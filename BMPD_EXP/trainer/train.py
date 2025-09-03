import os

import numpy
import torch
from utils.metrics import ConfMatrix,MIoU
from tqdm import tqdm
import torch.nn as nn
import numpy as np
import cv2

def decode_segmap(image, nc=5):
 
    label_colors = np.array([(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (128, 0, 128)])

    r = np.zeros_like(image).astype(np.uint8)
    g = np.zeros_like(image).astype(np.uint8)
    b = np.zeros_like(image).astype(np.uint8)

    for l in range(0, nc):
        idx = image == l

        r[idx] = label_colors[l, 0]
        g[idx] = label_colors[l, 1]
        b[idx] = label_colors[l, 2]

    rgb = np.stack([b, g, r], axis=2)
    return rgb

def save_seg_results(binary_scores, img_name, binary_score_map_path):

    # binary score map
    img_name1 = img_name.split('/')[-1]

    PRE = decode_segmap(binary_scores)

    cv2.imwrite(os.path.join(binary_score_map_path, "{}".format(img_name1)), PRE)



class Trainer:
    def __init__(self, loss, hyp_param):
        self.loss = loss
        # self.tensorboard = tensorboard
        self.hyp_param = hyp_param

    @staticmethod
    def CutMix(image, label, depth):
        def rand_bbox(size, lam=None):
            # past implementation
            w = size[2]
            h = size[3]
            b = size[0]
            cut_rat = numpy.sqrt(1. - lam)
            cut_w = int(w * cut_rat)
            cut_h = int(h * cut_rat)

            cx = numpy.random.randint(size=[b, ], low=int(w/8), high=h)
            cy = numpy.random.randint(size=[b, ], low=int(h/8), high=h)

            bbx1 = numpy.clip(cx - cut_w // 2, 0, w)
            bby1 = numpy.clip(cy - cut_h // 2, 0, h)

            bbx2 = numpy.clip(cx + cut_w // 2, 0, w)
            bby2 = numpy.clip(cy + cut_h // 2, 0, h)

            return bbx1, bby1, bbx2, bby2

        mix_image = image.clone()
        mix_label = label.clone()
        mix_depth = depth.clone()

        rand_index = torch.randperm(image.size()[0])[:image.size()[0]].cuda()
        u_bbx1, u_bby1, u_bbx2, u_bby2 = rand_bbox(image.size(), lam=numpy.random.beta(4, 4))

        for i in range(0, image.shape[0]):
            mix_image[i, :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                image[rand_index[i], :, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

            mix_label[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                label[rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

            mix_depth[i, u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]] = \
                depth[rand_index[i], u_bbx1[i]:u_bbx2[i], u_bby1[i]:u_bby2[i]]

        del image, label, depth

        return mix_image, mix_label, mix_depth

    def train(self, model, epoch, optimiser, dataloader):
        model.train()
        model.module.freeze_sam_parameters()
        miou = MIoU(self.hyp_param.num_classes, self.hyp_param.ignore_index)

        loader_len = len(dataloader)
        tbar = tqdm(range(loader_len)) if self.hyp_param.local_rank <= 0 else range(loader_len)

        for batch_idx in tbar:
            curr_idx = batch_idx + loader_len * epoch

            data = next(dataloader)
            image, label, depth = data['image'], data['label'], data['depth']

            image = image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            depth = depth.cuda(non_blocking=True)

            # depth is currently unused.
            outputs = model(image, depth)
            loss_dicts = self.loss(outputs, model.module.prepare_targets(label))
            curr_loss = loss_dicts["loss_ce"] + loss_dicts['loss_dice'] + loss_dicts['loss_mask'] * 15.

            optimiser.zero_grad()
            curr_loss.backward()
            optimiser.step()

            del outputs,loss_dicts,image, label, depth


            # update the learning rate based on poly decay
            curr_lr = self.hyp_param.lr * (1 - curr_idx / (loader_len * self.hyp_param.epochs)) ** 0.9

            # optimiser.param_groups[0]['lr'] = curr_lr
            for i, opt_group in enumerate(optimiser.param_groups[0:]):
                opt_group['lr'] = curr_lr # * 10.



    def validate(self, model, dataset, epoch,save_dir=None):
        # if self.hyp_param.local_rank > 0: return
        model.eval()
        conf_mat=ConfMatrix(self.hyp_param.num_classes)
        tbar = tqdm(range(len(dataset)))
        for index in tbar:
            data = dataset[index]
            image, label, depth = data['image'].unsqueeze(0), data['label'].unsqueeze(0), data['depth']
            image = image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            depth = depth.cuda(non_blocking=True).unsqueeze(0)
            names = data['name']
            pred = model.module(image, depth)['semantic_masks']
           
            _, predict = torch.max(pred, 1)
           
            conf_mat.update(predict.flatten(), label.flatten())

            if save_dir is not None:
                for j in range(label.size()[0]):
                    pr = predict[j].squeeze().cpu().numpy()
                    save_seg_results(pr, names, save_dir)
           

        mIoU, mAcc, mF1,  iu, acc, f1 = conf_mat.get_metrics_test()
       

        return mIoU, mAcc, mF1,iu,acc,f1

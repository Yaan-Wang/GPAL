import os

import numpy
import torch
from utils.metrics import ConfMatrix,MIoU
from tqdm import tqdm


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
            # print("unique target",torch.unique(label))
# 
            # depth is currently unused.
            outputs = model(image, depth)
            loss_dicts = self.loss(outputs, model.module.prepare_targets(label))
            curr_loss = loss_dicts["loss_ce"] + loss_dicts['loss_dice'] + loss_dicts['loss_mask'] * 15.
            # print("loss_ce",loss_dicts["loss_ce"])
            # print("loss_dice",loss_dicts['loss_dice'])
            # print("loss_mask",loss_dicts['loss_mask'])

            optimiser.zero_grad()
            curr_loss.backward()
            optimiser.step()

            del outputs,loss_dicts,image, label, depth

            # # 清空缓存，释放显存
            # torch.cuda.empty_cache()

            # update the learning rate based on poly decay
            curr_lr = self.hyp_param.lr * (1 - curr_idx / (loader_len * self.hyp_param.epochs)) ** 0.9

            # optimiser.param_groups[0]['lr'] = curr_lr
            for i, opt_group in enumerate(optimiser.param_groups[0:]):
                opt_group['lr'] = curr_lr # * 10.

            # if self.hyp_param.local_rank <= 0:
            #     iou, acc = miou(outputs['semantic_masks'], label)
            #     tbar.set_description('epoch {}, loss {}, miou (f) {}, miou (a) {}'.format(epoch, curr_loss.item(),
            #                                                                               numpy.round(iou[1:].mean().item(), 4),
            #                                                                               numpy.round(iou.mean().item(),
            #                                                                                           4)))

            #     self.tensorboard.upload_wandb_info(curr_idx, info_dict={'backbone_lr': curr_lr, 'head_lr': curr_lr * 10,
            #                                                             'iou (f)': numpy.round(iou[1:].mean().item(), 4),
            #                                                             'iou (a)': numpy.round(iou.mean().item(), 4),
            #                                                             'loss': numpy.round(curr_loss.item(), 2)})

            #     if not curr_idx % 100:
            #         self.tensorboard.upload_wandb_image(image, label, outputs['semantic_masks'])

        

    def validate(self, model, dataset, epoch,test=False):
        # if self.hyp_param.local_rank > 0: return
        model.eval()
        conf_mat=ConfMatrix(self.hyp_param.num_classes)
        tbar = tqdm(range(len(dataset)),disable=True)

        for index in tbar:
            data = dataset[index]
            image, label, depth = data['image'].unsqueeze(0), data['label'].unsqueeze(0), data['depth']
            image = image.cuda(non_blocking=True)
            label = label.cuda(non_blocking=True)
            depth = depth.cuda(non_blocking=True).unsqueeze(0)
            # print("image",image.shape)

            # depth is currently unused.
            pred = model.module(image, depth)['semantic_masks']
            # print(pred.shape)
            _, predict = torch.max(pred, 1)
            # print(predict.shape)
            # print(torch.unique(label))
            conf_mat.update(predict.flatten(), label.flatten())
            # iou, acc = miou(pred, label)
            # tbar.set_description('miou (f) {}, miou (a) {}'.format(numpy.round(iou[1:].mean().item(), 4),
            #                                                        numpy.round(iou.mean().item(), 4)))

            # if not index:
            #     self.tensorboard.upload_wandb_image(image.unsqueeze(0), label.unsqueeze(0), pred.unsqueeze(0))
        mIoU, mAcc, mF1,  iu, acc, f1 = conf_mat.get_metrics_test()

        # # torch.save(model.state_dict(), '/home/wyy/PycharmProjects/3ddetedction/RoadCrackSeg/save/mida.pth')
        # if test:
        #     return mIoU, mAcc, mF1,iu,acc,f1
        # else:
        #     self.tensorboard.upload_wandb_info(epoch, {'eval_miou': mIoU,
        #                                                'eval_acc': mAcc,
        #                                                'val_average_F1': mF1})
        #     return mIoU
        return mIoU, mAcc, mF1,iu,acc,f1


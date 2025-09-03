import os
import torch
import argparse
import random
import numpy
from easydict import EasyDict
from dataloader.dataset import RoadCrack
from dataloader.dataloader import Dataloader
from torch.utils.tensorboard import SummaryWriter


os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    numpy.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


def main(local_rank, ngpus_per_node, hyp_param):
    hyp_param.local_rank = local_rank
    torch.distributed.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=hyp_param.local_rank,
        world_size=hyp_param.gpus * 1
    )
    seed_it(local_rank + hyp_param.seed)

    # build dataloader
    from dataloader.augmentation import Augmentation
    train_loader = Dataloader(batch_size=hyp_param.batch_size, num_workers=hyp_param.num_workers,
                              shuffle=True, dataset=RoadCrack(split='train',
                                                              augmentation=Augmentation(image_mean=hyp_param.image_mean,
                                                                                        image_std=hyp_param.image_std,
                                                                                        image_width=hyp_param.image_size,
                                                                                        image_height=hyp_param.image_size,
                                                                                        split='train',
                                                                                        scale_list=hyp_param.scale_list),
                                                              root_path=hyp_param.data_root_path, name='find'))

    val_set = RoadCrack(split='eval', augmentation=Augmentation(image_mean=hyp_param.image_mean,
                                                                image_std=hyp_param.image_std,
                                                                image_width=hyp_param.image_size,
                                                                image_height=hyp_param.image_size,
                                                                split='eval', scale_list=None),
                        root_path=hyp_param.data_root_path, name='find')
    test_set = RoadCrack(split='test', augmentation=Augmentation(image_mean=hyp_param.image_mean,
                                                                image_std=hyp_param.image_std,
                                                                image_width=hyp_param.image_size,
                                                                image_height=hyp_param.image_size,
                                                                split='test', scale_list=None),
                        root_path=hyp_param.data_root_path, name='find')
 

    # build model
    from model.network import Models
    model = Models(hyp_param).cuda(hyp_param.local_rank)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[hyp_param.local_rank],
                                                      find_unused_parameters=True)
    torch.cuda.set_device(hyp_param.local_rank)

    # build optimiser
    from itertools import chain
    optimiser = torch.optim.AdamW([{'params': model.module.segment.image_encoder.parameters(),
                                    'lr': hyp_param.lr * 1.},
                                   {'params': model.module.segment.prompt_encoder.parameters(),
                                    'lr': hyp_param.lr * 1.},
                                   {'params': list(chain(*model.module.segment.mask_decoder.fine_tune_layers)),
                                    'lr': hyp_param.lr * 1.},
                                   {'params': list(chain(*model.module.segment.mask_decoder.retrain_layers)),
                                    'lr': hyp_param.lr * 1.},
                                  {'params': model.module.segment.depthen.parameters(),
                                   'lr': hyp_param.lr * 1.},
                                   {'params': list(chain(*model.module.segment.depth_retrain_layer)),
                                    'lr': hyp_param.lr * 1.}],
                                  lr=hyp_param.lr, betas=hyp_param.betas, weight_decay=hyp_param.weight_decay)

    # build loss
    from loss.losses import SetCriterion
    loss = SetCriterion(hyp_param.num_classes, losses=["labels", "masks"],
                        weight_dict=torch.ones(hyp_param.num_classes))

 
    writer = SummaryWriter(log_dir=os.path.join(hyp_param.save_model_path, 'logs')) if hyp_param.local_rank <= 0 else None

    # build trainer
    from trainer.train import Trainer
    trainer = Trainer(loss=loss, hyp_param=hyp_param)
    best_iou=0.0
    best_epoch=0
    best_model_path = os.path.join(hyp_param.save_model_path, 'best.pth')

    if hyp_param.local_rank == 0:
        checkpoint = torch.load(best_model_path, map_location='cuda:0')
        model.load_state_dict(checkpoint, strict=True)
        print("Loaded best model parameters from", best_model_path)
        mIoU, mAcc, mF1,iu,acc,f1 = trainer.validate(model, test_set, best_epoch, save_dir=os.path.join(hyp_param.save_model_path, 'seg_results'))

    results_file = os.path.join(hyp_param.save_model_path, "results.txt")
    with open(results_file, 'a') as f:
        f.write(f"\nMean IoU (excluding background): {mIoU}\n")
        f.write(f"Mean Accuracy (excluding background): {mAcc}\n")
        f.write(f"Mean F1 (excluding background): {mF1}\n")
        f.write(f"best iou:{best_iou}\n")
        f.write(f"best epoch:{best_epoch}\n")
        f.write("\nPer-class metrics:\n")
        for i, (iou, acc_val, f1_val) in enumerate(zip(iu, acc, f1)):
            f.write(f"Class {i}: IoU: {iou:.4f}, Accuracy: {acc_val:.4f}, F1: {f1_val:.4f}\n")



    print(f"Test results saved to {results_file}")

    return


if __name__ == '__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')

    parser.add_argument('--batch_size', default=4, type=int)

    parser.add_argument('--epochs', default=100, type=int)

    parser.add_argument('--lr', default=7.5e-5, type=float,
                        help='Default HEAD Learning rate for PSP, '
                             '*Note: the head layers lr will automatically divide by 10*'
                             '*Note: in ddp training, lr will automatically times by n_gpu')

    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')

    parser.add_argument("--local_rank", type=int, default=-1)

    parser.add_argument("--backbone", default='sam_l', type=str,
                        help="choose the SAM scales, currently default to be vit large.")
    parser.add_argument('--save_model_path', default=None, type=str)
    parser.add_argument('--results_file', default=None, type=str)
    # parser.add_argument("--ddp", action="store_true",
    #                     help="distributed data parallel training or not;"
    #                          "MUST SPECIFIED")

    args = parser.parse_args()
    from configs.config import C

    args = EasyDict({**C, **vars(args)})
    # at the moment, we set sam_l as default.
    args = EasyDict({**args, **C.sam_l})  # remember to modify this line after a while.
    args.lr *= args.gpus
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '9763'

    torch.multiprocessing.spawn(main, nprocs=args.gpus, args=(args.gpus, args))

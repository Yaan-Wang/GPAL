import os
import numpy
from easydict import EasyDict

C = EasyDict()
config = C
cfg = C

C.seed = 666

"""Root Directory Config"""
C.repo_name = 'RoadCrackSeg'
C.root_dir = os.path.realpath("../")

"""Data Dir and Weight Dir"""
C.data_root_path = os.path.join(C.root_dir,'dataset','NPO++') 


"""Network Config"""
C.fix_bias = True
C.bn_eps = 1e-5
C.bn_momentum = 0.1

"""Image Config"""
C.num_classes = 3

C.image_mean = numpy.array([0.485, 0.456, 0.406])  # 0.485, 0.456, 0.406
C.image_std = numpy.array([0.229, 0.224, 0.225])

C.image_size =[288,512]
C.scale_list =  [1., 1.1,1.2,1.3] # [.5, .75, 1., 1.25, 1.5] #
C.ignore_index = 255

"""Train Config"""
C.lr = 7.5e-5
C.batch_size = 4
C.energy_weight = .05

C.lr_power = 0.9
C.momentum = 0.9
C.betas = (0.9, 0.999)
C.weight_decay = 0.03

C.num_workers = 8

"""Display Config"""
C.record_info_iter = 20
C.display_iter = 50



# Your project [work_space] name
C.proj_name = "rgbdsegmentation"

C.experiment_name = "GPAL_MPO"


"""SAM settings"""
# "original SAM setup -> image_size: 1024; image_embedding_size: 64"
C.sam_b = EasyDict()
C.sam_l = EasyDict()
C.sam_h = EasyDict()

C.sam_b.vit_patch_size = 16
C.sam_b.prompt_embed_dim = 256
C.sam_b.encoder_embed_dim = 768
C.sam_b.image_embedding_size = [int(C.image_size[0]/16),int(C.image_size[1]/16)]
C.sam_b.encoder_depth = 12
C.sam_b.encoder_num_heads = 12
C.sam_b.encoder_global_attn_indexes = [2, 5, 8, 11]
C.sam_b.checkpoint_path =os.path.join(C.root_dir,'ckpts','sam_vit_b_01ec64.pth') 

C.sam_l.vit_patch_size = 16
C.sam_l.prompt_embed_dim = 256
C.sam_l.encoder_embed_dim = 1024
C.sam_l.image_embedding_size = [int(C.image_size[0]/16),int(C.image_size[1]/16)]
C.sam_l.encoder_depth = 24
C.sam_l.encoder_num_heads = 16
C.sam_l.encoder_global_attn_indexes = [5, 11, 17, 23]
C.sam_l.checkpoint_path = os.path.join(C.root_dir,'ckpts','sam_vit_l_0b3195.pth') 

C.sam_h.vit_patch_size = 16
C.sam_h.prompt_embed_dim = 256
C.sam_h.encoder_embed_dim = 1280
C.sam_h.image_embedding_size = [int(C.image_size[0]/16),int(C.image_size[1]/16)]
C.sam_h.encoder_depth = 32
C.sam_h.encoder_num_heads = 16
C.sam_h.encoder_global_attn_indexes = [7, 15, 23, 31]
C.sam_h.checkpoint_path = ""


"""Save Config"""
C.saved_dir = os.path.join(C.root_dir, 'ckpts', 'exp', C.experiment_name)

# if not os.path.exists(C.saved_dir):
#     os.mkdir(C.saved_dir)

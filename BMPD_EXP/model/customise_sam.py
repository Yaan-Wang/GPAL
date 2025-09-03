from functools import partial
import torch
# from model.sam.mask_decoder import MaskDecoder
from model.class_decoder import MaskDecoder
from model.sam.image_encoder import ImageEncoderViT
from model.sam.prompt_encoder import PromptEncoder
from model.sam.transformer import TwoWayTransformer
import torch.nn as nn
from modelsgemi.segformer import DE
from mmcv.cnn import build_norm_layer
import einops
from model.sam.image_encoder import PatchEmbed
from model.sam.transformer import Attention as DownAttention
from model.sam.common import MLPBlock
import torch.nn.functional as F

class InteractCrossSelfAttn(nn.Module):

    def __init__(self, embedding_dim=1024, num_heads=4, downsample_rate=4) -> None:
        super().__init__()

        self.self_attn = DownAttention(embedding_dim=embedding_dim, num_heads=num_heads, downsample_rate=downsample_rate)
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.mlp = MLPBlock(embedding_dim, mlp_dim=embedding_dim // 2)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.cross_attn = DownAttention(embedding_dim=embedding_dim, num_heads=num_heads, downsample_rate=downsample_rate)
        self.norm3 = nn.LayerNorm(embedding_dim)

    def forward(self, image_defect_token, prompt_token):

        # self-attn
        short_cut = image_defect_token
        image_defect_token = self.norm1(image_defect_token)
        image_defect_token = self.self_attn(q=image_defect_token, k=image_defect_token, v=image_defect_token)
        image_defect_token = short_cut + image_defect_token

        # mlp
        image_defect_token = image_defect_token + self.mlp(self.norm2(image_defect_token))

        # cross-attn
        short_cut = image_defect_token
        image_defect_token = self.norm3(image_defect_token)
        image_defect_token = self.cross_attn(q=image_defect_token, k=prompt_token, v=prompt_token)
        image_defect_token = short_cut + image_defect_token

        return image_defect_token
   

class semanticpro(nn.Module):
    def __init__(
        self,
        depth_channels,
        num_heads=4,
        downsample_rate=4,
        layer_num=3,
    ):
        super().__init__()

        self.layer_num = layer_num
        self.input_image_size = (512, 512)

        self.intra_layer = nn.ModuleList() 
        self.inter_layer = nn.ModuleList()
        self.conv_channels = nn.ModuleList()

        for i in range(self.layer_num):
            self.intra_layer.append(InteractCrossSelfAttn(embedding_dim=1024, num_heads=num_heads, downsample_rate=downsample_rate))
            self.inter_layer.append(InteractCrossSelfAttn(embedding_dim=1024, num_heads=num_heads, downsample_rate=downsample_rate))
            self.conv_channels.append(nn.Conv2d(depth_channels[i], 1024, kernel_size=1))

    def forward(self, image_feat, maskfeature_expanded, depth_feature, layer_index):
        B, H, W, C = image_feat.shape
       
        depth_feature = F.interpolate(depth_feature, size=(H, W), mode='bilinear', align_corners=True)
       
        if depth_feature.shape[1] != C:
            depth_feature = self.conv_channels[layer_index](depth_feature)
        depth_feature = depth_feature.permute(0, 2, 3, 1)

        for i in range(B):
            image_defect_token = image_feat[i:i + 1, maskfeature_expanded[i], :].detach()
            image_background_token = image_feat[i:i + 1, ~maskfeature_expanded[i], :].detach()
            depth_defect_token = depth_feature[i:i + 1, maskfeature_expanded[i], :].detach()

            image_feat[i, maskfeature_expanded[i], :] = (
                self.intra_layer[layer_index](image_defect_token, image_background_token) +
                0.4 * self.inter_layer[layer_index](image_defect_token, depth_defect_token)
            )
            

        return image_feat



class SamWithClassifier(torch.nn.Module):
    def __init__(self, sam_hyps):
        super().__init__()
        self.image_encoder = ImageEncoderViT(
            depth=sam_hyps.encoder_depth,
            embed_dim=sam_hyps.encoder_embed_dim,
            img_size=sam_hyps.image_size,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=sam_hyps.encoder_num_heads,
            patch_size=sam_hyps.vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=sam_hyps.encoder_global_attn_indexes,
            window_size=14,
            out_chans=sam_hyps.prompt_embed_dim,
        )
        self.prompt_encoder = PromptEncoder(
            embed_dim=sam_hyps.prompt_embed_dim,
            image_embedding_size=(sam_hyps.image_embedding_size, sam_hyps.image_embedding_size),
            input_image_size=(sam_hyps.image_size, sam_hyps.image_size),
            mask_in_chans=16,
        )
        self.mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=sam_hyps.prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=sam_hyps.prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            num_classes=sam_hyps.num_classes
        )
        self.depthen = DE("swin_tiny", sam_hyps.num_classes, 8, 0.2, 0.0)
        self.post_norm_cfg = dict(type='LN')
        self.norm_layer = build_norm_layer(self.post_norm_cfg, 256)[1]
        self.cls_embed = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, sam_hyps.num_classes+1)
        )
        self.proto_grow = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 256 * 2 * 2)###256 x the number of propotypes for each class x 2
        )
        self.prot_feat = nn.Embedding(sam_hyps.num_classes, 256)
        self.mask_embed = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256), nn.ReLU(inplace=True),
            nn.Linear(256, 256)
        )
        self.enpro_layer = nn.ModuleDict({
            'patch_embed': PatchEmbed(kernel_size=(16, 16), stride=(16, 16), in_chans=4, embed_dim=1024),
            'prompt_layer': semanticpro([96, 192, 384, 768], downsample_rate=2)
        })
        self.depth_retrain_layer = [
            self.norm_layer.parameters(),
            self.cls_embed.parameters(),
            self.proto_grow.parameters(),
            self.prot_feat.parameters(),
            self.mask_embed.parameters(),
            self.enpro_layer.parameters()
        ]

  
    def forward(self, x_, depth, points_=None, bbox_loc=None):
        batch_size = x_.shape[0]
        
        mdep, sdep = self.depthen([x_, depth])
        prot_feat = self.prot_feat.weight.unsqueeze(0).repeat((batch_size, 1, 1))
        prot_featf = self.norm_layer(prot_feat)

        cls_pred = self.cls_embed(prot_featf)
        prot_feats = self.proto_grow(prot_featf)
        prot_feats = einops.rearrange(prot_feats, 'b n_c (n_p c) -> b n_c n_p c', n_p=2)
        prot_featt = torch.sin(prot_feats[..., ::2]) + prot_feats[..., 1::2]

        mask_embed = self.mask_embed(prot_featf)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mdep)
        

        input_mask = mask_pred.detach()

        _, binpredict = torch.max(input_mask, 1)
      
        binary_mask = (binpredict != 0).float()
       
        input_masks = einops.repeat(input_mask, 'b n h w -> (b n) c h w', c=1)
        
        image_embeddings = self.image_encoder(x_, binary_mask, self.enpro_layer, sdep)
        dense_embeddings = self.prompt_encoder(points_, None, input_masks, size_=image_embeddings.shape)

        del points_

        semantic_mask, sailency_mask, class_embed = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.prompt_encoder.get_dense_pe(None if self.training else image_embeddings.shape[-2:]),
            sparse_prompt_embeddings=prot_featt,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False
        )

        return {
            "semantic_masks": semantic_mask,
            "sailency_masks": sailency_mask,
            "categories": class_embed,
            "dep_cls_pred": cls_pred,
            "mask_pred_plus": mask_pred
        }


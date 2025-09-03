# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from typing import Type

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F

from model.sam.common import LayerNorm2d
import einops

class MaskDecoder(nn.Module):
    def __init__(
            self,
            *,
            transformer_dim: int,
            transformer: nn.Module,
            num_multimask_outputs: int = 3,
            activation: Type[nn.Module] = nn.GELU,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 256,
            num_classes: int = 21
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.

        Arguments:
          transformer_dim (int): the channel dimension of the transformer
          transformer (nn.Module): the transformer used to predict masks
          num_multimask_outputs (int): the number of masks to predict
            when disambiguating masks
          activation (nn.Module): the type of activation to use when
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim
        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1

        self.transformer = transformer
        self.num_classes = num_classes
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),
            activation(),
        )
        self.output_hypernetworks_mlps = nn.ModuleList(
            [
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for _ in range(self.num_mask_tokens)
            ]
        )
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

        self.fine_tune_layers = []
        self.fine_tune_layers.append(self.iou_token.parameters())
        self.fine_tune_layers.append(self.transformer.parameters())
        self.fine_tune_layers.append(self.iou_prediction_head.parameters())
        self.fine_tune_layers.append(self.mask_tokens.parameters())
        # self.fine_tune_layers.append(self.output_upscaling.parameters())
        # self.fine_tune_layers.append(self.output_hypernetworks_mlps.parameters())

        # self.cls_token = nn.Embedding(1, transformer_dim)
        self.classifier = MLP(transformer_dim, transformer_dim, num_classes+1, 3)
        # self.adaptor = MLP(10**2 * 2, 12**2 * 2, num_classes*2, 1)
        # self.adaptor = MLP(10**2 * 2, 12**2 * 2, 50 * 2, 1)

        # self.adaptor = MLP(20*20, 20*20*3, num_classes*2, 3)
        self.retrain_layers = []
        # self.retrain_layers.append(self.cls_token.parameters())
        self.retrain_layers.append(self.classifier.parameters())
        # self.retrain_layers.append(self.adaptor.parameters())
        self.retrain_layers.append(self.output_hypernetworks_mlps.parameters())
        self.retrain_layers.append(self.output_upscaling.parameters())

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multimask_output: bool
    ):
        """
        Predict masks given image and prompt embeddings.

        Arguments:
          image_embeddings (torch.Tensor): the embeddings from the image encoder
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
          torch.Tensor: batched predicted masks
          torch.Tensor: batched predictions of mask quality
        """
        return self.predict_masks(image_embeddings=image_embeddings,
                                  image_pe=image_pe,
                                  sparse_prompt_embeddings=sparse_prompt_embeddings,
                                  dense_prompt_embeddings=dense_prompt_embeddings)
     

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
    ):
        batch_size = image_embeddings.shape[0]
        image_pe = torch.repeat_interleave(image_pe, batch_size, dim=0)
        """Predicts masks. See 'forward' for more details."""
       

        # Concatenate output tokens
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight],
                                  dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(1), -1, -1)
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1, -1)
        # torch.Size([num_classes, 5, 256]) torch.Size([num_classes, 2, 256])
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=2)
        # Expand per-image data in batch direction to be per-mask
        src = torch.repeat_interleave(image_embeddings.unsqueeze(1), tokens.shape[1], dim=1)
        b, q, c, h, w = src.shape
        dense_prompt_embeddings=einops.rearrange(dense_prompt_embeddings, '(b q) c h w -> b q c h w',q=5)
  
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe.unsqueeze(1), tokens.shape[1], dim=1)
 

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)
        # update the places of the bchw
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1, :]
        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b*q, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
    
        hyper_in = self.output_hypernetworks_mlps[0](mask_tokens_out).unsqueeze(1)
        b, c, h, w = upscaled_embedding.shape
        sailency_masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w).squeeze()
        class_embedding = self.classifier(mask_tokens_out)
       
        sailency_masks = sailency_masks.view(batch_size, int(sailency_masks.shape[0]/batch_size),
                                             sailency_masks.shape[1], sailency_masks.shape[2])
        class_embedding = class_embedding.view(batch_size, int(class_embedding.shape[0]/batch_size),
                                               class_embedding.shape[1])
        semantic_masks = torch.einsum("bqc, bqhw -> bchw", torch.softmax(class_embedding, dim=-1)[:, :, :-1],
                                      torch.sigmoid(sailency_masks)) / .1  # apply the sharpen factor.

        # iou_pred = self.iou_prediction_head(iou_token_out)

        return semantic_masks, sailency_masks, class_embedding




# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
            softmax_output: bool = False
    ) -> None:
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output
        self.softmax_output = softmax_output

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)[..., :-1]
        elif self.softmax_output:
            x = torch.softmax(x, dim=1)[..., :-1]
        return x

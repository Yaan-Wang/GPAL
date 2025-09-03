import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F

from model.customise_sam import SamWithClassifier


class Models(torch.nn.Module):
    def __init__(self, param):
        super().__init__()
        self.param = param
        # self.segment_s = SamWithClassifier(num_classes=self.paramnum_classes, sam_hyps=param.sam_hyps)
        self.segment = SamWithClassifier(sam_hyps=param)
        # load partial SAM parameters.
        from utils.sam_utils import partial_load_sam_model
        self.segment.load_state_dict(partial_load_sam_model(self.segment.state_dict(),
                                                            torch.load(self.param.checkpoint_path)),
                                     strict=True)
        self.freeze_sam_parameters()

    # def freeze_teacher_parameters(self):
    #     for p in self.segment_t.parameters():
    #         p.requires_grad = False

    def freeze_sam_parameters(self):
        for name, parameter in self.segment.image_encoder.named_parameters():
            parameter.requires_grad = False if "adaptor" not in name else True

    def forward(self, image, depth):
        # outputs: dict_keys(['semantic_masks', 'sailency_masks', 'categories'])
        outputs = self.segment(image,depth)

        outputs['semantic_masks'] = F.interpolate(outputs['semantic_masks'], size=(image.shape[-2:]), mode='bilinear',
                                                  align_corners=False)

        outputs['sailency_masks'] = F.interpolate(outputs['sailency_masks'], size=(image.shape[-2:]), mode='bilinear',
                                                  align_corners=False)

        return outputs

    # adopted from: https://github.com/zbwxp/SegVit/blob/master/losses/atm_loss.py
    def prepare_targets(self, targets):
        new_targets = []
        for targets_per_image in targets:
            # gt_cls
            gt_cls = targets_per_image.unique()
            gt_cls = gt_cls[gt_cls != self.param.ignore_index]
            masks = []
            for cls in gt_cls:
                masks.append(targets_per_image == cls)
            if len(gt_cls) == 0:
                masks.append(targets_per_image == self.param.ignore_index)
            masks = torch.stack(masks, dim=0)
            new_targets.append(
                {
                    "labels": gt_cls,
                    "masks": masks,
                }
            )
        return new_targets



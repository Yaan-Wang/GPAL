import numpy
import torch


def build_point_grid(n_per_side: int) -> numpy.ndarray:
    """Generates a 2D grid of points evenly spaced in [0,1]x[0,1]."""
    offset = 1 / (2 * n_per_side)
    points_one_side = numpy.linspace(offset, 1 - offset, n_per_side)
    points_x = numpy.tile(points_one_side[None, :], (n_per_side, 1))
    points_y = numpy.tile(points_one_side[:, None], (1, n_per_side))
    points = numpy.stack([points_x, points_y], axis=-1).reshape(-1, 2)
    return points


def partial_load_sam_model(model_dicts, checkpoint_dicts):
    from collections import OrderedDict
    customised_dict = OrderedDict()
    init_param_lists = []
    for k, v in model_dicts.items():
        if k not in checkpoint_dicts:
            customised_dict[k] = model_dicts[k]
        else:
            if any([i in k for i in init_param_lists]):
                customised_dict[k] = torch.nn.Parameter(torch.nn.init.kaiming_normal_(torch.zeros(v.shape))) \
                    if len(v.shape) >= 2 else torch.nn.Parameter(torch.zeros(v.shape))
            elif v.shape == checkpoint_dicts[k].shape:
                customised_dict[k] = checkpoint_dicts[k]
            else:
                if len(list(v.shape)) > 3:
                    customised_dict[k] = resize_pos_embedding(model_pos_embedding=model_dicts[k],
                                                              incoming_pos_embedding=checkpoint_dicts[k])
                else:
                    customised_dict[k] = torch.nn.Parameter(torch.zeros(v.shape))

    return customised_dict


def resize_pos_embedding(model_pos_embedding, incoming_pos_embedding):
    """
    shape => BxHxWxC
    """
    assert len(list(model_pos_embedding.shape)) == 4, "assert the pos embedding dimension follow SAM definition"
    return torch.nn.functional.interpolate(incoming_pos_embedding.permute(0, 3, 1, 2),
                                           size=(model_pos_embedding.shape[1:3]),
                                           align_corners=False,
                                           mode='bilinear').permute(0, 2, 3, 1)


def resize_rel_embedding(model_rel_embedding, incoming_rel_embedding):
    """
    shape => BxLxC
    """
    assert len(list(model_rel_embedding.shape)) == 2, "assert the pos embedding dimension follow SAM definition"
    return torch.nn.functional.interpolate(incoming_rel_embedding.T.unsqueeze(0),
                                           size=model_rel_embedding.shape[0], mode='linear',
                                           align_corners=False).squeeze().T

import os

import PIL
import matplotlib.pyplot as plt
import numpy
import torch
import torchvision
import wandb

# from utils.visualize import show_img


color_map = {"background": (0, 0, 0), "longitudinal": (128, 0, 0), "pothole": (0, 128, 0),
             "alligator": (128, 128, 0),  "transverse": (128, 0, 128), "ignore": (255, 255, 255)}


class Tensorboard:
    def __init__(self, config):
        os.environ['WANDB_API_KEY'] = config['wandb_key']
        os.system("wandb login")
        os.system("wandb {}".format("online" if config['wandb_online'] else "offline"))
        self.tensor_board = wandb.init(project=config['proj_name'], name=config['experiment_name'],
                                       config=config)

        self.restore_transform = torchvision.transforms.Compose([
            DeNormalize(config['image_mean'], config['image_std']),
            torchvision.transforms.ToPILImage()])

    def upload_wandb_info(self, current_step, info_dict):
        for i, info in enumerate(info_dict):
            self.tensor_board.log({info: info_dict[info], "global_step": current_step})
        return

    def upload_ood_image(self, current_step, energy_map, img_number=4, data_name="?", measure_way="energy"):
        self.tensor_board.log({"{}_focus_area_map".format(data_name): [wandb.Image(j, caption="id {}".format(str(i)))
                                                                       for i, j in enumerate(energy_map[:img_number])],
                               "global_step": current_step})

        return

    def upload_wandb_image(self, images, ground_truth, prediction, img_number=4):
        img_number = min(ground_truth.shape[0], img_number)
        images = images[:img_number]
        ground_truth = ground_truth[:img_number]
        prediction = prediction[:img_number]

        prediction = torch.argmax(prediction, dim=1)
        class_map_ = numpy.asarray(list(color_map.values()))
        ground_truth[ground_truth > len(class_map_)] = len(class_map_)
        upload_ground_truth = numpy.apply_along_axis(lambda x: class_map_[x], 1,
                                                     ground_truth[:, numpy.newaxis].cpu().numpy()).squeeze()
        upload_prediction = numpy.apply_along_axis(lambda x: class_map_[x], 1,
                                                   prediction[:, numpy.newaxis].cpu().numpy()).squeeze()

        images = self.de_normalize(images)

        self.tensor_board.log({"image": [wandb.Image(j, caption="id {}".format(str(i)))
                                         for i, j in enumerate(images)]})
        self.tensor_board.log({"label": [wandb.Image(j.transpose(1, 2, 0), caption="id {}".format(str(i)))
                                         for i, j in enumerate(upload_ground_truth)]})
        self.tensor_board.log({"prediction": [wandb.Image(j.transpose(1, 2, 0), caption="id {}".format(str(i)))
                                              for i, j in enumerate(upload_prediction)]})

    def de_normalize(self, image):
        return [self.restore_transform(i.detach().cpu()) if (isinstance(i, torch.Tensor) and len(i.shape) == 3)
                else colorize_mask(i.detach().cpu().numpy(), self.palette)
                for i in image]

    def finish(self):
        self.tensor_board.finish()


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def colorize_mask(mask, palette):
    zero_pad = 256 * 3 - len(palette)
    for i in range(zero_pad):
        palette.append(0)
    # palette[-6:-3] = [183, 65, 14]
    new_mask = PIL.Image.fromarray(mask.astype(numpy.uint8)).convert('P')
    new_mask.putpalette(palette)
    return new_mask

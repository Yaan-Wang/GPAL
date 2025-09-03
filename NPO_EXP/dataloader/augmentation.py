import random
import numpy
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms


class Augmentation(object):
    def __init__(self, image_mean, image_std, image_width, image_height, split, scale_list, ignore_index=255):
        self.split = split
        self.image_size = (image_height, image_width)
        self.image_norm = (image_mean, image_std)
        self.color_jitter = transforms.ColorJitter(brightness=.3, contrast=.15, saturation=.15, hue=0)
        # self.gaussian_blurring = transforms.GaussianBlur((3, 3))
        self.scale_list = scale_list
        self.normalise = transforms.Normalize(mean=image_mean, std=image_std)
        self.get_crop_pos = transforms.RandomCrop(self.image_size)
        self.to_tensor = transforms.ToTensor()
        self.ignore_index = ignore_index

        # if setup == "avs" or setup == "avss" or setup == "avss_binary":
        #     # AVS
        #     self.scale_list = [.5, .75, 1.]
        #     self.color_jitter = None
        # else:
        #     # COCO
        #     # self.scale_list = [.75, 1., 1.25, 1.5, 1.75, 2.]
        #     self.scale_list = [0.5,0.75,1.0,1.25,1.5,1.75,2.0]

    
    def resize(self, image_, label_, depth_):
        h_, w_ = self.image_size
        image_ = F.resize(image_, (h_, w_), transforms.InterpolationMode.NEAREST)
        label_ = F.resize(label_, (h_, w_), transforms.InterpolationMode.NEAREST)
        depth_ = F.resize(depth_, (h_, w_), transforms.InterpolationMode.NEAREST)
        return image_, label_, depth_

    def random_crop_with_padding(self, image_, label_, depth_):
        w_, h_ = image_.size
        if min(h_, w_) < min(self.image_size):
            res_w_ = max(self.image_size[0] - w_, 0)
            res_h_ = max(self.image_size[1] - h_, 0)
            # image_ = F.pad(image_, [0, 0, res_w_, res_h_], fill=(numpy.array(self.image_norm[0]) * 255.).tolist())
            image_ = F.pad(image_, [0, 0, res_w_, res_h_], fill=self.ignore_index) # if error, define the padding value.
            label_ = F.pad(label_, [0, 0, res_w_, res_h_], fill=self.ignore_index)
            depth_ = F.pad(depth_, [0, 0, res_w_, res_h_], fill=self.ignore_index)
            return image_, label_, depth_

        pos_ = self.get_crop_pos.get_params(image_, self.image_size)
        image_ = F.crop(image_, *pos_)
        label_ = F.crop(label_, *pos_)
        depth_ = F.crop(depth_, *pos_)

        return image_, label_, depth_

    # @staticmethod
    def random_scales(self, image_, label_, depth_):
        w_, h_ = image_.size
        chosen_scale = random.choice(self.scale_list)
        w_, h_ = int(w_ * chosen_scale), int(h_ * chosen_scale)
        image_ = F.resize(image_, (h_, w_), transforms.InterpolationMode.NEAREST)
        label_ = F.resize(label_, (h_, w_), transforms.InterpolationMode.NEAREST)
        depth_ = F.resize(depth_, (h_, w_), transforms.InterpolationMode.NEAREST)
        return image_, label_, depth_

    @staticmethod
    def random_flip_h(image_, label_, depth_):
        chosen_flip = random.random() > 0.5
        image_ = F.hflip(image_) if chosen_flip else image_
        label_ = F.hflip(label_) if chosen_flip else label_
        depth_ = F.hflip(depth_) if chosen_flip else depth_
        return image_, label_, depth_

    def train_aug(self, x, y, d):
        x, y, d = self.random_flip_h(x, y, d)
        if self.color_jitter is not None and random.random() < 0.6:
            x = self.color_jitter(x)
        # if self.gaussian_blurring is not None and random.random() < 0.5:
        #     x = self.gaussian_blurring(x)
        
        x, y, d = self.random_scales(x, y, d)
        x, y, d = self.random_crop_with_padding(x, y, d)
        
        x = self.to_tensor(x)
        y = torch.tensor(numpy.asarray(y)).long()
        d = torch.tensor(numpy.asarray(d)).float()

        x = self.normalise(x)
        return x, y, d

    def test_aug(self, x, y, d):
        x = self.to_tensor(x)
        y = torch.tensor(numpy.asarray(y)).long()
        d = torch.tensor(numpy.asarray(d)).float()

        x = self.normalise(x)
        return x, y, d

    def __call__(self, x, y, d):
        return self.train_aug(x, y, d) if self.split == "train" else self.test_aug(x, y, d)
    
    

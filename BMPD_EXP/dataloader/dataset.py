import os
import re
import PIL.Image
import numpy
import torch


class RoadCrack(torch.utils.data.Dataset):
    def __init__(self, split, augmentation, root_path='', name='find'):
        image_list1, label_list1, depth_list1 = self.organise_files(split, root_path, name)
        image_list2, label_list2, depth_list2 = self.organise_files(split, root_path, 'my')

        image_list = image_list1 + image_list2
        label_list = label_list1 + label_list2
        depth_list = depth_list1 + depth_list2

        self.image_list = image_list
        self.label_list = label_list
        self.depth_list = depth_list

        self.augment = augmentation

    def __getitem__(self, index):
        image, label, depth = self.load_data(self.image_list[index], self.label_list[index], self.depth_list[index])
        image, label, depth = self.augment(image, label, depth)
        '''
        depth = depth/depth.max()

        if len(depth.shape) == 3:
            depth = numpy.transpose(depth, (2, 0, 1))
        else:
            depth = numpy.expand_dims(depth, axis=0)
            depth = numpy.repeat(depth, 3, axis=0)
        '''
        depth = depth/depth.max()

        if len(depth.shape) == 3:
            depth = torch.permute(depth, (2, 0, 1))
        else:
            depth = torch.unsqueeze(depth, dim=0).repeat(3, 1, 1)
        return {'image': image, 'label': label, 'depth': depth,'name': self.image_list[index].split('/')[-1]}

    def __len__(self):
        return len(self.image_list)

    @staticmethod
    def load_data(image_path, label_path, depth_path):
        image_ = PIL.Image.open(image_path)
        label_ = PIL.Image.open(label_path)
        depth_ = PIL.Image.open(depth_path)
        return image_, label_, depth_

    @staticmethod
    def organise_files(split_, root_path_, name_):

        entire_files = os.listdir(os.path.join(root_path_, split_, name_))
        entire_files = [os.path.join(root_path_, split_, name_, x) for x in entire_files]
        label_files = list(filter(lambda x: 'lab' in x, entire_files))

        image_files = [i.replace('_lab', '') for i in label_files]
        depth_files = [i.replace('_lab', '_dep') for i in label_files]

        return image_files, label_files, depth_files



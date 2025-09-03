import os
import re
import PIL.Image
import numpy as np
import torch


class RoadCrack(torch.utils.data.Dataset):
    def __init__(self, split, augmentation, root_path=''):

        with open(os.path.join(root_path, split+'.txt'), 'r') as f:
            self.names = [name.strip() for name in f.readlines()]
        self.input_w=512
        self.input_h=288
        self.root_path=root_path
        
        self.augment = augmentation
    @staticmethod
    def read_image(name, folder,head,root_path):
        file_path = os.path.join(root_path, '%s/%s%s.png' % (folder, head,name))
        # image  = np.asarray(PIL.Image.open(file_path))
        image  = PIL.Image.open(file_path)

        return image

    def __getitem__(self, index):
        name  = self.names[index]
        image = self.read_image(name, 'left','left',self.root_path)
        # print("image",image.size)
        label = self.read_image(name, 'labels','label',self.root_path)
        depth = self.read_image(name, 'depth','depth',self.root_path)
        
        image, label, depth = self.augment(image, label, depth)


        M = depth.max()
        depth = depth/M

        if len(depth.shape) == 3:
            depth = torch.permute(depth, (2, 0, 1))
        else:
            depth = torch.unsqueeze(depth, dim=0).repeat(3, 1, 1)

      
        return {'image': image, 'label': label, 'depth': depth,"name":name}



    def __len__(self):
        return len(self.names)

    @staticmethod
    def load_data(image_path, label_path, depth_path):
        image_ = PIL.Image.open(image_path)
        label_ = PIL.Image.open(label_path)
        depth_ = PIL.Image.open(depth_path)
        return image_, label_, depth_

    @staticmethod
    def organise_files(split_, root_path_):

        entire_files = os.listdir(os.path.join(root_path_, split_))
        entire_files = [os.path.join(root_path_, split_, name_, x) for x in entire_files]
        label_files = list(filter(lambda x: 'lab' in x, entire_files))

        image_files = [i.replace('_lab', '') for i in label_files]
        depth_files = [i.replace('_lab', '_dep') for i in label_files]

        return image_files, label_files, depth_files



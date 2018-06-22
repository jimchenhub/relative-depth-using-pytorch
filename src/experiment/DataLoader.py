import os
import h5py
import math
import random
import torch
from common.NYU_params import *
from DataPointer import DataPointer
from torchvision import transforms
from PIL import Image


class DataLoader(object):
    """docstring for DataLoader"""
    def __init__(self, relative_depth_filename, folder_path):
        super(DataLoader, self).__init__()
        print(">>>>>>>>>>>>>>>>> Using DataLoader")
        self.folder_path = folder_path
        self.parse_depth(relative_depth_filename)
        self.data_ptr_relative_depth = DataPointer(self.n_relative_depth_sample)
        print("DataLoader init: \n \t{} relative depth samples \n ".format(self.n_relative_depth_sample))

    def parse_relative_depth_line(self, line):
        splits = line.split(',')
        sample = {}
        sample['img_filename'] = splits[0]
        sample['n_point'] = int(splits[2])
        return sample

    def parse_csv(self, filename, parsing_func):
        handle = {}
        if filename == None:
            return handle

        f = open(filename, 'r')
        sample_idx = 0
        for this_line in f:
            if this_line != '':
                handle[sample_idx] = parsing_func(this_line)
                sample_idx += 1
        f.close()

        return handle

    def parse_depth(self, relative_depth_filename):
        if relative_depth_filename is not None:
            simplified_relative_depth_filename = relative_depth_filename.replace('.csv', '_name.csv')
            if os.path.isfile(simplified_relative_depth_filename):
                print(simplified_relative_depth_filename+" already exists.")
            else:
                command = "grep '.png' "+ relative_depth_filename + " > " + simplified_relative_depth_filename
                print("executing:{}".format(command))
                os.system(command)

            self.relative_depth_handle = self.parse_csv(simplified_relative_depth_filename, self.parse_relative_depth_line)

            hdf5_filename = relative_depth_filename.replace('.csv', '.h5')
            self.relative_depth_handle['hdf5_handle'] = h5py.File(hdf5_filename, 'r')

        else:
            self.relative_depth_handle = {}

        self.n_relative_depth_sample = len(self.relative_depth_handle)-1

    def load_indices(self, depth_indices):
        if depth_indices is not None:
            n_depth = len(depth_indices)
        else:
            n_depth = 0

        batch_size = n_depth
        color = torch.Tensor(batch_size, 3, g_input_height, g_input_width)

        _batch_target_relative_depth_gpu = {}
        _batch_target_relative_depth_gpu['n_sample'] = n_depth

        transform = transforms.Compose([
            transforms.ToTensor()
            ])

        for i in range(0,n_depth):
            idx = int(depth_indices[i])
            _batch_target_relative_depth_gpu[i] = {}
            img_name = self.folder_path+self.relative_depth_handle[idx]['img_filename']
            # print(img_name)
            n_point = self.relative_depth_handle[idx]['n_point']
            
            image = Image.open(img_name)
            image = transform(image).float()
            # print(image)
            # print(image.size())
            # image = Variable(image, require_grad=True)
            color[i,:,:,:].copy_(image)

            _hdf5_offset = int(5*idx)
            _this_sample_hdf5 = self.relative_depth_handle['hdf5_handle']['/data'][_hdf5_offset:_hdf5_offset+5,0:n_point]

            assert(_this_sample_hdf5.shape[0] == 5)
            assert(_this_sample_hdf5.shape[1] == n_point)

            _batch_target_relative_depth_gpu[i]['y_A']= torch.Tensor(torch.from_numpy(_this_sample_hdf5[0]-1)).cuda()
            _batch_target_relative_depth_gpu[i]['x_A']= torch.Tensor(torch.from_numpy(_this_sample_hdf5[1]-1)).cuda()
            _batch_target_relative_depth_gpu[i]['y_B']= torch.Tensor(torch.from_numpy(_this_sample_hdf5[2]-1)).cuda()
            _batch_target_relative_depth_gpu[i]['x_B']= torch.Tensor(torch.from_numpy(_this_sample_hdf5[3]-1)).cuda()            
            _batch_target_relative_depth_gpu[i]['ordianl_relation']= torch.Tensor(torch.from_numpy(_this_sample_hdf5[4])).cuda()
            _batch_target_relative_depth_gpu[i]['n_point'] = n_point

        return torch.autograd.Variable(color.cuda()), _batch_target_relative_depth_gpu

    def load_next_batch(self, batch_size):
        depth_indices = self.data_ptr_relative_depth.load_next_batch(batch_size)
        return self.load_indices(depth_indices)

    def reset(self):
        self.current_pos = 1

    def mixed_sample_strategy1(self, batch_size):
        n_depth = random.randint(0,batch_size-1)
        return n_depth, batch_size - n_depth

    def mixed_sample_strategy2(self, batch_size):
        n_depth = math.floor(batch_size/2)
        return n_depth, batch_size - n_depth #careful about the index
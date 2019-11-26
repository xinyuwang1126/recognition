import torch
from torch.utils.data import Dataset
import h5py
import json
import os
import numpy as np

class FrameDataset(Dataset):
    """
    A PyTorch Dataset class to be used in a PyTorch DataLoader to create batches.
    """

    def __init__(self, data_folder, split, transform=None):
        """
        :param data_folder: folder where data files are stored
        :param data_name: base name of processed datasets
        :param split: split, one of 'TRAIN', 'VAL', or 'TEST'
        :param transform: image transform pipeline
        """
        self.split = split
        assert self.split in {'TRAIN', 'VAL', 'TEST'}
       # print('reading images')
        #self.hdf5file = os.path.join(data_folder, self.split + '_IMAGES'  + '.hdf5')
        # Open hdf5 file where images are stored
        #self.h = h5py.File(os.path.join(data_folder, self.split + '_IMAGES'  + '.hdf5'), 'r')
        #self.imgs = self.h['images']
        #print(self.h['images'].shape)
        # Captions per image
        #self.fpi = self.h.attrs['frames_per_image']

        # Load encoded captions (completely into memory)
        self.image_folder = os.path.join(data_folder,self.split)
        if self.split == 'TRAIN':
            filename_file = os.path.join(data_folder,'train_names.txt')
        elif self.split == 'VAL':
            filename_file = os.path.join(data_folder,'val_names.txt')
        else:
            filename_file = os.path.join(data_folder,'test_names.txt')
        self.filenames=open(filename_file).readlines()


        with open(os.path.join(data_folder, self.split + '_FRAMES' + '.json'), 'r') as j:
            self.frames = json.load(j)

        # Load caption lengths (completely into memory)
        with open(os.path.join(data_folder, self.split + '_FRAMELENS' +  '.json'), 'r') as j:
            self.framelens = json.load(j)

        with open(os.path.join(data_folder, self.split + '_ROLES' +  '.json'), 'r') as j:
            self.roles = json.load(j)

        # PyTorch transformation pipeline for the image (normalizing, etc.)
        self.transform = transform
        self.fpi = 3
        # Total number of  datapoints
        self.dataset_size = len(self.filenames)*3

    def __getitem__(self, i):
        # Remember, the Nth caption corresponds to the (N // captions_per_image)th image
        #print('getting item')
        filename = self.filenames[i//self.fpi].strip()
        filename = filename[:-4]+'.npy'
        img_path = os.path.join(self.image_folder, str(filename)) 
        img = np.load(img_path)
        img = torch.FloatTensor(img / 255.) 
        #with h5py.File(self.hdf5file, 'r') as db:
            #img = torch.FloatTensor(db['images'][i // self.fpi] / 255.) 
        #img = torch.FloatTensor(self.imgs[i // self.fpi] / 255.)
        if self.transform is not None:
            #print('transforming image')
            img = self.transform(img)
        #print('reading frames..')
        frame = torch.LongTensor(self.frames[i])

        framelen = torch.LongTensor([self.framelens[i]])
        #print(self.roles[i // self.fpi])
        role = torch.LongTensor(self.roles[i // self.fpi])
        #role = torch.LongTensor(self.roles[i])

        if self.split is 'TRAIN':
            #print('returning all frames')
            all_frames = torch.LongTensor(
                self.frames[((i // self.fpi) * self.fpi):(((i // self.fpi) * self.fpi) + self.fpi)])
            return img, frame, framelen, role, all_frames
        else:
            # For validation of testing, also return all 'captions_per_image' captions to find BLEU-4 score
            all_frames = torch.LongTensor(
                self.frames[((i // self.fpi) * self.fpi):(((i // self.fpi) * self.fpi) + self.fpi)])
            return img, frame, framelen, role, all_frames

    def __len__(self):
        return self.dataset_size

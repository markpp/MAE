from glob import glob
import torch
import torch.nn.functional as F
import os, sys
import numpy as np
import cv2
import random

class ImageFolderDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, folders, transforms=None):

        print(os.path.join(root_dir, '*.jpg'))
        self.image_paths = glob(os.path.join(root_dir, '*.jpg'))#[:10000]
        self.labels = [0] * len(self.image_paths)

        '''
        self.image_paths, self.labels = [], []
        for i, folder in enumerate(folders):
            files = glob(os.path.join(root_dir, folder + '/*.jpg'))
            self.image_paths.extend(files)
            self.labels.extend([i] * len(files))
        '''

        if os.path.isfile(self.image_paths[0]):
            print("found {} files such as {}".format(len(self.image_paths),self.image_paths[0]))
        else:
            print("file paths such as {} are incorrect".format(self.image_paths[0]))

        self.unique_labels, self.label_counts = np.unique(self.labels, return_counts=True)

        self.transforms = transforms

    def load_sample(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        #label = torch.as_tensor(self.labels[idx])
        #label = F.one_hot(torch.as_tensor(self.labels[idx]), num_classes=len(self.unique_labels)).float()
        return img, 0

    def __getitem__(self, idx):
        img, target = self.load_sample(idx)

        if self.transforms:
            x = self.transforms(image=img)['image']
        else:
            x = img/255.0
        x = x.transpose((2, 0, 1))


        return torch.from_numpy(x).float(), target

    def __len__(self):
        return len(self.image_paths)

#if __name__ == '__main__':

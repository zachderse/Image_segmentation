
import numpy as np
import torch
from PIL import Image
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class BasicDataset(Dataset):
    def __init__(self, images_dir, mask_dir):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
       

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
     
    
    #change preprocess depending on how much transformations you want
    def img_preprocess(self, image):

        image = np.asarray(image)
        return image
        
    def mask_preprocess(self, mask):

        mask = np.asarray(mask)
        return mask

    def __getitem__(self, idx):
        #name from idx e.g. [0]
        name = self.ids[idx]
        
        
        #the one mask file from the given idx
        mask_file = list(self.mask_dir.glob(name  + '.*'))
        #same but for img_file
        img_file = list(self.images_dir.glob(name + '.*'))

        #makes the file an image
        mask = load_image(mask_file[0])
        img = load_image(img_file[0])

 
        img = self.img_preprocess(img)
        mask = self.mask_preprocess(mask)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }

    def __len__(self):
        return len(self.ids)

#creating the dataset
dataset = BasicDataset("...image path",
                     "... labels path"
                    )

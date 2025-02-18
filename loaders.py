import os
from PIL import Image

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
import torchvision
import gin
from typing import Callable, Iterable, Sequence, Tuple
import pathlib
from tqdm import tqdm
import numpy as np

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor




torch.multiprocessing.set_sharing_strategy('file_system')

from datasets import load_dataset


def flatten(iterator: Iterable):
    for elm in iterator:
        for sub_elm in elm:
            yield sub_elm
            
            
def search_for_images(path_list: Sequence[str], extensions = ["png"]):
    paths = map(pathlib.Path, path_list)
    audios = []
    for p in paths:
        for ext in extensions:
            audios.append(p.rglob(f'*.{ext}'))
            audios.append(p.rglob(f'*.{ext.upper()}'))
            
    audios = flatten(audios)
    return audios


class ImageDataset(Dataset):
        def __init__(self, root_dir, transform=None, build_cache = False, means = None, stds = None, max_files=None):
            self.root_dir = root_dir
            self.transform = transform
            if means is not None:
                
                self.means = torch.tensor(means)[:, None, None]
                self.stds = torch.tensor(stds)[:, None, None]
            else:
                self.means, self.stds = None, None
            
            # Gather image file paths from all subject directories
            self.samples = list(search_for_images([root_dir]))
            
            if max_files is not None:
                self.samples = np.random.choice(self.samples, max_files)
            
            #if build_cache:
            #    print("Building Cache")
            #    self.cache = [Image.open(img_path).convert("RGB") for img_path in tqdm(self.samples)]
            #else:
            #    self.cache = None
            def load_image(img_path):
                return Image.open(img_path).convert("RGB")

            if build_cache:
                print("building")
                with ThreadPoolExecutor(max_workers = 32) as executor:
                    self.cache = list(tqdm(executor.map(load_image, self.samples), total=len(self.samples)))
            else:
                self.cache = None


        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, index):
            if self.cache is not None:
                image = self.cache[index]
            else:
                img_path = self.samples[index]
                image = Image.open(img_path).convert('RGB')
                
            if self.transform:
                image = self.transform(image)
                
            if self.means is not None:
                image = (image - self.means)/self.stds
            
            return image

@gin.configurable
def get_dataloaders(img_size, batch_size, num_workers, build_cache= False):
    
    faces_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2, hue=0.1),
        T.ToTensor()
    ])

    faces_dataset = ImageDataset('/data/nils/repos/NIMAT_PROJECT/data/faces', transform=faces_transform, means= [0.3898356258869171, 0.31504639983177185, 0.27228453755378723], stds=[0.22846238315105438, 0.21757572889328003, 0.2148447185754776], build_cache = build_cache, max_files = 100000)
    faces_dataloader = DataLoader(faces_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last = True)
    

    class RandomResize():
        def __init__(self):
            pass
        def __call__(self, image):
            random_size = int(torch.randint(low=128, high = 512, size=(1,))[0])
            return T.Resize(random_size)(image)
    
    
    hubble_transform = T.Compose([
        RandomResize(),
        T.RandomCrop((img_size, img_size)),
        T.ToTensor(),
    ])
    
    
    hubble_dataset = ImageDataset('data/hubble_processed/', transform=hubble_transform,means=[0.24119672179222107, 0.19881127774715424, 0.19402649998664856], 
                     stds=[0.2448999285697937, 0.21586428582668304, 0.21901872754096985], build_cache = build_cache)
    hubble_dataloader = DataLoader(hubble_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last = True)
    
    return faces_dataset, faces_dataloader, hubble_dataset, hubble_dataloader

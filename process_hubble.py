
import os
from PIL import Image

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
import gin
torch.multiprocessing.set_sharing_strategy('file_system')

from datasets import load_dataset



hubble_dataset = load_dataset(
        "Supermaxman/esa-hubble",
        split="train",
        cache_dir="./data/hubble",
        streaming=False,
       # features = "image",
        num_proc = 16,
    )

hubble_dataset.shuffle(seed=42)

def collate(batch):
    # batch is a list of samples
    #images = [hubble_transform(sample["image"]) for sample in batch]
    #images = torch.stack(images)
    return batch
hubble_dataloader = DataLoader(hubble_dataset, batch_size=1, num_workers=0, collate_fn=collate)



def crop(im,height,width):
    #im = Image.open(infile)
    print(im.size)
    imgwidth, imgheight = im.size

    for i in range(imgheight//height):
        for j in range(imgwidth//width):
            box = (j*width, i*height, (j+1)*width, (i+1)*height)
            yield im.crop(box)
            

idx = 0

import tqdm
pbar = tqdm.tqdm(len(hubble_dataloader))


for image in hubble_dataloader:
    image = image[0]["image"]
    
   
    if False:
        H, W = image.size[0], image.size[1]
        print(W, H)
        H = min(2048, H//512 * 512)
        W = min(2048, W//512 * 512)
        
        H, W = H // 2, W // 2 

        H, W = max(H, 512), max(W, 512)
    else:
        H, W = 512, 512

    image = image.resize((H, W))
    
    
    all_images= crop(image, height= 512, width = 512)
    
    pbar.update(1)
    
    
    for j, small_image in enumerate(all_images):
        small_image.save("./data/hubble_processed/" + str(idx)+ "_" + str(j)+ ".png")
        
        
        
    #break
    idx+=1
import os
from PIL import Image

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as F
import gin
torch.multiprocessing.set_sharing_strategy('file_system')

from datasets import load_dataset

@gin.configurable
def get_dataloaders(img_size, batch_size, num_workers):
    class FaceDataset(Dataset):
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            
            # Gather image file paths from all subject directories
            self.samples = []
            # Each subdirectory contains images from one subject
            #identities
            
            #for group in os.listdir(root_dir):
            
            #group_dir = os.path.join(root_dir, group)
            
            folders =[f for f in os.listdir(root_dir) if ".zip" not in f]
            for subject in sorted(folders):
                subject_dir = os.path.join(root_dir, subject)
                if os.path.isdir(subject_dir):
                    for fname in sorted(os.listdir(subject_dir)):
                        if fname.endswith('.png'):
                            self.samples.append(os.path.join(subject_dir, fname))
                        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, index):
            img_path = self.samples[index]
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            return image

    faces_transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        # T.Normalize(mean=[0.485, 0.456, 0.406], 
        #             std=[0.229, 0.224, 0.225])
    ])

    faces_dataset = FaceDataset('data/faces', transform=faces_transform)
    faces_dataloader = DataLoader(faces_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    hubble_dataset = load_dataset(
        "Supermaxman/esa-hubble",
        split="train",
        cache_dir="./data/hubble",
        streaming=False,
       # features = "image",
        num_proc = 16,
    )

    hubble_dataset.shuffle(seed=42)
    hubble_transform = T.Compose([
        T.Resize(1024),
        T.ToTensor(),
        T.RandomCrop(size = img_size),
        # T.Normalize(mean = (0,0,0), std = (1,1,1), inplace=False),
    ])

    def collate(batch):
        # batch is a list of samples
        images = [hubble_transform(sample["image"]) for sample in batch]
        images = torch.stack(images)
        return images

    hubble_dataloader = DataLoader(hubble_dataset, batch_size=batch_size, num_workers=num_workers, collate_fn=collate)

    return faces_dataloader, hubble_dataloader


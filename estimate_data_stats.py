from loaders import *
import json
from tqdm import tqdm

faces_dataloader, hubble_dataloader = get_dataloaders(128, 32, num_workers=4)

mu_f = torch.Tensor([0,0,0])
sigma_f = torch.Tensor([0,0,0])

mu_h = torch.Tensor([0,0,0])
sigma_h = torch.Tensor([0,0,0])

print(len(faces_dataloader))
print(len(hubble_dataloader))

pbar = tqdm(total = 2000)

with torch.no_grad():
    face_images = []
    hubble_images = []
    num_files = 0
    while num_files < 2000:
        for f_batch, h_batch in zip(faces_dataloader, hubble_dataloader):
            face_images.append(f_batch)
            hubble_images.append(h_batch)
            num_files+=32
            pbar.update(32)
            if num_files >2000:
                break

    face_images = torch.cat(face_images)
    hubble_images = torch.cat(hubble_images)

    mu_f = torch.mean(face_images, dim = (0,2,3))
    sigma_f = torch.std(face_images, dim = (0,2,3))
    mu_h = torch.mean(hubble_images, dim = (0,2,3))
    sigma_h = torch.std(hubble_images, dim = (0,2,3))
    print(mu_f.shape)

    stats = {"faces_mean": mu_f.tolist(), "faces_std": sigma_f.tolist(), "hubble_mean": mu_h.tolist(), "hubble_std": sigma_h.tolist()}

with open("dataset_stats.json", "w") as outfile: 
    json.dump(stats, outfile)
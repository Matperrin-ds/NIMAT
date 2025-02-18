import gin

gin.add_config_file_search_path('./configs')
import torch
import os
import numpy as np
from loaders import get_dataloaders
import argparse

parser = argparse.ArgumentParser()

# MDOEL
parser.add_argument("--name", type=str, default="test")
parser.add_argument("--restart", type=int, default=0)
parser.add_argument("--gpu", type=int, default=0)
parser.add_argument('--config', action="append", default=["base"])
parser.add_argument('--model', default="rectified")

# Training
parser.add_argument("--bsize", type=int, default=32)
parser.add_argument("--out_path", type=str, default="./runs")


# DATASET

def add_gin_extension(config_name: str) -> str:
    if config_name[-4:] != '.gin':
        config_name += '.gin'
    return config_name

def main(args):

    gin.parse_config_files_and_bindings(
        map(add_gin_extension, args.config),
        [],
    )

    if args.restart > 0:
        config_path = "./runs/" + args.name + "/config.gin"
        with gin.unlock_config():
            gin.parse_config_files_and_bindings([config_path], [])

    device = "cuda:" + str(args.gpu) if args.gpu >= 0 else "cpu"
    

    ######### BUILD MODEL #########
    

    if args.model == "rectified":
        from model import RectifiedFlow
        blender = RectifiedFlow(device=device)
    else:
        raise ValueError("Model not recognized")
    
    

    ######### GET THE DATASET #########
    dataset1, train_loader1, dataset2, train_loader2 = get_dataloaders(batch_size=args.bsize, num_workers= 0, build_cache = True)


    print(next(iter(train_loader1)).shape)
    print(next(iter(train_loader2)).shape)

    ######### SAVE CONFIG #########
    model_dir = os.path.join(args.out_path, args.name)
    os.makedirs(model_dir, exist_ok=True)

    ######### PRINT NUMBER OF PARAMETERS #########
    num_el = 0
    for p in blender.net.parameters():
        num_el += p.numel()
    print("Number of parameters - unet : ", num_el / 1e6, "M")

    ######### TRAINING #########

    blender.train(logdir =model_dir, train_dl1 = train_loader1, train_dl2 = train_loader2)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)

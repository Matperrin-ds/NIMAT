
import gin 
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
import gin
import matplotlib.pyplot as plt
import os


@gin.configurable
class Base(torch.nn.Module):

    def __init__(self, net, device = "cpu"):
        super().__init__()
        self.net = net
        self.device = device
        self.net.to(device)
        
        
    @gin.configurable
    def train(self, train_dl1, train_dl2, iterations, grad_norm=None, display_step=5000, save_step = 10000, logdir = None):
        
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-4)
        self.net.train()
        self.logger = SummaryWriter(os.path.join(logdir, "runs")) if logdir is not None else None
        pbar = tqdm(range(iterations))
        self.step = 0
        while self.step < iterations:
            print("starting epoch")
            for x1, x2 in zip(train_dl1, train_dl2):
                
                x1, x2 = x1.to(self.device), x2.to(self.device)
                

                loss = self.flow_step(x1, x2)
                self.optimizer.zero_grad()
                loss.backward()
                
                self.logger.add_scalar("Loss", loss.item(), global_step = self.step)

                if grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(list(
                            self.net.parameters()),max_norm=grad_norm)
                    
                self.optimizer.step()
                pbar.set_postfix({"Loss": loss.item()})
                pbar.update(1)

                if self.step % display_step == 0:
                    self.log_samples(x1)
                    
                    
                if self.step%save_step == 0:
                    d= {}
                    d["model_state"] = self.net.state_dict()
                    d["optimizer_state"] = self.optimizer.state_dict()
                    torch.save(d, os.path.join(logdir, f"checkpoint_{self.step}.pt"))
                self.step += 1
            print("Epoch done")
                
                
    @gin.configurable          
    def log_samples(self, x1, nb_steps, n_images):
        print("LOOOOG")
        f, ax = plt.subplots(len(nb_steps)+1, n_images, figsize=(5, 5))
        
        x1 = x1[:n_images].to(self.device)

        for i, nb_step in enumerate(nb_steps):
            x2 = self.sample(x0=x1, nb_steps=nb_step)
            
            x2 = x2.cpu().detach()
            
            for j in range(n_images):
                ax[i+1, j].imshow(x2[j].squeeze().permute(1, 2, 0))
                ax[i+1, j].axis('off')
                if i == 0:
                    ax[0, j].imshow(x1[j].cpu().detach().squeeze().permute(1, 2, 0))
                    ax[0, j].axis('off')
    
        if self.logger is not None:
            print("????")
            self.logger.add_figure(f"Samples", f, global_step = self.step)
        #else:
        plt.show()
                
                
@gin.configurable
class RectifiedFlow(Base):

    def __init__(self, net, device = "cpu"):
        super().__init__(net, device)

    def flow_step(self, x0, x1):
        target = x1 - x0
        
        t = torch.rand(x0.size(0)).to(self.device)

        t = t[:, None,None, None]
        interpolant = (1 - t) * x0 + t * x1

        model_output = self.net(interpolant, t.view(-1))

        loss = ((model_output - target)**2).mean()
        return loss

    @torch.no_grad()
    def sample(self, x0, nb_steps, return_trajectory=False):
        dt = 1 / nb_steps
        t_values = torch.linspace(0, 1, nb_steps+1)[:-1]

        x = x0.to(self.device)

        x_out = []
        for t in t_values:
            t = t.reshape(1).to(self.device)
            x = x + self.net(x, t.repeat(x.shape[0])) * dt

            x_out.append(x.cpu())

        if return_trajectory:
            return x, torch.stack(x_out, dim=-1)
        return x

    @torch.no_grad()
    def reverse_sample(self, x0, nb_steps, return_trajectory=False):
        dt = 1 / nb_steps
        t_values = torch.linspace(1, 0, nb_steps + 1)

        x = x0.to(self.device)

        x_out = []
        for t in t_values[:-1]:
            t = t.reshape(1).to(self.device)
            x = x - self.net(x, t.repeat(x.shape[0])) * dt

            x_out.append(x.cpu())

        if return_trajectory:
            return x, torch.stack(x_out, dim=-1)
        return x

    def reflow(self, pairs0, pairs1, iterations, bsize=128):
        self.net.train()
        it = 0
        pbar = tqdm(range(iterations))

        while it < iterations:
            perm = torch.randperm(pairs0.size(0))
            pairs_train0 = pairs0.clone()[perm]
            pairs_train1 = pairs1.clone()[perm]

            for i in range(pairs_train1.shape[0] // bsize):
                x0 = pairs_train0[i * bsize:(i + 1) * bsize].to(self.device)
                x1 = pairs_train1[i * bsize:(i + 1) * bsize].to(self.device)

                target = x1 - x0
                t = torch.rand(x0.size(0)).to(self.device)

                t = t[:, None]
                interpolant = (1 - t) * x0 + t * x1

                t_emb = self.net.spe(t)
                model_output = self.net(interpolant, t_emb)

                loss = ((model_output - target)**2).mean()
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.set_postfix({"Loss": loss.item()})
                pbar.update(1)
                it += 1

            
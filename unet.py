
import gin 
from tqdm import tqdm
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
import gin

class SPE(torch.nn.Module):

    def __init__(self, dim=128, max_positions=10000, scale=20):
        super().__init__()
        self.embedding_size = dim
        self.max_positions = max_positions
        self.scale = scale

    def forward(self, x):
        x = x * self.scale
        freqs = torch.arange(start=0,
                             end=self.embedding_size // 2,
                             dtype=torch.float32,
                             device=x.device)
        w = (1 / self.max_positions)**(2 * freqs / self.embedding_size)
        w = w[None, :]
        x = x[:, None]
        x = torch.cat([torch.sin(w * x), torch.cos(w * x)], dim=-1)
        return x.squeeze(1)
    
    
@gin.configurable
class ConvBlock2D(nn.Module):

    def __init__(self,
                 in_c,
                 out_c,
                 time_channels,
                 kernel_size,
                 act=nn.SiLU,
                 res=True, 
                 normalize = False):
        super().__init__()
        self.res = res

        self.conv1 = nn.Conv2d(in_c,
                               out_c,
                               kernel_size=kernel_size,
                               padding = "same")
        
        
        self.gn1 = nn.GroupNorm(min(16, in_c//4), in_c) if normalize else nn.Identity()
        
        self.conv2 = nn.Conv2d(out_c,
                               out_c,
                               kernel_size=kernel_size,
                               padding = "same")
        
        self.gn2 = nn.GroupNorm(min(16, (out_c)//4), out_c)  if normalize else nn.Identity()
        self.act = act()

        self.time_mlp = nn.Sequential(nn.Linear(time_channels, 128), act(),
                                      nn.Linear(128, 2 * out_c))
        
        if in_c != out_c:
            self.to_out = nn.Conv2d(in_c,
                                    out_c,
                                    kernel_size=1,
                                    padding="same")
        else:
            self.to_out = nn.Identity()

    def forward(self, x, time=None):
        res = x.clone()
            
        x = self.gn1(x)
        x = self.act(x) 
        x = self.conv1(x)
        time = self.time_mlp(time)
        time_mult, time_add = torch.split(time, time.shape[-1] // 2, -1)
        x = x * time_mult[:, :, None, None] + time_add[:, :, None, None]

        x = self.gn2(x)
        x = self.act(x)
        x = self.conv2(x)
    
        if self.res:
            return x + self.to_out(res)

        return x



@gin.configurable
class EncoderBlock2D(nn.Module):

    def __init__(self,
                 in_c,
                 out_c,
                 time_channels,
                 kernel_size=3,
                 ratio=2,
                 act=nn.SiLU):
        super().__init__()
        
        self.conv = ConvBlock2D(in_c=in_c,
                                out_c=out_c,
                                time_channels=time_channels,
                                kernel_size=kernel_size,
                                act=act)

        self.pool = nn.Conv2d(out_c, out_c, stride = ratio, kernel_size=3, padding=(ratio+1)//2)
        
    def forward(self, x, time):
        skip = self.conv(x, time=time)
        out = self.pool(skip)
        return out, skip

@gin.configurable
class MiddleBlock2D(nn.Module):

    def __init__(self,
                 in_c,
                 time_channels,
                 kernel_size=3,
                 act=nn.SiLU,
    ):
        super().__init__()
        
        self.conv = ConvBlock2D(in_c=in_c,
                                out_c=in_c,
                                time_channels=time_channels,
                                kernel_size=kernel_size,
                                act=act)
        
        
            
    def forward(self, x, time):
        x = self.conv(x, time=time)
        return x

@gin.configurable
class DecoderBlock2D(nn.Module):

    def __init__(self,
                 in_c,
                 out_c,
                 time_channels,
                 kernel_size,
                 res = True,
                 act=nn.SiLU,
                 ratio=2,
                 upsample_nearest = True):
        super().__init__()
        
        if ratio == 1:
            self.up = nn.Identity()
        else:        
            if upsample_nearest==True:
                conv = nn.Conv2d(in_c, in_c, kernel_size=3, stride=1,
                          padding= "same",
                          )
                self.up = nn.Sequential(
                    nn.Upsample(mode='nearest', scale_factor=ratio),
                    conv)
            else:
                 self.up = nn.ConvTranspose2d(in_channels=in_c,
                               out_channels=in_c,
                               kernel_size=3, stride=ratio, padding=ratio//2)
                        

        self.conv = ConvBlock2D(in_c=in_c + in_c,
                                out_c=out_c,
                                time_channels=time_channels,
                                kernel_size=kernel_size,
                                res = res,
                                act=act)        

    def forward(self, x, skip=None, time=None):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x, time=time)
        return x

@gin.configurable
class UNET2D(nn.Module):

    def __init__(self,
                 in_size,
                 channels,
                 ratios,
                 kernel_size,
                 time_channels,
                 out_size = None):
        
        super().__init__()
        
        self.channels = channels
        self.in_size = in_size
    
        if out_size is None:
            out_size = in_size
        
        n = len(self.channels)

        self.time_emb = SPE(time_channels)

        self.up_layers = nn.ModuleList()
        self.down_layers = nn.ModuleList()
        
        comp_ratios = []
        cur_ratio = 1
        for r in ratios:
            cur_ratio *= r
            comp_ratios.append(cur_ratio)

        self.down_layers.append(
            EncoderBlock2D(in_c=in_size,
                           out_c=channels[0],
                           time_channels=time_channels,
                           kernel_size=kernel_size,
                           ratio=ratios[0]))
            
        

        for i in range(1, n):
            self.down_layers.append(
                EncoderBlock2D(in_c=channels[i - 1],
                               out_c=channels[i],
                               time_channels=time_channels,
                               kernel_size=kernel_size,
                               ratio=ratios[i]))
            
        
        self.middle_block = MiddleBlock2D(
            in_c=channels[-1],
            time_channels=time_channels,
            kernel_size=kernel_size)
            
            
        for i in range(1,n):
            self.up_layers.append(
                DecoderBlock2D(in_c=channels[n - i],
                               out_c=channels[n - i - 1],
                               time_channels=time_channels,
                               kernel_size=kernel_size,
                               ratio=ratios[n-i],
                               res = True))

        self.up_layers.append(
            DecoderBlock2D(in_c=channels[0],
                           out_c=out_size,
                           time_channels=time_channels,
                           kernel_size=kernel_size,
                           res = False,
                           ratio=ratios[0]))
        

    def forward(self, x, time):
        time = self.time_emb(time).to(x) if time is not None else None
        skips = []
                
        for layer in self.down_layers:
            x, skip = layer(x, time=time)
            skips.append(skip)
            
        x = self.middle_block(x, time=time)
        
        for layer in self.up_layers:
            skip = skips.pop(-1)
            x = layer(x, skip = skip, time=time)
        return x

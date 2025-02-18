import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
import gin
import cached_conv as cc
from pqmf import CachedPQMF, DummyIdentity
from .cached_conv_2d import CachedConv2d, CachedConvTranspose2d, CachedPadding2d, AlignBranches2d


class SPE(torch.nn.Module):
    def __init__(self, dim = 128, max_positions=10000, scale = 1000):
        super().__init__()
        self.embedding_size = dim
        self.max_positions = max_positions
        self.scale = scale

    def forward(self, x):
        x = self.scale * x
        freqs = torch.arange(start=0, end=self.embedding_size//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.embedding_size // 2 - 1)
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        return x

    

def __prepare_scriptable__(self):
    for hook in self._forward_pre_hooks.values():
        # The hook we want to remove is an instance of WeightNorm class, so
        # normally we would do `if isinstance(...)` but this class is not accessible
        # because of shadowing, so we check the module name directly.
        # https://github.com/pytorch/pytorch/blob/be0ca00c5ce260eb5bcec3237357f7a30cc08983/torch/nn/utils/__init__.py#L3
        if hook.__module__ == "torch.nn.utils.weight_norm" and hook.__class__.__name__ == "WeightNorm":
            print("Removing weight_norm from %s", self.__class__.__name__)
            torch.nn.utils.remove_weight_norm(self)
    return self


def normalization(module: nn.Module, mode: str = 'weight_norm'):
    if mode == 'identity':
        return module
    elif mode == 'weight_norm':
        layer = torch.nn.utils.weight_norm(module)
        layer.__prepare_scriptable__ = __prepare_scriptable__.__get__(layer)
        return layer
    else:
        raise Exception(f'Normalization mode {mode} not supported')
    
    

def Conv1d(*args, **kwargs) -> nn.Module:
    return normalization(CachedConv2d(*args, **kwargs, bias=False))


def ConvTranspose1d(*args, **kwargs) -> nn.Module:
    return normalization(CachedConvTranspose2d(*args, **kwargs, bias=False))


def Downsample2d(in_channels: int,
                 out_channels: int,
                 ratio: int,
                 freq_ratio: int,
                 kernel_multiplier: int = 2, cumulative_delay : int =0) -> nn.Module:
    assert kernel_multiplier % 2 == 0, "Kernel multiplier must be even"
    return CachedConv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(freq_ratio * kernel_multiplier + 1, ratio*kernel_multiplier+1),
            stride=(freq_ratio, ratio),
            padding=((freq_ratio+1)//2 + (1 if freq_ratio > 1 else 0),(ratio+1)//2 + (1 if ratio > 1 else 0)),
            cumulative_delay=cumulative_delay,
        )
    


@gin.configurable
class ConvBlock2D(nn.Module):

    def __init__(self,
                 in_c,
                 out_c,
                 time_cond_channels,
                 time_channels,
                 kernel_size,
                 cond_channels = 0,
                 act=nn.SiLU,
                 res=True, 
                 normalize = False,
                 cumulative_delay = 0):
        super().__init__()
        self.res = res

        self.conv1 = CachedConv2d(in_c + time_cond_channels,
                               out_c,
                               kernel_size=kernel_size,
                               padding = (kernel_size-1)//2, cumulative_delay = cumulative_delay)
        
        
        self.gn1 = nn.GroupNorm(min(16, (in_c  + time_cond_channels)//4), in_c + time_cond_channels) if normalize else nn.Identity()
        
        self.conv2 = CachedConv2d(out_c,
                               out_c,
                               kernel_size=kernel_size,
                               padding = (kernel_size-1)//2, cumulative_delay = self.conv1.cumulative_delay)
        
        self.gn2 = nn.GroupNorm(min(16, (out_c)//4), out_c)  if normalize else nn.Identity()
        self.act = act()

        self.time_mlp = nn.Sequential(nn.Linear(time_channels, 128), act(),
                                      nn.Linear(128, 2 * out_c))
        
        if cond_channels > 0:
            self.cond_mlp = nn.Sequential(nn.Linear(cond_channels, 128), act(),
                                        nn.Linear(128, 2 * out_c))
        else: 
            self.cond_mlp = None

        if in_c != out_c:
            self.to_out = CachedConv2d(in_c,
                                    out_c,
                                    kernel_size=1,
                                    padding=0, cumulative_delay = 0)
            cd_to_out = self.to_out.cumulative_delay
        else:
            self.to_out = nn.Identity()
            cd_to_out = 0
            
        self.time_shape = out_c
        self.cond_shape = out_c
        
        self.cumulative_delay = self.conv2.cumulative_delay
        
        self.delay_res = CachedPadding2d(self.cumulative_delay - (cd_to_out+cumulative_delay), crop = True)

    def __prepare_scriptable__(self):
        torch.nn.utils.remove_weight_norm(self.conv1)
        torch.nn.utils.remove_weight_norm(self.conv2)
        return self


    def forward(self, x, time=None, zsem=None, time_cond=None):
        res = x.clone()

        if time_cond is not None:
            x = torch.cat([x, time_cond], dim=1)
            
        x = self.gn1(x)
        x = self.act(x) 
        x = self.conv1(x)


        if time is not None:
            time = self.time_mlp(time)
            time_mult, time_add = torch.split(time, time.shape[-1] // 2, -1)
            x = x * time_mult[:, :, None, None] + time_add[:, :, None, None]

        
        if zsem is not None:
            zsem = self.cond_mlp(zsem)
            zsem_mult, zsem_add = torch.split(zsem, zsem.shape[-1] // 2, -1)
            x = x * zsem_mult[:, :, None, None] + zsem_add[:, :, None, None]
         

        x = self.gn2(x)
        x = self.act(x)
        x = self.conv2(x)
    
        if self.res:
            return x + self.to_out(self.delay_res(res))

        return x

@gin.configurable
class EncoderBlock2D(nn.Module):

    def __init__(self,
                 in_c,
                 out_c,
                 time_cond_channels,
                 time_channels,
                 cond_channels = 0,
                 kernel_size=3,
                 ratio=2,
                 freq_ratio = 2,
                 act=nn.SiLU,
                 cumulative_delay = 0,
                 target_delay = None):
        super().__init__()
        
        
        
        if target_delay is not None:
            if target_delay > cumulative_delay:
                self.delay_x = CachedPadding2d(target_delay - cumulative_delay, crop = True)
                self.delay_time_cond = nn.Identity()
            else:
                self.delay_time_cond = CachedPadding2d(cumulative_delay - target_delay, crop = True)
                self.delay_x = nn.Identity()
            
            cumulative_delay = max(target_delay, cumulative_delay)
            
        else:
            self.delay_x = nn.Identity()
            self.delay_time_cond = nn.Identity()
            cumulative_delay = cumulative_delay
        
        
        
        self.conv = ConvBlock2D(in_c=in_c,
                                out_c=in_c,
                                time_cond_channels=time_cond_channels,
                                time_channels=time_channels,
                                cond_channels = cond_channels,
                                kernel_size=kernel_size,
                                act=act, 
                                cumulative_delay = cumulative_delay)

            
        self.pool = Downsample2d(in_c, out_c, ratio = ratio, freq_ratio  = freq_ratio, kernel_multiplier=2, cumulative_delay = self.conv.cumulative_delay)

        self.cumulative_delay = self.pool.cumulative_delay
        
        
       #print(cumulative_delay, self.conv.cumulative_delay, self.pool.cumulative_delay)


    def forward(self, x, time = None,time_cond=None):
        x = self.delay_x(x)
        time_cond = self.delay_time_cond(time_cond) if time_cond is not None else None
        skip = self.conv(x, time=time, time_cond=time_cond)
        skip = self.pool(skip)
        #x = torch.cat([skip, time_cond], dim=1) if time_cond is not None else skip
        return skip, skip

@gin.configurable
class MiddleBlock2D(nn.Module):

    def __init__(self,
                 in_c,
                 time_cond_channels,
                 time_channels,
                 kernel_size=3,
                 act=nn.SiLU,
                 cond_channels = 0,
                 cumulative_delay = 0,
                 target_delay = 0
    ):
        super().__init__()
        
        if target_delay is not None:
            if target_delay > cumulative_delay:
                self.delay_x = CachedPadding2d(target_delay - cumulative_delay, crop = True)
                self.delay_time_cond = nn.Identity()
            else:
                self.delay_time_cond = CachedPadding2d(cumulative_delay - target_delay, crop = True)
                self.delay_x = nn.Identity()
        else:
            self.delay_x = nn.Identity()
            self.delay_time_cond = nn.Identity()
            
            cumulative_delay = cumulative_delay
        
        
        self.conv = ConvBlock2D(in_c=in_c,
                                out_c=in_c,
                                time_cond_channels=time_cond_channels,
                                time_channels=time_channels,
                                cond_channels = cond_channels,
                                kernel_size=kernel_size,
                                act=act, cumulative_delay=cumulative_delay)
        
        
        self.cumulative_delay = self.conv.cumulative_delay
            
    def forward(self, x, time=None, time_cond=None):
        x = self.delay_x(x)
        time_cond = self.delay_time_cond(time_cond) if time_cond is not None else None
        x = self.conv(x, time=time, time_cond=time_cond)
        return x

@gin.configurable
class DecoderBlock2D(nn.Module):

    def __init__(self,
                 in_c,
                 out_c,
                 time_cond_channels,
                 time_channels,
                 kernel_size,
                 res = True,
                 act=nn.SiLU,
                 ratio=2,
                 freq_ratio = 2,
                 skip_size = None,
                 use_skip = True,
                 cond_channels = 0,
                 cumulative_delay = 0,
                 target_delay = None,
                 upsample_nearest = False):
        super().__init__()
        
        self.use_skip = use_skip
        if skip_size is None and use_skip:
            skip_size = in_c
        else:
            skip_size = 0
        
        if target_delay is not None:
            if target_delay > cumulative_delay:
                self.delay_x = CachedPadding2d(target_delay - cumulative_delay, crop = True)
                self.delay_time_cond = nn.Identity()
            else:
                self.delay_time_cond = CachedPadding2d(cumulative_delay - target_delay, crop = True)
                self.delay_x = nn.Identity()
            
            cumulative_delay = max(target_delay, cumulative_delay)
            
        else:
            self.delay_x = nn.Identity()
            self.delay_time_cond = nn.Identity()
            cumulative_delay = cumulative_delay
        
        
        if ratio == 1 and freq_ratio == 1:
            self.up = nn.Identity()
            up_cumulative_delay = ratio*cumulative_delay
        else:
            #self.up = nn.ConvTranspose1d(in_c, out_c, kernel_size=2, stride=2)
        
            if upsample_nearest==True:
                conv = CachedConv2d(in_c + skip_size + time_cond_channels, in_c + skip_size + time_cond_channels, kernel_size=3, stride=1,
                          padding= 1,
                          )
                self.up = nn.Sequential(
                    nn.Upsample(mode='nearest', scale_factor=[freq_ratio,ratio]),
                    conv)
            else:
                 self.up = CachedConvTranspose2d(in_channels=in_c + skip_size + time_cond_channels,
                               out_channels=in_c + skip_size + time_cond_channels,
                               kernel_size=(2*freq_ratio + (1 if freq_ratio == 1 else 0), 2*ratio + (1 if ratio == 1 else 0)), stride=(freq_ratio, ratio), padding=(freq_ratio//2 + (1 if freq_ratio == 1 else 0), ratio//2 + (1 if ratio == 1 else 0)))
            
            
            
            up_cumulative_delay = self.up.cumulative_delay + ratio*cumulative_delay
            
        self.up_delay = up_cumulative_delay

        self.conv = ConvBlock2D(in_c=in_c + skip_size + time_cond_channels,
                                out_c=out_c,
                                time_cond_channels=0,
                                time_channels=time_channels,
                                cond_channels = cond_channels,
                                kernel_size=kernel_size,
                                res = res,
                                act=act,
                                cumulative_delay = up_cumulative_delay)
        
        self.cumulative_delay = self.conv.cumulative_delay 

    def forward(self, x, skip=None, time=None, time_cond=None):
        
        x = self.delay_x(x)
        skip = self.delay_x(skip) if skip is not None else None
        time_cond = self.delay_time_cond(time_cond) if time_cond is not None else None
        
        
        if self.use_skip:
            x = torch.cat([x, skip], dim=1)
            
        if time_cond is not None:
            x = torch.cat([x, time_cond], dim=1)
            
        x = self.up(x)
        x = self.conv(x, time=time, time_cond=None)
        return x

@gin.configurable
class UNET2D(nn.Module):

    def __init__(self,
                 in_size=128,
                 out_size = None,
                 channels=[128, 128, 128, 128],
                 time_ratios=[1, 1, 1, 2, 1],
                 freq_ratios=[2, 2, 2, 2, 1],
                 kernel_size=3,
                 time_channels=64,
                 time_cond_channels=None,
                 cond_channels=32,
                 use_skip = True,
                 bottleneck = nn.Identity,
                 target_delays = None):
        
        super().__init__()
        
        self.channels = channels
        self.time_cond_channels = time_cond_channels
        
        self.bottleneck = bottleneck()
        
        self.in_size = in_size
        if out_size is None:
            out_size = in_size
        self.use_skip = use_skip
        
        if time_cond_channels is None:
            time_cond_channels = [0]*len(channels)

        n = len(self.channels)
        
        if time_channels == 0:
            self.time_emb = lambda _: torch.empty(0)
        else:
            self.time_emb = SPE(time_channels)

        self.up_layers = nn.ModuleList()
        self.down_layers = nn.ModuleList()


        self.down_layers.append(
            EncoderBlock2D(in_c=in_size,
                           out_c=channels[0],
                           time_channels=time_channels,
                           time_cond_channels=0,
                           cond_channels=cond_channels,
                           kernel_size=kernel_size,
                           ratio=time_ratios[0],
                           freq_ratio = freq_ratios[0],
                           cumulative_delay=0,
                           target_delay = 0))
        comp_ratios = []
        cur_ratio = 1
        for r in time_ratios:
            cur_ratio *= r
            comp_ratios.append(cur_ratio)

        for i in range(1, n):
            self.down_layers.append(
                EncoderBlock2D(in_c=channels[i - 1],
                               out_c=channels[i],
                               time_channels=time_channels,
                               time_cond_channels=time_cond_channels[i-1],
                               cond_channels=cond_channels,
                               kernel_size=kernel_size,
                               ratio=time_ratios[i],
                               freq_ratio = freq_ratios[i],
                               cumulative_delay=self.down_layers[-1].cumulative_delay,
                               target_delay = target_delays[n-i] if target_delays is not None else None))
            
            
        
        self.middle_block = MiddleBlock2D(
            in_c=channels[-1],
            time_channels=time_channels,
            cond_channels=cond_channels,
            time_cond_channels=0,#time_cond_channels[-1],
            kernel_size=kernel_size,
            cumulative_delay=self.down_layers[-1].cumulative_delay,
            target_delay =None)# target_delays[0] if target_delays is not None else None)
            
            
        for i in range(1,n):
            self.up_layers.append(
                DecoderBlock2D(in_c=channels[n - i],
                               out_c=channels[n - i - 1],
                               time_channels=time_channels,
                               time_cond_channels=time_cond_channels[n-i],
                               cond_channels=cond_channels,
                               kernel_size=kernel_size,
                               ratio=time_ratios[n-i],
                               freq_ratio = freq_ratios[n-i],
                               res = True,
                               use_skip = self.use_skip,
                               cumulative_delay=self.up_layers[-1].cumulative_delay if i>1 else self.middle_block.cumulative_delay,
                               target_delay = target_delays[i-1] if target_delays is not None else None))

        self.up_layers.append(
            DecoderBlock2D(in_c=channels[0],
                           out_c=out_size,
                           skip_size = None,
                           time_channels=time_channels,
                           time_cond_channels=time_cond_channels[0],
                           cond_channels=cond_channels,
                           kernel_size=kernel_size,
                           res = False,
                           use_skip = self.use_skip,
                           ratio=time_ratios[0],
                           freq_ratio = freq_ratios[0],
                           cumulative_delay = self.up_layers[-1].cumulative_delay,
                           target_delay = target_delays[-1] if target_delays is not None else None))
        
        self.delays = nn.ModuleList()
        
        
        for i in range(0, n):
            cur_skip_delay = self.down_layers[i].cumulative_delay
            cur_up_delay = self.up_layers[n-i-2].cumulative_delay if n-i-2>=0 else self.middle_block.cumulative_delay
            self.delays.append(AlignBranches2d(nn.Identity(), nn.Identity(), delays = [cur_skip_delay - cur_up_delay, 0]))
                        
        self.cumulative_delay = self.up_layers[-1].cumulative_delay
        

    def forward(self, x, time=None, time_conds=None):
        time = self.time_emb(time).to(x) if time is not None else None
        skips = []
                
        if time_conds is None:
            time_conds = [None]*len(self.down_layers)
            
        time_conds = time_conds + [None]
                    
        for layer, time_cond in zip(self.down_layers, reversed(time_conds[1:])):

            x, skip = layer(x, time=time, time_cond=time_cond)
            skips.append(skip)
            #print("downloop", x.shape)
            
        x = self.middle_block(x, time=time)
        
        x = self.bottleneck(x)
        #print("middle", x.shape)
       
        for layer, delay, time_cond in zip(self.up_layers, reversed(self.delays), time_conds):
            if self.use_skip:
                skip = skips.pop(-1)
                skip = delay(skip)[0]
            else:
                skip = None
            x = layer(x, skip = skip, time=time, time_cond = time_cond)
            #print(x.shape)
        return x
    
    def encode(self, x):
        
        for layer in self.down_layers:
            x, _ = layer(x)
            
        x = self.middle_block(x)
        x = self.bottleneck(x)
        return x
    
    def decode(self, x):
        out = [x]
        for layer in self.up_layers[:-1]:
            x = layer(x)
            out.append(x.clone())
        return out
    


@gin.configurable
class UNET_AE2D(nn.Module):
    def __init__(self, unet_encoder, unet):
        super().__init__()
        self.unet_encoder =  unet_encoder
        self.unet = unet
        
    def to(self, device):
        self.unet_encoder.to(device)
        self.unet.to(device)        
        
    def encode(self, x):
        z = self.unet_encoder.encode(x)
        latents = self.unet_encoder.decode(z)
        return latents

    def forward(self, x, time, x_encode=None):
        latents = self.encode(x_encode)
        out = self.unet(x, time = time, time_conds=latents)
        return out
    
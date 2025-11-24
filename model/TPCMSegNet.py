import torch
import torch.nn as nn
import math
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.models.layers import DropPath, trunc_normal_
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
import clip
#from utils.text_process import get_text
import json

def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
        h_w: Height of window
        w_w: Width of window
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B,windows.shape[2], H, W)
    return x

class Fuse(nn.Module):
    def __init__(self,dim):
        super().__init__()

        self.process = nn.Conv2d(dim*2,dim,1)



    def forward(self,x_in, x_recive):
        x = torch.cat([x_in,x_recive],dim=1)
        x = self.process(x)

        x = x + x_recive
        return x        


class Downsample(nn.Module):
    """
    Down-sampling block"
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
                 ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x



class Upsample(nn.Module):
    """
    Up-sampling block"
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
                 ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = dim // 2
            self.expansion = nn.Sequential(
                        nn.ConvTranspose2d(dim, dim_out, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            # nn.BatchNorm2d(dim_out)  # Optionally add batch normalization after upsampling
        )

    def forward(self, x):
        x = self.expansion(x)
        return x





class PatchEmbed(nn.Module):
    """
    Patch embedding block"
    """

    def __init__(self, in_chans=3, in_dim=32, dim=64):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        # in_dim = 1
        super().__init__()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 1, 1),
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 1, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU()
            )

    def forward(self, x):
        x = self.conv_down(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self, dim,
                 drop_path=0.,
                 layer_scale=None,
                 kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = input + self.drop_path(x)
        return x


class MambaVisionMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True, 
        layer_idx=None,
        device=None,
        dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)    
        self.x_proj = nn.Linear(
            self.d_inner//2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner//2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner//2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner//2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner//2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner//2,
            out_channels=self.d_inner//2,
            bias=conv_bias//2,
            kernel_size=d_conv,
            groups=self.d_inner//2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        # ??????
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same', groups=self.d_inner//2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        # ?????x ???? ??????silu
        # x = F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same', groups=self.d_inner//2)
        # x = x*z # Hadamard product simzhang added
        
        y = selective_scan_fn(x, 
                              dt, 
                              A, 
                              B, 
                              C, 
                              self.D.float(), 
                              z=None, 
                              delta_bias=self.dt_proj.bias.float(), 
                              delta_softplus=True, 
                              return_last_state=None)
        
        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out
    

class Attention(nn.Module):

    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_norm=False,
            attn_drop=0.,
            proj_drop=0.,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = True

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
             q, k, v,
                dropout_p=self.attn_drop.p,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, 
                 dim, 
                 num_heads, 
                 counter, 
                 transformer_blocks, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=False, 
                 drop=0., 
                 attn_drop=0.,
                 drop_path=0., 
                 act_layer=nn.GELU, 
                 norm_layer=nn.LayerNorm, 
                 Mlp_block=Mlp,
                 layer_scale=None,
                 ):
        super().__init__()

        self.norm1 = norm_layer(dim)
        if counter in transformer_blocks:
            self.mixer = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
            norm_layer=norm_layer,
        )
        else:
            self.mixer = MambaVisionMixer(d_model=dim, 
                                          d_state=8,  
                                          d_conv=3,    
                                          expand=1
                                          )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        

    def forward(self, x):
        x = x + self.drop_path(self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class MambaVisionLayer(nn.Module):
    """
    MambaVision layer"
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 conv=False,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 transformer_blocks = [],
    ):
        super().__init__()
        self.conv = conv
        self.transformer_block = False
        if conv:
            self.blocks = nn.ModuleList([ConvBlock(dim=dim,
                                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                                   layer_scale=layer_scale_conv)
                                                   for i in range(depth)])
            self.transformer_block = False
        else:
            self.blocks = nn.ModuleList([Block(dim=dim,
                                               counter=i, 
                                               transformer_blocks=transformer_blocks,
                                               num_heads=num_heads,
                                               mlp_ratio=mlp_ratio,
                                               qkv_bias=qkv_bias,
                                               qk_scale=qk_scale,
                                               drop=drop,
                                               attn_drop=attn_drop,
                                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                               layer_scale=layer_scale)
                                               for i in range(depth)])
            self.transformer_block = True

        self.downsample = None if not downsample else Downsample(dim=dim)
        self.do_gt = False
        self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape
        

        if not self.transformer_block:
            for id, blk in enumerate(self.blocks):
                x = blk(x)
                if id == len(self.blocks)//2:
                    x2 = x
            if self.downsample is None:
                return x, x2
            return self.downsample(x), x2
        


        if self.transformer_block:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = torch.nn.functional.pad(x, (0,pad_r,0,pad_b))
                _, _, Hp, Wp = x.shape
            else:
                Hp, Wp = H, W
            x = window_partition(x, self.window_size)
            residual = x
            for _, blk in enumerate(self.blocks):
                x = blk(x)
        
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()

            if self.downsample is None:
                return x
            return self.downsample(x)





class MambaVisionLayer_up(nn.Module):
    """
    MambaVision layer"
    """

    def __init__(self,
                 dim,
                 depth,
                 num_heads,
                 window_size,
                 conv=False,
                 upsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 transformer_blocks = [],
    ):
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            window_size: window size in each stage.
            conv: bool argument for conv stage flag.
            upsample: bool argument for up-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
            layer_scale_conv: conv layer scaling coefficient.
            transformer_blocks: list of transformer blocks.
        """

        super().__init__()
        self.conv = conv
        self.transformer_block = False
        if conv:
            self.blocks = nn.ModuleList([ConvBlock(dim=dim,
                                                   drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                                   layer_scale=layer_scale_conv)
                                                   for i in range(depth)])
            self.transformer_block = False
        else:
            self.blocks = nn.ModuleList([Block(dim=dim,
                                               counter=i, 
                                               transformer_blocks=transformer_blocks,
                                               num_heads=num_heads,
                                               mlp_ratio=mlp_ratio,
                                               qkv_bias=qkv_bias,
                                               qk_scale=qk_scale,
                                               drop=drop,
                                               attn_drop=attn_drop,
                                               drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                               layer_scale=layer_scale)
                                               for i in range(depth)])
            self.transformer_block = True

        self.upsample = None if not upsample else Upsample(dim=dim*2)
        self.downsample = None if not upsample else Downsample(dim=dim)
        self.do_gt = False
        self.window_size = window_size

    def forward(self, x, skip):
        if self.upsample is not None:
            x = self.upsample(x)+skip

        _, _, H, W = x.shape

        if not self.transformer_block:
            for id, blk in enumerate(self.blocks):
                x = blk(x)
                if id == len(self.blocks)//2:
                    x2=x
                    if self.downsample is not None:
                        x2 = self.downsample(x2)         
            return x, x2

        if self.transformer_block:
            pad_r = (self.window_size - W % self.window_size) % self.window_size
            pad_b = (self.window_size - H % self.window_size) % self.window_size
            if pad_r > 0 or pad_b > 0:
                x = torch.nn.functional.pad(x, (0,pad_r,0,pad_b))
                _, _, Hp, Wp = x.shape
            else:
                Hp, Wp = H, W
            x = window_partition(x, self.window_size)

            for _, blk in enumerate(self.blocks):
                x = blk(x)
        
            x = window_reverse(x, self.window_size, Hp, Wp)
            if pad_r > 0 or pad_b > 0:
                x = x[:, :, :H, :W].contiguous()

            
            return x


class Layer(nn.Module):

    def __init__(
            self,
            dim,
            depth,
            num_heads,
            window_size,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop,
            attn_drop,
            drop_path,
            downsample,
            layer_scale,
            layer_scale_conv,
            transformer_blocks,
            ):
        super().__init__()
        conv = True
        self.level_cnn = MambaVisionLayer(dim=dim,
                                     depth=depth,
                                     num_heads=num_heads,
                                     window_size=window_size,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     conv=conv,
                                     drop=drop,
                                     attn_drop=attn_drop,
                                     drop_path=drop_path,
                                     downsample=downsample,
                                     layer_scale=layer_scale,
                                     layer_scale_conv=layer_scale_conv,
                                     transformer_blocks=transformer_blocks,
                                     )
        conv = False
        self.level_mt =  MambaVisionLayer(dim=dim,
                                     depth=depth,
                                     num_heads=num_heads,
                                     window_size=window_size,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     conv=conv,
                                     drop=drop,
                                     attn_drop=attn_drop,
                                     drop_path=drop_path,
                                     downsample=downsample,
                                     layer_scale=layer_scale,
                                     layer_scale_conv=layer_scale_conv,
                                     transformer_blocks=transformer_blocks,
                                     )
        self.fuse1 = Fuse(dim)
        if downsample:
            dim = dim*2
        self.fuse2 = Fuse(dim)
    def forward(self, x, mt):
        # mt = x
        x1, x2 = self.level_cnn(x)
        mt = self.fuse1(x2, mt)
        mt = self.level_mt(mt)
        x = self.fuse2(mt,x1)
        return x, mt    



class Layer_UP(nn.Module):

    def __init__(
            self,
            dim,
            depth,
            num_heads,
            window_size,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop,
            attn_drop,
            drop_path,
            upsample,
            layer_scale,
            layer_scale_conv,
            transformer_blocks,
            ):
        super().__init__()
        conv = True
        self.level_cnn_up = MambaVisionLayer_up(dim=dim,
                                     depth=depth,
                                     num_heads=num_heads,
                                     window_size=window_size,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     conv=conv,
                                     drop=drop,
                                     attn_drop=attn_drop,
                                     drop_path=drop_path,
                                     upsample=upsample,
                                     layer_scale=layer_scale,
                                     layer_scale_conv=layer_scale_conv,
                                     transformer_blocks=transformer_blocks,
                                     )
        conv = False
        self.level_mt_up =  MambaVisionLayer_up(dim=dim,
                                     depth=depth,
                                     num_heads=num_heads,
                                     window_size=window_size,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     conv=conv,
                                     drop=drop,
                                     attn_drop=attn_drop,
                                     drop_path=drop_path,
                                     upsample=upsample,
                                     layer_scale=layer_scale,
                                     layer_scale_conv=layer_scale_conv,
                                     transformer_blocks=transformer_blocks,
                                     )
        if not upsample:
            self.fuse1 = Fuse(dim)
        else:
            self.fuse1 = Fuse(dim*2)
        self.fuse2 = Fuse(dim)
        
    def forward(self, inx, x, mt, skip_cnn, skip_mt):

        # print(f"x: {x.shape}, skip: {skip_cnn[2-inx].shape}")
        # print(x)
        x1, x2 = self.level_cnn_up(x,skip_cnn[2-inx])
        mt = self.fuse1(x2, mt)

        mt = self.level_mt_up(mt,skip_mt[2-inx])
        # print(f'skip{inx}: {skip_cnn[-inx].shape}')
        x = self.fuse2(mt,x1)
        # x = x1+mt
        return x, mt



class network(nn.Module):
    """
    MambaVision sim for unet,
    """

    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 mlp_ratio,
                 num_heads,
                 drop_path_rate=0.2,
                 in_chans=1,
                 num_classes=9,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.2,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 **kwargs):
        
        super().__init__()
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # each depth drop rates
        self.pos_drop = nn.Dropout(p=drop_rate) # train drop rate
        self.encoder_depths = depths
        self.decoder_depths = depths[::-1]  #???depths
        
        self.level =  nn.ModuleList()
        for i in range(len(depths)): # i --> 0,1,2,3
            level = Layer(dim=int(dim * 2 ** i), # up layer ?? ???layer ??dim ?????????
                            depth=depths[i],
                            num_heads=num_heads[i],
                            window_size=window_size[i],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop=drop_rate,
                            attn_drop=attn_drop_rate,
                            drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                            downsample=(i < 3),
                            layer_scale=layer_scale,
                            layer_scale_conv=layer_scale_conv,
                            transformer_blocks=list(range(depths[i]//2+1, depths[i])) if depths[i]%2!=0 else list(range(depths[i]//2, depths[i])),
                            )
            self.level.append(level)

        self.level_up = nn.ModuleList()
        for i in range(len(depths)-1):
            level_up = Layer_UP(dim=int( dim * 2 ** (2-i)),
                                     depth=depths[2-i],
                                     num_heads=num_heads[2-i],
                                     window_size=window_size[2-i],
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     drop=drop_rate,
                                     attn_drop=attn_drop_rate,
                                     drop_path=dpr[sum(self.decoder_depths[:i]):sum(self.decoder_depths[:i + 1])],
                                     upsample=True,
                                     layer_scale=layer_scale,
                                     layer_scale_conv=layer_scale_conv,
                                     transformer_blocks=list(range(depths[2-i]//2+1, depths[2-i])) if depths[2-i]%2!=0 else list(range(depths[2-i]//2, depths[2-i])), 
                                     )
            self.level_up.append(level_up)   
    
        self.norm_layer = nn.LayerNorm
        self.final_up = Final_PatchExpand2D(dim=dim, dim_scale=4, norm_layer=self.norm_layer)
        self.final_conv = nn.Conv2d(dim*4, num_classes, 1)

        self.image_pool = nn.MaxPool2d(2)
        self.classify_conv_stem = nn.Conv2d(dim,num_classes,1)
        self.cascade_fuse_stem = nn.Conv2d(in_chans+num_classes,dim,1)

        self.classify_conv_x = nn.ModuleList()
        self.cascade_fuse_x = nn.ModuleList()
        self.classify_conv_mt = nn.ModuleList()
        self.cascade_fuse_mt = nn.ModuleList()

        for i in range(len(depths)-1):
            cls_conv_x = nn.Conv2d(dim*2**(i+1),num_classes,1)
            cls_conv_mt = nn.Conv2d(dim*2**(i+1),num_classes,1)
            cas_fuse_x = nn.Conv2d(in_chans+num_classes,dim*2**(i+1),1)
            cas_fuse_mt = nn.Conv2d(in_chans+num_classes,dim*2**(i+1),1)
            self.classify_conv_x.append(cls_conv_x) 
            self.cascade_fuse_x.append(cas_fuse_x)  
            self.classify_conv_mt.append(cls_conv_mt) 
            self.cascade_fuse_mt.append(cas_fuse_mt)

        self.classify_conv_x.append(nn.Identity()) 
        self.cascade_fuse_x.append(nn.Identity())  
        self.classify_conv_mt.append(nn.Identity()) 
        self.cascade_fuse_mt.append(nn.Identity())  

        self.clip_model, _ = clip.load("ViT-B/32", device='cpu', jit=False) 

        
        self.fuse_text_encoder = nn.ModuleList()
        self.text_liner_encoder = nn.ModuleList()
        for i in range(len(depths)):
            fuse_encoder = FuseModule(img_dim=int(dim * 2 ** i), text_dim=int(dim * 2 ** i), num_heads=8)
            text_liner = nn.Linear(512,int(dim * 2 ** i))
            self.fuse_text_encoder.append(fuse_encoder)
            self.text_liner_encoder.append(text_liner)


        self.fuse_text_decoder = nn.ModuleList()
        self.text_liner_decoder = nn.ModuleList()
        for i in range(len(depths)-1):
            fuse_decoder = FuseModule(img_dim=int(dim * 2 ** (3-i)), text_dim=int(dim * 2 ** (3-i)), num_heads=8)
            text_liner = nn.Linear(512,int(dim * 2 ** (3-i)))
            self.fuse_text_decoder.append(fuse_decoder)
            self.text_liner_decoder.append(text_liner)

        self.fuse_decoder_cls = FuseModule(img_dim=int(dim), text_dim=int(dim), num_heads=8)
        self.text_liner_cls = nn.Linear(512,int(dim))

        self.apply(self._init_weights)


        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}

    def forward_features(self, x, text):

        image = []
        skip_list_cnn = []
        skip_list_mt = []
        x_image = x

        for i in range(3):
            x_image = self.image_pool(x_image)
            image.append(x_image)
        
        image.append(x_image)
        image_stem = x
        x = self.patch_embed(x) 
        x = self.pos_drop(x)
        x_feature = x
        x = self.classify_conv_stem(x) #predict
        x = torch.cat([image_stem,x],dim=1)
        x = self.cascade_fuse_stem(x)
        x = x + x_feature
        mt = x


        for inx, (level, fuse, text_liner, image, cls_x, cls_mt, cascade_x, cascade_mt) in enumerate(zip(self.level, self.fuse_text_encoder, self.text_liner_encoder,image,
                                                                   self.classify_conv_x, self.classify_conv_mt, self.cascade_fuse_x, self.cascade_fuse_mt)):

            skip_list_cnn.append(x)
            skip_list_mt.append(mt)

            text_feature = text_liner(text)
            text_feature = text_feature.unsqueeze(1)
            x, mt = level(fuse(x,text_feature), fuse(mt,text_feature))
            
            if inx <3:
                x_feature = x
                mt_feature = mt
                x = cascade_x(torch.cat([image,cls_x(x)],dim=1)) + x_feature
                mt = cascade_mt(torch.cat([image,cls_mt(mt)],dim=1)) + mt_feature   

        return x, mt ,skip_list_cnn, skip_list_mt


    def forward_features_up(self, x, mt, skip_list_cnn, skip_list_mt, text):
        for inx, (level_up, fuse, text_liner) in enumerate(zip(self.level_up, self.fuse_text_decoder, self.text_liner_decoder)):
            
            text_feature = text_liner(text)
            text_feature = text_feature.unsqueeze(1)

            x = fuse(x,text_feature)
            mt = fuse(mt,text_feature)
            x, mt = level_up(inx, x, mt, skip_list_cnn, skip_list_mt)

        text_feature = self.text_liner_cls(text)
        text_feature = text_feature.unsqueeze(1)
        x = self.fuse_decoder_cls(x,text_feature)
        mt = self.fuse_decoder_cls(mt,text_feature)

        return x + skip_list_cnn[0], mt + skip_list_mt[0]

    def forward_final(self, x):
        x = self.final_up(x)

        x = self.final_conv(x)
        return x

    def forward(self, x, text):
    
        with torch.no_grad():
            text_features = self.clip_model.encode_text(text)  

        x, mt, skip_list_cnn, skip_list_mt = self.forward_features(x, text_features)

        x, mt = self.forward_features_up(x, mt, skip_list_cnn, skip_list_mt, text_features)

        x = self.forward_final(x+mt)
        
        return x


class CrossAttentionModule(nn.Module):
    def __init__(self, dim_q, dim_kv, num_heads=8):
        super(CrossAttentionModule, self).__init__()
        self.num_heads = num_heads
        self.dim_q = dim_q
        self.dim_kv = dim_kv
        self.dim_head_q = dim_q // num_heads
        self.dim_head_kv = dim_kv // num_heads

        self.q_linear = nn.Linear(dim_q, dim_q)
        self.k_linear = nn.Linear(dim_kv, dim_q)
        self.v_linear = nn.Linear(dim_kv, dim_q)
        self.out_linear = nn.Linear(dim_q, dim_q)

    def forward(self, query, key, value):

        batch_size = query.size(0)

        Q = self.q_linear(query)  
        K = self.k_linear(key)    
        V = self.v_linear(value)  
    
        Q = Q.view(batch_size, -1, self.num_heads, self.dim_head_q).transpose(1, 2)  
        K = K.view(batch_size, -1, self.num_heads, self.dim_head_q).transpose(1, 2)  
        V = V.view(batch_size, -1, self.num_heads, self.dim_head_q).transpose(1, 2) 
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.dim_head_q ** 0.5)  
        attn = F.softmax(scores, dim=-1)  
        context = torch.matmul(attn, V)   

        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.dim_head_q) 

        out = self.out_linear(context)  

        return out

class FuseModule(nn.Module):
    def __init__(self, img_dim, text_dim, num_heads=8):
        super(FuseModule, self).__init__()
        self.cross_attn = CrossAttentionModule(dim_q=img_dim, dim_kv=text_dim, num_heads=num_heads)
        self.linear = nn.Linear(img_dim, img_dim)
        self.norm = nn.LayerNorm(img_dim)

    def forward(self, img_features, text_features):

        batch_size, channels, H, W = img_features.shape
        img_seq = img_features.view(batch_size, channels, -1).permute(0, 2, 1)  
        N = img_seq.size(1)
        text_seq = text_features  
        fused_seq = self.cross_attn(img_seq, text_seq, text_seq)  
        fused_seq = self.norm(fused_seq + img_seq)  
        fused_features = fused_seq.permute(0, 2, 1).contiguous().view(batch_size, channels, H, W)  

        return fused_features



class Final_PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale*self.dim, bias=False)
        self.norm = norm_layer(self.dim * dim_scale)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0,2,3,1)  # b h w c
        #print(f'before liner: {x.shape}')
        x = self.expand(x)
        #print(f'after liner: {x.shape}')
        # x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//self.dim_scale)
        #x = rearrange(x, 'b h w c -> b h w (c)', c=C//self.dim_scale)
        x= self.norm(x)
        x = x.permute(0,3,1,2)  # b c h w

        return x



class TPCMSegNet(nn.Module):
    
    def __init__(self, 
                    input_channels=1,
                    num_classes=9,
                    depths=[2, 4, 4, 8],
                    num_heads=[2, 4, 8, 16],
                    window_size=[8, 8, 14, 7],
                    dim=64,
                    in_dim=64,
                    mlp_ratio=4,
                    resolution=224,
                    drop_path_rate=0.2,
                    load_ckpt_path=None,
                    **kwargs):
        
        super().__init__()
        
        self.load_ckpt_path = load_ckpt_path
        self.num_classes = num_classes
        
        self.model = network(
                depths=depths,
                num_heads=num_heads,
                window_size=window_size,
                in_chans=input_channels,
                num_classes = num_classes,
                dim=dim,
                in_dim=in_dim,
                mlp_ratio=mlp_ratio,
                resolution=resolution,
                drop_path_rate=drop_path_rate,
        )
    
    def forward(self, x, text):
        return self.model(x, text)


    
    
    
    
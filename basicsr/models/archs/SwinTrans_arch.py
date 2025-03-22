'''
SwinEventDeblur: A Swin Transformer-based architecture for event-based deblurring
Created by: Bingyu-Huang (2025-03-22)
For NTIRE 2025 Event Deblurring Challenge
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
from einops import rearrange

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CrossWindowAttention(nn.Module):
    """Cross Window Attention between event and image features"""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # Cross-modal projection layers
        self.q_img = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_event = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_img, x_event, mask=None):
        """
        Args:
            x_img: image features [B_, N, C]
            x_event: event features [B_, N, C]
            mask: mask for attention [nW, N, N] or None
        """
        B_, N, C = x_img.shape

        q = self.q_img(x_img).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # B_, nH, N, C/nH
        kv = self.kv_event(x_event).reshape(B_, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]  # B_, nH, N, C/nH

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B_, nH, N, N

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class EventImageSwinTransformerBlock(nn.Module):
    """Swin Transformer Block with cross-attention between event and image features"""

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        # Normalization layers
        self.norm1_img = norm_layer(dim)
        self.norm1_event = norm_layer(dim)
        self.norm2 = norm_layer(dim)

        # Cross-modal attention
        self.cross_attn = CrossWindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # DropPath and MLP
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Attention mask for shifted window attention
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x_img, x_event):
        """
        Args:
            x_img: image features [B, H*W, C]
            x_event: event features [B, H*W, C]
        """
        H, W = self.input_resolution
        B, L, C = x_img.shape
        assert L == H * W, "input feature has wrong size"

        # Save original inputs for residual connections
        img_shortcut = x_img

        # Apply normalization
        x_img = self.norm1_img(x_img)
        x_event = self.norm1_event(x_event)

        # Reshape to spatial dimensions
        x_img = x_img.view(B, H, W, C)
        x_event = x_event.view(B, H, W, C)

        # Apply cyclic shift
        if self.shift_size > 0:
            shifted_img = torch.roll(x_img, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            shifted_event = torch.roll(x_event, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_img = x_img
            shifted_event = x_event

        # Partition windows
        img_windows = window_partition(shifted_img, self.window_size)  # nW*B, window_size, window_size, C
        event_windows = window_partition(shifted_event, self.window_size)

        img_windows = img_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        event_windows = event_windows.view(-1, self.window_size * self.window_size, C)

        # Cross-modal window attention
        attn_windows = self.cross_attn(img_windows, event_windows,
                                       mask=self.attn_mask)  # nW*B, window_size*window_size, C

        # Merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # Reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # Apply FFN with residual connection
        x = img_shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class PatchMerging(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim)

    def forward(self, x):
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=1, c=C // 2)
        x = x.view(B, -1, C // 2)

        x = self.norm(x)
        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16 * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(16 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // (self.dim_scale ** 2))
        x = x.view(B, -1, C // (self.dim_scale ** 2))
        x = self.norm(x)

        return x


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = torch.utils.checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class EventImageFusionLayer(nn.Module):
    """A Swin Transformer layer that fuses event and image data using cross-attention"""

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # Self-attention blocks for event stream
        self.event_blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # Cross-attention blocks between event and image
        self.fusion_blocks = nn.ModuleList([
            EventImageSwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                           num_heads=num_heads, window_size=window_size,
                                           shift_size=0 if (i % 2 == 0) else window_size // 2,
                                           mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias, qk_scale=qk_scale,
                                           drop=drop, attn_drop=attn_drop,
                                           drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                           norm_layer=norm_layer)
            for i in range(depth)])

        # Downsampling operations
        if downsample is not None:
            self.downsample_img = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
            self.downsample_event = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample_img = None
            self.downsample_event = None

    def forward(self, x_img, x_event):
        """
        Forward pass with both image and event features

        Args:
            x_img: Image features [B, H*W, C]
            x_event: Event features [B, H*W, C]

        Returns:
            Tuple of processed image and event features
        """
        if torch.is_grad_enabled() and self.training:
            print(f"[FusionLayer] Input - x_img: {x_img.shape}, x_event: {x_event.shape}")

        # Process event stream with self-attention
        for i, event_blk in enumerate(self.event_blocks):
            if self.training and torch.is_grad_enabled():
                print(f"[FusionLayer] Before event block {i} - event shape: {x_event.shape}")

            if self.use_checkpoint:
                x_event = torch.utils.checkpoint.checkpoint(event_blk, x_event)
            else:
                x_event = event_blk(x_event)

            if self.training and torch.is_grad_enabled():
                print(f"[FusionLayer] After event block {i} - event shape: {x_event.shape}")
                print(f"[FusionLayer] Before fusion block {i} - img: {x_img.shape}, event: {x_event.shape}")

            # After each event self-attention, apply cross-attention fusion
            fusion_blk = self.fusion_blocks[i]
            if self.use_checkpoint:
                x_img = torch.utils.checkpoint.checkpoint(fusion_blk, x_img, x_event)
            else:
                x_img = fusion_blk(x_img, x_event)

            if self.training and torch.is_grad_enabled():
                print(f"[FusionLayer] After fusion block {i} - img shape: {x_img.shape}")

        # Apply downsampling if needed
        if self.downsample_img is not None:
            if self.training and torch.is_grad_enabled():
                print(f"[FusionLayer] Before downsampling - img: {x_img.shape}, event: {x_event.shape}")

            x_img = self.downsample_img(x_img)
            x_event = self.downsample_event(x_event)

            if self.training and torch.is_grad_enabled():
                print(f"[FusionLayer] After downsampling - img: {x_img.shape}, event: {x_event.shape}")

        return x_img, x_event


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x


class SkipAdapter(nn.Module):
    def __init__(self, in_dim, out_dim, input_resolution):
        super().__init__()
        self.input_resolution = input_resolution
        self.adapter = nn.Linear(in_dim, out_dim) if in_dim != out_dim else nn.Identity()
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x, target_shape):
        B, L, C = x.shape
        target_L = target_shape[1]

        # If sequence lengths match, only adapt channels
        if L == target_L:
            return self.norm(self.adapter(x))

        # Handle sequence length mismatch
        H, W = self.input_resolution
        # Calculate target resolution
        target_H = int(math.sqrt(target_L))
        target_W = target_L // target_H

        # Reshape to 2D, adapt spatial dimensions, then reshape back
        x = x.transpose(1, 2).view(B, C, H, W)
        x = F.interpolate(x, size=(target_H, target_W), mode='bilinear', align_corners=False)
        x = x.flatten(2).transpose(1, 2)

        # Adapt channels if needed
        x = self.adapter(x)
        return self.norm(x)

class SwinEventDeblur(nn.Module):
    """Swin Transformer based architecture for event-based deblurring

    Args:
        img_size (int | tuple(int)): Input image size. Default 256
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        event_chans (int): Number of input event channels. Default: 6 (voxel grid)
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=256, patch_size=4, in_chans=3, event_chans=6,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=8, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, skip_fusion=True, **kwargs):
        super().__init__()

        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.skip_fusion = skip_fusion

        # Split image into non-overlapping patches
        self.patch_embed_img = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # Event patch embedding
        self.patch_embed_event = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=event_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # Calculate properties
        num_patches = self.patch_embed_img.num_patches
        patches_resolution = self.patch_embed_img.patches_resolution
        self.patches_resolution = patches_resolution

        # Absolute position embedding (optional)
        if self.ape:
            self.absolute_pos_embed_img = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            self.absolute_pos_embed_event = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed_img, std=.02)
            trunc_normal_(self.absolute_pos_embed_event, std=.02)

        self.pos_drop_img = nn.Dropout(p=drop_rate)
        self.pos_drop_event = nn.Dropout(p=drop_rate)

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Build encoder layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = EventImageFusionLayer(
                dim=int(embed_dim * 2 ** i_layer),
                input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                  patches_resolution[1] // (2 ** i_layer)),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        # Skip connection adapters
        self.skip_adapters = nn.ModuleList()
        for i_layer in range(self.num_layers - 1):
            in_dim = int(embed_dim * 2 ** i_layer)
            out_dim = int(embed_dim * 2 ** (self.num_layers - i_layer - 1))
            input_resolution = (patches_resolution[0] // (2 ** i_layer),
                                patches_resolution[1] // (2 ** i_layer))

            self.skip_adapters.append(SkipAdapter(in_dim, out_dim, input_resolution))

        # Build decoder layers
        self.decoder_layers = nn.ModuleList()
        for i_layer in range(self.num_layers - 1, -1, -1):
            dim = int(embed_dim * 2 ** i_layer)
            input_resolution = (patches_resolution[0] // (2 ** i_layer),
                                patches_resolution[1] // (2 ** i_layer))

            if i_layer > 0:
                layer = EventImageFusionLayer(
                    dim=dim,
                    input_resolution=input_resolution,
                    depth=depths[i_layer],
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                    norm_layer=norm_layer,
                    downsample=None,
                    use_checkpoint=use_checkpoint)
                self.decoder_layers.append(layer)

                # Upsampling layer
                expand = PatchExpand(input_resolution=input_resolution,
                                     dim=dim,
                                     dim_scale=2,
                                     norm_layer=norm_layer)
                self.decoder_layers.append(expand)
            else:
                # Final upsampling to original resolution
                final_layer = EventImageFusionLayer(
                    dim=dim,
                    input_resolution=input_resolution,
                    depth=1,  # Reduced depth for the final layer
                    num_heads=num_heads[i_layer],
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate,
                    drop_path=0.0,  # No drop path in final layer
                    norm_layer=norm_layer,
                    downsample=None,
                    use_checkpoint=use_checkpoint)
                self.decoder_layers.append(final_layer)

                # Final patch expanding for full resolution
                final_expand = FinalPatchExpand_X4(
                    input_resolution=input_resolution,
                    dim=dim,
                    dim_scale=patch_size,
                    norm_layer=norm_layer)
                self.decoder_layers.append(final_expand)

        # Final normalization
        self.norm = norm_layer(self.num_features)

        # Final convolutional layer to produce output image
        self.final_conv = nn.Sequential(
            nn.Conv2d(embed_dim, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, in_chans, kernel_size=3, padding=1)
        )

        # SAM (Supervised Attention Module) for intermediate supervision
        self.sam = nn.Sequential(
            nn.Conv2d(embed_dim, 32, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, in_chans, kernel_size=3, padding=1)
        )

        self.apply(self._init_weights)

        # Skip connection adapters
        self.skip_adapters = nn.ModuleList()
        for i_layer in range(self.num_layers):  # Change from range(self.num_layers - 1)
            in_dim = int(embed_dim * 2 ** i_layer)
            # For the last layer, we need a special case since there's no next layer
            if i_layer == self.num_layers - 1:
                out_dim = in_dim  # Same dimension for the last skip connection
            else:
                out_dim = int(embed_dim * 2 ** (self.num_layers - i_layer - 1))

            input_resolution = (patches_resolution[0] // (2 ** i_layer),
                                patches_resolution[1] // (2 ** i_layer))

            self.skip_adapters.append(SkipAdapter(in_dim, out_dim, input_resolution))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed_img', 'absolute_pos_embed_event'}

    def forward_features(self, img, event):
        """Forward pass through feature extraction part of the network"""
        if self.training:
            print(f"[Features] Input shapes - img: {img.shape}, event: {event.shape}")

        # Embed patches
        img_features = self.patch_embed_img(img)  # B, L, C
        event_features = self.patch_embed_event(event)  # B, L, C

        if self.training:
            print(f"[Features] After patch embedding - img: {img_features.shape}, event: {event_features.shape}")

        # Add position embedding if used
        if self.ape:
            img_features = img_features + self.absolute_pos_embed_img
            event_features = event_features + self.absolute_pos_embed_event

        img_features = self.pos_drop_img(img_features)
        event_features = self.pos_drop_event(event_features)

        # Store skip connections
        img_skips = []
        event_skips = []

        # Forward through encoder layers
        for i, layer in enumerate(self.layers):
            img_skips.append(img_features)
            event_skips.append(event_features)
            if self.training:
                print(f"[Features] Before layer {i} - img: {img_features.shape}, event: {event_features.shape}")

            img_features, event_features = layer(img_features, event_features)

            if self.training:
                print(f"[Features] After layer {i} - img: {img_features.shape}, event: {event_features.shape}")

        # Normalize features
        img_features = self.norm(img_features)

        if self.training:
            print(f"[Features] Final normalized img features: {img_features.shape}")

        return img_features, event_features, img_skips, event_skips

    def forward_decoder(self, img_features, event_features, img_skips, event_skips):
        """Forward pass through decoder part of the network"""
        if self.training:
            print(f"[Decoder] Initial shapes - img: {img_features.shape}, event: {event_features.shape}")
            print(f"[Decoder] Skip connection counts - img: {len(img_skips)}, event: {len(event_skips)}")
            print(f"[Decoder] Number of skip adapters: {len(self.skip_adapters)}")

        # Loop through decoder layers
        i = 0
        sam_output = None

        while i < len(self.decoder_layers):
            # Process fusion layer if it's a transformer block
            if isinstance(self.decoder_layers[i], EventImageFusionLayer):
                # Add skip connections if available and enabled
                # Combine with skip connection
                if len(img_skips) > 0 and self.skip_fusion:
                    skip_idx = len(img_skips) - 1

                    # Ensure skip_idx is within range of self.skip_adapters
                    if skip_idx < len(self.skip_adapters):
                        skip_img = self.skip_adapters[skip_idx](img_skips.pop(), img_features.shape)
                        skip_event = self.skip_adapters[skip_idx](event_skips.pop(), event_features.shape)

                        if self.training:
                            print(
                                f"[Decoder] Skip connection {skip_idx} - skip_img: {skip_img.shape}, skip_event: {skip_event.shape}")

                        img_features = img_features + skip_img
                        event_features = event_features + skip_event
                    else:
                        # Skip connection index out of range, just pop and discard
                        if self.training:
                            print(f"[Decoder] Skip connection {skip_idx} - SKIPPED (adapter index out of range)")
                        img_skips.pop()
                        event_skips.pop()


                if self.training:
                    print(
                        f"[Decoder] Before fusion layer {i} - img: {img_features.shape}, event: {event_features.shape}")

                # Process with fusion layer
                img_features, event_features = self.decoder_layers[i](img_features, event_features)

                if self.training:
                    print(
                        f"[Decoder] After fusion layer {i} - img: {img_features.shape}, event: {event_features.shape}")

                # Save intermediate output for SAM (at 1/4 resolution)
                if i == len(self.decoder_layers) - 2:  # Before the final expansion
                    B, L, C = img_features.shape
                    H = W = int(math.sqrt(L))
                    sam_features = img_features.transpose(1, 2).view(B, C, H, W)
                    sam_output = self.sam(sam_features)

                    if self.training:
                        print(f"[Decoder] SAM features shape: {sam_features.shape}, output: {sam_output.shape}")

                i += 1

            # Process expansion layer
            elif isinstance(self.decoder_layers[i], (PatchExpand, FinalPatchExpand_X4)):
                if self.training:
                    print(
                        f"[Decoder] Before expansion layer {i} - img: {img_features.shape}, event: {event_features.shape}")

                img_features = self.decoder_layers[i](img_features)
                event_features = self.decoder_layers[i](event_features)

                if self.training:
                    print(
                        f"[Decoder] After expansion layer {i} - img: {img_features.shape}, event: {event_features.shape}")

                i += 1

        # Final processing
        B, L, C = img_features.shape
        H = W = int(math.sqrt(L))
        output_features = img_features.transpose(1, 2).view(B, C, H, W)

        if self.training:
            print(f"[Decoder] Final output features shape: {output_features.shape}")

        return output_features, sam_output

    def forward(self, x, event):
        if self.training:
            print(f"[Forward] Input image shape: {x.shape}")
            print(f"[Forward] Input event shape: {event.shape}")

        img = x
        # Forward through feature extraction
        img_features, event_features, img_skips, event_skips = self.forward_features(img, event)

        if self.training:
            print(f"[Forward] After features - img_features shape: {img_features.shape}")
            print(f"[Forward] After features - event_features shape: {event_features.shape}")
            print(f"[Forward] Number of image skip connections: {len(img_skips)}")

        # Forward through decoder
        decoded_features, sam_output = self.forward_decoder(img_features, event_features, img_skips, event_skips)

        if self.training:
            print(f"[Forward] Decoded features shape: {decoded_features.shape}")
            if sam_output is not None:
                print(f"[Forward] SAM output shape: {sam_output.shape}")

        # Final output
        output = self.final_conv(decoded_features)
        if self.training:
            print(f"[Forward] Before residual connection shape: {output.shape}")

        output = output + img  # Residual connection

        if self.training:
            print(f"[Forward] Final output shape: {output.shape}")
            return [sam_output, output]  # Return both outputs during training
        else:
            return output  # Return only final output during inference
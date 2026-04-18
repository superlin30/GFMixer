import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace

import torch.backends.cuda
import torch.nn.functional as F
from torch import einsum

from typing import (
    Callable,
    Dict,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    cast,
    Union
)

import math


class RotaryEmbedding(nn.Module):
    """
    [Rotary positional embeddings (RoPE)](https://arxiv.org/abs/2104.09864).
    """

    def __init__(
        self, 
        config,
        #cache: BufferCache, 
        dim: int = None,
        n_channels: int = None,
        prefix: str = "attn",
        freq_distribution: str = None,
        freq_learnable: bool = None,
        use_rope_cache: bool = True,
        include_neg_freq: bool = None,
        floor_freq_ratio: float = None,
        clamp_floor_freq: bool = None,
        clamp_floor_to_zero: bool = None,
        upper_freq_ratio: float = None,
        clamp_upper_freq: float = None,
        clamp_upper_to_zero: bool = None,
        zero_freq_ratio: float = None,
        clamp_to_linear: bool = None,
        clamp_to_linear_mode: str = None,
        init_upper_freq: float = None,
        init_floor_freq: float = None,
    ):
        super().__init__()
        self.config = config
        #self.__cache = cache
        
        self.prefix = prefix
        self.suffix = "rope"
        self.use_rope_cache = use_rope_cache
        
        self.freq_distribution = freq_distribution if freq_distribution is not None else self.config.rope_init_distribution
        if self.freq_distribution not in ["constant", "linear", "uniform", "gaussian", "exponential"]:
            raise ValueError(f"Unsupported frequency distribution: {self.freq_distribution}")
        
        self.freq_learnable = freq_learnable if freq_learnable is not None else self.config.rope_learnable
        self.include_neg_freq = include_neg_freq if include_neg_freq is not None else self.config.rope_include_neg_freq
        
        if dim is not None:
            self.dim = dim
        elif self.prefix == "attn":
            self.dim = self.config.d_model // self.config.n_heads
        else:
            self.dim = self.config.d_model
        
        if n_channels is not None:
            self.n_channels = n_channels
        elif self.prefix == "attn" and self.config.rope_learnable and self.config.rope_no_repetition:
            self.n_channels = self.config.n_heads
        else:
            self.n_channels = 1
            
        self.init_floor_freq = init_floor_freq if init_floor_freq is not None else self.config.rope_init_floor_freq
        self.init_upper_freq = init_upper_freq if init_upper_freq is not None else self.config.rope_init_upper_freq
        
        self.clamp_floor_freq = clamp_floor_freq if clamp_floor_freq is not None else self.config.rope_clamp_floor_freq
        if self.clamp_floor_freq:
            self.floor_freq_ratio = floor_freq_ratio if floor_freq_ratio is not None else self.config.rope_floor_freq_ratio
            self.floor_freq = 2*torch.pi/self.config.max_sequence_length * self.floor_freq_ratio
            
            self.clamp_floor_to_zero = clamp_floor_to_zero if clamp_floor_to_zero is not None else self.config.rope_clamp_floor_to_zero
            self.clamp_floor_value = 0.0 if self.clamp_floor_to_zero else 2*torch.pi/self.config.max_sequence_length*self.clamp_floor_ratio
        else:
            self.floor_freq = 0.0
        
        self.clamp_upper_freq = clamp_upper_freq if clamp_upper_freq is not None else self.config.rope_clamp_upper_freq
        if self.clamp_upper_freq:
            self.upper_freq_ratio = upper_freq_ratio if upper_freq_ratio is not None else self.config.rope_upper_freq_ratio
            self.upper_freq = torch.pi * self.upper_freq_ratio
            
            self.clamp_upper_to_zero = clamp_upper_to_zero if clamp_upper_to_zero is not None else self.config.rope_clamp_upper_to_zero
            self.clamp_upper_value = 0.0 if self.clamp_upper_to_zero else torch.pi * self.clamp_upper_ratio
        else:
            self.upper_freq = 1.0
            
        if self.clamp_floor_freq or self.clamp_upper_freq:
            assert self.upper_freq >= self.floor_freq
        
        self.zero_freq_ratio = zero_freq_ratio if zero_freq_ratio is not None else self.config.rope_zero_freq_ratio
        
        self.clamp_to_linear = clamp_to_linear if clamp_to_linear is not None else self.config.rope_clamp_to_linear
        self.clamp_to_linear_mode = clamp_to_linear_mode if clamp_to_linear_mode is not None else self.config.rope_clamp_to_linear_mode
        
        if self.freq_learnable:
            self.inv_freq = nn.Parameter(
                self.get_inv_freq(self.dim, _non_meta_init_device(config)),
                requires_grad=True
            )
        else:
            self.inv_freq = self.get_inv_freq(self.dim, _non_meta_init_device(config))
            
            if self.use_rope_cache:
                # Warm up cache.
                self.get_rotary_embedding(config.max_sequence_length, _non_meta_init_device(config))


    
        
    def extra_yarn(self, inv_freq = None) -> torch.Tensor:
        assert self.config.len_extra_orig_length != 0
        
        def find_correction_dim(num_rotations, dim, base=self.config.rope_theta, orig_max_position_embeddings=self.config.len_extra_orig_length):
            return (dim * math.log(orig_max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base))

        def find_correction_range(low_rot, high_rot, dim, base=self.config.rope_theta, orig_max_position_embeddings=self.config.len_extra_orig_length):
            low = math.floor(find_correction_dim(
                low_rot, dim, base, orig_max_position_embeddings))
            high = math.ceil(find_correction_dim(
                high_rot, dim, base, orig_max_position_embeddings))
            return max(low, 0), min(high, dim-1)  # Clamp values just in case

        def linear_ramp_mask(low, fast, dim):
            if low == fast:
                fast += 0.001  # Prevent singularity
            linear_func = (torch.arange(dim, dtype=torch.float32) - low) / (fast - low)
            ramp_func = torch.clamp(linear_func, 0, 1)
            return ramp_func
        
        if inv_freq is None:
            pos_freqs = self.config.rope_theta ** (torch.arange(0, self.dim, 2, device=_non_meta_init_device(self.config), dtype=torch.float) / self.dim)
            inv_freq_extra = 1.0 / pos_freqs
            inv_freq_inter = 1.0 / (self.config.len_extra_yarn_scale * pos_freqs)
        else:
            inv_freq_extra = inv_freq
            inv_freq_inter = inv_freq / self.config.len_extra_yarn_scale
       
        low, high = find_correction_range(self.config.len_extra_yarn_beta_fast, self.config.len_extra_yarn_beta_slow, self.dim)
        inv_freq_mask = (1 - linear_ramp_mask(low, high, self.dim // 2).float().to(_non_meta_init_device(self.config))) * self.config.len_extra_yarn_factor
        inv_freq = inv_freq_inter * (1 - inv_freq_mask) + inv_freq_extra * inv_freq_mask
        
        return inv_freq
    
    def extra_pi(self, inv_freq = None) -> torch.Tensor:
        assert self.config.len_extra_orig_length != 0
            
        self.pi_ratio = self.config.len_extra_orig_length / self.config.max_sequence_length
        
        inv_freq = inv_freq * self.pi_ratio
        
        return inv_freq
    
    def get_inv_freq(self, dim: int, device: torch.device) -> torch.Tensor:
        if self.freq_distribution == "constant": # self.config.rope_no_rotary:
            torch.ones_like(torch.arange(0, dim, 2, device=device, dtype=torch.float))
        
        #if self.freq_distribution == "constant":
            #inv_freq = torch.ones(dim//2, device=device, dtype=torch.float)

        elif self.freq_distribution == "linear": # self.config.rope_linear:
            inv_freq = 2*torch.pi/self.config.max_sequence_length * torch.arange(0, dim, 2, device=device, dtype=torch.float) 
            
            inv_freq[inv_freq > self.clamp_upper_ratio * torch.pi] = self.clamp_upper_value
            
            inv_freq = inv_freq.flip(0)
            
        elif self.freq_distribution == "uniform": # self.config.rope_uniform:
            inv_freq = 1.0 * torch.rand(dim//2, device=device, dtype=torch.float)
            
        elif self.freq_distribution == "gaussian": # self.config.rope_gaussian:
            inv_freq = torch.randn(dim//2, device=device, dtype=torch.float).abs()
            inv_freq = inv_freq / inv_freq.max()
        else:
            inv_freq = 1.0 / (
                self.config.rope_theta ** (torch.arange(0, dim, 2, device=device, dtype=torch.float) / dim)
            )
        
        inv_freq = self.init_floor_freq + inv_freq * (self.init_upper_freq - self.init_floor_freq)
        
        if self.config.len_extra and self.config.len_extra_before_clamp:
            if self.config.len_extra_type == "PI":
                inv_freq = self.extra_pi(inv_freq)
        
            elif self.config.len_extra_type == "YARN":
                inv_freq = self.extra_yarn(inv_freq)
        
        if self.clamp_floor_freq:
            inv_freq[inv_freq < self.floor_freq] = self.clamp_floor_value
        if self.clamp_upper_freq:
            inv_freq[inv_freq > self.upper_freq] = self.clamp_upper_value
            
        if self.clamp_to_linear:
            freq_ratio = inv_freq / (2*torch.pi/self.config.max_sequence_length)
            if self.clamp_to_linear_mode == "ceil":
                inv_freq = 2*torch.pi/self.config.max_sequence_length * freq_ratio.ceil()
            elif self.clamp_to_linear_mode == "floor":
                inv_freq = 2*torch.pi/self.config.max_sequence_length * freq_ratio.floor()
            elif self.clamp_to_linear_mode == "half":
                _half = (freq_ratio != 0).float() * 0.5
                inv_freq = 2*torch.pi/self.config.max_sequence_length * (freq_ratio.floor() + _half)
            elif self.clamp_to_linear_mode == "arange":
                _linear = (freq_ratio != 0).float()
                _linear[_linear!=0] = torch.arange(0, 1, 1/_linear.sum(), device=device, dtype=torch.float)
                
                inv_freq = 2*torch.pi/self.config.max_sequence_length * (freq_ratio.floor() + _linear)    
            elif self.clamp_to_linear_mode == "flip_arange":
                _linear = (freq_ratio != 0).float()
                _linear[_linear!=0] = (torch.arange(0, 1, 1/_linear.sum(), device=device, dtype=torch.float)).flip(0)
                
                inv_freq = 2*torch.pi/self.config.max_sequence_length * (freq_ratio.floor() + _linear)  
            else:
                raise ValueError(f"Unsupported clamp_to_linear_mode: {self.clamp_to_linear_mode}")
        
        if self.zero_freq_ratio >= 0.0:
            zero_freq_num = (inv_freq == 0.0).sum()
            zero_freq_num_ceil = math.ceil(dim//2 * self.zero_freq_ratio)
            
            if zero_freq_num > zero_freq_num_ceil:
                zero_freq_indeces = torch.nonzero(inv_freq == 0.0, as_tuple=False).squeeze()
                reset_freq_num = zero_freq_num - zero_freq_num_ceil

                inv_freq[zero_freq_indeces[:reset_freq_num]] = \
                    2*torch.pi/self.config.max_sequence_length*torch.arange(1, reset_freq_num+1, device=device, dtype=torch.float)
            else:
                non_zero_freq_indeces = torch.nonzero(inv_freq != 0.0, as_tuple=False).squeeze()
                reset_freq_num = zero_freq_num_ceil - zero_freq_num
                
                if self.clamp_upper_to_zero:
                    inv_freq[non_zero_freq_indeces[:reset_freq_num]] = 0.0
                else:
                    inv_freq[non_zero_freq_indeces[-reset_freq_num:]] = 0.0
                    
        if self.config.len_extra and not self.config.len_extra_before_clamp:
            if self.config.len_extra_type == "PI":
                inv_freq = self.extra_pi(inv_freq)
        
            elif self.config.len_extra_type == "YARN":
                inv_freq = self.extra_yarn(inv_freq)
        
        if self.include_neg_freq:
            inv_freq *= (-1) ** torch.arange(0, dim//2, device=device, dtype=torch.float)
        
        if self.config.fourier and self.config.fourier_ignore_zero:
            inv_freq = inv_freq[inv_freq != 0.0]
        
        if self.prefix == "embed":
            inv_freq = inv_freq # shape: (dim//2, )
        elif self.prefix == "attn":
            if self.config.rope_learnable and self.config.rope_no_repetition:
                inv_freq = inv_freq.repeat(self.n_channels, 1) # shape: (n_heads, dim//2)
            else:
                inv_freq = inv_freq[None, :] # shape: (1, dim//2)
        else:
            raise ValueError(f"Unsupported prefix: {self.prefix}")
        
        return inv_freq
        
    def get_rotary_embedding(
        self, 
        seq_len: int, 
        device: torch.device, 
        layer_idx: Optional[int] = None,
        use_rope_cache: bool = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        use_rope_cache = use_rope_cache or ((not self.freq_learnable) and self.use_rope_cache)
        
        if use_rope_cache:
            if (
                (pos_sin := self.__cache.get(f"{self.prefix}_{self.suffix}_pos_sin")) is not None
                and (pos_cos := self.__cache.get(f"{self.prefix}_{self.suffix}_pos_cos")) is not None
                and pos_sin.shape[-2] >= seq_len
                and pos_cos.shape[-2] >= seq_len
            ):
                if pos_sin.device != device:
                    pos_sin = pos_sin.to(device)
                    self.__cache[f"{self.prefix}_{self.suffix}_pos_sin"] = pos_sin
                if pos_cos.device != device:
                    pos_cos = pos_cos.to(device)
                    self.__cache[f"{self.prefix}_{self.suffix}_pos_cos"] = pos_cos
                    
                if self.prefix == "embed":
                    return pos_sin[:, :seq_len, :], pos_cos[:, :seq_len, :]
                elif self.prefix == "attn":
                    return pos_sin[:, :, :seq_len, :], pos_cos[:, :, :seq_len, :]
                else:
                    raise ValueError(f"Unsupported prefix: {self.prefix}")

        with torch.autocast(device.type, enabled=False):
            
            if self.freq_learnable:
                if self.clamp_floor_freq or self.clamp_upper_freq:
                    sign = self.inv_freq.data.sign()
                    self.inv_freq.data.abs_().clamp_(
                        2*torch.pi/self.config.max_sequence_length*self.clamp_floor_ratio, torch.pi*self.clamp_upper_ratio
                    ).mul_(sign)
                else:
                    self.inv_freq.data.clamp_(-torch.pi, torch.pi)
            
            if self.config.rope_no_pos:
                seq = torch.ones(seq_len, device=device, dtype=torch.float) # ablation
            else:
                seq = torch.arange(seq_len, device=device, dtype=torch.float) # original
            
            if self.prefix == "embed":
                freqs = torch.einsum("t, d -> td", seq, self.inv_freq) # shape: (seq_len, dim//2)
            elif self.prefix == "attn":
                freqs = torch.einsum("t, hd -> htd", seq, self.inv_freq) # shape: (1 or n_heads, seq_len, dim//2)
            else:
                raise ValueError(f"Unsupported prefix: {self.prefix}")
            
            if self.suffix == "fourier": 
                positions = freqs.unsqueeze(0) # shape: (1, 1 or n_heads, seq_len, dim//2) or (1, seq_len, dim//2)
            else:
                positions = torch.cat((freqs, freqs), dim=-1).unsqueeze(0) # shape: (1, 1 or n_heads, seq_len, dim) or (1, seq_len, dim)
                
            pos_sin, pos_cos = positions.sin(), positions.cos()

        if (not self.freq_learnable) and self.use_rope_cache:
            self.__cache[f"{self.prefix}_{self.suffix}_pos_sin"] = pos_sin
            self.__cache[f"{self.prefix}_{self.suffix}_pos_cos"] = pos_cos
            
        return pos_sin, pos_cos

    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        if self.prefix == "embed":
            B, T, hs = x.size()
            x = x.view(B, T, 2, hs // 2)
            
        elif self.prefix == "attn":
            B, nh, T, hs = x.size()
            x = x.view(B, nh, T, 2, hs // 2)
            
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        if not inverse:
            return ((t * pos_cos) + (self.rotate_half(t) * pos_sin)).to(t.dtype)
        else:
            return ((t * pos_cos) - (self.rotate_half(t) * pos_sin)).to(t.dtype)
    
    def forward(
        self, 
        x: torch.Tensor, 
        all_len: int, 
        layer_idx: Optional[int] = None, 
        inverse: bool = False,
        use_rope_cache: bool = None
    ) -> torch.Tensor:
        
        if self.config.rope_full_precision:
            x_ = x.float()
        else:
            x_ = x
        
        with torch.autocast(x.device.type, enabled=False):
            x_len = x_.shape[-2]
            pos_sin, pos_cos = self.get_rotary_embedding(all_len, x_.device, layer_idx=layer_idx, use_rope_cache=use_rope_cache)
            pos_sin = pos_sin.type_as(x_)
            pos_cos = pos_cos.type_as(x_)
            
            if self.prefix == "embed":
                x_ = self.apply_rotary_pos_emb(
                    pos_sin[:, all_len - x_len : x_len, :], 
                    pos_cos[:, all_len - x_len : x_len, :], 
                    x_,
                    inverse
                )
            elif self.prefix == "attn":
                x_ = self.apply_rotary_pos_emb(
                    pos_sin[:, :, all_len - x_len : all_len, :], 
                    pos_cos[:, :, all_len - x_len : all_len, :], 
                    x_,
                    inverse
                )
            else:
                raise ValueError(f"Unsupported prefix: {self.prefix}")
            
        return x_.type_as(x)




class FourierEmbedding(RotaryEmbedding):
    """
    [FOPE](https://arxiv.org/pdf/2412.17739).
    """
    def __init__(self, config, *args, **kwargs):
        self.config = config
        
        self.head_dim = self.config.d_model // self.config.n_heads
        dim = self.config.fourier_dim if self.config.fourier_dim > self.head_dim else self.head_dim
        
        super().__init__(config, dim=dim, *args, **kwargs)
        
        self.suffix = "fourier"
        
        if self.config.fourier_ignore_zero:
            self.input_dim = self.inv_freq.size(-1)
            self.output_dim = min(self.input_dim, self.head_dim//4) # TODO: self.head_dim//8
        else:
            self.input_dim = self.dim // 2
            self.output_dim = self.head_dim // 2
       

        
        if self.prefix == "embed":
            self.input_shape = "btD"
            self.output_shape = "btd"
        elif self.prefix == "attn":
            self.input_shape = "bhtD"
            self.output_shape = "bhtd"
            
        if self.prefix == "attn" and self.config.fourier_separate_head:
            size = (self.config.n_heads, self.input_dim, self.output_dim)
            self.coef_shape = "hDd"
        else:
            size = (self.input_dim, self.output_dim)
            self.coef_shape = "Dd"
        
        if self.config.fourier_separate_basis:
            self.sin_coef = nn.Parameter(
                torch.randn(size=size, device=_non_meta_init_device(self.config), dtype=torch.float),
                requires_grad=self.config.fourier_learnable
            )
            self.cos_coef = nn.Parameter(
                torch.randn(size=size, device=_non_meta_init_device(self.config), dtype=torch.float),
                requires_grad=self.config.fourier_learnable
            )
        else:
            self.fourier_coef = nn.Parameter(
                torch.randn(size=size, device=_non_meta_init_device(self.config), dtype=torch.float),
                requires_grad=self.config.fourier_learnable
            )
        
        self.reset_parameters()
    
    def apply_rotary_pos_emb(self, pos_sin, pos_cos, t, inverse = False):
        if self.config.fourier_separate_basis:
            if self.config.fourier_norm:
                fourier_sin = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_sin, self.sin_coef / self.sin_coef.sum(dim=-2, keepdim=True))
                fourier_cos = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_cos, self.cos_coef / self.cos_coef.sum(dim=-2, keepdim=True))
            else:
                fourier_sin = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_sin, self.sin_coef)
                fourier_cos = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_cos, self.cos_coef)
        else:
            if self.config.fourier_norm:
                fourier_sin = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_sin, self.fourier_coef / self.fourier_coef.sum(dim=-2, keepdim=True))
                fourier_cos = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_cos, self.fourier_coef / self.fourier_coef.sum(dim=-2, keepdim=True))
            else:
                fourier_sin = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_sin, self.fourier_coef)
                fourier_cos = torch.einsum(f"{self.input_shape}, {self.coef_shape} -> {self.output_shape}", pos_cos, self.fourier_coef)
        
        if self.config.fourier_ignore_zero:
            fourier_sin = F.pad(input=fourier_sin, pad=(0, self.head_dim//2-fourier_sin.size(-1)), mode="constant", value=1)
            fourier_cos = F.pad(input=fourier_cos, pad=(0, self.head_dim//2-fourier_cos.size(-1)), mode="constant", value=1)
        
        fourier_sin = torch.cat((fourier_sin, fourier_sin), dim=-1)
        fourier_cos = torch.cat((fourier_cos, fourier_cos), dim=-1)
        
        if not inverse:
            return ((t * fourier_cos) + (self.rotate_half(t) * fourier_sin)).to(t.dtype)
        else:
            return ((t * fourier_cos) - (self.rotate_half(t) * fourier_sin)).to(t.dtype)
    
    def get_step_eye(self, _param):
        _param = torch.zeros_like(_param)
        
        step = math.ceil(self.input_dim / self.output_dim)
        for i in range(self.output_dim):
            if i*step < self.input_dim:
                _param[..., i*step, i] = 1.0
        
        return _param
    
    def reset_parameters(self):
        with torch.no_grad():
            if self.config.fourier_separate_basis:
                if self.config.fourier_init == "eye":
                    if self.input_dim == self.output_dim:
                        if self.prefix == "attn" and self.config.fourier_separate_head:
                            for i in range(self.sin_coef.size(0)):
                                torch.nn.init.eye_(self.sin_coef[i]) 
                                torch.nn.init.eye_(self.cos_coef[i])
                        else:
                            torch.nn.init.eye_(self.sin_coef)
                            torch.nn.init.eye_(self.cos_coef)
                    else:   
                        self.sin_coef.data = self.get_step_eye(self.sin_coef)
                        self.cos_coef.data = self.get_step_eye(self.cos_coef)
                        
                elif self.config.fourier_init == "eye_norm":
                    torch.nn.init.normal_(self.sin_coef, std=self.config.fourier_init_norm_gain)
                    torch.nn.init.normal_(self.cos_coef, std=self.config.fourier_init_norm_gain)
                    
                    if self.input_dim == self.output_dim:
                        self.sin_coef += torch.eye(self.input_dim, device=self.sin_coef.device)
                        self.cos_coef += torch.eye(self.input_dim, device=self.cos_coef.device)
                    else:
                        self.sin_coef += self.get_step_eye(self.sin_coef)
                        self.cos_coef += self.get_step_eye(self.cos_coef)
                    
                elif self.config.fourier_init == "eye_xavier_norm":
                    torch.nn.init.xavier_normal_(self.sin_coef, gain=self.config.fourier_init_norm_gain)
                    torch.nn.init.xavier_normal_(self.cos_coef, gain=self.config.fourier_init_norm_gain)
                    
                    if self.input_dim == self.output_dim:    
                        self.sin_coef += torch.eye(self.input_dim, device=self.sin_coef.device)
                        self.cos_coef += torch.eye(self.input_dim, device=self.cos_coef.device)
                    else:
                        self.sin_coef += self.get_step_eye(self.sin_coef)
                        self.cos_coef += self.get_step_eye(self.cos_coef)
                
                elif self.config.fourier_init == "eye_xavier_uniform":
                    torch.nn.init.xavier_uniform_(self.sin_coef, gain=self.config.fourier_init_norm_gain)
                    torch.nn.init.xavier_uniform_(self.cos_coef, gain=self.config.fourier_init_norm_gain)
                    
                    if self.input_dim == self.output_dim:
                        self.sin_coef += torch.eye(self.input_dim, device=self.sin_coef.device)
                        self.cos_coef += torch.eye(self.input_dim, device=self.cos_coef.device)
                    else:
                        self.sin_coef += self.get_step_eye(self.sin_coef)
                        self.cos_coef += self.get_step_eye(self.cos_coef)
                
                elif self.config.fourier_init == "xavier_norm":
                    torch.nn.init.xavier_normal_(self.sin_coef)
                    torch.nn.init.xavier_normal_(self.cos_coef)
                elif self.config.fourier_init == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(self.sin_coef)
                    torch.nn.init.xavier_uniform_(self.cos_coef)
                else:
                    raise ValueError(f"Unsupported init method: {self.config.fourier_init}")
            else:
                if self.config.fourier_init == "eye":
                    
                    if self.input_dim == self.output_dim:
                        if self.prefix == "attn" and self.config.fourier_separate_head:
                            for i in range(self.fourier_coef.size(0)):
                                torch.nn.init.eye_(self.fourier_coef[i])
                        else:
                            torch.nn.init.eye_(self.fourier_coef)
                    else:
                            
                        self.fourier_coef.data = self.get_step_eye(self.fourier_coef)
                        
                elif self.config.fourier_init == "eye_norm":
                    torch.nn.init.normal_(self.fourier_coef, std=self.config.fourier_init_norm_gain)
                    
                    if self.input_dim == self.output_dim:    
                        self.fourier_coef += torch.eye(self.input_dim, device=self.fourier_coef.device)
                    else:
                        self.fourier_coef += self.get_step_eye(self.fourier_coef)
                    
                elif self.config.fourier_init == "eye_xavier_norm":
                    torch.nn.init.xavier_normal_(self.fourier_coef, gain=self.config.fourier_init_norm_gain)
                    
                    if self.input_dim == self.output_dim:
                        self.fourier_coef += torch.eye(self.input_dim, device=self.fourier_coef.device)
                    else:
                        self.fourier_coef += self.get_step_eye(self.fourier_coef)
                    
                elif self.config.fourier_init == "eye_xavier_uniform":
                    torch.nn.init.xavier_uniform_(self.fourier_coef, gain=self.config.fourier_init_norm_gain) 
                    
                    if self.input_dim == self.output_dim:
                        self.fourier_coef += torch.eye(self.input_dim, device=self.fourier_coef.device)
                    else:
                        self.fourier_coef += self.get_step_eye(self.fourier_coef)
                            
                elif self.config.fourier_init == "xavier_norm":
                    torch.nn.init.xavier_normal_(self.fourier_coef)
                elif self.config.fourier_init == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(self.fourier_coef)
                else:
                    raise ValueError(f"Unsupported init method: {self.config.fourier_init}")

    def __repr__(self):
        return f"{self.__class__.__name__}(fourier_dim={self.input_dim})"
    
def _non_meta_init_device(config):
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

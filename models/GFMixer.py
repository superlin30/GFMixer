__all__ = ['GFMixer']

# Cell
from typing import Optional

from torch import nn
from torch import Tensor
import torch.nn.functional as F
from layers.GFMixerBackbone import GFMixerBackbone

from layers.Conv_Blocks import Inception_Block_V1
import torch
from layers.TGB import TGB





class Model(nn.Module):
    def __init__(self, configs, max_seq_len: Optional[int] = 1024, d_k: Optional[int] = None, d_v: Optional[int] = None,
                 norm: str = 'BatchNorm', attn_dropout: float = 0.,
                 act: str = "gelu", key_padding_mask: bool = 'auto', padding_var: Optional[int] = None,
                 attn_mask: Optional[Tensor] = None, res_attention: bool = True,
                 pre_norm: bool = False, store_attn: bool = False, pe: str = 'zeros', learn_pe: bool = True,
                 pretrain_head: bool = False, head_type='flatten', verbose: bool = False, **kwargs):

        super().__init__()

        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len

        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout

        individual = configs.individual

        add = configs.add
        wo_conv = configs.wo_conv
        serial_conv = configs.serial_conv

        kernel_list = configs.kernel_list
        patch_len = configs.patch_len
        period = configs.period
        stride = configs.stride

        padding_patch = configs.padding_patch

        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        

        num_kernels = configs.num_kernels
        batch = configs.batch_size
        use_fat=configs.use_FAT
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

        # model
        self.model_with_FAB = GFMixerBackbone(c_in=c_in, context_window=context_window, target_window=target_window,
                                           wo_conv=wo_conv, serial_conv=serial_conv, add=add,
                                           patch_len=patch_len, kernel_list=kernel_list, period=period, stride=stride,
                                           max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                           n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm,
                                           attn_dropout=attn_dropout,
                                           dropout=dropout, act=act, key_padding_mask=key_padding_mask,
                                           padding_var=padding_var,
                                           attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm,
                                           store_attn=store_attn,
                                           pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout,
                                           head_dropout=head_dropout,
                                           padding_patch=padding_patch,
                                           pretrain_head=pretrain_head, head_type=head_type, individual=individual,
                                           revin=revin, affine=affine,
                                           subtract_last=subtract_last, verbose=verbose,use_fat =use_fat, **kwargs)   
        
        self.use_tgb = bool(getattr(configs, 'TGB', 0))
        self.tgb = TGB(configs) if self.use_tgb else None

        self.alpha = nn.Parameter(torch.tensor(0.5))  # Learnable gating parameter




    def forward(self, x):  # x: [Batch, Input length, Channel]

        B, T, N = x.size() # [Batch, Input length, Channel]
        x_fab= self.model_with_FAB(x.permute(0, 2, 1))

        if self.use_tgb:
            x_out = x_fab + self.tgb(x)                  
        else:
            x_out = x_fab                                 
       
        

        return x_out.permute(0, 2, 1)





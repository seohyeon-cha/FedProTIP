'''
 * Copyright (c) 2022, salesforce.com, inc.
 * All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 * For full license text, see LICENSE.txt file in the repo root or https://opensource.org/licenses/BSD-3-Clause
 * By Junnan Li
 * Based on timm code base
 * https://github.com/rwightman/pytorch-image-models/tree/master/timm
 * Prompt modifications from CODA_P code by James Seale Smith
'''
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math 
from typing import Any, Callable, Dict, Optional, Set, Tuple, Type, Union, List
from collections import OrderedDict

from timm.models.vision_transformer import _cfg, PatchEmbed
from timm.layers import resample_abs_pos_embed
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath
from timm.models.helpers import named_apply, adapt_input_conv
import matplotlib.pyplot as plt
#from fairscale.nn.checkpoint.checkpoint_activations import checkpoint_wrapper

class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act_layer = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.act = OrderedDict()
        self.count = 0

    def forward(self, x):
        self.count = self.count % 2 
        self.act['mlp_{}'.format(self.count)] = x
        self.count +=1
        x = self.fc1(x)
        x = self.act_layer(x)
        x = self.drop(x)

        self.count = self.count % 2 
        self.act['mlp_{}'.format(self.count)] = x
        self.count +=1
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., args=None):
        super().__init__()
        self.args = args
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_gradients = None
        self.attention_map = None
        self.act = OrderedDict()
        self.count = 0

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients
        
    def get_attn_gradients(self):
        return self.attn_gradients
    
    def save_attention_map(self, attention_map):
        self.attention_map = attention_map
        
    def get_attention_map(self):
        return self.attention_map
    
    def forward(self, x, register_hook=False, prompt=None, register_blk=None, ep_g=None, client_index=None):
        #print(x.shape)
        B, N, C = x.shape

        self.count = self.count % 2 
        self.act['attn_{}'.format(self.count)] = x
        self.count +=1
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        if prompt is not None:
            pk, pv = prompt
            pk = pk.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            pv = pv.reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            if "v2" not in self.args["method"]:
                k = torch.cat((pk,k), dim=2)
                v = torch.cat((pv,v), dim=2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn) # dropout 
                
        if register_hook:
            self.save_attention_map(attn)
            attn.register_hook(self.save_attn_gradients)    

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        
        self.count = self.count % 2 
        self.act['attn_{}'.format(self.count)] = x
        self.count +=1
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, use_grad_checkpointing=False, args=None):
        super().__init__()
        self.args=args
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, args=self.args)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.act = OrderedDict()
        self.count = 0

    def forward(self, x, register_hook=False, prompt=None, register_blk=None, ep_g=None, client_index=None):
        
        if "v2" in self.args["method"]:
            if prompt is not None:
                pk, pv = prompt
                pkv = torch.cat((pk, pv), dim=1)
                x = torch.cat((pkv, x), dim=1)
        
        x = x + self.drop_path(self.attn(self.norm1(x), register_hook=register_hook, prompt=prompt, register_blk=register_blk, ep_g=ep_g, client_index=client_index))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

    
class VisionTransformer(nn.Module):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=None, 
                 use_grad_checkpointing=False, ckpt_layer=0, device='cuda:0', args=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            norm_layer: (nn.Module): normalization layer
        """
        super().__init__()
        self.args = args
        self.device = args["device"][0]
        img_size = args["img_size"]
        self.act = OrderedDict() 

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        # image into patch embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches

        # class token and position embedding 
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        # encoder blocks 
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                use_grad_checkpointing=(use_grad_checkpointing and i>=depth-ckpt_layer), args=self.args
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        self.mapping =  nn.Linear(768, 512) # just for attention in lander 

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

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
        return {'pos_embed', 'cls_token'}
    
    def getAttention(self, x, register_blk=-1, prompt=None, q=None, task=None):
        attention = []
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
  
        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            p_list, attention_layer, x = prompt.forward_with_attention(q, i, x, train=False, task_id=task)
            #print(i)
            x = blk(x, register_blk=i, prompt=p_list)
            if attention_layer is not None:
                attention.append(attention_layer)
        #print(len(attention))
        return attention


    def get_aqk(self, x, prompt=None, client_index = None, q=None, task_id=None, trained_task_id=None, finished_task=None):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
  
        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)

        mean_aqk_list = None
        for i,blk in enumerate(self.blocks):
            if prompt is not None:
                p_list, x, mean_aqk = prompt.get_aqk(q, i, x, trained_task_id=trained_task_id)
                if p_list is not None:
                    if mean_aqk_list is None:
                        mean_aqk_list = mean_aqk.unsqueeze(0)
                    else:
                        mean_aqk_list = torch.cat((mean_aqk_list, mean_aqk.unsqueeze(0)), dim=0)
            x = blk(x, register_blk=i, prompt=p_list, ep_g=0, client_index=client_index)

        return mean_aqk_list

    def forward_sharedprompt(self, x, prompt, global_task_id, client_index=0, client_global_task_id=0):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
  
        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)

        Gk = prompt[-1].unsqueeze(0).repeat(B, 1, 1)[:,:4,:]
        Gv = prompt[-1].unsqueeze(0).repeat(B, 1, 1)[:,4:,:]
        
        
        for i,blk in enumerate(self.blocks):
            if i in [4,5,6]:
                if global_task_id != -1 and client_global_task_id == global_task_id:   
                    Ek = prompt[global_task_id].unsqueeze(0).repeat(B, 1, 1)[:,:4,:]
                    Ev = prompt[global_task_id].unsqueeze(0).repeat(B, 1, 1)[:,4:,:]
                    Pk = torch.cat((Ek, Gk), dim=1)
                    Pv = torch.cat((Ev, Gv), dim=1)
                elif global_task_id != -1 and client_global_task_id != global_task_id:
                    Ek = prompt[global_task_id].clone().detach().unsqueeze(0).repeat(B, 1, 1)[:,:4,:]
                    Ev = prompt[global_task_id].clone().detach().unsqueeze(0).repeat(B, 1, 1)[:,4:,:]
                    Pk = torch.cat((Ek, Gk.clone().detach()), dim=1)
                    Pv = torch.cat((Ev, Gv.clone().detach()), dim=1)
                else:
                    Pk = Gk
                    Pv = Gv

                p_list = [Pk, Pv]
            else:
                p_list = None
            x = blk(x, register_blk=i, prompt=p_list, ep_g=0, client_index=client_index)
        x = self.norm(x)
        return x
    
    def forward_sharedcodap(self, x, prompt, global_prompt, global_task_id, q, train, task_id, ep_g=None, client_index=None):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
  
        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)
        if global_task_id == -1:
            for i,blk in enumerate(self.blocks):
                if train:
                    p_list, loss, x, P_, indices_taskchoosing, mean_aqk = prompt.forward(q, i, x, train=True, task_id=task_id, aq_k=None)                     
                else:
                    p_list, _, x, indices_taskchoosing = prompt.forward(q, i, x, train=False, task_id=task_id, aq_k=None)
                x = blk(x, register_blk=i, prompt=p_list, ep_g=ep_g, client_index=client_index)
        else:
            for i,blk in enumerate(self.blocks):
                if train:
                    p_list, x, same = prompt.forward_sharedcodap(q, i, x, train=True, task_id=task_id, aq_k=None, global_task_id=global_task_id)                     
                else:
                    p_list, x, same = prompt.forward_sharedcodap(q, i, x, train=False, task_id=task_id, aq_k=None, global_task_id=global_task_id)
                
                if p_list is not None:
                    if same:
                        Gk = global_prompt.unsqueeze(0).repeat(B, 1, 1)[:,:4,:]
                        Gv = global_prompt.unsqueeze(0).repeat(B, 1, 1)[:,4:,:]
                    else:
                        Gk = global_prompt.clone().detach().unsqueeze(0).repeat(B, 1, 1)[:,:4,:]
                        Gv = global_prompt.clone().detach().unsqueeze(0).repeat(B, 1, 1)[:,4:,:]
                    Pk = torch.cat((p_list[0], Gk), dim=1)
                    Pv = torch.cat((p_list[1], Gv), dim=1)
                    p_list = [Pk, Pv]

                x = blk(x, register_blk=i, prompt=p_list, ep_g=ep_g, client_index=client_index)
        x = self.norm(x)  
        return x

    def forward(self, x, register_blk=-1, prompt=None, q=None, train=False, task_id=None, aq_k=None, ep_g=None, client_index=None):
        
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
  
        x = x + self.pos_embed[:,:x.size(1),:]
        x = self.pos_drop(x)

        
        # if isinstance(self.device, int):
        prompt_loss = torch.zeros((1,), requires_grad=True).to(self.device)

        prompt_client = []
        mean_aqk_list = None
        indices_taskchoosings = []
        feature_map = []

        for i, blk in enumerate(self.blocks):
            if prompt is not None:
                if train:
                    if "classincremental" in self.args["method"]:
                        p_list, loss, x, P_, indices_taskchoosing, mean_aqk = prompt.forward(q, i, x, train=True, task_id=task_id, aq_k=aq_k)
                    else:
                        p_list, loss, x, P_, indices_taskchoosing, mean_aqk = prompt.forward(q, i, x, train=True, task_id=task_id, aq_k=aq_k)
                    indices_taskchoosings.append(indices_taskchoosing)
                    prompt_loss += loss
                    if p_list is not None:
                        prompt_client.append(P_)
                        if mean_aqk is not None:
                            if mean_aqk_list is None:
                                mean_aqk_list = mean_aqk.unsqueeze(0)
                            else:
                                mean_aqk_list = torch.cat((mean_aqk_list, mean_aqk.unsqueeze(0)), dim=0)
                        
                else:
                    if "classincremental" in self.args["method"]:
                        p_list, _, x, indices_taskchoosing = prompt.forward(q, i, x, train=False, task_id=task_id, aq_k=aq_k)
                    else:
                        p_list, _, x, indices_taskchoosing = prompt.forward(q, i, x, train=False, task_id=task_id, aq_k=aq_k)
                    indices_taskchoosings.append(indices_taskchoosing)
            else:
                p_list = None

            x = blk(x, register_blk=i, prompt=p_list, ep_g=ep_g, client_index=client_index)
            feature_map.append(x[:, -197:, :])

        #print(x.size())
        x = self.norm(x)

        # check if this is correct 
        feature = x[:,0,:]
        feature = feature.view(feature.size(0), -1)

        att = self.mapping(feature)
        self.act['space'] = feature

        out = {}
        out["x"] = x
        # out["prompt_loss"] = prompt_loss
        # out["prompt_client"] = prompt_client
        # out["task_indices"] = indices_taskchoosing[0]
        # out["mean_aqk"] = mean_aqk_list
        out["features"] = feature
        out["att"] = att

        return out 
    
    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        _load_weights(self, checkpoint_path, prefix)
        

@torch.no_grad()
def _load_weights(model: VisionTransformer, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    model.cls_token.copy_(_n2p(w[f'{prefix}cls'], t=False))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))

    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(
        posemb: torch.Tensor,
        posemb_new: torch.Tensor,
        num_prefix_tokens: int = 1,
        gs_new: Tuple[int, int] = (),
        interpolation: str = 'bicubic',
        antialias: bool = False,
    ) -> torch.Tensor:
    """ Rescale the grid of position embeddings when loading from state_dict.
    *DEPRECATED* This function is being deprecated in favour of using resample_abs_pos_embed
    """
    ntok_new = posemb_new.shape[1] - num_prefix_tokens
    ntok_old = posemb.shape[1] - num_prefix_tokens
    gs_old = [int(math.sqrt(ntok_old))] * 2
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    return resample_abs_pos_embed(
        posemb, gs_new, gs_old,
        num_prefix_tokens=num_prefix_tokens,
        interpolation=interpolation,
        antialias=antialias,
        verbose=True,
    )
           
def interpolate_pos_embed(pos_embed_checkpoint, visual_encoder):        
    # interpolate position embedding
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = visual_encoder.patch_embed.num_patches
    num_extra_tokens = visual_encoder.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)

    if orig_size!=new_size:
        # class_token and dist_token are kept unchanged
        extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
        # only the position tokens are interpolated
        pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
        pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
        pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
        pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
        new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
        print('reshape position embedding from %d to %d'%(orig_size ** 2,new_size ** 2))
        
        return new_pos_embed    
    else:
        return pos_embed_checkpoint

def vit224(pretrained, **kwargs):
    model = VisionTransformer(**kwargs)
    if pretrained:
        import timm
        from timm.models import vit_base_patch16_224_in21k, vit_base_patch16_224
        load_dict = vit_base_patch16_224_in21k(pretrained=True).state_dict()

        pr_model = timm.create_model('vit_base_patch16_224.augreg_in21k', pretrained=True)
        load_dict = pr_model.state_dict()
        del load_dict['head.weight']; del load_dict['head.bias']
        model.load_state_dict(load_dict, strict=False) # due to self.mapping 
    return model

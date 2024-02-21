import math
import logging
from typing import List, Tuple
from ray.rllib.utils.framework import TensorType

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.models.modelv2 import ModelV2
logger = logging.getLogger(__name__)
import numpy as np
from collections import OrderedDict
from types import SimpleNamespace
def unflatten_vec(obs,grid_size):
    s = grid_size*grid_size
    sizes = [s, 2, s, 2, s, 2,   1, s, s, 2, 2, s, 1,     s, 1, 1]
    sizesum = [0]
    for _s in sizes:
        sizesum.append(sizesum[-1]+_s)
    
    names = [
        "clip","clip_dim",
        "grid","grid_dim",
        "input","input_dim",
            "active",
            "background",
            "object","object_dim","object_pos","object_sel",
            "rotation_parity",
        "selected",
        "terminated",
        "trials_remain",
        ]
    req_reshape = ["clip", "grid", "input", "background", "object", "object_sel", "selected"]

    splits = [obs[:, st:ed].long() for st, ed in zip(sizesum[:-1],sizesum[1:])]
    res =  OrderedDict(zip(names,splits))
    for k in req_reshape:
        res[k] = res[k].reshape(-1,grid_size,grid_size)
    return res

class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, cfg):
        super().__init__()
        assert cfg.model.n_embd % cfg.model.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(cfg.model.n_embd, cfg.model.n_embd)
        self.query = nn.Linear(cfg.model.n_embd, cfg.model.n_embd)
        self.value = nn.Linear(cfg.model.n_embd, cfg.model.n_embd)

        # regularization
        self.attn_drop = nn.Dropout(cfg.model.attn_pdrop)
        self.resid_drop = nn.Dropout(cfg.model.resid_pdrop)

        # output projection
        self.proj = nn.Linear(cfg.model.n_embd, cfg.model.n_embd)
        self.n_head = cfg.model.n_head

    def forward(self, x, key_padding_mask):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #att[key_padding_mask[:, None, :, None].tile(1, self.n_head, 1, T)] = float('-inf')
        att[key_padding_mask[:, None, None, :].tile(1, self.n_head, T, 1)] = float('-inf')

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, cfg, num_fixed_tokens, num_all_tokens):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.model.n_embd)
        self.ln2 = nn.LayerNorm(cfg.model.n_embd)
        self.resid_drop = nn.Dropout(cfg.model.resid_pdrop)
        self.attn = CausalSelfAttention(cfg)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.model.n_embd, 4 * cfg.model.n_embd),
            GELU(),
            nn.Linear(4 * cfg.model.n_embd, cfg.model.n_embd),
            nn.Dropout(cfg.model.resid_pdrop),
        )

    def forward(self, inp):
        x, mask = inp
        x = x + self.attn(self.ln1(x), key_padding_mask=mask)
        x = x + self.mlp(self.ln2(x))
        return x, mask

class Periodic(nn.Module):
    def __init__(self, inp_dim, n_freq, oup_dim, sigma) -> None:
        super().__init__()
        coefficients = torch.normal(0.0, sigma, (inp_dim, n_freq))
        self.coefficients = nn.Parameter(coefficients)
        self.encoder = nn.Sequential(nn.Linear(inp_dim * n_freq * 2, oup_dim), GELU())

    def forward(self, x):
        assert x.ndim == 2
        x = 2 * torch.pi * self.coefficients[None] * x[..., None]
        x = torch.cat([torch.cos(x), torch.sin(x)], -1).reshape(x.shape[0], -1)
        return self.encoder(x)

class GPTPolicy(TorchModelV2, nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, 
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name, **kwargs):
        
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)
        cfg = SimpleNamespace()
        cfg.model = SimpleNamespace(**model_config["custom_model_config"]["model"])
        cfg.env = SimpleNamespace(**model_config["custom_model_config"]["env"])

        self.cfg = cfg
        #self.device = device

        self.num_pixel = cfg.env.grid_x * cfg.env.grid_y
        self.grid_shape = nn.Parameter(torch.tensor([self.cfg.env.grid_x, self.cfg.env.grid_y]),requires_grad=False)

        self.num_fixed_tokens = self.num_pixel * 2 + 2 # info_tkn, cls_tkn
        self.num_action_recur = 5
        self.num_all_tokens = self.num_fixed_tokens + self.num_action_recur

        self.pos_emb = nn.Parameter(torch.randn(1, self.num_pixel, cfg.model.n_embd) * 0.02)
        self.global_pos_emb = nn.Parameter(
            torch.randn(1, self.num_all_tokens, cfg.model.n_embd) * 0.02)
        self.state_emb = nn.Parameter(torch.randn(8, 1, cfg.model.n_embd) * 0.02)
        self.bbox_emb = nn.Linear(4, cfg.model.n_embd)
        self.drop = nn.Dropout(cfg.model.embd_pdrop)
        self.cls_tkn = nn.Parameter(torch.randn(1, 1, cfg.model.n_embd) * 0.02)
        self.color_action_tkn = nn.Parameter(torch.randn(1, 1, cfg.model.n_embd) * 0.02)
        # transformer
        self.blocks = nn.Sequential(*[Block(
            cfg, self.num_fixed_tokens, self.num_all_tokens
            ) for _ in range(cfg.model.n_layer)])

        # decoder head
        self.ln_f = nn.LayerNorm(cfg.model.n_embd)
        self.head_critic = nn.Linear(cfg.model.n_embd, cfg.env.num_actions)

        self.color_encoder = nn.Embedding(cfg.env.num_colors, cfg.model.n_embd)
        self.binary_encoder = nn.Embedding(2, cfg.model.n_embd)
        self.term_encoder = nn.Embedding(2, cfg.model.n_embd)
        self.trials_encoder = nn.Embedding(4, cfg.model.n_embd)
        self.active_encoder = nn.Embedding(2, cfg.model.n_embd)
        self.rotation_encoder = nn.Embedding(4, cfg.model.n_embd)
    
        self.operation_encoder = nn.Embedding(cfg.env.num_actions, cfg.model.n_embd)
        self.bbox_encoder = Periodic(4, cfg.model.n_embd // 8, cfg.model.n_embd, 0.15)

        self.apply(self._transformer_init_weights)
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        # rnn policy
        def linear_with_orthogonal_init(inp_dim, oup_dim, scale):
            linear = nn.Linear(inp_dim, oup_dim)
            torch.nn.init.orthogonal_(linear.weight, scale)
            torch.nn.init.zeros_(linear.bias)
            return linear
            
        head_factory = lambda last_dim, oup_init_scale: nn.Sequential(
            linear_with_orthogonal_init(cfg.model.n_embd, cfg.model.n_embd, math.sqrt(2)), GELU(), 
            linear_with_orthogonal_init(cfg.model.n_embd, cfg.model.n_embd, math.sqrt(2)), GELU(), 
            linear_with_orthogonal_init(cfg.model.n_embd, last_dim, oup_init_scale))
        self.head_operation = head_factory(1, 0.01)
        self.head_bbox_mean = head_factory(4, 0.01)
        self.head_bbox_std = head_factory(4, 0.01)
        self.head_critic = head_factory(1, 1)
        self.head_aux_rtm1 = head_factory(1, 1)
        self.head_aux_reward = head_factory(1, 1)
        self.head_aux_transition = head_factory(cfg.env.num_colors, 1)

    def _transformer_init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding, nn.Parameter)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    @override(ModelV2)
    def value_function(self) -> TensorType:
        return self.head_critic(self.last_forward[:, -1]).squeeze(1)
    
    # for compat (optim_group vs .parameters())
    def get_optim_group(self):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """
        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()

        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, torch.nn.MultiheadAttention, nn.GRUCell)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                """
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                """
                if 'bias' in pn:
                    # 원래 모든 bias는 학습 안되는데 여기서는 학습 OK
                    no_decay.add(fpn)
                elif 'weight' in pn and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif 'weight' in pn and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')
        no_decay.add('state_emb')
        no_decay.add('cls_tkn')
        no_decay.add('color_action_tkn')
        no_decay.add('bbox_encoder.coefficients')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.1},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return optim_groups
    
    def forward(self, input_dict, state, seq_lens):
        obs = unflatten_vec(input_dict['obs'],30)
        self.last_forward = self._forward(
            obs["grid"], obs["grid_dim"], obs["selected"], obs["clip"], obs["clip_dim"],
            obs["terminated"], obs["trials_remain"], obs["active"], obs["object"], obs["object_sel"]
            , obs["object_dim"], obs["object_pos"], obs["background"], obs["rotation_parity"], obs["input"], obs["input_dim"]
            ,[], [], []
        )
        self.last_forward = self.last_forward[:, -1-self.cfg.env.num_actions: ]
        return self.last_forward, state

    def _forward(self, grid, grid_dim, selected, clip, clip_dim, terminated,
                trials_remain, active, object, object_sel, object_dim, object_pos,
                background, rotation_parity, input, input_dim, answer, answer_dim,
                additional_tokens):
        if grid.get_device()<0:
            self.device = 'cpu'
        else:
            self.device = f'cuda:{grid.get_device()}'
        B = grid.shape[0]
        def compute_mask(base, end_dim, start_dim=None):

            if start_dim is None:
                active = torch.ones_like(base)
                tran = torch.zeros_like(end_dim)
                tran[:, 0] = end_dim[:, 0] - self.cfg.env.grid_x
                tran[:, 1] = end_dim[:, 1] - self.cfg.env.grid_y
                active = core_translate(active, tran)
            else:
                active = torch.ones_like(base)
                tran = torch.zeros_like(end_dim)
                tran[:, 0] = torch.minimum(start_dim[:, 0] + end_dim[:, 0] - self.cfg.env.grid_x, torch.zeros_like(start_dim[:, 0]))
                tran[:, 1] = torch.minimum(start_dim[:, 1] + end_dim[:, 1] - self.cfg.env.grid_y, torch.zeros_like(start_dim[:, 0]))
                active = core_translate(active, tran)

                opposite = torch.ones_like(base)
                tran[:, 0] = - start_dim[:, 0]
                tran[:, 1] = - start_dim[:, 1]
                opposite = core_translate(opposite, tran)

                opposite = torch.flip(opposite, [1, 2])
                active = torch.logical_and(active, opposite)
            return (~active.bool()).reshape(B, -1)

        def core_translate(base, pos):
            translate = torch.eye(2, 2,device=self.device)[None].tile(B, 1, 1)
            rate = - torch.flip(pos * 2 / self.grid_shape, [1])[..., None]
            translate = torch.concat([translate, rate], axis=2)
            ff = torch.nn.functional.affine_grid(translate, [B, 1, self.cfg.env.grid_x, self.cfg.env.grid_y], align_corners=False)
            res = torch.nn.functional.grid_sample(
                base.reshape([B, 1, self.cfg.env.grid_x, self.cfg.env.grid_y]).float(), ff, align_corners=False).round().long().squeeze(1)
            return res

        def translate(base, pos):
            pos[:, 0] = torch.remainder(pos[:, 0] + self.cfg.env.grid_x, self.cfg.env.grid_x)
            pos[:, 1] = torch.remainder(pos[:, 1] + self.cfg.env.grid_y, self.cfg.env.grid_y)
            return core_translate(base, pos)
        

        active_grid = compute_mask(grid, grid_dim)
        #active_ans = compute_mask(answer, answer_dim)

        grid = self.color_encoder(grid.reshape(B, -1))
        grid = grid + self.pos_emb + self.state_emb[0]

        selected = self.binary_encoder(selected.reshape(B, -1)) # follows active_grid
        selected = selected + self.pos_emb + self.state_emb[1]

        active_clip = compute_mask(clip, clip_dim)

        clip = self.color_encoder(clip.reshape(B, -1))
        clip = clip + self.pos_emb + self.state_emb[2]

        active_obj = compute_mask(object, object_dim, object_pos)

        object = self.color_encoder(translate(object, object_pos).reshape(B, -1))
        object = object + self.pos_emb + self.state_emb[3]

        object_sel = self.binary_encoder(translate(object_sel, object_pos).reshape(B, -1)) # follows active_obj
        object_sel = object_sel + self.pos_emb + self.state_emb[4]

        background = self.color_encoder(background.reshape(B, -1)) # follows active_grid
        background = background + self.pos_emb + self.state_emb[5]

        active_inp = compute_mask(input, input_dim)

        input = self.color_encoder(input.reshape(B, -1))
        input = input + self.pos_emb + self.state_emb[6]

        #answer = self.color_encoder(answer.reshape(B, -1))
        #answer = answer + self.pos_emb + self.state_emb[7]

        info_tkn = (
            #self.term_encoder(terminated) + 
            self.trials_encoder(trials_remain) +
            self.active_encoder(active) #+ 
            #self.rotation_encoder(rotation_parity)
            ).reshape(-1, 1, self.cfg.model.n_embd)

        cls_tkn = self.cls_tkn.tile(len(active_grid), 1, 1)
        color_tkns = torch.cat([self.color_action_tkn + self.color_encoder.weight[None]]).tile(len(active_grid), 1, 1)
        operation_tkns = torch.cat([self.operation_encoder.weight[None]]).tile(len(active_grid), 1, 1)
        operation_tkns[:, :10] += color_tkns
        #additional_tokens = [
        #    each.reshape(-1, 1, self.cfg.model.n_embd) for each in additional_tokens]
        # inputs = torch.cat([
        #     grid, selected, clip, object, 
        #     object_sel, background, input, answer, info_tkn, 
        #     cls_tkn] + additional_tokens, axis=1)
        inputs = torch.cat([
            grid, input, info_tkn, operation_tkns, cls_tkn], axis=1)
        
        #inputs = inputs + self.global_pos_emb[:, :inputs.shape[1]]

        # masks = torch.cat([
        #     active_grid, active_grid, active_clip, active_obj,
        #     active_obj, active_grid + 1 - active.reshape(-1, 1), active_inp, active_ans, 
        #     torch.zeros((len(active_grid), 2 + len(additional_tokens)),
        #                  device=self.device)], axis=1)
        masks = torch.cat([
            active_grid,  active_inp,
            torch.zeros((len(active_grid), 2 + self.cfg.env.num_actions ), #+ len(additional_tokens)
                        device= self.device)], axis=1)

        x = self.drop(inputs)
        x, _ = self.blocks((x, masks.bool()))
        x = self.ln_f(x)
        
        return x

    def act(self, **kwargs):

        # compute value, rtm1
        x = self.forward(**kwargs, additional_tokens=[])
        value = self.head_critic(x[:, -1]).squeeze(1)
        rtm1_pred = self.head_aux_rtm1(x[:, -1]).squeeze(1)

        # sample operation
        dist = Categorical(logits=self.head_operation(x[:, -1-self.cfg.env.num_colors:-1]).squeeze(-1))
        operation = dist.sample()
        target_x = x[:, -1-self.cfg.env.num_colors:-1][torch.arange(len(x)), operation]
        log_prob = dist.log_prob(operation)
        enc_op = self.operation_encoder(operation)

        # sample bbox
        bbox_mean = torch.nn.functional.sigmoid(self.head_bbox_mean(target_x))
        bbox_std = torch.exp(torch.clamp(self.head_bbox_std(target_x), -20, 2))
        dist = TruncatedNormal(bbox_mean, bbox_std, 0, 1)
        bbox = dist.sample()
        log_prob += dist.log_prob(bbox).sum(1)
        enc_bb = self.bbox_encoder(bbox)

        x = self.forward(**kwargs, additional_tokens=[enc_op, enc_bb])
        r_pred = self.head_aux_reward(x[:, -1]).squeeze(1)
        g_pred = self.head_aux_transition(x[:, :self.num_pixel])

        return operation, bbox, -log_prob, value, rtm1_pred, r_pred, g_pred

    def evaluate(self, operation, bbox, **kwargs):

        # compute value, rtm1
        x = self.forward(**kwargs, additional_tokens=[])
        vpred = self.head_critic(x[:, -1]).squeeze(1)
        rtm1_pred = self.head_aux_rtm1(x[:, -1]).squeeze(1)

        # sample operation
        dist = Categorical(logits=self.head_operation(x[:, -1-self.cfg.env.num_colors:-1]).squeeze(-1))
        log_prob = dist.log_prob(operation)
        target_x = x[:, -1-self.cfg.env.num_colors:-1][torch.arange(len(x)), operation]
        enc_op = self.operation_encoder(operation)
        entropy = dist.entropy()

        # sample bbox
        x = self.forward(**kwargs, additional_tokens=[enc_op])
        bbox_mean = torch.nn.functional.sigmoid(self.head_bbox_mean(target_x))
        bbox_std = torch.exp(torch.clamp(self.head_bbox_std(target_x), -20, 2))
        dist = TruncatedNormal(bbox_mean, bbox_std, 0, 1)
        log_prob += dist.log_prob(bbox).sum(1)
        enc_bb = self.bbox_encoder(bbox)
        entropy += dist.entropy.sum(1)

        x = self.forward(**kwargs, additional_tokens=[enc_op, enc_bb])
        r_pred = self.head_aux_reward(x[:, -1]).squeeze(1)
        g_pred = self.head_aux_transition(x[:, :self.num_pixel])

        return -log_prob, vpred, entropy, rtm1_pred, r_pred, g_pred

    def get_grads(self):
        grads = []
        for p in self.parameters():
            if p.grad is None:
                grads.append(torch.zeros_like(p).flatten())
            else:
                grads.append(p.grad.data.clone().flatten())
        return torch.cat(grads)

    def set_grads(self, new_grads):
        start = 0
        for p in self.parameters():
            dims = p.shape
            end = start + dims.numel()
            if p.grad is not None:
                p.grad.data = new_grads[start:end].reshape(dims)
            start = end

from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.torch_action_dist import TorchMultiCategorical
"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

"""
DT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

# Under Construction

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

import numpy as np
from ray.rllib.utils.annotations import override
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)

class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size*config.number_of_tokens, config.block_size*config.number_of_tokens))
                                     .view(1, 1, config.block_size*config.number_of_tokens, config.block_size*config.number_of_tokens))
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size + 1, config.block_size + 1))
        #                              .view(1, 1, config.block_size + 1, config.block_size + 1))
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class DT(TorchModelV2, nn.Module):
    """  the full DT language model, with a context size of block_size """

    def __init__(self,  
                obs_space,
                action_space,
                num_outputs,
                model_config,
                name,
                config):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        self.config = config
        
        # input embedding stem
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd) # action -> embed
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.n_embd))

        self.global_pos_emb = nn.Parameter(torch.zeros(1, config.max_timestep+1, config.n_embd))
        self.drop = nn.Dropout(config.embd_pdrop)

        # transformer
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        # self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # loss, adv head -> 더 복잡한 모델로 변경하여 사용!
        self.loss_head = nn.Sequential(nn.Conv2d(in_channels=config.number_of_tokens, out_channels=1, kernel_size=1),
                                       nn.Linear(config.n_embd, config.loss_dim), 
                                       nn.ReLU())
        self.adv_head = nn.Sequential(nn.Conv2d(in_channels=config.number_of_tokens, out_channels=1, kernel_size=1),
                                            nn.Linear(config.n_embd, config.adv_dim), 
                                            nn.ReLU())
        self.vf_head = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1),
                                       nn.Linear(config.n_embd, config.adv_dim), 
                                       nn.ReLU())
        
        self.block_size = config.block_size
        
        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        # Grid Encoder -> 더 복잡한 모델 사용 가능, 추후 변형하여 사용!
        self.state_grid_encoder = nn.Sequential(nn.Linear(900, config.n_embd), nn.Tanh())
        self.state_input_encoder = nn.Sequential(nn.Linear(900, config.n_embd), nn.Tanh())
        self.state_clip_encoder = nn.Sequential(nn.Linear(900, config.n_embd), nn.Tanh())
        self.state_object_encoder = nn.Sequential(nn.Linear(900, config.n_embd), nn.Tanh())
        self.action_embeddings = nn.Sequential(nn.Embedding(config.vocab_size, config.n_embd), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)
        
        self.x_encoder = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())
        self.y_encoder = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())
        self.h_encoder = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())
        self.w_encoder = nn.Sequential(nn.Linear(1, config.n_embd), nn.Tanh())
        
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root DT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, obs, actions=None, timesteps=None):
        # state_grid: (batch, block_size, 900)
        # state_mask: (batch, block_size, 900)
        # actions: (batch, block_size, 5)
        # timesteps: (batch, 1, 1)
        
        batch_size = obs.shape[0]
        if actions == None:
            if obs.ndim == 2:
                obs = torch.unsqueeze(obs, dim = 1)
            if timesteps == None:
                timesteps = torch.zeros(size=(batch_size, 1, 1), dtype=torch.int64).to(obs.device)
            elif timesteps.ndim == 2:
                timesteps = torch.unsqueeze(timesteps, dim = 1)
           
            state_grid = obs[:,:,:900]
            # state_mask = obs[:,:,900:]

            state_grid_embeddings = self.state_grid_encoder(state_grid.type(torch.float32).contiguous())
            # state_mask_embeddings = self.state_mask_encoder(state_mask.type(torch.float32).contiguous())

            batch_size = state_grid.shape[0]
            token_embeddings = torch.zeros((batch_size, state_grid.shape[1], self.config.n_embd), dtype=torch.float32, device=state_grid_embeddings.device)
            token_embeddings[:,::1,:] = state_grid_embeddings
            #token_embeddings[:,1::1,:] = state_mask_embeddings

            all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0) # batch_size, traj_length, n_embd
            position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)) + self.pos_emb[:, :token_embeddings.shape[1], :]
            
            x = self.drop(token_embeddings + torch.repeat_interleave(position_embeddings, 1, dim=1))
            x = self.blocks(x)
            x = self.ln_f(x)
            x = x.view(batch_size, 1, -1, self.config.n_embd)

            logits_loss = torch.squeeze(self.vf_head(x)[:, -1, :, :], 1)
            
            return logits_loss

        else:
            if obs.ndim == 2:
                obs = torch.unsqueeze(obs, dim = 1)
            if actions.ndim == 2:
                actions = torch.unsqueeze(actions, dim = 1)
            if timesteps == None:
                timesteps = torch.zeros(batch_size, 1, 1, dtype=torch.int64).to(obs.device)
            elif timesteps.ndim == 2:
                timesteps = torch.unsqueeze(timesteps, dim = 1)

            state_grid = obs[:,:,:900]
            # state_mask = obs[:,:,900:]

            action_number = torch.unsqueeze(actions[:,:,0], dim = -1)
            x = torch.unsqueeze(actions[:,:,1], dim = -1)
            y = torch.unsqueeze(actions[:,:,2], dim = -1)
            h = torch.unsqueeze(actions[:,:,3], dim = -1)
            w = torch.unsqueeze(actions[:,:,4], dim = -1)

            state_grid_embeddings = self.state_grid_encoder(state_grid.type(torch.float32).contiguous())
            # state_mask_embeddings = self.state_mask_encoder(state_mask.type(torch.float32).contiguous())
            
            action_embeddings = self.action_embeddings(action_number.type(torch.long).squeeze(-1))
            x_embeddings = self.x_encoder(x.type(torch.float32))
            y_embeddings = self.y_encoder(y.type(torch.float32))
            h_embeddings = self.h_encoder(h.type(torch.float32))
            w_embeddings = self.w_encoder(w.type(torch.float32))

            batch_size = state_grid.shape[0]
            token_embeddings = torch.zeros((batch_size, state_grid.shape[1] * self.config.number_of_tokens, self.config.n_embd), dtype=torch.float32, device=state_grid_embeddings.device)
            token_embeddings[:,::self.config.number_of_tokens,:] = state_grid_embeddings
            #token_embeddings[:,1::self.config.number_of_tokens,:] = state_mask_embeddings
            token_embeddings[:,1::self.config.number_of_tokens,:] = action_embeddings
            token_embeddings[:,2::self.config.number_of_tokens,:] = x_embeddings
            token_embeddings[:,3::self.config.number_of_tokens,:] = y_embeddings
            token_embeddings[:,4::self.config.number_of_tokens,:] = h_embeddings
            token_embeddings[:,5::self.config.number_of_tokens,:] = w_embeddings

            all_global_pos_emb = torch.repeat_interleave(self.global_pos_emb, batch_size, dim=0) # batch_size, traj_length, n_embd
            position_embeddings = torch.gather(all_global_pos_emb, 1, torch.repeat_interleave(timesteps, self.config.n_embd, dim=-1)) + self.pos_emb[:, :token_embeddings.shape[1], :]
            
            x = self.drop(token_embeddings + torch.repeat_interleave(position_embeddings, self.config.number_of_tokens, dim=1))
            x = self.blocks(x)
            x = self.ln_f(x)
            x = x.view(batch_size, self.config.number_of_tokens, -1, self.config.n_embd)

            try:
                logits_loss = torch.squeeze(self.loss_head(x)[:, -1, :, :], 1)
                logits_adv = torch.squeeze(self.adv_head(x)[:, -1, :, :], 1)
            except:
                import pdb; pdb.set_trace()
            
            return logits_loss, logits_adv


# Hyperparameters
class Config:
    def __init__(self):
        # Select device
        if(torch.cuda.is_available()):
            self.device = torch.device("cuda")
            print("GPU:", torch.cuda.get_device_name(0))
        else:
            self.device = torch.device("cpu")
            print("CPU:", torch.get_num_threads())
        
        # Model
        self.number_of_tokens = 6
        self.embd_pdrop = 0.1
        self.resid_pdrop = 0.1
        self.attn_pdrop = 0.1
        
        self.n_layer = 12
        self.n_head = 12
        self.n_embd = 768
        
        self.vocab_size = 34        # action 개수
        self.block_size = 1         # trajectory 길이 (a, s, r 페어 개수)
        self.max_timestep = 25
        
        # Training
        self.max_epochs = 10
        self.batch_size = 4
        self.learning_rate = 3e-4
        self.betas = (0.9, 0.95)
        self.grad_norm_clip = 1.0
        self.weight_decay = 0.1 # only applied on matmul weights
        # learning rate decay params: linear warmup followed by cosine decay to 10% of original
        self.lr_decay = False
        self.warmup_tokens = 375e6 # these two numbers come from the GPT-3 paper, but may not be good defaults elsewhere
        self.final_tokens = 260e9 # (at what point we reach 10% of original LR)
        # checkpoint settings
        self.ckpt_path = None
        self.num_workers = 0 # for DataLoader
        
        # Output
        self.adv_dim = 1
        self.loss_dim = 5

if(__name__ == "__main__"):
    config = Config()
    
    state_grid = torch.randint(-1, 10, (config.batch_size, 900)).to(config.device)
    state_mask = torch.randint(-1, 10, (config.batch_size, 900)).to(config.device)
    action = torch.randint(0, config.vocab_size, (config.batch_size, 1)).to(config.device)
    x = torch.randint(0, 30, (config.batch_size, 1)).to(config.device)
    y = torch.randint(0, 30, (config.batch_size, 1)).to(config.device)
    h = torch.randint(0, 30, (config.batch_size, 1)).to(config.device)
    w = torch.randint(0, 30, (config.batch_size, 1)).to(config.device)
    timesteps = torch.randint(0, config.max_timestep, (config.batch_size, 1)).to(config.device)

    # state_grid = torch.randint(-1, 10, (config.batch_size, config.block_size, 900)).to(config.device)
    # state_mask = torch.randint(-1, 10, (config.batch_size, config.block_size, 900)).to(config.device)
    # action = torch.randint(0, config.vocab_size, (config.batch_size, config.block_size, 1)).to(config.device)
    # x = torch.randint(0, 30, (config.batch_size, config.block_size, 1)).to(config.device)
    # y = torch.randint(0, 30, (config.batch_size, config.block_size, 1)).to(config.device)
    # h = torch.randint(0, 30, (config.batch_size, config.block_size, 1)).to(config.device)
    # w = torch.randint(0, 30, (config.batch_size, config.block_size, 1)).to(config.device)
    # timesteps = torch.randint(0, config.max_timestep, (config.batch_size, config.block_size, 1)).to(config.device)

    model = DT(config).to(config.device)

    print("\n====== Input Size ======")
    print("state_grid:", state_grid.shape)
    print("state_mask:", state_mask.shape)
    print("action:", action.shape)
    print("x:", x.shape)
    print("y:", y.shape)
    print("h:", h.shape)
    print("w:", w.shape)
    print("timesteps:", timesteps.shape)
    print("==========================\n")
    
    obs = torch.cat([state_grid, state_mask], dim = -1)
    actions = torch.cat([action, x, y, h, w], dim = -1)

    print("\n====== Input Size ======")
    print("obs:", obs.shape)
    print("actions:", actions.shape)
    print("timesteps:", timesteps.shape)
    print("==========================\n")

    # logits_loss = model(obs, actions = None, timesteps = timesteps)
    logits_loss, logits_adv = model(obs, actions = actions, timesteps = timesteps)
    
    print("\n====== Output Size ======")
    print("logits_loss:", logits_loss.shape)
    print("logits_adv:", logits_adv.shape)
    print("==========================\n")
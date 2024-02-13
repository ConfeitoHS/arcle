import functools
import gymnasium as gym
from math import log
import numpy as np
import tree  # pip install dm_tree
from typing import Optional

from ray.rllib.models.action_dist import ActionDistribution
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override, DeveloperAPI, ExperimentalAPI
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.numpy import SMALL_NUMBER, MIN_LOG_NN_OUTPUT, MAX_LOG_NN_OUTPUT
from ray.rllib.utils.spaces.space_utils import get_base_struct_from_space
from ray.rllib.utils.typing import TensorType, List, Union, Tuple, ModelConfigDict
from ray.rllib.models.torch.torch_action_dist import TorchDistributionWrapper, TorchCategorical
from .truncated_normal import TruncatedNormal
torch, nn = try_import_torch()

#autoregressive OP and BBox
class AROPandBBox(TorchDistributionWrapper):
    

    def __init__(self, inputs, model: TorchModelV2):
        super().__init__(inputs, model)
        self.cfg = model.cfg
        self.dist_op = torch.distributions.Categorical(logits=model.head_operation(inputs[:, -1-self.cfg.env.num_actions:-1]).squeeze(-1))
        

    def sample(self) :
        operation = self.dist_op.sample().long()
        target_x = self.inputs[:, -1-self.cfg.env.num_actions:-1][torch.arange(len(self.inputs)), operation]

        bbox_mean = torch.nn.functional.sigmoid(self.model.head_bbox_mean(target_x))
        bbox_std = torch.exp(torch.clamp(self.model.head_bbox_std(target_x), -20, 2))
        dist_bbox = TruncatedNormal(bbox_mean, bbox_std, 0, 1)
        bbox =(dist_bbox.sample()*30).floor().long()
        self.last_sample = torch.concat([bbox,operation.unsqueeze(-1)], dim=-1)
        return self.last_sample

    def deterministic_sample(self):
        operation = torch.argmax(self.dist_op.logits,dim=1).long()
        target_x = self.inputs[:, -1-self.cfg.env.num_actions:-1][torch.arange(len(self.inputs)), operation]

        bbox_mean = torch.nn.functional.sigmoid(self.model.head_bbox_mean(target_x))
        bbox_std = torch.exp(torch.clamp(self.model.head_bbox_std(target_x), -20, 2))
        dist_bbox = TruncatedNormal(bbox_mean, bbox_std, 0, 1)
        bbox = (dist_bbox._mean*30).floor().long()
        self.last_sample = torch.concat([bbox,operation.unsqueeze(-1)], dim=-1)
        return self.last_sample

    def logp(self, actions):
        bbox = actions[:, :4]
        op = actions[:, -1].squeeze(-1)
        print(bbox,op)
        target_x = self.inputs[:, -1-self.cfg.env.num_actions:-1][torch.arange(len(self.inputs)), op]

        bbox_mean = torch.nn.functional.sigmoid(self.model.head_bbox_mean(target_x))
        bbox_std = torch.exp(torch.clamp(self.model.head_bbox_std(target_x), -20, 2))
        
        return self.dist_op.log_prob(op)+TruncatedNormal(bbox_mean, bbox_std, 0, 1).log_prob(bbox/30.).sum(1)

    @staticmethod
    @override(ActionDistribution)
    def required_model_output_shape(action_space: gym.Space, model_config: ModelConfigDict):
        cfg = model_config["custom_model_config"]
        (cfg["env"]["num_actions"] + 1) * cfg["model"]["n_embd"]

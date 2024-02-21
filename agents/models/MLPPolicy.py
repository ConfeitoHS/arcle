import torch
import torch.nn as nn
from typing import List, Callable, Optional
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2

class WLinear(nn.Module):
    def __init__(
        self, in_features: int, out_features: int, bias_size: Optional[int] = None
    ):
        super().__init__()

        if bias_size is None:
            bias_size = out_features

        dim = 1000
        self.z = nn.Parameter(torch.empty(dim).normal_(0, 1.0 / out_features))
        self.fc = nn.Linear(dim, in_features * out_features + out_features)
        self.seq = self.fc
        self.w_idx = in_features * out_features
        self.weight = self.fc.weight
        self._linear = self.fc
        self.out_f = out_features

    def adaptation_parameters(self):
        return [self.z]

    def forward(self, x: torch.tensor):
        theta = self.fc(self.z)
        w = theta[: self.w_idx].view(x.shape[-1], -1)
        b = theta[self.w_idx :]

        ret = x @ w + b
        # import pdb; pdb.set_trace()
        return x @ w + b


class MLP(TorchModelV2, nn.Module):
    def __init__(
        self, 
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        layer_widths: List[int],
        final_activation: Callable = lambda x: x,
        bias_linear: bool = False,
        extra_head_layers: List[int] = None,
        w_linear: bool = False,
    ):
        nn.Module.__init__(self)
        super().__init__(obs_space,action_space,num_outputs,model_config,name)
        

        if len(layer_widths) < 2:
            raise ValueError(
                "Layer widths needs at least an in-dimension and out-dimension"
            )

        self._final_activation = final_activation
        self.seq = nn.Sequential()
        self._head = extra_head_layers is not None

        # if not w_linear:
        #     linear = BiasLinear if bias_linear else nn.Linear
        # else:
        #     linear = WLinear
        # self.bias_linear = bias_linear
        linear = WLinear
        self.aparams = []

        for idx in range(len(layer_widths) - 1):
            w = linear(layer_widths[idx], layer_widths[idx + 1])
            self.seq.add_module(f"fc_{idx}", w)
            if idx < len(layer_widths) - 2:
                self.seq.add_module(f"relu_{idx}", nn.ReLU())

        if extra_head_layers is not None:
            self.pre_seq = self.seq[:-2]
            self.post_seq = self.seq[-2:]

            self.head_seq = nn.Sequential()
            extra_head_layers = [
                layer_widths[-2] + layer_widths[-1]
            ] + extra_head_layers

            for idx, (infc, outfc) in enumerate(
                zip(extra_head_layers[:-1], extra_head_layers[1:])
            ):
                self.head_seq.add_module(f"relu_{idx}", nn.ReLU())
                w = linear(extra_head_layers[idx], extra_head_layers[idx + 1])
                self.head_seq.add_module(f"fc_{idx}", w)


    def forward(self, obs: torch.tensor, actions: Optional[torch.tensor] = None):
        if self._head and actions is not None:
            h = self.pre_seq(obs)
            head_input = torch.cat((h, actions), -1)
            
            # if torch.isnan(self._final_activation(self.post_seq(h))[0][0]): 
            #     import pdb; pdb.set_trace()
            return self._final_activation(self.post_seq(h)), self.head_seq(head_input)
        else:
            return self._final_activation(self.seq(obs))


if __name__ == "__main__":
    mlp = MLP([1, 5, 8, 2])
    x = torch.empty(10, 1).normal_()
    print(mlp(x).shape)
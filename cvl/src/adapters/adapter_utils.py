"""Implementation of different utility functions for adapter layers."""

import torch
import torch.nn as nn
from transformers.activations import get_activation
import torch.nn.functional as F

class Activations(nn.Module):
    def __init__(self, activation_type):
        super().__init__()
        self.f = get_activation(activation_type)

    def forward(self, x):
        return self.f(x)


def init_linear_layer(linear_layer, std=1e-2):
    """Initializes the given linear module as explained in adapter paper."""
    nn.init.normal_(linear_layer.weight, std=std)
    nn.init.zeros_(linear_layer.bias)


def linear_layer(input_dim, output_dim, std=1e-2):
    """Generates a linear module and initializes it."""
    linear = nn.Linear(input_dim, output_dim)
    init_linear_layer(linear, std=std)
    return linear

class Router(nn.Module):
    def __init__(self,config, in_dim=None):
        super().__init__()
        self.config = config
        self.model_dim = self.config.model_dim
        self.n_routers = self.config.num_routers
        self.num_adapters = self.config.num_adapters
        if in_dim is None:
            self.in_dim = self.model_dim
        else:
            self.in_dim = in_dim

        # (M,C,N)
        self.router_weights = nn.Parameter(torch.zeros(self.n_routers, self.in_dim, self.num_adapters))
        nn.init.normal_(self.router_weights, std=config.router_init_scale)
        if not config.no_router_bias:
            # (M,N)
            self.router_bias = nn.Parameter(torch.zeros(self.n_routers, self.num_adapters))

        self.multi_router_layer_norms = []
        for i in range(self.n_routers):
            self.multi_router_layer_norms.append(nn.LayerNorm(self.in_dim))
        self.multi_router_layer_norms = nn.ModuleList(self.multi_router_layer_norms)
        self.multi_router_weights_layer_norms = []
        for i in range(self.n_routers):
            self.multi_router_weights_layer_norms.append(nn.LayerNorm(self.in_dim))
        self.multi_router_weights_layer_norms = nn.ModuleList(self.multi_router_weights_layer_norms)
        for i in range(self.n_routers):
            self.multi_router_weights_layer_norms[i].weight = nn.Parameter(torch.ones(self.in_dim)*config.router_init_scale)
    
    def forward(self,x):
        # x shape (B,C)
        if self.config.use_load_balancing:
            new_x = x
        else:    
            new_x = []
            for i in range(self.n_routers):
                new_x.append(self.multi_router_layer_norms[i](x))
        #(M,B,C)
        new_x = torch.cat([y.unsqueeze(0) for y in new_x], dim=0)

        if self.config.normalize_router_weights:
            router_weights = []
            for i in range(self.n_routers):
                router_weights.append(self.multi_router_weights_layer_norms[i](self.router_weights[i].T).T)
            router_weights = torch.cat([y.unsqueeze(0) for y in router_weights], dim=0)
        elif self.config.cosine_router:
            router_weights = []
            for i in range(self.n_routers):
                router_weights.append(self.router_weights[i] / torch.norm(self.router_weights[i], dim=0, keepdim=True))
            router_weights = torch.cat([y.unsqueeze(0) for y in router_weights], dim=0)
        else:
            # (M,C,N)
            router_weights = self.router_weights

        if not self.config.no_router_bias:
            adapter_logits = torch.matmul(new_x, router_weights) + self.router_bias[:,None,:]
        else:
            adapter_logits = torch.matmul(new_x, router_weights)
        # (M,B,N)
        adapter_probs = F.softmax(adapter_logits, dim=-1)
        

        if self.training and self.config.same_init_then_branch > 0:
            adapter_probs = torch.ones(adapter_probs.shape).to(adapter_probs.device)
            adapter_probs = adapter_probs / torch.sum(adapter_probs, dim=-1, keepdim=True)
            return new_x, adapter_logits, adapter_probs
            
        if self.training and self.config.epsilon_greedy != 0:
            p = torch.rand(1).item()
            if p < self.config.epsilon_greedy:
                adapter_probs = torch.ones(adapter_probs.shape).to(adapter_probs.device)
                adapter_probs = adapter_probs / torch.sum(adapter_probs, dim=-1, keepdim=True)

        if self.training and self.config.expert_dropout != 0:
            ones = torch.ones(adapter_probs.shape)
            zeros = torch.zeros(adapter_probs.shape)
            random_tensor = torch.rand(adapter_probs.shape)
            mask = torch.where(random_tensor < self.config.expert_dropout, zeros, ones)
            mask = mask.to(adapter_probs.device)
            adapter_probs = adapter_probs * mask
            if self.config.renormalize_adapter_probs:
                adapter_probs = (adapter_probs+1e-10) / torch.sum((adapter_probs+1e-10), dim=-1, keepdim=True)
        
            neg_infs = - torch.ones(adapter_probs.shape) * 1e10
            mask2 = torch.where(random_tensor < self.config.expert_dropout, neg_infs, zeros)
            mask2 = mask2.to(adapter_probs.device)
            adapter_logits = adapter_logits + mask2

        return new_x, adapter_logits, adapter_probs

class Baseline(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, config, in_dim=None):
        super().__init__()
        self.config = config
        self.model_dim = self.config.model_dim
        self.n_routers = self.config.num_routers
        self.num_adapters = self.config.num_adapters
        if in_dim is None:
            self.in_dim = self.model_dim
        else:
            self.in_dim = in_dim
        self.down_sample_size = self.in_dim // self.config.bl_reduction_factor
        #(M,C,down_size)
        self.down_samplers_weights = nn.Parameter(torch.zeros(self.n_routers, self.in_dim, self.down_sample_size))
        nn.init.normal_(self.down_samplers_weights, std=1e-2)
        #(M,down_size)
        self.down_samplers_bias = nn.Parameter(torch.zeros(self.n_routers, self.down_sample_size))

        #(M,down_size,1)
        self.up_samplers_weights = nn.Parameter(torch.zeros(self.n_routers, self.down_sample_size, 1))
        nn.init.normal_(self.up_samplers_weights, std=1e-2)
        #(M,1)
        self.up_samplers_bias = nn.Parameter(torch.zeros(self.n_routers, 1))
        
    def forward(self, new_x):
        z = torch.matmul(new_x, self.down_samplers_weights) + self.down_samplers_bias[:,None,:]
        # (M,B,1)
        val = torch.matmul(z, self.up_samplers_weights) + self.up_samplers_bias[:,None,:]
        return val

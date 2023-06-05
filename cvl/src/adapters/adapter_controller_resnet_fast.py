"""Implements Adapter Controller, a module that keeps multiple
layers of Adapters, and controls which adapter layer to use."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from src.adapters.adapter_utils import Activations
import ipdb
import copy
import math
from sparsemax import Sparsemax
from .adapter_utils import Router, Baseline
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli

class Adapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, config, channel_dim=None):
        super().__init__()
        self.config = config
        self.channel_dim = channel_dim
        n_routers = self.config.num_routers
        num_adapters = self.config.num_adapters
        self.down_sample_size = self.config.down_sample_size
        if self.config.same_init_then_branch != -1:
                #(M,N,C,down_size)
                self.down_samplers_weights = nn.Parameter(torch.zeros(n_routers, num_adapters, self.channel_dim, self.down_sample_size))
                initial_value = torch.randn(self.down_samplers_weights[:,0,:,:].shape) * 1e-2
                for i in range(num_adapters):
                    self.down_samplers_weights[:,i,:,:].data.copy_(initial_value)
                #(M,N,down_size)
                self.down_samplers_bias = nn.Parameter(torch.zeros(n_routers, num_adapters, self.down_sample_size))
                nn.init.zeros_(self.down_samplers_bias)
                #(M,N,down_size,C)
                self.up_samplers_weights = nn.Parameter(torch.zeros(n_routers, num_adapters, self.down_sample_size, self.model_dim))
                initial_value = torch.randn(self.up_samplers_weights[:,0,:,:].shape) * 1e-2
                for i in range(num_adapters):
                    self.up_samplers_weights[:,i,:,:].data.copy_(initial_value)
                if self.config.bias_in_up_sampler:
                    #(M,N,C)
                    self.up_samplers_bias = nn.Parameter(torch.zeros(n_routers, num_adapters, self.channel_dim))
                    nn.init.zeros_(self.up_samplers_bias)
        else:
            if self.config.same_experts_across_routers:
                #(M,N,C,down_size)
                self.down_samplers_weights = nn.Parameter(torch.zeros(num_adapters, self.channel_dim, self.down_sample_size))
                nn.init.normal_(self.down_samplers_weights, std=1e-2)
                #(M,N,down_size)
                self.down_samplers_bias = nn.Parameter(torch.zeros(num_adapters, self.down_sample_size))
                nn.init.zeros_(self.down_samplers_bias)
            else:
                #(M,N,C,down_size)
                self.down_samplers_weights = nn.Parameter(torch.zeros(n_routers, num_adapters, self.channel_dim, self.down_sample_size))
                nn.init.normal_(self.down_samplers_weights, std=1e-2)
                #(M,N,down_size)
                self.down_samplers_bias = nn.Parameter(torch.zeros(n_routers, num_adapters, self.down_sample_size))
                nn.init.zeros_(self.down_samplers_bias)

            if self.config.same_experts_across_routers:
                #(M,N,down_size,C)
                self.up_samplers_weights = nn.Parameter(torch.zeros(num_adapters, self.down_sample_size, self.channel_dim))
                nn.init.normal_(self.up_samplers_weights, std=1e-2)

                #(M,N,C)
                self.up_samplers_bias = nn.Parameter(torch.zeros(num_adapters, self.channel_dim))
                nn.init.zeros_(self.up_samplers_bias)
            else:
                #(M,N,down_size,C)
                self.up_samplers_weights = nn.Parameter(torch.zeros(n_routers, num_adapters, self.down_sample_size, self.channel_dim))
                nn.init.normal_(self.up_samplers_weights, std=1e-2)
                if self.config.bias_in_up_sampler:
                    #(M,N,C)
                    self.up_samplers_bias = nn.Parameter(torch.zeros(n_routers, num_adapters, self.channel_dim))
                    nn.init.zeros_(self.up_samplers_bias)
        self.activation = Activations("swish")

    def forward(self, x, expert_index, prob_dist=None):
        (batch_size,seq_len,channel_dim) = x.shape
        # x shape (B,S,C)
        # expert_index shape (M, B)
        # prob_dist shape (M, B, N)
        if self.config.average_domain_adapters:
            down_samplers_weights = torch.mean(self.down_samplers_weights, dim=1, keepdim=True)
            down_samplers_bias = torch.mean(self.down_samplers_bias, dim=1, keepdim=True)
            up_samplers_weights = torch.mean(self.up_samplers_weights, dim=1, keepdim=True)
        else:
            down_samplers_weights = self.down_samplers_weights
            down_samplers_bias = self.down_samplers_bias
            up_samplers_weights = self.up_samplers_weights

        M, N, _, _ = down_samplers_weights.shape
        if type(expert_index) == type(()):
            if self.training:
                #(M,B,C,down_size)
                batch_down_samplers_weights = torch.gather(down_samplers_weights, 1, expert_index[0][:,:,None, None].repeat(1,1,channel_dim, self.down_sample_size))
                #(M,B,down_size)
                batch_down_samplers_bias = torch.gather(down_samplers_bias, 1, expert_index[0][:,:,None].repeat(1,1,self.down_sample_size))
                #(M,B,S,down_size)
                z = torch.matmul(x[None,:,:,:], batch_down_samplers_weights) + batch_down_samplers_bias[:,:,None,:]
                z = self.activation(z)
                #(M,B,down_size,C)
                batch_up_samplers_weights = torch.gather(up_samplers_weights, 1, expert_index[1][:,:,None, None].repeat(1,1, self.down_sample_size, channel_dim))
                #(M,B,S,C)
                u = torch.matmul(z,batch_up_samplers_weights)
                return u
            else:
                down_samplers_weights = torch.mean(self.down_samplers_weights, dim=1, keepdim=True).squeeze(0).squeeze(0)
                down_samplers_bias = torch.mean(self.down_samplers_bias, dim=1, keepdim=True).squeeze(0).squeeze(0)
                up_samplers_weights = torch.mean(self.up_samplers_weights, dim=1, keepdim=True).squeeze(0).squeeze(0)
                z = torch.matmul(x, down_samplers_weights) + down_samplers_bias
                z = self.activation(z)
                u = torch.matmul(z, up_samplers_weights)
                u = u.unsqueeze(0)
                return u

        if N == 1 and M == 1:
            down_samplers_weights = down_samplers_weights.squeeze(0).squeeze(0)
            down_samplers_bias = down_samplers_bias.squeeze(0).squeeze(0)
            up_samplers_weights = up_samplers_weights.squeeze(0).squeeze(0)
            z = torch.matmul(x, down_samplers_weights) + down_samplers_bias
            z = self.activation(z)
            u = torch.matmul(z, up_samplers_weights)
            u = u.unsqueeze(0)
            return u

        if prob_dist is not None:
            if self.config.routing_estimator == 'soft_routing':
                M, N, C, down_size = down_samplers_weights.shape
                # (M, B, N, C, down_size)
                batch_down_samplers_weights = down_samplers_weights.unsqueeze(1).repeat(1, batch_size, 1, 1, 1)
                # (M, B, N, down_size)
                batch_down_samplers_bias = down_samplers_bias.unsqueeze(1).repeat(1, batch_size, 1, 1)
            else:
                M, N, C, down_size = down_samplers_weights.shape
                # (M,N,C*down_size)
                down_samplers_weights = down_samplers_weights.reshape(M, N, -1)
                # (M, B, C*down_size)
                batch_down_samplers_weights = torch.matmul(prob_dist, down_samplers_weights)
                # (M, B, C, down_size)
                batch_down_samplers_weights = batch_down_samplers_weights.unsqueeze(3).reshape(M,batch_size,C,down_size)
                # (M, B, down_size)
                batch_down_samplers_bias = torch.matmul(prob_dist, down_samplers_bias)
        else:
                #(M,B,C,down_size)
                batch_down_samplers_weights = torch.gather(down_samplers_weights, 1, expert_index[:,:,None, None].repeat(1,1,channel_dim, self.down_sample_size))
                #(M,B,down_size)
                batch_down_samplers_bias = torch.gather(down_samplers_bias, 1, expert_index[:,:,None].repeat(1,1,self.down_sample_size))

        if self.config.routing_estimator == 'soft_routing':
            inp = x.unsqueeze(1).repeat(1, self.config.num_adapters, 1, 1)
            # (M, B, N, S, down_size)
            z = torch.matmul(inp[None, :,:,:,:], batch_down_samplers_weights) + batch_down_samplers_bias[:,:,:,None,:]
        else:
            #(M,B,S,down_size)
            z = torch.matmul(x[None,:,:,:], batch_down_samplers_weights) + batch_down_samplers_bias[:,:,None,:]
            z = self.activation(z)

        if prob_dist is not None:
            if self.config.routing_estimator == 'soft_routing':
                M, N, down_size, C = up_samplers_weights.shape
                # (M, B, N, down_size, C)
                batch_up_samplers_weights = up_samplers_weights.unsqueeze(1).repeat(1, batch_size, 1, 1, 1)
            else:
                M, N, down_size, C = up_samplers_weights.shape
                # (M, N, down_size*C)
                up_samplers_weights = up_samplers_weights.reshape(M, N, -1)
                # (M, B, down_size*C)
                batch_up_samplers_weights = torch.matmul(prob_dist, up_samplers_weights)
                # (M, B, down_size, C)
                batch_up_samplers_weights = batch_up_samplers_weights.unsqueeze(3).reshape(M,batch_size,down_size,C)
        else:
                #(M,B,down_size,C)
                batch_up_samplers_weights = torch.gather(up_samplers_weights, 1, expert_index[:,:,None, None].repeat(1,1, self.down_sample_size, channel_dim))
        
        if self.config.routing_estimator == 'soft_routing':
            # (M, B, N, S, C)
            u = torch.matmul(z, batch_up_samplers_weights)
            u = prob_dist[:,:,:,None,None] * u 
            # (M, B, S, C)
            u = torch.sum(u, dim=2)
        else:
            #(M,B,S,C)
            u = torch.matmul(z,batch_up_samplers_weights)
        return u


class AdapterController(nn.Module):
    """Implements Adapter controller module which controls the logics of
    putting adapter layers within transformer's layers."""

    def __init__(self, config, channel_dim):
        super().__init__()
        self.config = config
        if self.config.num_adapters == 1:
            self.n_routers = 1
        else:
            self.n_routers = self.config.num_routers
        self.multi_adapters = Adapter(config, channel_dim)
        self.multi_routers = Router(config, channel_dim)
        if self.config.routing_estimator == 'reinf_bl_routing':
            self.multi_router_baselines = Baseline(config, channel_dim)
        if self.config.routing_estimator == "skill_routing":
            self.skill_weights = nn.Parameter(torch.empty((self.config.num_domains, self.config.num_adapters)).uniform_(-1e-3, 1e-3))
            
        self.add_batch_norm_before_adapter = True
        self.add_batch_norm_after_adapter = False
        if self.add_batch_norm_before_adapter:
            self.pre_batch_norm = nn.BatchNorm2d(channel_dim)
        if self.add_batch_norm_after_adapter:
            self.post_batch_norm = nn.BatchNorm2d(channel_dim)
        if self.config.supervised_loss_weight != 0:
            self.supervised_loss_fn = nn.CrossEntropyLoss()

    def forward(self, inputs, domain_lbls, orig_domain_lbls, hash_lbls):
        """Retrieves the adapter layer corresponding to the given
        task. It freezes the adapter layers for all the other tasks
        and call the selected adapter layer.
        Args:
            task: the name of the current task.
            inputs: the inputs to feed in in the adapter layer.
        Returns:
            outputs of the adapter layer."""
        # inputs (B,C,H,W)
        (batch_size, channel_dim, height, width) = inputs.shape
        z = self.pre_batch_norm(inputs) if self.add_batch_norm_before_adapter else inputs
        #(B,H,W,C)
        z = z.permute(0,2,3,1)
        #(B,S,C)
        z = torch.flatten(z,start_dim=1,end_dim=2)
        routing_estimator = self.config.routing_estimator
        load_loss = torch.tensor(0.0).to(self.config.device)
        supervised_loss = torch.tensor(0.0).to(self.config.device)

        if self.config.train_layer != -1:
            if layer_index != self.config.train_layer:
                routing_estimator = 'task_routing'

        if routing_estimator is None:
            assert self.config.num_adapters == 1
            #(M,B)
            expert_index = torch.zeros((self.n_routers, batch_size)).long().to(self.config.device)
        elif self.config.average_domain_adapters:
            #(M,B)
            expert_index = torch.zeros((self.n_routers, batch_size)).long().to(self.config.device)
        else:
            if routing_estimator == 'task_routing':
                x_indices = torch.arange(batch_size).to(self.config.device)
                mask = torch.zeros(batch_size, self.config.num_adapters).to(self.config.device)
                if self.config.same_qdr_skt:
                    # qdr is 3 and skt is 5
                    domain_lbls[domain_lbls==3] = 5
                if self.config.same_pnt_rel:
                    # pnt is 2 and rel is 4
                    domain_lbls[domain_lbls==2] = 4

                mask[x_indices, domain_lbls] = 1.0
                #(M,B,N)
                mask = mask.unsqueeze(0).repeat(self.n_routers, 1,1)
                expert_index = torch.argmax(mask, dim=-1)
            
            elif routing_estimator == "adamix_routing":
                expert_index = (torch.randint(self.config.num_adapters, (self.n_routers, batch_size)).to(self.config.device), 
                torch.randint(self.config.num_adapters, (self.n_routers, batch_size)).to(self.config.device))

            elif routing_estimator == 'hash_routing':
                assert self.n_routers == 1
                x_indices = torch.arange(batch_size).to(self.config.device)
                mask = torch.zeros(batch_size, self.config.num_adapters).to(self.config.device)
                mask[x_indices, hash_lbls[:,self.router_index]] = 1.0
                #(M,B,N)
                mask = mask.unsqueeze(0).repeat(self.n_routers, 1,1)
                expert_index = torch.argmax(mask, dim=-1)

            else:
                if self.config.token_dropout != 0:
                    z = self.token_dropout(z)
                z_sent = torch.mean(z, dim=1)
                if self.training and self.config.jitter_noise > 0:
                    r1 = 1 - self.config.jitter_noise
                    r2 = 1 + self.config.jitter_noise
                    noise = (r1 - r2) * torch.rand(z_sent.shape).to(self.config.device) + r2
                    z_sent = z_sent * noise
                #(M,B,N)
                new_x, adapter_logits, adapter_probs = self.multi_routers(z_sent)

                if self.training:
                    if routing_estimator == 'gs_st_routing':
                        U = torch.rand(adapter_logits.shape).to(self.config.device)
                        y = adapter_logits + (-torch.log(-torch.log(U + 1e-20) + 1e-20))

                        y = F.softmax(y / self.config.adapter_temp, dim=-1)
                        probs, expert_index = y.max(dim=-1)
                        val = torch.ones_like(expert_index) - probs.detach() + probs
                    
                    elif routing_estimator == "reinf_bl_routing":
                        y_dist = torch.distributions.categorical.Categorical(logits = adapter_logits)
                        y = y_dist.sample()
                        hard_mask = F.one_hot(y, num_classes=self.config.num_adapters).float()
                        baseline_vals = self.multi_router_baselines(new_x.detach())
                        self.config.adapter_probs_list.extend([t.squeeze(0) for t in torch.split(adapter_probs, split_size_or_sections=1,dim=0)])
                        self.config.baseline_vals_list.extend([t.squeeze(0) for t in torch.split(baseline_vals, split_size_or_sections=1,dim=0)])
                        self.config.adapter_samples_list.extend([t.squeeze(0) for t in torch.split(hard_mask, split_size_or_sections=1,dim=0)])
                        _, expert_index = hard_mask.max(dim=-1)
                    
                    elif routing_estimator == 'switch_routing':
                        probs, expert_index = adapter_probs.max(dim=-1)
 
                    else:
                        probs, expert_index = adapter_probs.max(dim=-1)

                    if self.config.use_load_balancing:
                        for router_index in range(self.config.num_routers):
                            indices_load = expert_index[router_index]
                            probs_load = adapter_probs[router_index]
                            mask_load = torch.zeros_like(probs_load)
                            mask_load[torch.arange(batch_size).to(self.config.device), indices_load] = 1.0
                            avg_probs_load = torch.mean(probs_load, dim=0)
                            avg_mask_load = torch.mean(mask_load, dim=0)
                            load_loss += (batch_size * torch.sum(avg_probs_load * avg_mask_load))

                    if self.config.supervised_loss_weight != 0:
                        for router_index in range(self.config.num_routers):
                            supervised_loss += self.supervised_loss_fn(adapter_logits[router_index], domain_lbls)

                else:
                    #(M,B)
                    probs, expert_index = adapter_probs.max(dim=-1)
        
        if routing_estimator == 'parameter_averaging_routing':
            #(M,B,S,C)
            outputs = self.multi_adapters(z, None, adapter_probs)
        elif routing_estimator == 'soft_routing':
            #(M,B,S,C)
            outputs = self.multi_adapters(z, None, adapter_probs) 
        elif routing_estimator == "skill_routing":
            if self.training:
                # u = torch.rand(self.skill_weights.shape).to(self.config.device)
                # skill_allocation_matrix = torch.sigmoid(torch.log(torch.sigmoid(self.skill_weights) * u / ((1 - torch.sigmoid(self.skill_weights))*(1-u))))
                skill_allocation_matrix = RelaxedBernoulli(temperature=1., logits=self.skill_weights).rsample()
            else:
                skill_allocation_matrix = torch.sigmoid(self.skill_weights)
            adapter_probs = torch.index_select(skill_allocation_matrix, dim=0, index=domain_lbls)
            adapter_probs = adapter_probs / (torch.sum(adapter_probs, dim=-1, keepdim=True) + 1e-12)
            adapter_probs = adapter_probs.unsqueeze(0).repeat(self.config.num_routers, 1,1)
            #(M,B,S,C)
            outputs = self.multi_adapters(z, None, adapter_probs)  
        else:
            #(M,B,S,C)
            outputs = self.multi_adapters(z, expert_index)
        if self.config.analyze_model:
            # need to use expert_index to accumulate values for analysis
            self.config.analysis_list.append(expert_index)
            self.config.analysis_list.append(adapter_probs)
            self.config.analysis_list.append(orig_domain_lbls)

        if routing_estimator == 'switch_routing':
            outputs = outputs * probs[:,:,None,None]
        if routing_estimator == 'gs_st_routing' and self.training:
            outputs = outputs * val[:,:,None,None]
        
        #(B,S,C)
        outputs = torch.mean(outputs, dim=0)
        unflatten = torch.nn.Unflatten(1,(height, width))
        #(B,H,W,C)
        multi_outputs = unflatten(outputs)
        #(B,C,H,W)
        multi_outputs = multi_outputs.permute(0,3,1,2)
        multi_outputs = multi_outputs + inputs
        if self.add_batch_norm_after_adapter:
            multi_outputs = self.post_batch_norm(multi_outputs)
        if self.training:
            return multi_outputs, load_loss, supervised_loss
        else:
            return multi_outputs
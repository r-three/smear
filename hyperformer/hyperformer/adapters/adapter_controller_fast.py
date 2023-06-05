"""Implements Adapter Controller, a module that keeps multiple
layers of Adapters, and controls which adapter layer to use."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from .adapter_utils import Activations
import pdb
import copy
import math
from .adapter_utils import Router, Baseline
from torch.distributions.relaxed_bernoulli import RelaxedBernoulli


class Adapter(nn.Module):
    """Conventional Adapter layer, in which the weights of up and down sampler modules
    are parameters and are optimized."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_dim = self.config.model_dim
        n_routers = self.config.num_routers
        num_adapters = self.config.num_adapters
        self.down_sample_size = self.config.model_dim // self.config.reduction_factor
        if self.config.same_init_then_branch != -1:
                #(M,N,C,down_size)
                self.down_samplers_weights = nn.Parameter(torch.zeros(n_routers, num_adapters, self.model_dim, self.down_sample_size))
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
                    self.up_samplers_bias = nn.Parameter(torch.zeros(n_routers, num_adapters, self.model_dim))
                    nn.init.zeros_(self.up_samplers_bias)
        else:
            if self.config.same_experts_across_routers:
                #(M,N,C,down_size)
                self.down_samplers_weights = nn.Parameter(torch.zeros(num_adapters, self.model_dim, self.down_sample_size))
                nn.init.normal_(self.down_samplers_weights, std=1e-2)
                #(M,N,down_size)
                self.down_samplers_bias = nn.Parameter(torch.zeros(num_adapters, self.down_sample_size))
                nn.init.zeros_(self.down_samplers_bias)
            else:
                #(M,N,C,down_size)
                self.down_samplers_weights = nn.Parameter(torch.zeros(n_routers, num_adapters, self.model_dim, self.down_sample_size))
                nn.init.normal_(self.down_samplers_weights, std=1e-2)
                #(M,N,down_size)
                self.down_samplers_bias = nn.Parameter(torch.zeros(n_routers, num_adapters, self.down_sample_size))
                nn.init.zeros_(self.down_samplers_bias)

            if self.config.same_experts_across_routers:
                #(M,N,down_size,C)
                self.up_samplers_weights = nn.Parameter(torch.zeros(num_adapters, self.down_sample_size, self.model_dim))
                nn.init.normal_(self.up_samplers_weights, std=1e-2)

                #(M,N,C)
                self.up_samplers_bias = nn.Parameter(torch.zeros(num_adapters, self.model_dim))
                nn.init.zeros_(self.up_samplers_bias)
            else:
                #(M,N,down_size,C)
                self.up_samplers_weights = nn.Parameter(torch.zeros(n_routers, num_adapters, self.down_sample_size, self.model_dim))
                nn.init.normal_(self.up_samplers_weights, std=1e-2)
                if self.config.bias_in_up_sampler:
                    #(M,N,C)
                    self.up_samplers_bias = nn.Parameter(torch.zeros(n_routers, num_adapters, self.model_dim))
                    nn.init.zeros_(self.up_samplers_bias)
        self.activation = Activations("swish")

    def forward(self, x, expert_index, prob_dist=None):
        (batch_size,seq_len,embed_dim) = x.shape
        # x shape (B,S,C)
        # expert_index shape (M, B)
        # prob_dist shape (M, B, N)
        down_samplers_weights = self.down_samplers_weights
        down_samplers_bias = self.down_samplers_bias
        up_samplers_weights = self.up_samplers_weights

        M, N, C, down_size = down_samplers_weights.shape
        if type(expert_index) == type(()):
            if self.training:
                #(M,B,C,down_size)
                batch_down_samplers_weights = torch.gather(down_samplers_weights, 1, expert_index[0][:,:,None, None].repeat(1,1,embed_dim, down_size))
                #(M,B,down_size)
                batch_down_samplers_bias = torch.gather(down_samplers_bias, 1, expert_index[0][:,:,None].repeat(1,1,down_size))
                #(M,B,S,down_size)
                z = torch.matmul(x[None,:,:,:], batch_down_samplers_weights) + batch_down_samplers_bias[:,:,None,:]
                z = self.activation(z)
                #(M,B,down_size,C)
                batch_up_samplers_weights = torch.gather(up_samplers_weights, 1, expert_index[1][:,:,None, None].repeat(1,1, down_size, embed_dim))
                #(M,B,S,C)
                u = torch.matmul(z,batch_up_samplers_weights)
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
                # (M, B, N, C, down_size)
                batch_down_samplers_weights = down_samplers_weights.unsqueeze(1).repeat(1, batch_size, 1, 1, 1)
                # (M, B, N, down_size)
                batch_down_samplers_bias = down_samplers_bias.unsqueeze(1).repeat(1, batch_size, 1, 1)
            else:
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
            batch_down_samplers_weights = torch.gather(down_samplers_weights, 1, expert_index[:,:,None, None].repeat(1,1,embed_dim, down_size))
            #(M,B,down_size)
            batch_down_samplers_bias = torch.gather(down_samplers_bias, 1, expert_index[:,:,None].repeat(1,1,down_size))

        if self.config.routing_estimator == 'soft_routing':
            inp = x.unsqueeze(1).repeat(1, self.config.num_adapters, 1, 1)
            # (M, B, N, S, down_size)
            z = torch.matmul(inp[None, :,:,:,:], batch_down_samplers_weights) + batch_down_samplers_bias[:,:,:,None,:]
        else:        
            #(M,B,S,down_size)
            z = torch.matmul(x[None,:,:,:], batch_down_samplers_weights) + batch_down_samplers_bias[:,:,None,:]
            z = self.activation(z)

        M, N, down_size, C = up_samplers_weights.shape
        if prob_dist is not None:
            if self.config.routing_estimator == 'soft_routing':
                # (M, B, N, down_size, C)
                batch_up_samplers_weights = up_samplers_weights.unsqueeze(1).repeat(1, batch_size, 1, 1, 1)
            else:
                # (M, N, down_size*C)
                up_samplers_weights = up_samplers_weights.reshape(M, N, -1)
                # (M, B, down_size*C)
                batch_up_samplers_weights = torch.matmul(prob_dist, up_samplers_weights)
                # (M, B, down_size, C)
                batch_up_samplers_weights = batch_up_samplers_weights.unsqueeze(3).reshape(M,batch_size,down_size,C)
        else:
            #(M,B,down_size,C)
            batch_up_samplers_weights = torch.gather(up_samplers_weights, 1, expert_index[:,:,None, None].repeat(1,1, down_size, embed_dim))

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

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tasks = config.tasks
        self.model_dim = self.config.model_dim
        if self.config.num_adapters == 1:
            self.n_routers = 1
        else:
            self.n_routers = self.config.num_routers
        self.multi_adapters = Adapter(config)
        self.multi_routers = Router(config)
        if self.config.routing_estimator == 'reinf_bl_routing':
            self.multi_baselines = Baseline(config)
        if self.config.routing_estimator == "skill_routing":
            self.skill_weights = nn.Parameter(torch.empty((len(self.config.tasks), self.config.num_adapters)).uniform_(-1e-3, 1e-3))
        if self.config.routing_estimator == 'cluster_routing':
            self.clusters = torch.randn((self.config.num_adapters, self.config.model_dim)).to(self.config.device) * self.config.cluster_initscale
            self.momentum = self.config.cluster_momentum

        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        if self.add_layer_norm_before_adapter:
            self.pre_layer_norm = nn.LayerNorm(self.config.model_dim)
        if self.add_layer_norm_after_adapter:
            self.post_layer_norm = nn.LayerNorm(self.config.model_dim)
        self.task_to_adapter = {task: index for index, task in enumerate(self.tasks)}
        self.better_task_to_adapter = {task: index for index, task in enumerate(self.tasks)}
        if self.config.token_dropout != 0:
            self.token_dropout = nn.Dropout(p=self.config.token_dropout)
        if 'rte' in self.tasks and 'mnli' in self.tasks:
            self.better_task_to_adapter['rte'] = self.better_task_to_adapter['mnli']
        if self.config.probe_input_features:
            self.probe_mlp = nn.Linear(self.config.model_dim, len(self.tasks))
            self.probe_loss = nn.CrossEntropyLoss(reduction="none")
        if self.config.supervised_loss_weight != 0:
            self.supervised_loss_fn = nn.CrossEntropyLoss()

    def forward(self, tasks, orig_tasks, hash_lbls, inputs, given_hidden_states=None):
        """Retrieves the adapter layer corresponding to the given
        task. It freezes the adapter layers for all the other tasks
        and call the selected adapter layer.
        Args:
            task: the name of the current task.
            inputs: the inputs to feed in in the adapter layer.
        Returns:
            outputs of the adapter layer."""
        batch_size = inputs.shape[0]
        z = self.pre_layer_norm(inputs) if self.add_layer_norm_before_adapter else inputs
        if self.config.probe_input_features and (self.config.num_adapters > 1):
            z_sent = torch.mean(z, dim=1)
            z_logits = self.probe_mlp(z_sent)
            task_indices = [self.task_to_adapter[task] for task in tasks]
            task_indices = torch.tensor(task_indices).to(self.config.device)
            if self.training:
                loss = self.probe_loss(z_logits, task_indices)
                return inputs, torch.mean(loss)
            else:
                z_pred = torch.argmax(z_logits, dim=-1)
                correct_pred = z_pred == task_indices
                self.config.num_count_task_pred = torch.sum(correct_pred).item()
                self.config.den_count_task_pred = task_indices.shape[0]
                return inputs

        routing_estimator = self.config.routing_estimator
        load_loss = torch.tensor(0.0).to(self.config.device) 
        supervised_loss = torch.tensor(0.0).to(self.config.device) 

        if routing_estimator is None:
            assert self.config.num_adapters == 1
            #(M,B)
            expert_index = torch.zeros((self.n_routers, batch_size)).long().to(self.config.device)
        else:
            if routing_estimator == 'task_routing':
                if given_hidden_states is None:
                    # print('Inside encoder')
                    if self.config.same_rte_mnli:
                        # print(self.better_task_to_adapter)
                        task_indices = [self.better_task_to_adapter[task] for task in tasks]
                    else:
                        # print(self.task_to_adapter)
                        task_indices = [self.task_to_adapter[task] for task in tasks]
                else:
                    # print('Inside decoder')
                    if self.config.same_rte_mnli and not self.config.only_in_encoder:
                        # print(self.better_task_to_adapter)
                        task_indices = [self.better_task_to_adapter[task] for task in tasks]
                    else:
                        # print(self.task_to_adapter)
                        task_indices = [self.task_to_adapter[task] for task in tasks]

                x_indices = torch.arange(batch_size).to(self.config.device)
                mask = torch.zeros(batch_size, self.config.num_adapters).to(self.config.device)
                mask[x_indices, task_indices] = 1.0
                #(M,B,N)
                mask = mask.unsqueeze(0).repeat(self.n_routers, 1,1)
                expert_index = torch.argmax(mask, dim=-1)
            
            elif routing_estimator == 'random_routing':
                expert_index = torch.randint(self.config.num_adapters, (self.n_routers, batch_size)).to(self.config.device)

            elif routing_estimator == 'adamix_routing':
                expert_index = (torch.randint(self.config.num_adapters, (self.n_routers, batch_size)).to(self.config.device),
                torch.randint(self.config.num_adapters, (self.n_routers, batch_size)).to(self.config.device))
            
            elif routing_estimator == 'cluster_routing':
                # only works when number of routers is 1
                #(B,C)
                z_sent = torch.mean(z, dim=1)
                # self.clusters (6,C) | z_sent (B,C)
                if self.config.cluster_distance_metric == 'l2':
                    # (B,6,C)
                    distances = (z_sent.unsqueeze(1) - self.clusters.unsqueeze(0))**2
                    # (B,6)
                    distances = torch.mean(distances, dim=-1)
                    expert_index = torch.argmin(distances, dim=-1)
                elif self.config.cluster_distance_metric == 'cosine':
                    # (B,6)
                    distances = torch.matmul(z_sent, torch.transpose(self.clusters, 0,1)) / ((torch.norm(z_sent,dim=1)).unsqueeze(1) * torch.norm(self.clusters, dim=1).unsqueeze(0))
                    expert_index = torch.argmax(distances, dim=-1)
                for cluster_index in range(self.config.num_adapters):
                    cluster_sents = z_sent[expert_index==cluster_index]
                    if cluster_sents.shape[0] != 0:
                        self.clusters[cluster_index] = (1-self.momentum)*(self.clusters[cluster_index]) + self.momentum*(torch.mean(cluster_sents, dim=0))
                # (M=1,B)
                expert_index = expert_index.unsqueeze(0)
            
            elif routing_estimator == 'hash_routing':
                x_indices = torch.arange(batch_size).to(self.config.device)
                mask = torch.zeros(batch_size, self.config.num_adapters).to(self.config.device)
                mask[x_indices, hash_lbls[:,self.router_index]] = 1.0
                #(M,B,N)
                mask = mask.unsqueeze(0).repeat(self.n_routers, 1,1)
                expert_index = torch.argmax(mask, dim=-1)

            else:
                if given_hidden_states is not None:
                    if self.config.token_dropout != 0:
                        given_hidden_states = self.token_dropout(given_hidden_states)
                    z_sent = torch.mean(given_hidden_states, dim=1)
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
                        baseline_vals = self.multi_baselines(new_x.detach())
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
                        task_labels = [-100 if task=='None' else self.task_to_adapter[task] for task in tasks]
                        task_labels = torch.tensor(task_labels).to(self.config.device)
                        for router_index in range(self.config.num_routers):
                            supervised_loss += self.supervised_loss_fn(adapter_logits[router_index], task_labels)
                else:
                    #(M,B)
                    probs, expert_index = adapter_probs.max(dim=-1)
                    # y_dist = torch.distributions.categorical.Categorical(logits = adapter_logits)
                    # y = y_dist.sample()
                    # hard_mask = F.one_hot(y, num_classes=self.config.num_adapters).float()
                    # _, expert_index = hard_mask.max(dim=-1)
                    # if self.router_index == 11:
                    #     print(f"the tasks are {tasks}")
                    #     print(f"the expert indices are {expert_index}")
                    # if given_hidden_states is None and self.router_index == 20:
                    #     print(adapter_probs[0,0])
                    #     print(torch.norm(self.multi_routers.router_weights, dim=1))
                    #     import pdb; pdb.set_trace()

        if routing_estimator == 'parameter_averaging_routing':
            #(M,B,S,C)
            outputs = self.multi_adapters(z, None, adapter_probs)
            # rand_index = torch.randint(batch_size,(1,)).item()
            # print(f"router probs of example {rand_index} is {adapter_probs[0,rand_index]}")
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
            task_indices = torch.tensor([self.task_to_adapter[task] for task in tasks]).to(self.config.device)
            adapter_probs = torch.index_select(skill_allocation_matrix, dim=0, index=task_indices)
            adapter_probs = adapter_probs / (torch.sum(adapter_probs, dim=-1, keepdim=True) + 1e-12)
            adapter_probs = adapter_probs.unsqueeze(0).repeat(self.config.num_routers, 1,1)
            #(M,B,S,C)
            outputs = self.multi_adapters(z, None, adapter_probs)  
        else:
            #(M,B,S,C)
            outputs = self.multi_adapters(z, expert_index)
        if self.config.analyze_model and self.config.num_adapters != 1:
            # need to use expert_index to accumulate values for analysis
            if self.router_index not in self.config.analysis_list and given_hidden_states is None:
                self.config.analysis_list[self.router_index] = [expert_index, adapter_probs, [self.task_to_adapter[task] for task in orig_tasks]]
            elif given_hidden_states is not None:
                self.config.analysis_list[self.router_index] = [expert_index, adapter_probs, [self.task_to_adapter[task] for task in orig_tasks]]  
            # print(f'router index is {self.router_index}')
            # print(f"encoder is {given_hidden_states==None}")
            # print(f"length of analysis list is {len(self.config.analysis_list)}")
            # import pdb; pdb.set_trace()
        if routing_estimator == 'switch_routing':
            outputs = outputs * probs[:,:,None,None]
        if routing_estimator == 'gs_st_routing' and self.training:
            outputs = outputs * val[:,:,None,None]
        
        #(B,S,C)
        outputs = torch.mean(outputs, dim=0)

        if self.add_layer_norm_after_adapter:
            outputs = self.post_layer_norm(outputs)
        outputs = outputs + inputs
        if self.training:
            return outputs, load_loss, supervised_loss
        else:
            return outputs
"""Implements Adapter Controller, a module that keeps multiple
layers of Adapters, and controls which adapter layer to use."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from transformers.activations import get_activation

from .adapter_configuration import AdapterConfig, MetaAdapterConfig
from .adapter_modeling import Adapter, AdapterHyperNet
from .adapter_utils import LayerNormHyperNet
import pdb
import copy

class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.
    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.
    There are two functions:
    dispatch - take an input Tensor and create input Tensors for each expert.
    combine - take output Tensors from each expert and form a combined output
      Tensor.  Outputs from different experts for the same batch element are
      summed together, weighted by the provided "gates".
    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.
    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().
    Example use:
    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.
    dispatcher = SparseDispatcher(num_experts, gates)
    expert_inputs = dispatcher.dispatch(inputs)
    expert_outputs = [experts[i](expert_inputs[i]) for i in range(num_experts)]
    outputs = dispatcher.combine(expert_outputs)
    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))
    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates, device, training=False):
        """Create a SparseDispatcher."""

        self._gates = gates
        self._num_experts = num_experts
        # sort experts
        sorted_experts, index_sorted_experts = torch.nonzero(gates).sort(0)
        # drop indices
        _, self._expert_index = sorted_experts.split(1, dim=1)
        # get according batch index for each expert
        self._batch_index = torch.nonzero(gates)[index_sorted_experts[:, 1],0]
        # calculate num samples that each expert gets
        self._part_sizes = (gates > 0).sum(0).tolist()
        # expand gates to match with self._batch_index
        gates_exp = gates[self._batch_index.flatten()]
        self._nonzero_gates = torch.gather(gates_exp, 1, self._expert_index)
        self.device = device
        self.training = training

    def dispatch(self, inp):
        """Create one input Tensor for each expert.
        The `Tensor` for a expert `i` contains the slices of `inp` corresponding
        to the batch elements `b` where `gates[b, i] > 0`.
        Args:
          inp: a `Tensor` of shape "[batch_size, <extra_input_dims>]`
        Returns:
          a list of `num_experts` `Tensor`s with shapes
            `[expert_batch_size_i, <extra_input_dims>]`.
        """

        # assigns samples to experts whose gate is nonzero

        # expand according to batch index so we can just split by _part_sizes
        inp_exp = inp[self._batch_index].squeeze(1)
        return torch.split(inp_exp, self._part_sizes, dim=0)


    def combine(self, expert_out, seq_len, dim, multiply_by_gates=True):
        """Sum together the expert output, weighted by the gates.
        The slice corresponding to a particular batch element `b` is computed
        as the sum over all experts `i` of the expert output, weighted by the
        corresponding gate values.  If `multiply_by_gates` is set to False, the
        gate values are ignored.
        Args:
          expert_out: a list of `num_experts` `Tensor`s, each with shape
            `[expert_batch_size_i, <extra_output_dims>]`.
          multiply_by_gates: a boolean
        Returns:
          a `Tensor` with shape `[batch_size, <extra_output_dims>]`.
        """
        # apply exp to expert outputs, so we are not longer in log space
        stitched = torch.cat(expert_out, 0)
        if multiply_by_gates:
            if seq_len == 1:
                stitched = stitched.mul(self._nonzero_gates).unsqueeze(1)
            else:
                stitched = stitched.mul(self._nonzero_gates.unsqueeze(1))

        if not self.training:
            zeros = torch.zeros(self._gates.size(0), dim).unsqueeze(1).repeat(1,seq_len,1).to(self.device)
        else:
            zeros = torch.zeros(self._gates.size(0), dim, requires_grad=True).unsqueeze(1).repeat(1,seq_len, 1).to(self.device)
        # combine samples that have been processed by the same k experts
        combined = zeros.index_add(0, self._batch_index, stitched.float()).to(self.device)

        return combined


    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert `Tensor`s.
        Returns:
          a list of `num_experts` one-dimensional `Tensor`s with type `tf.float32`
              and shapes `[expert_batch_size_i]`
        """
        # split nonzero gates for each expert
        return torch.split(self._nonzero_gates, self._part_sizes, dim=0)

class Router(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.ff = nn.Linear(self.config.model_dim, self.config.num_adapters)
        
    def forward(self,x):
        adapter_logits = self.ff(x)
        adapter_probs = F.softmax(adapter_logits, dim=-1)
        return adapter_logits, adapter_probs

class Baseline(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.ff1 = nn.Linear(self.config.model_dim, self.config.model_dim // self.config.bl_reduction_factor)
        self.ff2 = nn.Linear(self.config.model_dim // self.config.bl_reduction_factor, 1)
        
    def forward(self,x):
        return self.ff2(self.ff1(x))

class AdapterController(nn.Module):
    """Implements Adapter controller module which controls the logics of
    putting adapter layers within transformer's layers."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tasks = config.tasks
        if self.config.num_adapters == 1:
            self.n_routers = 1
        else:
            self.n_routers = self.config.num_routers
        self.multi_adapters = []
        self.multi_routers = []
        self.multi_router_batch_norms = []
        self.multi_baselines = []
        for i in range(self.n_routers):
            self.multi_adapters.append(self.construct_adapters(self.config.num_adapters))
            self.multi_routers.append(Router(config))
            if self.config.routing_estimator == 'reinf_bl_routing':
                self.multi_baselines.append(Baseline(config))
            self.multi_router_batch_norms.append(nn.BatchNorm1d(self.config.model_dim))
        self.multi_adapters = nn.ModuleList(self.multi_adapters)
        self.multi_routers = nn.ModuleList(self.multi_routers)
        self.multi_router_batch_norms = nn.ModuleList(self.multi_router_batch_norms)
        if self.config.routing_estimator == 'reinf_bl_routing':
            self.multi_baselines = nn.ModuleList(self.multi_baselines)

        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        if self.add_layer_norm_before_adapter:
            self.pre_layer_norm = nn.LayerNorm(config.input_dim)
        if self.add_layer_norm_after_adapter:
            self.post_layer_norm = nn.LayerNorm(config.input_dim)
        if self.config.routing_estimator == 'task_routing':
            self.task_to_adapter = {task: index for index, task in enumerate(self.tasks)}

    def construct_adapters(self, num_adapters):
        """
        Constructs adapter layers and adds them to a dictionary for the given
        number of adapters.
        Args:
            num_adapters: number of adapters in our model.
        """
        adapters = nn.ModuleDict(dict())
        for i in range(num_adapters):
            adapters['adapter' + str(i)] = Adapter(self.config)
        return adapters

    def forward(self, tasks, inputs):
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
        z_pre_sent = torch.mean(z, dim=1)
        multi_outputs = []
        multi_load_loss = torch.tensor(0).to(self.config.device)
        for index in range(self.n_routers):
            z_sent = self.multi_router_batch_norms[index](z_pre_sent)
            adapter_logits, adapter_probs = self.multi_routers[index](z_sent)
            if self.config.routing_estimator is None:
                assert self.config.num_adapters == 1
                outputs = self.multi_adapters[index]['adapter' + str(0)](z)
                load_loss = torch.tensor(0.0).to(self.config.device)
            else:
                if self.config.routing_estimator == 'task_routing':
                    task_indices = [self.task_to_adapter[task] for task in tasks]
                    x_indices = torch.arange(batch_size).to(self.config.device)
                    task_indices = torch.tensor(task_indices).to(self.config.device)
                    mask = torch.zeros(batch_size, self.config.num_adapters).to(self.config.device)
                    mask[x_indices, task_indices] = 1.0
                    load_loss = torch.tensor(0.0).to(self.config.device)
                else:
                    if self.training:
                        if self.config.routing_estimator == 'gs_st_routing':
                            U = torch.rand(adapter_logits.shape).to(self.config.device)
                            y = adapter_logits + (-torch.log(-torch.log(U + 1e-20) + 1e-20))

                            y = F.softmax(y / self.config.adapter_temp, dim=-1)
                            _, ind = y.max(dim=-1)
                            shape = y.size()
                            hard_mask = torch.zeros_like(y).reshape(-1, shape[-1])
                            hard_mask.scatter_(1, ind.unsqueeze(1), 1)
                            hard_mask = hard_mask.reshape(*shape)
                            # Set gradients w.r.t. hard_mask gradients w.r.t. y
                            mask = hard_mask - y.detach() + y

                        elif self.config.routing_estimator == "reinf_bl_routing":
                            y_dist = torch.distributions.categorical.Categorical(logits = adapter_logits)
                            y = y_dist.sample()
                            hard_mask = F.one_hot(y, num_classes=self.config.num_adapters).float()
                            baseline_vals = self.multi_baselines[index](z_sent)
                            self.config.adapter_probs_list.append(adapter_probs)
                            self.config.baseline_vals_list.append(baseline_vals)
                            self.config.adapter_samples_list.append(hard_mask)
                            mask = hard_mask
                        elif self.config.routing_estimator == 'switch_routing':
                            with torch.no_grad():
                                val, ind = adapter_probs.max(dim=-1)
                            hard_mask = torch.zeros_like(adapter_probs)
                            hard_mask.scatter_(1, ind.unsqueeze(1), 1)

                            mask = (adapter_probs * hard_mask)

                        else:
                            pass
                        input_f = torch.sum(hard_mask, dim=0)/ batch_size
                        prob_f = torch.sum(adapter_probs, dim = 0)/batch_size
                        load_loss = torch.sum(input_f*prob_f, dim = -1) * self.config.num_adapters
                        multi_load_loss = multi_load_loss + load_loss
                        
                    else:
                        with torch.no_grad():
                            val, ind = adapter_probs.max(dim=-1)
                        hard_mask = torch.zeros_like(adapter_probs)
                        hard_mask.scatter_(1, ind.unsqueeze(1), 1)
                        if self.config.routing_estimator == 'switch_routing':
                            mask = (adapter_probs * hard_mask)
                        else:
                            mask = hard_mask
                dispatcher = SparseDispatcher(self.config.num_adapters, mask, self.config.device, self.training)
                expert_inputs = dispatcher.dispatch(z)
                expert_outputs = [self.multi_adapters[index]['adapter' + str(i)](expert_inputs[i]) for i in range(self.config.num_adapters)]
                outputs = dispatcher.combine(expert_outputs, z.shape[1], z.shape[2])
            multi_outputs.append(outputs)
        
        multi_outputs = torch.cat([outputs.unsqueeze(1) for outputs in multi_outputs], dim = 1)
        multi_outputs = torch.mean(multi_outputs, dim = 1)
        if self.add_layer_norm_after_adapter:
            multi_outputs = self.post_layer_norm(multi_outputs)
        multi_outputs = multi_outputs + inputs
        if self.training:
            return multi_outputs, multi_load_loss
        else:
            return multi_outputs


class MetaAdapterController(nn.Module):
    """Implements Meta Adapter controller module, in which
    the adapter layers' weights are generated from a hyper-network.
    In this case, task-embeddings are fixed, and the task
    embeddings will be initialized to random."""

    def __init__(self, config):
        super().__init__()
        self.device = config.device
        self.adapters = nn.ModuleDict(dict())
        self.config = config
        self.input_dim = config.input_dim
        self.down_sample_size = self.input_dim // config.reduction_factor
        self.meta_up_sampler = AdapterHyperNet(config, self.input_dim, self.down_sample_size)
        self.meta_down_sampler = AdapterHyperNet(config, self.down_sample_size, self.input_dim)
        self.activation_type = config.non_linearity.lower()
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter
        self.conditional_layer_norm = config.conditional_layer_norm
        if self.add_layer_norm_after_adapter:
            if self.conditional_layer_norm:
                self.post_layernorm_hypernet = LayerNormHyperNet(config)
            else:
                self.post_layer_norm = nn.LayerNorm(self.input_dim)
        if self.add_layer_norm_before_adapter:
            if self.conditional_layer_norm:
                self.pre_layernorm_hypernet = LayerNormHyperNet(config)
            else:
                self.pre_layer_norm = nn.LayerNorm(self.input_dim)

    def call_adapter(self, inputs, task_embedding):
        weight_up, bias_up = self.meta_up_sampler(task_embedding)
        weight_down, bias_down = self.meta_down_sampler(task_embedding)
        down = F.linear(inputs, weight=weight_down, bias=bias_down)
        middle = get_activation(self.activation_type)(down)
        output = F.linear(middle, weight=weight_up, bias=bias_up)
        return output

    def apply_pre_layer_norm(self, inputs, task_embeddings):
        """Applies pre layer norm to the inputs."""
        if self.conditional_layer_norm:
            weight, bias = self.pre_layernorm_hypernet(task_embeddings)
            return torch.nn.functional.layer_norm(inputs, (self.input_dim,), weight=weight, bias=bias)
        else:
            return self.pre_layer_norm(inputs)

    def apply_post_layer_norm(self, inputs, task_embeddings):
        """Applies post layer norm to the inputs."""
        if self.conditional_layer_norm:
            weight, bias = self.post_layernorm_hypernet(task_embeddings)
            return torch.nn.functional.layer_norm(inputs, (self.input_dim,), weight=weight, bias=bias)
        else:
            return self.post_layer_norm(inputs)

    def forward(self, task_embedding, inputs):
        """Retrieves the adapter layer corresponding to the given
        task. It freezes the adapter layers for all the other tasks
        and call the selected adapter layer.
        Args:
            task: the name of the current task.
            inputs: the inputs to feed in in the adapter layer.
        Returns:
            outputs of the adapter layer."""
        z = self.apply_pre_layer_norm(inputs, task_embedding) if self.add_layer_norm_before_adapter else inputs
        outputs = self.call_adapter(z, task_embedding)
        if self.add_layer_norm_after_adapter:
            outputs = self.apply_post_layer_norm(outputs, task_embedding)
        outputs = outputs + inputs
        return outputs


class MetaLayersAdapterController(nn.Module):
    """Implements Meta Adapter controller module, in which
    the adapter layers' weights are generated from a unique hyper-network."""

    def __init__(self, config):
        super().__init__()
        self.activation_type = config.non_linearity.lower()
        self.input_dim = config.input_dim
        self.add_layer_norm_before_adapter = config.add_layer_norm_before_adapter
        self.add_layer_norm_after_adapter = config.add_layer_norm_after_adapter

    def apply_layer_norm(self, inputs, layer_norm_weights):
        """Applies layer norm to the inputs."""
        return torch.nn.functional.layer_norm(inputs, (self.input_dim,),
                                              weight=layer_norm_weights.weight,
                                              bias=layer_norm_weights.bias)

    def call_adapter(self, inputs, adapter_weights):
        """Computes the output of the adapter layers."""
        down = F.linear(inputs, weight=adapter_weights.down.weight,
                        bias=adapter_weights.down.bias)
        middle = get_activation(self.activation_type)(down)
        output = F.linear(middle, weight=adapter_weights.up.weight,
                          bias=adapter_weights.up.bias)
        return output

    def forward(self, inputs, adapter_weights):
        z = self.apply_layer_norm(inputs, adapter_weights.pre_norm) if self.add_layer_norm_before_adapter else inputs
        outputs = self.call_adapter(z, adapter_weights)
        if self.add_layer_norm_after_adapter:
            outputs = self.apply_layer_norm(outputs, adapter_weights.post_norm)
        outputs = outputs + inputs
        return outputs


class AutoAdapterController(nn.Module):
    """Generic adapter controller class to instantiate different adapter
    controller classes."""

    @classmethod
    def get(cls, config):
        if isinstance(config, MetaAdapterConfig):
            return MetaAdapterController(config)
        elif isinstance(config, AdapterConfig):
            return AdapterController(config)
        raise ValueError("Unrecognized adapter config", config)

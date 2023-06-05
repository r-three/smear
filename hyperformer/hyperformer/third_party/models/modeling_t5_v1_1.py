""" PyTorch T5 model. """

import copy
import torch
import warnings
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import BaseModelOutput
from transformers.modeling_t5 import (T5PreTrainedModel, T5LayerNorm, T5Block)
from transformers.utils import logging

from hyperformer.adapters import (AutoAdapterController, MetaAdapterConfig,
                              TaskEmbeddingController, LayerNormHyperNet,
                              AdapterLayersHyperNetController,
                              MetaLayersAdapterController,
                              AdapterLayersOneHyperNetController)
from hyperformer.adapters.adapter_controller_fast import AdapterController
from hyperformer.adapters.lora_controller import LoRAController
from hyperformer.adapters.ia3_controller import IA3Controller
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput
)
import math
import pdb
from transformers.activations import ACT2FN

logger = logging.get_logger(__name__)

class T5DenseGatedActDense(nn.Module):
    def __init__(self, config,adapter_config):
        super().__init__()
        self.adapter_config = adapter_config
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN["gelu_new"]
        if self.adapter_config.train_lora:
            self.wi_0_lora_controller = LoRAController(adapter_config, config.d_model, config.d_ff, config.is_decoder)
            self.wi_1_lora_controller = LoRAController(adapter_config, config.d_model, config.d_ff, config.is_decoder)
            self.wo_lora_controller = LoRAController(adapter_config, config.d_ff, config.d_model, config.is_decoder)
        if self.adapter_config.train_ia3:
            self.l_ffi_controller = IA3Controller(adapter_config, config.d_ff, config.is_decoder)
            self.l_ffo_controller = IA3Controller(adapter_config, config.d_model, config.is_decoder)

    def forward(self, hidden_states,encoder_hidden_states=None,task=None,hash_lbl=None,orig_task=None):
        if self.adapter_config.train_lora:
            if self.training:
                wi_0_lora, load_loss, supervised_loss = self.wi_0_lora_controller(task, orig_task, hash_lbl, hidden_states, encoder_hidden_states)
                self.adapter_config.load_loss_accm += load_loss
                self.adapter_config.supervised_loss_accm += supervised_loss
            else:
                wi_0_lora = self.wi_0_lora_controller(task, orig_task, hash_lbl, hidden_states, encoder_hidden_states)
            hidden_gelu = self.act(self.wi_0(hidden_states) + wi_0_lora)

            if self.training:
                wi_1_lora, load_loss, supervised_loss = self.wi_1_lora_controller(task, orig_task, hash_lbl, hidden_states, encoder_hidden_states)
                self.adapter_config.load_loss_accm += load_loss
                self.adapter_config.supervised_loss_accm += supervised_loss
            else:
                wi_1_lora = self.wi_1_lora_controller(task, orig_task, hash_lbl, hidden_states, encoder_hidden_states)
            
            hidden_linear = self.wi_1(hidden_states) + wi_1_lora
            hidden_states = hidden_gelu * hidden_linear
            hidden_states = self.dropout(hidden_states)

            if self.training:
                wo_lora, load_loss, supervised_loss = self.wo_lora_controller(task, orig_task, hash_lbl, hidden_states, encoder_hidden_states)
                self.adapter_config.load_loss_accm += load_loss
                self.adapter_config.supervised_loss_accm += supervised_loss
            else:
                wo_lora = self.wo_lora_controller(task, orig_task, hash_lbl, hidden_states, encoder_hidden_states)

            hidden_states = self.wo(hidden_states) + wo_lora
        else:
            hidden_gelu = self.act(self.wi_0(hidden_states))
            hidden_linear = self.wi_1(hidden_states)
            hidden_states = hidden_gelu * hidden_linear
            if self.adapter_config.train_ia3:
                if self.training:
                    hidden_states, load_loss, supervised_loss = self.l_ffi_controller(task, orig_task, hash_lbl, hidden_states, encoder_hidden_states)
                    self.adapter_config.load_loss_accm += load_loss
                    self.adapter_config.supervised_loss_accm += supervised_loss
                else:
                    hidden_states = self.l_ffi_controller(task, orig_task, hash_lbl, hidden_states, encoder_hidden_states)
            hidden_states = self.dropout(hidden_states)
            hidden_states = self.wo(hidden_states)
            if self.adapter_config.train_ia3:
                if self.training:
                    hidden_states, load_loss, supervised_loss = self.l_ffo_controller(task, orig_task, hash_lbl, hidden_states, encoder_hidden_states)
                    self.adapter_config.load_loss_accm += load_loss
                    self.adapter_config.supervised_loss_accm += supervised_loss
                else:
                    hidden_states = self.l_ffo_controller(task, orig_task, hash_lbl, hidden_states, encoder_hidden_states)
        return hidden_states

class T5LayerFF(nn.Module):
    def __init__(self, config, adapter_config=None):
        super().__init__()
        self.DenseReluDense = T5DenseGatedActDense(config,adapter_config)
        self.train_adapters = config.train_adapters
        self.adapter_config = adapter_config
        if self.train_adapters:
            self.adapter_controller = AdapterController(adapter_config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states, encoder_hidden_states=None,task=None, hash_lbl=None, orig_task=None):
        norm_x = self.layer_norm(hidden_states)
        y = self.DenseReluDense(norm_x,encoder_hidden_states,task=task,hash_lbl=hash_lbl, orig_task=orig_task)
        if self.train_adapters:
            if self.training:
                y, load_loss, supervised_loss = self.adapter_controller(task, orig_task, hash_lbl, y, encoder_hidden_states)
                self.adapter_config.load_loss_accm += load_loss
                self.adapter_config.supervised_loss_accm += supervised_loss
            else:
                y = self.adapter_controller(task, orig_task, hash_lbl, y, encoder_hidden_states)

        layer_output = hidden_states + self.dropout(y)
        return layer_output

class T5Attention(nn.Module):
    def __init__(self, config, adapter_config, has_relative_attention_bias=False, is_bidirectional=False):
        super().__init__()
        self.is_bidirectional = is_bidirectional
        self.adapter_config = adapter_config
        self.is_decoder = config.is_decoder
        self.has_relative_attention_bias = has_relative_attention_bias

        self.relative_attention_num_buckets = config.relative_attention_num_buckets
        self.d_model = config.d_model
        self.d_kv = config.d_kv
        self.n_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.inner_dim = self.n_heads * self.d_kv

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.k = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.v = nn.Linear(self.d_model, self.inner_dim, bias=False)
        self.o = nn.Linear(self.inner_dim, self.d_model, bias=False)
        if self.adapter_config.train_lora:
            self.q_lora_controller = LoRAController(adapter_config, self.d_model, self.inner_dim, config.is_decoder)
            self.k_lora_controller = LoRAController(adapter_config, self.d_model, self.inner_dim, config.is_decoder)
            self.v_lora_controller = LoRAController(adapter_config, self.d_model, self.inner_dim, config.is_decoder)
            self.o_lora_controller = LoRAController(adapter_config, self.inner_dim, self.d_model, config.is_decoder)
        if self.adapter_config.train_ia3:
            self.l_k_controller = IA3Controller(adapter_config, self.inner_dim, config.is_decoder)
            self.l_v_controller = IA3Controller(adapter_config, self.inner_dim, config.is_decoder)
            self.l_o_controller = IA3Controller(adapter_config, self.d_model, config.is_decoder)
        if self.has_relative_attention_bias:
            self.relative_attention_bias = nn.Embedding(self.relative_attention_num_buckets, self.n_heads)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(heads, self.n_heads, self.d_kv, self.pruned_heads)
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.n_heads = self.n_heads - len(heads)
        self.inner_dim = self.d_kv * self.n_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    @staticmethod
    def _relative_position_bucket(relative_position, bidirectional=True, num_buckets=32, max_distance=128):
        """
        Adapted from Mesh Tensorflow:
        https://github.com/tensorflow/mesh/blob/0cb87fe07da627bf0b7e60475d59f95ed6b5be3d/mesh_tensorflow/transformer/transformer_layers.py#L593

        Translate relative position to a bucket number for relative attention. The relative position is defined as
        memory_position - query_position, i.e. the distance in tokens from the attending position to the attended-to
        position. If bidirectional=False, then positive relative positions are invalid. We use smaller buckets for
        small absolute relative_position and larger buckets for larger absolute relative_positions. All relative
        positions >=max_distance map to the same bucket. All relative positions <=-max_distance map to the same bucket.
        This should allow for more graceful generalization to longer sequences than the model has been trained on

        Args:
            relative_position: an int32 Tensor
            bidirectional: a boolean - whether the attention is bidirectional
            num_buckets: an integer
            max_distance: an integer

        Returns:
            a Tensor with the same shape as relative_position, containing int32 values in the range [0, num_buckets)
        """
        ret = 0
        n = -relative_position
        if bidirectional:
            num_buckets //= 2
            ret += (n < 0).to(torch.long) * num_buckets  # mtf.to_int32(mtf.less(n, 0)) * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))
        # now n is in the range [0, inf)

        # half of the buckets are for exact increments in positions
        max_exact = num_buckets // 2
        is_small = n < max_exact

        # The other half of the buckets are for logarithmically bigger bins in positions up to max_distance
        val_if_large = max_exact + (
            torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).to(torch.long)
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def compute_bias(self, qlen, klen):
        """ Compute binned relative position bias """
        context_position = torch.arange(qlen, dtype=torch.long)[:, None]
        memory_position = torch.arange(klen, dtype=torch.long)[None, :]
        relative_position = memory_position - context_position  # shape (qlen, klen)
        rp_bucket = self._relative_position_bucket(
            relative_position,  # shape (qlen, klen)
            bidirectional=self.is_bidirectional,
            num_buckets=self.relative_attention_num_buckets,
        )
        rp_bucket = rp_bucket.to(self.relative_attention_bias.weight.device)
        values = self.relative_attention_bias(rp_bucket)  # shape (qlen, klen, num_heads)
        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, qlen, klen)
        return values

    def forward(
        self,
        input,
        encoder_hidden_states=None,
        mask=None,
        kv=None,
        position_bias=None,
        past_key_value=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        task=None,
        orig_task=None,
        hash_lbl=None,
    ):
        """
        Self-attention (if kv is None) or attention over source sentence (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        # past_key_value[0] is (bs, n_heads, q_len - 1, dim_per_head)
        bs, qlen, dim = input.size()

        if past_key_value is not None:
            assert self.is_decoder is True, "Encoder cannot cache past key value states"
            assert (
                len(past_key_value) == 2
            ), "past_key_value should have 2 past states: keys and values. Got {} past states".format(
                len(past_key_value)
            )
            real_qlen = qlen + past_key_value[0].shape[2] if query_length is None else query_length
        else:
            real_qlen = qlen

        if kv is None:
            klen = real_qlen
        else:
            klen = kv.size(1)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, self.d_kv).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.inner_dim)

        if self.adapter_config.train_lora:
            if self.training:
                q_lora, load_loss, supervised_loss = self.q_lora_controller(task, orig_task, hash_lbl, input, encoder_hidden_states)
                self.adapter_config.load_loss_accm += load_loss
                self.adapter_config.supervised_loss_accm += supervised_loss
            else:
                q_lora = self.q_lora_controller(task, orig_task, hash_lbl, input, encoder_hidden_states)
            q = shape(self.q(input) + q_lora)
        else:
            q = shape(self.q(input))  # (bs, n_heads, qlen, dim_per_head)

        if kv is None:
            if self.adapter_config.train_lora:
                if self.training:
                    k_lora, load_loss, supervised_loss = self.k_lora_controller(task, orig_task, hash_lbl, input, encoder_hidden_states)
                    self.adapter_config.load_loss_accm += load_loss
                    self.adapter_config.supervised_loss_accm += supervised_loss
                else:
                    k_lora = self.k_lora_controller(task, orig_task, hash_lbl, input, encoder_hidden_states)
                k = shape(self.k(input) + k_lora)

                if self.training:
                    v_lora, load_loss, supervised_loss = self.v_lora_controller(task, orig_task, hash_lbl, input, encoder_hidden_states)
                    self.adapter_config.load_loss_accm += load_loss
                    self.adapter_config.supervised_loss_accm += supervised_loss
                else:
                    v_lora = self.v_lora_controller(task, orig_task, hash_lbl, input, encoder_hidden_states)
                v = shape(self.v(input) + v_lora)
            elif self.adapter_config.train_ia3:
                if self.training:
                    k, load_loss, supervised_loss = self.l_k_controller(task, orig_task, hash_lbl, self.k(input), encoder_hidden_states)
                    self.adapter_config.load_loss_accm += load_loss
                    self.adapter_config.supervised_loss_accm += supervised_loss
                    k = shape(k)

                    v, load_loss, supervised_loss = self.l_v_controller(task, orig_task, hash_lbl, self.v(input), encoder_hidden_states)
                    self.adapter_config.load_loss_accm += load_loss
                    self.adapter_config.supervised_loss_accm += supervised_loss 
                    v = shape(v)
                else:
                    k = shape(self.l_k_controller(task, orig_task, hash_lbl, self.k(input), encoder_hidden_states))
                    v = shape(self.l_v_controller(task, orig_task, hash_lbl, self.v(input), encoder_hidden_states))
            else:
                k = shape(self.k(input))  # (bs, n_heads, qlen, dim_per_head)
                v = shape(self.v(input))  # (bs, n_heads, qlen, dim_per_head)
        elif past_key_value is None:
            k = v = kv
            if self.adapter_config.train_lora:
                if self.training:
                    k_lora, load_loss, supervised_loss = self.k_lora_controller(task, orig_task, hash_lbl, k, encoder_hidden_states)
                    self.adapter_config.load_loss_accm += load_loss
                    self.adapter_config.supervised_loss_accm += supervised_loss
                else:
                    k_lora = self.k_lora_controller(task, orig_task, hash_lbl, k, encoder_hidden_states)
                k = shape(self.k(k) + k_lora)

                if self.training:
                    v_lora, load_loss, supervised_loss = self.v_lora_controller(task, orig_task, hash_lbl, v, encoder_hidden_states)
                    self.adapter_config.load_loss_accm += load_loss
                    self.adapter_config.supervised_loss_accm += supervised_loss
                else:
                    v_lora = self.v_lora_controller(task, orig_task, hash_lbl, v, encoder_hidden_states)
                v = shape(self.v(v) + v_lora)
            elif self.adapter_config.train_ia3:
                if self.training:
                    k, load_loss, supervised_loss = self.l_k_controller(task, orig_task, hash_lbl, self.k(k), encoder_hidden_states)
                    self.adapter_config.load_loss_accm += load_loss
                    self.adapter_config.supervised_loss_accm += supervised_loss
                    k = shape(k)

                    v, load_loss, supervised_loss = self.l_v_controller(task, orig_task, hash_lbl, self.v(v), encoder_hidden_states)
                    self.adapter_config.load_loss_accm += load_loss
                    self.adapter_config.supervised_loss_accm += supervised_loss 
                    v = shape(v)
                else:
                    k = shape(self.l_k_controller(task, orig_task, hash_lbl, self.k(k), encoder_hidden_states))
                    v = shape(self.l_v_controller(task, orig_task, hash_lbl, self.v(v), encoder_hidden_states))
            else:
                k = shape(self.k(k))  # (bs, n_heads, qlen, dim_per_head)
                v = shape(self.v(v))  # (bs, n_heads, qlen, dim_per_head)

        if past_key_value is not None:
            if kv is None:
                k_, v_ = past_key_value
                k = torch.cat([k_, k], dim=2)  # (bs, n_heads, klen, dim_per_head)
                v = torch.cat([v_, v], dim=2)  # (bs, n_heads, klen, dim_per_head)
            else:
                k, v = past_key_value

        if self.is_decoder and use_cache is True:
            present_key_value_state = ((k, v),)
        else:
            present_key_value_state = (None,)

        # (bs, n_heads, qlen, klen)
        scores = torch.matmul(
            q, k.transpose(3, 2)
        )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", q, k), compatible with onnx op>9

        if position_bias is None:
            if not self.has_relative_attention_bias:
                raise ValueError("No position_bias provided and no weights to compute position_bias")
            position_bias = self.compute_bias(real_qlen, klen)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value is not None:
                position_bias = position_bias[:, :, -qlen:, :]

            if mask is not None:
                position_bias = position_bias + mask  # (bs, n_heads, qlen, klen)
        scores += position_bias
        weights = F.softmax(scores.float(), dim=-1).type_as(scores)  # (bs, n_heads, qlen, klen)
        weights = F.dropout(weights, p=self.dropout, training=self.training)  # (bs, n_heads, qlen, klen)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask

        context = torch.matmul(weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        context = self.o(context)
        if self.adapter_config.train_ia3:
            if self.training:
                context, load_loss, supervised_loss = self.l_o_controller(task, orig_task, hash_lbl, context, encoder_hidden_states)
                self.adapter_config.load_loss_accm += load_loss
                self.adapter_config.supervised_loss_accm += supervised_loss
            else:
                context = self.l_o_controller(task, orig_task, hash_lbl, context, encoder_hidden_states)
        outputs = (context,) + present_key_value_state

        if output_attentions:
            outputs = outputs + (weights,)
        if self.has_relative_attention_bias:
            outputs = outputs + (position_bias,)
        return outputs


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False, adapter_config=None):
        super().__init__()
        self.SelfAttention = T5Attention(
            config, adapter_config=adapter_config, has_relative_attention_bias=has_relative_attention_bias,
            is_bidirectional=not config.is_decoder
        )
        self.adapter_config = adapter_config
        self.train_adapters = config.train_adapters
        if self.train_adapters:
            self.adapter_controller = AdapterController(adapter_config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
            self,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            position_bias=None,
            head_mask=None,
            past_key_value=None,
            use_cache=False,
            output_attentions=False,
            task=None,
            hash_lbl=None,
            orig_task=None,
    ):
        norm_x = self.layer_norm(hidden_states)
        attention_output = self.SelfAttention(
            norm_x,
            encoder_hidden_states=encoder_hidden_states,
            mask=attention_mask,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            task=task,
            orig_task=orig_task,
            hash_lbl=hash_lbl
        )
        y = attention_output[0]
        if self.train_adapters:
            if self.training:
                y, load_loss, supervised_loss = self.adapter_controller(task, orig_task, hash_lbl, y, encoder_hidden_states)
                self.adapter_config.load_loss_accm += load_loss
                self.adapter_config.supervised_loss_accm += supervised_loss
            else:
                y = self.adapter_controller(task, orig_task, hash_lbl, y, encoder_hidden_states)

        layer_output = hidden_states + self.dropout(y)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs

class T5LayerCrossAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False, adapter_config=None):
        super().__init__()
        self.EncDecAttention = T5Attention(
            config, adapter_config=adapter_config, has_relative_attention_bias=has_relative_attention_bias, is_bidirectional=True
        )
        self.adapter_config = adapter_config
        self.train_adapters = config.train_adapters
        if self.train_adapters:
            self.adapter_controller = AdapterController(adapter_config)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        kv,
        attention_mask=None,
        position_bias=None,
        head_mask=None,
        past_key_value=None,
        use_cache=False,
        query_length=None,
        output_attentions=False,
        encoder_hidden_states=None,
        task=None,
        orig_task=None,
        hash_lbl=None
    ):
        norm_x = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            norm_x,
            mask=attention_mask,
            kv=kv,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
            encoder_hidden_states=encoder_hidden_states,
            task=task,
            orig_task=orig_task,
            hash_lbl=hash_lbl
        )
        y = attention_output[0]
        if self.train_adapters:
            if self.training:
                y, load_loss, supervised_loss = self.adapter_controller(task, orig_task, hash_lbl, y, encoder_hidden_states)
                self.adapter_config.load_loss_accm += load_loss
                self.adapter_config.supervised_loss_accm += supervised_loss
            else:
                y = self.adapter_controller(task, orig_task, hash_lbl, y, encoder_hidden_states)

        layer_output = hidden_states + self.dropout(y)
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False, adapter_config=None):
        super().__init__()
        self.adapter_config = adapter_config
        self.is_decoder = config.is_decoder
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, \
                                               has_relative_attention_bias=has_relative_attention_bias,
                                               adapter_config=self.adapter_config))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config, \
                                                    has_relative_attention_bias=has_relative_attention_bias,
                                                    adapter_config=self.adapter_config))
            # self.layer.append(T5LayerCrossAttention(config, \
            #                                         has_relative_attention_bias=has_relative_attention_bias))
        self.layer.append(T5LayerFF(config, self.adapter_config))

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            position_bias=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            encoder_decoder_position_bias=None,
            head_mask=None,
            past_key_value=None,
            use_cache=False,
            output_attentions=False,
            return_dict=False,
            task=None,
            hash_lbl=None,
            orig_task=None,
    ):
        if past_key_value is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            error_message = "There should be {} past states. 2 (past / key)\
            for self attention.{} Got {} past key / value states".format(
                expected_num_past_key_values, "2 (past / key) for cross \
                attention" if expected_num_past_key_values == 4 else "", \
                len(past_key_value),
            )
            assert len(past_key_value) == expected_num_past_key_values, \
                error_message

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            head_mask=head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            task=task,
            hash_lbl=hash_lbl,
            orig_task=orig_task
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        # Keep self-attention outputs and relative position weights
        attention_outputs = self_attention_outputs[2:]

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                kv=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                head_mask=head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
                encoder_hidden_states=encoder_hidden_states,
                task=task,
                orig_task=orig_task,
                hash_lbl=hash_lbl
            )
            hidden_states = cross_attention_outputs[0]
            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + \
                                          cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states, encoder_hidden_states=encoder_hidden_states, task=task,hash_lbl=hash_lbl, orig_task=orig_task)
        outputs = (hidden_states,)

        outputs = outputs + (present_key_value_state,) + attention_outputs
        return outputs  # hidden-states, present_key_value_states,
        # (self-attention weights), (self-attention position bias),
        # (cross-attention weights), (cross-attention position bias)


class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None, adapter_config=None):
        super().__init__(config)
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder
        self.adapter_config = adapter_config

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0), adapter_config=self.adapter_config)
             for i in range(config.num_layers)]
        )
        self.train_adapters = config.train_adapters
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            inputs_embeds=None,
            head_mask=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            task=None,
            task_embedding=None,
            hash_lbl=None,
            orig_task=None,
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}inputs and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You have to specify either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = past_key_values[0][0].shape[2] + seq_length if past_key_values is not None else seq_length

        if use_cache is True:
            assert self.is_decoder, ":obj:`use_cache` can only be set to `True` if {} is used as a decoder".format(
                self
            )

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None
        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, past_key_value) in enumerate(zip(self.block, past_key_values)):

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,
                head_mask=head_mask[i],
                past_key_value=past_key_value,
                use_cache=use_cache,
                output_attentions=output_attentions,
                task=task,
                hash_lbl=hash_lbl,
                orig_task=orig_task
            )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights),
            # (self-attention position bias), (cross-attention weights),
            # (cross-attention position bias)
            hidden_states, present_key_value_state = layer_outputs[:2]

            if i == 0:
                # We share the position biases between the layers - the first layer store them
                # layer_outputs = hidden-states, key-value-states (self-attention weights),
                # (self-attention position bias), (cross-attention weights), (cross-attention position bias)
                position_bias = layer_outputs[3 if output_attentions else 2]
                if self.is_decoder and encoder_hidden_states is not None:
                    encoder_decoder_position_bias = layer_outputs[5 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[2],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[4 if i == 0 else 3],)
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )

class T5ForConditionalGeneration(T5PreTrainedModel):
    def __init__(self, config, adapter_config=None):
        super().__init__(config)
        # Computes the task-embeddings.
        self.train_adapters = config.train_adapters
        self.adapter_config = adapter_config
        self.model_dim = config.d_model
        self.shared = nn.Embedding(config.vocab_size, config.d_model)
        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        if config.train_adapters:
            encoder_config.train_adapters = True
        self.encoder = T5Stack(encoder_config, self.shared, adapter_config=adapter_config)
        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        if config.train_adapters or adapter_config.train_lora or adapter_config.train_ia3:
            self.adapter_config.analysis_list = {}
            self.adapter_config.complete_analysis_list = []
            self.adapter_config.load_loss_accm = torch.tensor(0.0).to(self.adapter_config.device)
            self.adapter_config.supervised_loss_accm = torch.tensor(0.0).to(self.adapter_config.device)
            if self.adapter_config.routing_estimator == "reinf_bl_routing":
                self.adapter_config.adapter_probs_list = []
                self.adapter_config.baseline_vals_list = []
                self.adapter_config.adapter_samples_list = []
            
            decoder_adapter_config = copy.deepcopy(adapter_config)
            self.decoder = T5Stack(decoder_config, self.shared, adapter_config=decoder_adapter_config)
            self.decoder_adapter_config = decoder_adapter_config
        else:
            self.decoder = T5Stack(decoder_config, self.shared, adapter_config=adapter_config)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.init_weights()
        self.assign_router_indices()
    
    def assign_router_indices(self):
        router_index = 0
        if self.train_adapters:
            for block in self.encoder.block:
                block.layer[0].adapter_controller.router_index = router_index
                router_index += 1
                block.layer[-1].adapter_controller.router_index = router_index
                router_index +=1 
            for block in self.decoder.block:
                block.layer[0].adapter_controller.router_index = router_index
                router_index += 1
                block.layer[1].adapter_controller.router_index = router_index
                router_index +=1 
                block.layer[-1].adapter_controller.router_index = router_index
                router_index +=1 

        if self.adapter_config.train_lora:
            for name, sub_module in self.named_modules():
                if isinstance(sub_module, LoRAController):
                    sub_module.router_index = router_index
                    router_index += 1

        if self.adapter_config.train_ia3:
            for name, sub_module in self.named_modules():
                if isinstance(sub_module, IA3Controller):
                    sub_module.router_index = router_index
                    router_index += 1

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            head_mask=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            task=None,
            task_embedding=None,
            hash_lbl=None,
            orig_task=None,
            **kwargs,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`,
        `optional`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to ``-100`` are ignored
            (masked), the loss is only computed for
            labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.

        Returns:

        Examples::

            >>> from transformers import T5Tokenizer, T5ForConditionalGeneration

            >>> tokenizer = T5Tokenizer.from_pretrained('t5-small')
            >>> model = T5ForConditionalGeneration.from_pretrained('t5-small',
            return_dict=True)

            >>> input_ids = tokenizer('The <extra_id_0> walks in <extra_id_1>
            park', return_tensors='pt').input_ids
            >>> labels = tokenizer('<extra_id_0> cute dog <extra_id_1> the
            <extra_id_2> </s>', return_tensors='pt').input_ids
            >>> outputs = model(input_ids=input_ids, labels=labels)
            >>> loss = outputs.loss
            >>> logits = outputs.logits

            >>> input_ids = tokenizer("summarize: studies have shown that owning
            a dog is good for you ", return_tensors="pt").input_ids# Batch size 1
            >>> outputs = model.generate(input_ids)
        """
        if "lm_labels" in kwargs:
            warnings.warn(
                "The `lm_labels` argument is deprecated and will be removed in a future version, use `labels` instead.",
                FutureWarning,
            )
            labels = kwargs.pop("lm_labels")
        if "decoder_past_key_value_states" in kwargs:
            warnings.warn(
                "The `decoder_past_key_value_states` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_value_states")
        if "decoder_past_key_values" in kwargs:
            warnings.warn(
                "The `decoder_past_key_values` argument is deprecated and will be removed in a future version, use `past_key_values` instead.",
                FutureWarning,
            )
            past_key_values = kwargs.pop("decoder_past_key_values")
        assert kwargs == {}, f"Unexpected keyword arguments: {list(kwargs.keys())}."
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.train_adapters or self.adapter_config.train_lora or self.adapter_config.train_ia3:
            # If we do copy.copy(adapter_config) for decoder, then we might not need this if condition
            # Need to check if it will break other code
            self.decoder_adapter_config.same_init_then_branch = self.adapter_config.same_init_then_branch
            if self.adapter_config.routing_estimator == "gs_st_routing":
                self.decoder_adapter_config.adapter_temp = self.adapter_config.adapter_temp

        if self.train_adapters or self.adapter_config.train_lora or self.adapter_config.train_ia3:
            self.adapter_config.load_loss_accm = torch.tensor(0.0).to(self.adapter_config.device)
            self.adapter_config.supervised_loss_accm = torch.tensor(0.0).to(self.adapter_config.device)
            self.decoder_adapter_config.load_loss_accm = torch.tensor(0.0).to(self.adapter_config.device)
            self.decoder_adapter_config.supervised_loss_accm = torch.tensor(0.0).to(self.adapter_config.device)
            if self.adapter_config.routing_estimator == "reinf_bl_routing":
                self.adapter_config.adapter_probs_list = []
                self.adapter_config.baseline_vals_list = []
                self.adapter_config.adapter_samples_list = []
                self.decoder_adapter_config.adapter_probs_list = []
                self.decoder_adapter_config.baseline_vals_list = []
                self.decoder_adapter_config.adapter_samples_list = []

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                task=task,
                task_embedding=None,
                hash_lbl = hash_lbl,
                orig_task = orig_task
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
        hidden_states = encoder_outputs[0]
        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if past_key_values is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            task=task,
            task_embedding=None,
            hash_lbl = hash_lbl,
            orig_task = orig_task
        )

        sequence_output = decoder_outputs[0]
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        sequence_output = sequence_output * (self.model_dim ** -0.5)
        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666
        
        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            if self.train_adapters or self.adapter_config.train_lora or self.adapter_config.train_ia3:
                final_load_loss_accm = self.adapter_config.load_loss_accm + self.decoder_adapter_config.load_loss_accm
                final_supervised_loss_accm = self.adapter_config.supervised_loss_accm + self.decoder_adapter_config.supervised_loss_accm
                # print(f'length of encoder analysis list is {len(self.adapter_config.analysis_list)}')
                # print(f'length of decoder analysis list is {len(self.decoder_adapter_config.analysis_list)}')

                if self.adapter_config.routing_estimator == 'reinf_bl_routing':
                    final_probs_list = self.adapter_config.adapter_probs_list + self.decoder_adapter_config.adapter_probs_list
                    final_baseline_vals_list = self.adapter_config.baseline_vals_list + self.decoder_adapter_config.baseline_vals_list
                    final_samples_list = self.adapter_config.adapter_samples_list + self.decoder_adapter_config.adapter_samples_list
                    
                    return ((loss,) + output) if loss is not None else output, (final_probs_list, final_baseline_vals_list, final_samples_list, final_load_loss_accm, final_supervised_loss_accm)
                else:
                    return ((loss,) + output) if loss is not None else output, (final_load_loss_accm, final_supervised_loss_accm)
            else:
                return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions
        )

    def prepare_inputs_for_generation(
            self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]
        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
            "task": kwargs["task"],
            "hash_lbl": kwargs['hash_lbl'],
            "orig_task": kwargs['orig_task'],
            "task_embedding": kwargs["task_embedding"]
        }

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past is None:
            logger.warning("You might want to consider setting `use_cache=True` to speed up decoding")
            return past

        reordered_decoder_past = ()
        for layer_past_states in past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),
                )

            assert reordered_layer_past_states[0].shape == layer_past_states[0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (reordered_layer_past_states,)
        return reordered_decoder_past

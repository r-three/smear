"""Defines the arguments used for training and evaluation."""

import logging
from dataclasses import dataclass, field
from hyperformer.adapters import ADAPTER_CONFIG_MAPPING
from transformers import TrainingArguments
from transformers.optimization import (
    get_constant_schedule,
    get_constant_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    get_linear_schedule_with_warmup,
    get_polynomial_decay_schedule_with_warmup,
)
from typing import Optional, List

arg_to_scheduler = {
    "cosine_w_restarts": get_cosine_with_hard_restarts_schedule_with_warmup,
    "polynomial": get_polynomial_decay_schedule_with_warmup,
    "constant": get_constant_schedule,
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
    "constant_w_warmup": get_constant_schedule_with_warmup,
}

logger = logging.getLogger(__name__)


@dataclass
class Seq2SeqTrainingArguments(TrainingArguments):
    """
    Contains different training parameters such as dropout, optimizers parameters, ... .
    """
    label_smoothing: Optional[float] = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (if not zero)."}
    )
    predict_with_generate: bool = field(
        default=False, metadata={"help": "Whether to use generate to calculate generative metrics (ROUGE, BLEU)."}
    )
    adafactor: bool = field(default=False, metadata={"help": "whether to use adafactor"})
    encoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Encoder layer dropout probability. Goes into model.config."}
    )
    decoder_layerdrop: Optional[float] = field(
        default=None, metadata={"help": "Decoder layer dropout probability. Goes into model.config."}
    )
    dropout: Optional[float] = field(default=None, metadata={"help": "Dropout probability. Goes into model.config."})
    attention_dropout: Optional[float] = field(
        default=None, metadata={"help": "Attention dropout probability. Goes into model.config."}
    )
    lr_scheduler: Optional[str] = field(
        default="linear",
        metadata={"help": f"Which lr scheduler to use. Selected in {sorted(arg_to_scheduler.keys())}"},
    )
    temperature: Optional[int] = field(default=1, metadata={"help": "Defines the temperature"
                                                                    "value for sampling across the multiple datasets."})
    train_adapters: Optional[bool] = field(default=False, metadata={"help":
                                                                        "Train an adapter instead of the full model."})
    do_test: bool = field(default=False, metadata={"help": "Whether to comptue evaluation metrics on the test sets."})
    eval_output_dir: Optional[str] = field(default=None, metadata={
        "help": "The output directory where the evaluation of the model and checkpoints during "
        "evaluation will be written. Would use the original output_dir if not specified."})
    generate_classifier_weights: Optional[bool] = field(default=False, metadata={
        "help": "If set, generates the weights of the classifier by using a hyper-network."})
    optimize_from_scratch: Optional[bool] = field(default=False,
                                                  metadata={"help": "If set, this does not load the optimizers from"
                                                                    "the given model path."})
    optimize_from_scratch_with_loading_model: Optional[bool] = field(default=False, metadata={
        "help": "If set, it loads the model still but optimize from scratch."})
    split_validation_test: Optional[bool] = field(default=False,
                                                  metadata={"help": "If set, for the datasets which do not"
                                                                    "have the test set, we use validation set as their"
                                                                    "test set and make a validation set from either"
                                                                    "splitting the validation set into half (for smaller"
                                                                    "than 10K samples datasets), or by using 1K examples"
                                                                    "from training set as validation set (for larger"
                                                                    " datasets)."})
    print_num_parameters: Optional[str] = field(default=False,
                                                metadata={
                                                    "help": "If specified, prints the total number of parameters."})
    compute_memory: Optional[bool] = field(default=False,
                                           metadata={"help": "If specified, measures the memory needed."})
    compute_time: Optional[bool] = field(default=False, metadata={"help": "If specified, measures the time needed."})


@dataclass
class ModelArguments:
    """
    Contains the arguments defining model, tokenizer, and config which we use for finetuning.
    Also, it defines which parameters of the model needs to be freezed during finetuning.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    not_load_t5_checkpoint: bool = field(default=False, metadata={"help": "whether to load the checkpoint."})
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    freeze_encoder: bool = field(default=False, metadata={"help": "Whether tp freeze the encoder."})
    freeze_embeds: bool = field(default=False, metadata={"help": "Whether  to freeze the embeddings."})
    freeze_model: bool = field(default=False, metadata={"help": "Whether  to freeze the model."})
    freeze_model_but_lm_head: bool = field(default=False, metadata={"help": "Whether to freeze the"
        "whole model and only keep the language model head as the training parameter."})
    unfreeze_lm_head: bool = field(default=False, metadata={"help": "Whether  to unfreeze the lm_head."})
    freeze_model_but_task_embeddings: bool = field(default=False,
                                                   metadata={"help": "freezes the whole model but task-embedding."})
    unfreeze_layer_norms: bool = field(default=False, metadata={"help": "unfreezes the layer norms."})
    unfreeze_model: bool = field(default=False, metadata={"help": "Whether  to unfreeze the model."})
    model_dim: Optional[int] = field(default=512, metadata={"help": "d_dim of the pretrained model"})


@dataclass
class DataTrainingArguments:
    """
    Arguments related to data used for training and evaluation.
    """
    tasks: Optional[List[str]] = field(
        default="MRPC",
        metadata={"help": "Task name from the list of registered tasks."},
    )
    eval_tasks: Optional[List[str]] = field(
        default="MRPC",
        metadata={"help": "Evaluation task name from the list of registered tasks."},
    )
    adapters: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Defines a dictionary from adapters to the tasks."}
    )
    task_embeddings: Optional[List[str]] = field(
        default=None,
        metadata={"help": "Defines a dictionary from tasks to the tasks embeddings."}
    )
    max_source_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    test_max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for test target text after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    n_train: Optional[int] = field(default=-1, metadata={"help": "# training examples. -1 means use all."})
    n_val: Optional[int] = field(default=-1, metadata={"help": "# validation examples. -1 means use all."})
    n_test: Optional[int] = field(default=-1, metadata={"help": "# test examples. -1 means use all."})
    eval_beams: Optional[int] = field(default=None, metadata={"help": "# num_beams to use for evaluation."})
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={"help": "If only pad tokens should be ignored. This assumes that `config.pad_token_id` is defined."},
    )
    data_seed: Optional[int] = field(default=42, metadata={"help": "The seed used to subsample the datasets."})
    semi_supervised_ratio: Optional[float] = field(default=1.0, metadata={"help": "percentage of train examples with task labels"})
    check_mode: Optional[bool] = field(default=False, metadata={"help": "makes output dir in debug"})
    eval_all_templates: Optional[bool] = field(default=False, metadata={"help": "evaluate on all templates of p3"})


@dataclass
class AdapterTrainingArguments:
    """Defines the adapters parameters."""
    adapter_config_name: Optional[str] = field(
        default="meta-adapter", metadata={"help": "config name for the adapter layers, should be selected "
        f"in {sorted(ADAPTER_CONFIG_MAPPING.keys())}."}
    )
    task_embedding_dim: Optional[int] = field(default=None, metadata={"help": "task embedding dimensions."})
    add_layer_norm_before_adapter: Optional[bool] = field(default=False,
                                                          metadata={
                                                              "help": "whether to have layer-norm before adapter."})
    add_layer_norm_after_adapter: Optional[bool] = field(default=True,
                                                         metadata={"help": "whether to have layer-norm after adapter."})
    hidden_dim: Optional[int] = field(default=128, metadata={"help": "defines the default hidden dimension for "
                                                                     "adapter layers."})
    reduction_factor: Optional[int] = field(default=16, metadata={"help": "defines the default reduction factor for "
                                                                          "adapter layers."})
    non_linearity: Optional[str] = field(default="swish", metadata={"help": "Defines nonlinearity for adapter layers."})
    train_task_embeddings: Optional[bool] = field(default=False, metadata={"help": "If specified learns the tasks "
                                                                                   "embeddings from given task seedings."})
    projected_task_embedding_dim: Optional[int] = field(default=64,
                                                        metadata={"help": "Defines the task embedding dimension"
                                                                          " after projection layer. "})
    task_hidden_dim: Optional[int] = field(default=128, metadata={
        "help": "defines the hidden dimension for task embedding projector."})
    conditional_layer_norm: Optional[bool] = field(default=False,
                                                   metadata={"help": "Implements conditional layer norms "
                                                                     "modulated based on task embeddings."})
    train_adapters_blocks: bool = field(default=False, metadata={"help": "If set, uses adapter blocks."})
    unique_hyper_net: bool = field(default=False, metadata={"help": "If set, uses one hyper network"
                                                                    "to generates the adapter weights"
                                                                    "for all the layers."})
    efficient_unique_hyper_net: bool = field(default=False, metadata={"help": "If set, uses one hyper network"
                                                                              "for all adapters in each layer."})
    unique_hyper_net_layer_norm: bool = field(default=True, metadata={"help": "If set, applies a layer"
                                                                              "norm after computing the "
                                                                              "embeddings for the unique "
                                                                              "hyper-net."})
    num_adapters: Optional[int] = field(default=2, metadata={"help": "number of adapters in the model"})
    num_routers: Optional[int] = field(default=1, metadata={"help": "number of routers in the model"})
    load_loss_weight: Optional[int] = field(default=0, metadata={"help": "weight to balance load among adapters"})
    load_loss_accm: Optional[float] = field(default=0.0, metadata={"help": "accumulate load loss in training"})
    supervised_loss_weight: Optional[int] = field(default=0, metadata={"help": "weight for supervision"})
    supervised_loss_accm: Optional[float] = field(default=0.0, metadata={"help": "accumulate supervised loss among layers"})
    routing_estimator: Optional[str] = field(default="switch_routing", metadata={"help": "estimator to learn the router"})
    adapter_temp: Optional[float] = field(default=10.0, metadata={"help": "initial temperature in gumbel softmax before starting annealing"})
    anneal_rate: Optional[float] = field(default=1e-6, metadata={"help": "anneal rate for temperature softmax in gumbel softmax"})
    min_temp: Optional[float] = field(default=0.5, metadata={"help": "lowest temperature for which we stop annealing the adapter temperature"})
    adapter_probs_list: Optional[float] = field(default=0.0, metadata={"help": "accumulate probs across encoder layers for final policy loss"})
    baseline_vals_list: Optional[float] = field(default=0.0, metadata={"help": "accumulate baseline values across encoder layers for final policy loss"})
    adapter_samples_list: Optional[float] = field(default=0.0, metadata={"help": "accumulate samples across encoder layers for final policy loss"})
    bl_reduction_factor: Optional[int] = field(default=8, metadata={"help": "hidden dim in baseline for policy value function"})
    policy_weight: Optional[float] = field(default=0.01, metadata={"help": "weight for policy loss"})
    policy_entropy_weight: Optional[float] = field(default=0.01, metadata={"help": "weight for policy loss"})
    value_function_weight: Optional[float] = field(default=0.01, metadata={"help": "weight for learning value function"})
    value_loss_type: Optional[str] = field(default="Huber", metadata={"help": "loss type for learning value function"})
    same_rte_mnli : bool = field(default=False, metadata={"help": "If set, rte and mnli use same adapters"})
    only_in_encoder : bool = field(default=False, metadata={"help": "If set, rte and mnli use same adapters in encoder only"})
    analyze_model : bool = field(default=False, metadata={"help": "If set, gets expert indices and task labels for all examples"}) 
    analysis_list: Optional[float] = field(default=0.0, metadata={"help": "accumulate expert indices and task labels"})
    complete_analysis_list: Optional[float] = field(default=0.0, metadata={"help": "accumulate expert indices and task labels"})
    weight_path: str = field(default="", metadata={"help": "Path to task routed checkpoint"})
    only_train_router: bool = field(default=False, metadata={"help": "If set, only router parameters (with baseline if included) and decoder adapters are trained"})
    same_experts_across_routers: bool = field(default=False, metadata={"help": "If set, experts are tied across routers"})
    use_load_balancing: bool = field(default=False, metadata={"help": "Use load balancing loss during training"})
    probe_input_features: bool = field(default=False, metadata={"help": "If set, we probe input features at each layer to see if they have task information in them"})
    num_count_task_pred: Optional[int] = field(default=0, metadata={"help": "accumulate correct predictions in num"})
    den_count_task_pred: Optional[int] = field(default=0, metadata={"help": "accumulate all counts in den"})
    cluster_initscale: Optional[float] = field(default=0.01, metadata={"help": "init scale for clusters"})
    cluster_momentum: Optional[float] = field(default=0.1, metadata={"help": "momentum for clustering"})
    cluster_distance_metric: Optional[str] = field(default='cosine', metadata={"help": "distance metric used for clustering"})
    bias_in_up_sampler: bool = field(default=False, metadata={"help": "Use bias in up sampler of adapter"})
    jitter_noise: Optional[float] = field(default=0.0, metadata={"help": "multiplicate jitter noise for switch training, switch uses 0.01"})
    token_dropout: Optional[float] = field(default=0.0, metadata={"help": "token dropout before the router"})
    non_linear_router: bool = field(default=False, metadata={"help": "Is the router non-linear?"})
    train_lora: bool = field(default=False, metadata={"help": "use lora adapters"})
    train_ia3: bool = field(default=False, metadata={"help": "use ia3 adapters"})
    router_init_scale: Optional[float] = field(default=1e-2, metadata={"help": "init scale for router weights"})
    expert_dropout: Optional[float] = field(default=0.0, metadata={"help": "drop out rate of experts"})
    normalize_router_weights: bool = field(default=True, metadata={"help": "normalize router weights"})
    no_router_bias: bool = field(default=True, metadata={"help": "If set, router does not have bias"})
    renormalize_adapter_probs: bool = field(default=True, metadata={"help": "renormalize router probs"})
    epsilon_greedy: Optional[float] = field(default=0.0, metadata={"help": "uses all experts for epsilon fraction"}) 
    same_init_then_branch: Optional[int] = field(default=-1, metadata={"help": "all experts weights are initialized to same value and do unifrom routing for k steps"}) 
    lora_tieB: Optional[bool] = field(default=False, metadata={"help": "tie B matrix of lora"})
    eval_time: Optional[float] = field(default=0.0, metadata={"help": "times up the eval method"})
    cosine_router: Optional[bool] = field(default=False, metadata={"help": "use cosine when calculating the routing distribution"})
    skill_lr_ratio: Optional[float] = field(default=10.0, metadata={"help": "ratio of lr for skill adapters"})



    
import sys
import torch
import datasets
import json
import logging
import os
from pathlib import Path

from transformers import AutoTokenizer, HfArgumentParser, set_seed
from transformers.trainer_utils import EvaluationStrategy

from hyperformer.third_party.models import T5Config, T5ForConditionalGeneration
from hyperformer.third_party.trainers import T5Trainer
from hyperformer.adapters import AdapterController, AutoAdapterConfig
from hyperformer.data import AutoTask
from hyperformer.third_party.utils import TaskCollator, check_output_dir
from hyperformer.metrics import build_compute_metrics_fn
from hyperformer.training_args import Seq2SeqTrainingArguments, ModelArguments, DataTrainingArguments, \
    AdapterTrainingArguments
from hyperformer.utils import freezing_params, get_last_checkpoint_path, create_dir,\
    handle_metrics, get_training_args

import pdb
import pickle
logger = logging.getLogger(__name__)


def remove_rank_info_from_argv(args):
    extra_parameters = {}
    if args[1].startswith("--local_rank"):
        extra_parameters.update({'local_rank': int(args[1].split('=')[-1])})
        del args[1]
    return extra_parameters

def main():
    # See all possible arguments in src/transformers/training_args.py or by passing
    # the --help flag to this script. We now keep distinct sets of args, for a cleaner
    # separation of concerns.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, AdapterTrainingArguments))

    # For running on multiple gpus with torch.distributed.launch, it adds a local_rank paramter, to allow the parser
    # still use the config file, we add the local_rank to the config file.
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".json"):
        json_file = os.path.abspath(sys.argv[1])

        if len(sys.argv) > 3 and sys.argv[2] == "-k":
            with open(sys.argv[1]) as f:
                args_json = json.loads(f.read())
            output_dir = args_json['output_dir'].split("/")[-2]

            override_eqs = sys.argv[3:]
            override_json = {}
            for eq in override_eqs:
                name, value = eq.split("=")
                if name == "learning_rate":
                    override_json[name] = float(value)
                elif name == "lr_scheduler":
                    override_json[name] = value
                elif name == "max_steps":
                    override_json[name] = int(value)
                elif name == "adapter_temp":
                    override_json[name] = float(value)
                elif name == "gradient_accumulation_steps":
                    override_json[name] = int(value)
                elif name == "per_device_train_batch_size":
                    override_json[name] = int(value)
                elif name == "per_device_eval_batch_size":
                    override_json[name] = int(value)
                elif name == "reduction_factor":
                    override_json[name] = int(value)
                elif name == "num_routers":
                    override_json[name] = int(value)
                elif name == "num_adapters":
                    override_json[name] = int(value)
                elif name == "router_init_scale":
                    override_json[name] = float(value)
                elif name == "data_seed":
                    override_json["data_seed"] = int(value)
                elif name == "semi_supervised_ratio":
                    override_json[name] = float(value)
                elif name == "expert_dropout":
                    override_json[name] = float(value)
                elif name == "jitter_noise":
                    override_json[name] = float(value)
                elif name == "epsilon_greedy":
                    override_json[name] = float(value)
                elif name == "load_loss_weight":
                    override_json[name] = float(value)
                elif name == "skill_lr_ratio":
                    override_json[name] = float(value)
                elif name == "same_init_then_branch":
                    override_json[name] = int(value)
                elif name == "save_total_limit":
                    override_json[name] = int(value)
                elif name == "weight_path":
                    override_json[name] = value
                elif name == "routing_estimator":
                    override_json[name] = value
                elif name == "output_dir":
                    override_json["output_dir"] = value
                elif name == "split_validation_test":
                    override_json[name] = (value == 'True')
                elif name == "cosine_router":
                    override_json[name] = (value == 'True')
                elif name == "analyze_model":
                    override_json[name] = (value == 'True')
                elif name == "do_train":
                    override_json[name] = (value == 'True')
                elif name == "do_eval":
                    override_json[name] = (value == 'True')
                elif name == "do_test":
                    override_json[name] = (value == 'True')
                elif name == "lora_tieB":
                    override_json[name] = (value == 'True')
                elif name == "check_mode":
                    override_json[name] = (value == 'True')
                elif name == "eval_all_templates":
                    override_json[name] = (value == 'True')
                elif name == "use_load_balancing":
                    override_json[name] = (value == 'True')
                elif name == "normalize_router_weights":
                    override_json[name] = (value == 'True')
                elif name == "dselectk1_mode":
                    override_json[name] = (value == 'True')
                else:
                    raise ValueError(f"Can not override {name}")
            
            def make_new_name(output_dir, args_json):
                new_dir = output_dir
                val_to_str = {1e-4: "1e4",3e-4: "3e4", 1e-3: "1e3", 3e-3: "3e3", 5e-3: "5e3", 1e-2: "1e2", 0: "0", 0.01: "001", 0.1: "01", 0.3: "03", 262144: "18", 400000: "400k", 600000: "600k", 1: "1", 10: "10", 100: "100", 25000: "25k", 50000: "50k"}
                str_to_str = {"linear": "linear", "constant": "const", "constant_w_warmup": "const"}
                for key,value in args_json.items():
                    if key == "lora_tieB":
                        if value == True:
                            new_dir += f"_tieB"
                    if key == "cosine_router":
                        if value == True:
                            new_dir += f"_cosine_router"
                    if key == "learning_rate":
                        new_dir += f"_lr{val_to_str[value]}"
                    if key == "lr_scheduler":
                        new_dir += f"_{str_to_str[value]}shd"
                    if key == "max_steps":
                        new_dir += f"_steps{val_to_str[value]}"
                    if key == "reduction_factor":
                        if value not in [512, 768]:
                            new_dir += f"_rd{value}"
                    if key == "num_adapters":
                        if value not in [8]:
                            new_dir += f"_exps{value}"
                    if key == "skill_lr_ratio":
                        if value != 1:
                            new_dir += f"_slr{val_to_str[value]}"
                    if key == "router_init_scale":
                        if value != 1e-2:
                            new_dir += f"_rinit{val_to_str[value]}"
                    if key == "data_seed":
                        new_dir += f"_seed{value}"
                    if key == "expert_dropout":
                        if value != 0:
                            new_dir += f"_rndp{val_to_str[value]}"
                    if key == "jitter_noise":
                        if value != 0:
                            new_dir += f"_jn{val_to_str[value]}"
                    if key == "load_loss_weight":
                        if value != 0:
                            new_dir += f"_lw{val_to_str[value]}"
                    if key == "epsilon_greedy":
                        if value != 0:
                            new_dir += f"_eps{val_to_str[value]}"
                    if key == "same_init_then_branch":
                        if value != -1:
                            new_dir += f"_stb{val_to_str[value]}"
                return new_dir
            
            logger.warning(f"override args: {override_json.__repr__()}")
            args_json.update(override_json)
            if args_json["do_train"] and ("output_dir" not in override_json):
                output_dir = make_new_name(output_dir, args_json)
                if args_json['check_mode']:
                    args_json["output_dir"] = "/".join(args_json["output_dir"].split("/")[:-2] + ["debug", output_dir, ""])
                else:
                    args_json["output_dir"] = "/".join(args_json["output_dir"].split("/")[:-2] + [output_dir, ""])
                assert "?" not in output_dir

            temp_json_file = f"temp_jsons/{output_dir}.json"
            temp_json_file = os.path.abspath(temp_json_file)
            with open(temp_json_file, "w") as f:
                f.write(json.dumps(args_json))
            json_file = temp_json_file
            print(args_json)
        logger.warning("config path: %s", json_file)
        model_args, data_args, training_args, adapter_args = parser.parse_json_file(
            json_file=json_file)
            
    else:
        model_args, data_args, training_args, adapter_args = parser.parse_args_into_dataclasses()
    check_output_dir(training_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(data_args.data_seed)
    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = T5Config.from_pretrained(
        model_args.config_name if model_args.config_name else \
            model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )
    
    extra_model_params = ("encoder_layerdrop", "decoder_layerdrop", "dropout",
                          "attention_dropout",  "train_adapters")
    for p in extra_model_params:
        if getattr(training_args, p, None):
            assert hasattr(config, p), f"({config.__class__.__name__}) doesn't have a `{p}` attribute"
            setattr(config, p, getattr(training_args, p))
    # Gets the adapter config and updates the specified parameters.
    adapter_config = AutoAdapterConfig.get(adapter_args.adapter_config_name)
    adapter_config.input_dim = config.d_model
    adapter_config.tasks = data_args.tasks
    adapter_config.num_adapters = adapter_args.num_adapters
    adapter_config.routing_estimator = adapter_args.routing_estimator
    adapter_config.load_loss_weight = adapter_args.load_loss_weight
    adapter_config.load_loss_accm = adapter_args.load_loss_accm
    adapter_config.supervised_loss_weight = adapter_args.supervised_loss_weight
    adapter_config.supervised_loss_accm = adapter_args.supervised_loss_accm
    adapter_config.model_dim = model_args.model_dim
    adapter_config.num_routers = adapter_args.num_routers
    adapter_config.same_rte_mnli = adapter_args.same_rte_mnli
    adapter_config.only_in_encoder = adapter_args.only_in_encoder
    adapter_config.analyze_model = adapter_args.analyze_model
    adapter_config.analysis_list = adapter_args.analysis_list
    adapter_config.complete_analysis_list = adapter_args.complete_analysis_list
    adapter_config.only_train_router = adapter_args.only_train_router
    adapter_config.same_experts_across_routers = adapter_args.same_experts_across_routers
    adapter_config.use_load_balancing = adapter_args.use_load_balancing
    adapter_config.probe_input_features = adapter_args.probe_input_features
    adapter_config.num_count_task_pred = adapter_args.num_count_task_pred
    adapter_config.den_count_task_pred = adapter_args.den_count_task_pred
    adapter_config.cluster_initscale = adapter_args.cluster_initscale
    adapter_config.cluster_momentum = adapter_args.cluster_momentum
    adapter_config.cluster_distance_metric = adapter_args.cluster_distance_metric
    adapter_config.bias_in_up_sampler = adapter_args.bias_in_up_sampler
    adapter_config.jitter_noise = adapter_args.jitter_noise
    adapter_config.token_dropout = adapter_args.token_dropout
    adapter_config.non_linear_router = adapter_args.non_linear_router
    adapter_config.train_lora = adapter_args.train_lora
    adapter_config.train_ia3 = adapter_args.train_ia3
    adapter_config.router_init_scale = adapter_args.router_init_scale
    adapter_config.expert_dropout = adapter_args.expert_dropout
    adapter_config.normalize_router_weights = adapter_args.normalize_router_weights
    adapter_config.no_router_bias = adapter_args.no_router_bias
    adapter_config.renormalize_adapter_probs = adapter_args.renormalize_adapter_probs
    adapter_config.epsilon_greedy = adapter_args.epsilon_greedy
    adapter_config.same_init_then_branch = adapter_args.same_init_then_branch
    adapter_config.lora_tieB = adapter_args.lora_tieB
    adapter_config.eval_time = adapter_args.eval_time
    adapter_config.cosine_router = adapter_args.cosine_router
    adapter_config.skill_lr_ratio = adapter_args.skill_lr_ratio
    adapter_config.dselectk1_mode = adapter_args.dselectk1_mode
    
    if adapter_config.routing_estimator == "gs_st_routing":
        adapter_config.adapter_temp = adapter_args.adapter_temp
        adapter_config.anneal_rate = adapter_args.anneal_rate
        adapter_config.min_temp = adapter_args.min_temp
    if adapter_config.routing_estimator == "reinf_bl_routing":
        adapter_config.adapter_probs_list = adapter_args.adapter_probs_list
        adapter_config.baseline_vals_list = adapter_args.baseline_vals_list
        adapter_config.adapter_samples_list = adapter_args.adapter_samples_list
        adapter_config.bl_reduction_factor = adapter_args.bl_reduction_factor
        adapter_config.policy_weight = adapter_args.policy_weight
        adapter_config.policy_entropy_weight = adapter_args.policy_entropy_weight
        adapter_config.value_function_weight = adapter_args.value_function_weight
        adapter_config.value_loss_type = adapter_args.value_loss_type

    num_layers = 60
    if adapter_config.train_lora:
        num_layers = 216 
    elif adapter_config.train_ia3:
        num_layers = 156
    # adapter_config.task_to_adapter = {task:adapter for task, adapter in zip(data_args.tasks, data_args.adapters)} if data_args.adapters is not None else None
    # If this is a parametric task embedding this mapping makes sense, but in case we use any task embeddings,
    # then, we do not need any mapping as we use the pretrained task embeddings.
    # adapter_config.task_to_embeddings = {task:embedding for task, embedding in zip(data_args.tasks, data_args.task_embeddings)}\
        #  if (data_args.task_embeddings is not None) else None
    extra_adapter_params = ("task_embedding_dim",
                            "add_layer_norm_before_adapter",
                            "add_layer_norm_after_adapter",
                            "reduction_factor",
                            "hidden_dim",
                            "non_linearity",
                            "train_task_embeddings",
                            "projected_task_embedding_dim",
                            "task_hidden_dim",
                            "conditional_layer_norm",
                            "train_adapters_blocks",
                            "unique_hyper_net",
                            "unique_hyper_net_layer_norm",
                            "efficient_unique_hyper_net")
    for p in extra_adapter_params:
        if hasattr(adapter_args, p) and hasattr(adapter_config, p):
            setattr(adapter_config, p, getattr(adapter_args, p))
        else:
            logger.warning(f"({adapter_config.__class__.__name__}) doesn't have a `{p}` attribute")
    adapter_config.device = training_args.device

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else \
            model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    if model_args.not_load_t5_checkpoint:
        model = T5ForConditionalGeneration(config=config, adapter_config=adapter_config)
    else:
        if training_args.do_train:
            last_checkpoint_path = get_last_checkpoint_path(training_args.output_dir)
        else:
            last_checkpoint_path = training_args.output_dir
        model_path = model_args.model_name_or_path if ((training_args.optimize_from_scratch and not training_args.optimize_from_scratch_with_loading_model) or not os.path.exists(os.path.join(last_checkpoint_path, 'pytorch_model.bin')))\
            else last_checkpoint_path
        if adapter_args.weight_path != "":
            logger.warning("model path loaded from : %s", adapter_args.weight_path)
            model = T5ForConditionalGeneration.from_pretrained(
                adapter_args.weight_path,
                from_tf=".ckpt" in model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir,
                adapter_config=adapter_config
            )
        else:
            logger.warning("model path loaded from : %s", model_path)
            model = T5ForConditionalGeneration.from_pretrained(
                model_path,
                from_tf=".ckpt" in model_args.model_name_or_path,
                config=config,
                cache_dir=model_args.cache_dir,
                adapter_config=adapter_config
            )
            
    # set num_beams for evaluation
    if data_args.eval_beams is None:
        data_args.eval_beams = model.config.num_beams

    # freezing the parameters.
    if training_args.do_train:
        freezing_params(model, training_args, model_args, adapter_args)

    if training_args.print_num_parameters:
        logger.info(model)
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info("Parameter name %s", name)
        total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        logger.info("Total trainable parameters %s", total_trainable_params)
        logger.info("Total parameters %s", total_params)
    # Gets the training/test/validation datasets.

    dataset_class = AutoTask
    if training_args.do_train:
        # train_datasets = [dataset_class.get(task, seed=data_args.data_seed).get_dataset(
        #     split="train", n_obs=data_args.n_train, add_prefix=False if training_args.train_adapters else True)
        #     for task in data_args.tasks]
        train_datasets = [dataset_class.get(task,  semi_supervised_ratio=data_args.semi_supervised_ratio, seed=data_args.data_seed, num_layers=num_layers).get_dataset(
            split="train", n_obs=data_args.n_train, add_prefix=True, split_validation_test=training_args.split_validation_test)
            for task in data_args.tasks]
        dataset_sizes = [len(train_dataset) for train_dataset in train_datasets]
        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        # train_dataset = datasets.concatenate_datasets(train_datasets)
    training_args.remove_unused_columns = False
    # eval_datasets = ({task: dataset_class.get(task, seed=data_args.data_seed).get_dataset(
    #     split="validation", n_obs=data_args.n_val,
    #     add_prefix=False if training_args.train_adapters else True,
    #     split_validation_test=training_args.split_validation_test)
    #                      for task in data_args.eval_tasks}
    #                  if training_args.do_eval or training_args.evaluation_strategy != EvaluationStrategy.NO
    #                  else None)

    eval_datasets = ({task: dataset_class.get(task, seed=data_args.data_seed, num_layers=num_layers, eval_all_templates=data_args.eval_all_templates).get_dataset(
        split="validation", n_obs=data_args.n_val,
        add_prefix=True,
        split_validation_test=training_args.split_validation_test)
                         for task in data_args.eval_tasks}
                     if training_args.do_eval or training_args.evaluation_strategy != EvaluationStrategy.NO
                     else None)

    # test_dataset = (
    #     {task: dataset_class.get(task, seed=data_args.data_seed).get_dataset(
    #         split="test", n_obs=data_args.n_test,
    #         add_prefix=False if training_args.train_adapters else True,
    #         split_validation_test=training_args.split_validation_test)
    #         for task in data_args.eval_tasks} if training_args.do_test else None
    # )
    test_dataset = (
        {task: dataset_class.get(task, seed=data_args.data_seed, num_layers=num_layers, eval_all_templates=data_args.eval_all_templates).get_dataset(
            split="test", n_obs=data_args.n_test,
            add_prefix=True,
            split_validation_test=training_args.split_validation_test)
            for task in data_args.eval_tasks} if training_args.do_test else None
    )
    # Defines the metrics for evaluation.
    compute_metrics_fn = (
        build_compute_metrics_fn(data_args.eval_tasks, tokenizer) if training_args.predict_with_generate else None
    )
    # Defines the trainer.
    trainer = T5Trainer(
        model=model,
        config=config,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_datasets,
        data_collator=TaskCollator(tokenizer, data_args, tpu_num_cores=training_args.tpu_num_cores),
        compute_metrics=None,
        multi_task_compute_metrics=compute_metrics_fn,
        data_args=data_args,
        dataset_sizes=dataset_sizes if training_args.do_train else None,
        adapter_config=adapter_config
    )
    if trainer.is_world_process_zero():
        arguments = get_training_args([model_args, data_args, training_args, adapter_args])
        handle_metrics("arguments", arguments, training_args.output_dir)

    # Trains the model.
    if training_args.do_train:
        if trainer.is_world_process_zero():
           last_checkpoint_path = get_last_checkpoint_path(training_args.output_dir)
           model_path = model_args.model_name_or_path if (training_args.optimize_from_scratch or not os.path.exists(os.path.join(last_checkpoint_path, 'pytorch_model.bin')))\
             else last_checkpoint_path
        if training_args.compute_time:
           torch.cuda.synchronize()  # wait for move to complete
           start = torch.cuda.Event(enable_timing=True)
           end = torch.cuda.Event(enable_timing=True)
           start.record()
        trainer.train(
            #get_last_checkpoint_path(training_args.output_dir) \
            model_path=model_path \
                if (os.path.exists(training_args.output_dir) and not training_args.optimize_from_scratch) else None,
        )
        if training_args.compute_time: 
           torch.cuda.synchronize()  # wait for all_reduce to complete
           end.record()
           total_time = {"total_time": start.elapsed_time(end)}
           print("###### total_time ", total_time)
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            trainer.state.save_to_json(os.path.join(training_args.output_dir, "trainer_state.json"))
            tokenizer.save_pretrained(training_args.output_dir)
     
    # Evaluation
    all_metrics = {}
    if training_args.do_eval or training_args.do_test:
        if trainer.is_world_process_zero():
            # By default we load  the model from last checkpoint path,
            # in case of saving the model with the best metrics, make sure to
            # set save_total = 1 so the best model is loaded here.
            # if not exists returns the path to the output_dir.
            # last_checkpoint_path = get_last_checkpoint_path(training_args.output_dir)
            last_checkpoint_path = training_args.output_dir
            config = T5Config.from_pretrained(
                last_checkpoint_path,
                cache_dir=model_args.cache_dir)
            model = T5ForConditionalGeneration.from_pretrained(
                last_checkpoint_path,
                from_tf=".ckpt" in training_args.output_dir,
                config=config,
                cache_dir=model_args.cache_dir,
                adapter_config=adapter_config
            )
            # NOTE: if trainer is not re-defined, there is a bug in the codes, that making
            # huggingface codes does not using the best checkpoint.
            trainer = T5Trainer(
                model=model,
                config=config,
                args=training_args,
                train_dataset=train_dataset if training_args.do_train else None,
                eval_dataset=eval_datasets,
                data_collator=TaskCollator(tokenizer, data_args, tpu_num_cores=training_args.tpu_num_cores),
                compute_metrics=None,
                multi_task_compute_metrics=compute_metrics_fn,
                data_args=data_args,
                dataset_sizes=dataset_sizes if training_args.do_train else None,
                adapter_config=adapter_config
            )

        # if training_args.train_adapters:
        #     if adapter_args.adapter_config_name == "adapter" and data_args.adapters is not None:
        #         for name, sub_module in model.named_modules():
        #             task_to_adapter = {eval_task: adapter for eval_task, adapter in
        #                                zip(data_args.eval_tasks, data_args.adapters)}
        #             if isinstance(sub_module, AdapterController):
        #                 sub_module.set_task_to_adapter_map(task_to_adapter)
    if training_args.do_eval:
        metrics = trainer.evaluate()
        if trainer.is_world_process_zero():
            handle_metrics("val", metrics, training_args.output_dir)
            all_metrics.update(metrics)

    if training_args.do_test:
        print(f"time taken is {round(adapter_config.eval_time,2)} sec")
        adapter_config.eval_time = 0
        metrics = trainer.evaluate(test_dataset)
        if trainer.is_world_process_zero():
            handle_metrics("test", metrics, training_args.output_dir)
            all_metrics.update(metrics)

    if torch.cuda.is_available() and training_args.compute_memory:
        peak_memory = torch.cuda.max_memory_allocated()/1024**2
        print(
            "Memory utilization",
            peak_memory,
            "MB"
        )
        memory_usage = {"peak_memory": peak_memory}
    print(f"time taken is {round(adapter_config.eval_time,2)} sec")

    return all_metrics


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

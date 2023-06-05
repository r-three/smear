# Repository Setup and Usage

This repository contains the code and instructions for training and testing various routing methods in the T5-GLUE experiments.

## Installation

To install the required packages, run the following command:

```shell
python setup.py develop
```

Make sure to have the packages version same as the ones specified in `requirements.txt`. The above command will ensure that.

## Temporary JSON Files

Before running the code, create a directory named `temp_jsons` inside the `hyperformer/hyperformer` directory. This directory will be used to save temporary JSON files during the execution of the code.

## Training and Testing

The repository provides different commands for training and testing the models with various routing methods. Use the following commands to run on a single GPU to ensure the same hyperparameters as we used:

### SMEAR

**Training:**
```shell
python ./finetune_t5_trainer.py configs/adapter/parameter_averaging_routing.json -k data_seed=42 max_steps=600000 save_total_limit=5 expert_dropout=0.1
```

**Testing:**
```shell
python ./finetune_t5_trainer.py configs/adapter/parameter_averaging_routing.json -k data_seed=42 do_train=False eval_all_templates=True output_dir=<path/to/trained_directory>
```

### Tag

**Training:**
```shell
python ./finetune_t5_trainer.py configs/adapter/task_routing_same_rte_mnli_onlyencoder.json -k data_seed=42 max_steps=600000 save_total_limit=5
```

**Testing:**
```shell
python ./finetune_t5_trainer.py configs/adapter/task_routing_same_rte_mnli_onlyencoder.json -k data_seed=42 do_train=False eval_all_templates=True output_dir=<path/to/trained_directory>
```

### REINFORCE

**Training:**
```shell
python ./finetune_t5_trainer.py configs/adapter/reinf_bl_routing.json -k data_seed=42 max_steps=600000 save_total_limit=5
```

**Testing:**
```shell
python ./finetune_t5_trainer.py configs/adapter/reinf_bl_routing.json -k data_seed=42 do_train=False eval_all_templates=True output_dir=<path/to/trained_directory>
```

### 1x Compute

**Training:**
```shell
python ./finetune_t5_trainer.py configs/adapter/one_adapter.json -k data_seed=42 max_steps=600000 save_total_limit=5
```

**Testing:**
```shell
python ./finetune_t5_trainer.py configs/adapter/one_adapter.json -k data_seed=42 do_train=False eval_all_templates=True output_dir=<path/to/trained_directory>
```

### Hash

**Training:**
```shell
python ./finetune_t5_trainer.py configs/adapter/hash_routing.json -k data_seed=42 max_steps=600000 save_total_limit=5
```

**Testing:**
```shell
python ./finetune_t5_trainer.py configs/adapter/hash_routing.json -k data_seed=42 do_train=False eval_all_templates=True output_dir=<path/to/trained_directory>
```

### Top-K

**Training:**
```shell
python ./finetune_t5_trainer.py configs/adapter/switch_routing.json -k data_seed=42 max_steps=600000 save_total_limit=5 expert_dropout=0.1
```

**Testing:**
```shell
python ./finetune_t5_trainer.py configs/adapter/switch_routing.json -k data_seed=42 do_train=False eval_all_templates=True output_dir=<path/to/trained_directory>
```

### ST-Gumbel

**Training:**
```shell
python ./finet

une_t5_trainer.py configs/adapter/gs_st_routing.json -k data_seed=42 max_steps=600000 save_total_limit=5
```

**Testing:**
```shell
python ./finetune_t5_trainer.py configs/adapter/gs_st_routing.json -k data_seed=42 do_train=False eval_all_templates=True output_dir=<path/to/trained_directory>
```

### 1x Parameters

**Training:**
```shell
python ./finetune_t5_trainer.py configs/adapter/one_adapter.json -k data_seed=42 max_steps=600000 save_total_limit=5 reduction_factor=4
```

**Testing:**
```shell
python ./finetune_t5_trainer.py configs/adapter/one_adapter.json -k reduction_factor=4 data_seed=42 do_train=False eval_all_templates=True output_dir=<path/to/trained_directory>
```

### Adamix

**Training:**
```shell
python ./finetune_t5_trainer.py configs/adapter/adamix_routing.json -k data_seed=42 max_steps=600000 save_total_limit=5
```

**Testing:**
```shell
python ./finetune_t5_trainer.py configs/adapter/adamix_routing.json -k data_seed=42 do_train=False eval_all_templates=True output_dir=<path/to/trained_directory>
```

### Latent Skill

**Training:**
```shell
python ./finetune_t5_trainer.py configs/adapter/skill_routing.json -k data_seed=42 max_steps=600000 save_total_limit=5
```

**Testing:**
```shell
python ./finetune_t5_trainer.py configs/adapter/skill_routing.json -k data_seed=42 do_train=False eval_all_templates=True output_dir=<path/to/trained_directory>
```

### SMEAR 2x

**Training:**
```shell
python ./finetune_t5_trainer.py configs/adapter/parameter_averaging.json -k data_seed=42 max_steps=600000 save_total_limit=5 expert_dropout=0.1 num_adapters=16
```

**Testing:**
```shell
python ./finetune_t5_trainer.py configs/adapter/parameter_averaging.json -k num_adapters=16 data_seed=42 do_train=False eval_all_templates=True output_dir=<path/to/trained_directory>
```

Replace `<path/to/trained_directory>` with the path to the trained model directory.


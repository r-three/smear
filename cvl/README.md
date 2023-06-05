# Repository Setup and Usage

This repository contains the code and instructions for running ResNet-DomainNet experiments.

## Installation

To set up the required packages, run the following command:

```
pip install -r requirements.txt
```

## Setting up the Python Path

To properly set up the Python path, execute the following command:

```
source bin/setup.sh
```

## Dataset

Download the DomainNet dataset from [this link](http://ai.bu.edu/M3SDA/).

Once the dataset is downloaded, create a validation split using the provided code snippet:

```python
import random

domains = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
random.seed(42)

for domain in domains:
    with open(domain + '_train.txt', 'r') as f:
        domain_data = f.readlines()
        random.shuffle(domain_data)
        valid_data = domain_data[0:int(len(domain_data)*0.2)]
        train_data = domain_data[int(len(domain_data)*0.2):]

        with open(domain + '_train_fold.txt', 'w') as f:
            f.write("".join(train_data))
        with open(domain + '_valid_fold.txt', 'w') as f:
            f.write("".join(valid_data))
```

Place the dataset in the `data/DomainNet/` directory. The files `<domain>_train_fold.txt`, `<domain>_valid_fold.txt`, and `<domain>_test.txt` should be located within the `data/DomainNet` directory.

## Data Directory Structure

The final structure of the data directory should resemble the following:

```
- data/
  - DomainNet/
    - domain/
      - class/
        - image.png
```

## Training and Testing

The repository provides several scripts for training and testing different routing methods. Use the following commands to run the desired experiments:

### Training Multiple Routing Methods

**SMEAR:**
```
python src/train.py -c configs/default.json -k seed=42 routing_estimator="parameter_averaging_routing" expert_dropout=0.1 exp_name="parameter_averaging_routing_resnet18_lr1e3_bd32_seed42_steps100k"
```

**Tag:**
```
python src/train.py -c configs/default.json -k seed=42 routing_estimator="task_routing" exp_name="task_routing_resnet18_lr1e3_bd32_seed42_steps100k"
```

**REINFORCE:**
```
python src/train.py -c configs/default.json -k seed=42 routing_estimator="reinf_bl_routing" exp_name="reinf_bl_routing_resnet18_lr1e3_bd32_seed42_steps100k"
```

**1x Compute:**
```
python src/train.py -c configs/default.json -k seed=42 routing_estimator=None num_adapters=1 exp_name="one_adapter_resnet18_lr1e3_bd32_seed42_steps100k"
```

**Hash:**
```
python src/train.py -c configs/default.json -k seed=42 routing_estimator="hash_routing" exp_name="hash_routing_resnet18_lr1e3_bd32_seed42_steps100k"
```

**Top-K:**
```
python src/train.py -c configs/default.json -k seed=42 routing_estimator="switch_routing" expert_dropout=0.1 exp_name="switch_routing_resnet18_lr1e3_bd32_seed42_steps100k"
```

**ST-Gumbel:**
```
python src/train.py -c configs/default.json -k seed=42 routing_estimator="gs_st_routing" exp_name="gs_st_routing_resnet

18_lr1e3_bd32_seed42_steps100k"
```

**1x Parameters:**
```
python src/train.py -c configs/default.json -k seed=42 routing_estimator=None num_adapters=1 down_sample_size=192 exp_name="one_adapter_resnet18_lr1e3_bd192_seed42_steps100k"
```

**Adamix:**
```
python src/train.py -c configs/default.json -k seed=42 routing_estimator="adamix_routing" exp_name="adamix_routing_resnet18_lr1e3_bd32_seed42_steps100k"
```

**Latent Skill:**
```
python src/train.py -c configs/default.json -k seed=42 routing_estimator="skill_routing" skill_lr_ratio=10 exp_name="skill_routing_resnet18_lr1e3_bd32_seed42_steps100k"
```

**SMEAR 2x:**
```
python src/train.py -c configs/default.json -k seed=42 routing_estimator="parameter_averaging_routing" expert_dropout=0.1 num_adapters=12 exp_name="parameter_averaging_routing_resnet18_lr1e3_bd32_seed42_exps12_steps100k"
```

### Testing

To test the trained models, use the following command:

```python
python src/test.py -e <path/to/trained_directory> -k test_mode=True
```

Replace `<path/to/trained_directory>` with the path to the trained model directory.


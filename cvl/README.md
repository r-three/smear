Install the packages in requirements.txt file

Run following commands:
setup:
source bin/setup.sh

Training:
```python src/train.py -c configs/default.json  -k seed=42 routing_estimator="parameter_averaging_routing" expert_dropout=0.1 exp_name="parameter_averaging_routing_resnet18_lr1e3_bd32_seed42_steps100k"```

Test:
```python src/test.py -e <path/to/directory> -k test_mode=True```
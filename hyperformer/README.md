Install the packages
``` python setup.py develop ```

Run following commands:

Training:
``` python ./finetune_t5_trainer.py configs/adapter/parameter_averaging_routing.json -k data_seed=42 max_steps=600000 save_total_limit=5 expert_dropout=0.1```

Test:
```python ./finetune_t5_trainer.py configs/adapter/parameter_averaging_routing.json -k data_seed=42 do_train=False eval_all_templates=True output_dir=<path/to/directory>```
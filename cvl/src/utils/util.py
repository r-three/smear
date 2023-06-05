import argparse
import logging
import re
import random
import datetime
import os
import numpy as np
import torch
import torch.nn as nn

from shutil import copytree, ignore_patterns
from src.adapters.adapter_controller_resnet_fast import AdapterController as ResNetController
import ipdb


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


class ParseKwargs(argparse.Action):
    '''
    Parse Kwargs into dictionary
    '''
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            getattr(namespace, self.dest)[key] = value


def update_dict_val_store(dict_val_store, dict_update_val, grad_accum_factor):
    '''
    Update dictionary storing values

    :param dict_val_store: current dictionary storing values
    :param dict_update_val: new dictionary with values
    :param grad_accum_factor: grad accum factor
    :return:
    '''
    if dict_val_store is None:
        dict_val_store = {}
        for k in dict_update_val.keys():
            dict_val_store[k] = dict_update_val[k].detach().cpu().item() / grad_accum_factor
    else:
        for k in dict_val_store.keys():
            dict_val_store[k] += dict_update_val[k].detach().cpu().item() / grad_accum_factor

    return dict_val_store

def get_avg_dict_val_store(dict_val_store, eval_every):
    '''
    Get average dictionary value

    :param dict_val_store:
    :param eval_every:
    :return:
    '''
    dict_avg_val = {}
    for k in dict_val_store.keys():
        old_val = dict_val_store[k]
        dict_avg_val[k] = float('%.3f' % (old_val / eval_every))

    return dict_avg_val

def set_seeds(seed):
    '''
    Set random seeds

    :param seed:
    :return:
    '''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def make_dir(dir_name):
    '''
    Makes a directory if it doesn't exists yet

    Args:
        dir_name: directory name
    '''
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)


def make_exp_dir(base_exp_dir):
    '''
    Makes an experiment directory with timestamp

    Args:
        base_output_dir_name: base output directory name

    Returns:
        exp_dir_name: experiment directory name
    '''
    now = datetime.datetime.now()
    ts = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(now.year, now.month, now.day, now.hour, now.minute,
                                                            now.second)
    exp_dir_name = os.path.join(base_exp_dir, ts)
    make_dir(exp_dir_name)

    src_file = os.path.join(exp_dir_name, 'src')

    copytree(os.path.join(os.environ['PC_ROOT'], "src"), src_file,  ignore=ignore_patterns('*.pyc', 'tmp*'))

    return exp_dir_name

def freeze_params(model: nn.Module):
    """Set requires_grad=False for each of model.parameters()"""
    for par in model.parameters():
        par.requires_grad = False

def freezing_params(model, config):
    """
    Freezes the model parameters based on the given setting in the arguments.
    """
    # If we are training adapters, we freeze all parameters except the
    # parameters of adapter controllers.
    if config.full_finetune:
        for par in model.parameters():
            par.requires_grad = True
        print(sum(p.numel() for p in model.parameters() if p.requires_grad))
        return
    else:
        trainable_params = []
        non_router_trainable_params = []
        non_router_params = []
        freeze_params(model)
        if config.train_only_router:
            for name, param in model.named_parameters():
                # if ('router' in name) or ('baseline' in name):
                if 'router' in name:
                    param.requires_grad = True
                    trainable_params.append(name)
        elif config.train_layer != -1:
            for name, param in model.named_parameters():
                layer_strs = ['1.0','1.1','2.0','2.1','3.0','3.1','4.0','4.1']
                trainable_param_name = f'model.layer{layer_strs[config.train_layer]}.adapter_controller'
                if trainable_param_name in name:
                    param.requires_grad = True
                    trainable_params.append(name)
        elif config.probe_input_features:
            for name, param in model.named_parameters():
                # if 'probe' in name:
                param.requires_grad = True
                trainable_params.append(name)
        else:
            if config.train_adapters: 
                for name, sub_module in model.named_modules():
                    if isinstance(sub_module, (ResNetController)):
                        for param_name, param in sub_module.named_parameters():
                            param.requires_grad = True
                            full_name = name+'.'+param_name
                            trainable_params.append(full_name)
                            if ('router' not in full_name):
                                non_router_params.append(param)
                                non_router_trainable_params.append(full_name)

            for name, param in model.named_parameters():
                if 'classification_layer' in name:
                    param.requires_grad = True
                    trainable_params.append(name)
                    non_router_params.append(param)
                    non_router_trainable_params.append(name)
                    
        print('These parameters are trainable: ')
        print(trainable_params)
        print(f'There are {len(non_router_trainable_params)} non router trainable parameters')
        print(f"Total trainable params are {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
        print(f'Total non router trainable parameters are {sum(p.numel() for p in non_router_params)}')


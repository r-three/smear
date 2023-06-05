import torch
import torch.nn as nn
import os
import numpy as np
import argparse
import logging
import time
import re
import sys

from src.eval.eval_model import dev_eval, test_eval

from src.data.Batcher import Batcher
from src.data.DatasetReader import DatasetReader
from src.utils.Config import Config
from src.utils.get_optimizer import get_optimizer, get_lr
from src.utils.get_scheduler import get_scheduler
from src.utils.get_model import get_model
from src.utils.util import get_avg_dict_val_store, update_dict_val_store, ParseKwargs, set_global_logging_level, set_seeds, freezing_params
import ipdb
from tqdm import tqdm
import math

set_global_logging_level(logging.ERROR)

def train(config, device):
    '''
    Trains the model

    :param config:
    :return:
    '''
    
    dataset_reader = DatasetReader(config, device)
    batcher = Batcher(config, dataset_reader)

    train_iter = batcher.get_train_batch()
    model = get_model(config, dataset_reader, device)
    if config.weight_path != "":
        if config.train_layer == 7:
            state_dict = torch.load(config.weight_path, map_location=torch.device("cpu"))
            del state_dict['model.layer4.1.adapter_controller.multi_adapters.down_samplers_weights']
            del state_dict['model.layer4.1.adapter_controller.multi_adapters.down_samplers_bias']
            del state_dict['model.layer4.1.adapter_controller.multi_adapters.up_samplers_weights']
            del state_dict['model.layer4.1.adapter_controller.multi_adapters.up_samplers_bias']
            del state_dict['model.layer4.1.adapter_controller.pre_batch_norm.weight']
            del state_dict['model.layer4.1.adapter_controller.pre_batch_norm.bias']
            load_result = model.load_state_dict(state_dict, strict=False)
            print(f'Unexpected keys are {load_result.unexpected_keys.__str__()}')
        else:
            print(f'Loading the model from {config.weight_path}')
            state_dict = torch.load(config.weight_path, map_location=torch.device("cpu"))
            if config.save_for_gitcml:
                router_keys = [key for key in state_dict.keys() if 'multi_router' in key]
                for key in router_keys:
                    del state_dict[key]
            load_result = model.load_state_dict(state_dict, strict=False)
            print(f'Unexpected keys are {load_result.unexpected_keys.__str__()}')
    
    if config.save_for_gitcml:
        torch.save(model.state_dict(), os.path.join(config.exp_dir, f"model_n{config.num_adapters}.pt"))
        return         
    if config.dataset != "Shapes":
        freezing_params(model, config)
    optimizer = get_optimizer(model, config)
    
    # Create scheduler and calculate number of batches per epoch to know how frequently to step
    if config.scheduler is not None:
        scheduler = get_scheduler(optimizer, config)

    tot_num_batches = config.num_batches * config.grad_accum_factor
    dict_val_store = None
    best_dev_acc = 0 if config.dataset!="Shapes" else -50000

    for i in tqdm(range(tot_num_batches)):
        batch_idx = i // (config.grad_accum_factor)
        model.train()
        train_batch = next(train_iter)
        loss, dict_val_update = model(train_batch)
        loss = loss / config.grad_accum_factor
        loss.backward()

        dict_val_store = update_dict_val_store(dict_val_store, dict_val_update, config.grad_accum_factor)
        if (batch_idx + 1) % config.log_every == 0 and i % config.grad_accum_factor == 0:
            for k in dict_val_update.keys():
                print(f'The value of {k} is {round(dict_val_update[k].item(),3)}')                

        if (i+1) % config.grad_accum_factor == 0:
            if config.clip_grad_per_module:
                for param in model.parameters():
                    torch.nn.utils.clip_grad_norm_(param, config.grad_clip_norm)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            if config.scheduler is not None:
                scheduler.step()
            optimizer.zero_grad()

        if (batch_idx + 1) % config.eval_every == 0 and i % config.grad_accum_factor == 0: 
            if config.probe_input_features:
                test_eval(config, model, batcher)
            else:
                config.adapter_temp = np.maximum(config.adapter_temp*np.exp(-config.anneal_rate*batch_idx), config.min_temp) 
                dict_avg_val  = get_avg_dict_val_store(dict_val_store, config.eval_every)
                dev_acc, dict_dev_scores = dev_eval(config, model, batcher, batch_idx, dict_avg_val)
                print("Global Step: %d Acc: %.3f" % (batch_idx, dev_acc) + '\n')
                if dev_acc > best_dev_acc:
                    best_dev_acc = dev_acc
                    if config.save_model:
                        torch.save(model.state_dict(), os.path.join(config.exp_dir, "best_model.pt"))

                dict_val_store = None
        if (batch_idx + 1) == tot_num_batches and config.save_last_checkpoint:
            torch.save(model.state_dict(), os.path.join(config.exp_dir, "last_checkpoint.pt"))
        if config.forget_relearn == True:
            if (batch_idx+1) % (config.forget_every) == 0:
                for name, param in model.named_parameters():
                    if 'router_weights' in name:
                        nn.init.kaiming_uniform_(param.data, a=math.sqrt(5))
                    if 'router_bias' in name:
                        nn.init.zeros_(param.data)

        if config.freeze_router_after != -1:
            if (batch_idx+1) == config.freeze_router_after:
                for name, param in model.named_parameters():
                    if 'router_weights' in name:
                        param.requires_grad = False
                    if 'router_bias' in name:
                        param.requires_grad = False
                


    if config.eval_test and config.save_model:
        model.load_state_dict(torch.load(os.path.join(config.exp_dir, "best_model.pt")))
        test_eval(config, model, batcher)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', "--config_file", required=True)
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs, default={})
    args = parser.parse_args()

    config = Config(args.config_file, args.kwargs, mkdir=True)

    set_seeds(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config.device = device
    train(config, device)
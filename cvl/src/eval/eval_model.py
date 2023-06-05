import json
import torch
import numpy as np
from src.eval.Scorer import Scorer
import ipdb
import time

def eval_model(config, model, scorer, batch_iter):
    '''
    Evaluate model for online setting
    Returns:
    '''
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(batch_iter):
            if config.dataset == 'Shapes':
                recons_loss = model.predict(batch)
                batch_idx = batch["input"]["idx"]
                scorer.add_batch(batch_idx, recons_loss.detach().cpu())
            else:
                start_time = time.time()
                pred_lbl, _ = model.predict(batch)
                end_time = time.time()
                config.eval_time += (end_time - start_time)
                batch_idx = batch["input"]["idx"]
                true_lbl = batch["output"]["lbl"]
                domain_lbl = batch['output']['orig_domain_lbl']
                scorer.add_batch(batch_idx, pred_lbl.detach().cpu(), true_lbl, domain_lbl)

def dev_eval(config, model, batcher, num_batches, dict_avg_val=None):
    '''
    Evaluates the accuracy on the dev partition

    :param config:
    :param model:
    :param num_batches:
    :param loss:

    :return: currrent dev accuracy or average of previous split dev accuracy
    '''

    dict_eval = {}
    dict_eval["num_batches"] = num_batches

    if dict_avg_val is not None:
        dict_eval.update(dict_avg_val)

    dev_score = num_batches
    # Get dev Score
    dev_scorer = Scorer(config)
    batch_iter = batcher.get_dev_batch()
    eval_model(config, model, dev_scorer, batch_iter)
    dev_score, dict_dev_scores = dev_scorer.get_score()
    dict_eval.update({"dev": dict_dev_scores})
    if config.test_mode == False:
        with open(config.dev_score_file, 'a+') as f_out:
            f_out.write(json.dumps(dict_eval))
            f_out.write('\n')
    else:
        print('Dev score is ', dev_score)
        print('Dev dict is ', dict_dev_scores)

    return dev_score, dict_dev_scores


def test_eval(config, model, batcher,num_batches=0):
    '''
    Evaluates the accuracy on the dev partition
    :param config:
    :param model:
    :param batcher:
   '''

    dict_eval = {}
    # Get test Score
    test_scorer = Scorer(config)
    batch_iter = batcher.get_test_batch()
    eval_model(config, model, test_scorer, batch_iter)
    if config.probe_input_features:
        for key in config.num_count_domain_pred:
            acc = round (config.num_count_domain_pred[key] / config.den_count_domain_pred[key], 4)
            print(f'Accuracy at layer {key} is {acc}')
            with open(config.test_score_file, 'a+') as f_out:
                f_out.write(json.dumps(dict_eval))
                f_out.write(f'Accuracy at layer {key} is {acc}')
                f_out.write('\n')
    else:
        test_score, dict_test_scores = test_scorer.get_score()
        dict_eval.update({"test": dict_test_scores})
        if not config.test_mode:
            with open(config.test_score_file, 'a+') as f_out:
                f_out.write(json.dumps(dict_eval))
                f_out.write('\n')
        print('Test score is ', test_score)
        print('Test dict is ', dict_test_scores)
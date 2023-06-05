import argparse
import os
import torch
from src.data.DatasetReader import DatasetReader
from src.utils.util import ParseKwargs

from src.data.Batcher import Batcher
from src.utils.get_model import get_model
from src.utils.Config import Config
from src.eval.eval_model import test_eval
import ipdb
import pickle

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', "--exp_dir", required=True)
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs,  default={})
    args = parser.parse_args()

    config_file = os.path.join(args.exp_dir, "config.json")
    config = Config(config_file, args.kwargs, mkdir=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset_reader = DatasetReader(config, device)
    batcher = Batcher(config, dataset_reader)
    model = get_model(config, dataset_reader, device)
    if config.dataset == 'Shapes':
        load_result = model.load_state_dict(torch.load(os.path.join(args.exp_dir, "last_checkpoint.pt"), map_location=torch.device("cpu")), strict=False)
    else:
        load_result = model.load_state_dict(torch.load(os.path.join(args.exp_dir, "best_model.pt"), map_location=torch.device("cpu")), strict=False)
    print(f'Unexpected keys are {load_result.unexpected_keys.__str__()}')
    test_eval(config, model, batcher)

    print('time taken in seconds ', config.eval_time)
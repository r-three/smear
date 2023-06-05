import argparse
import os
import torch

from src.utils.util import ParseKwargs


from src.data.DatasetReader import DatasetReader
from src.data.Batcher import Batcher
from src.utils.Config import Config
from src.eval.eval_model import dev_eval
from src.utils.get_model import get_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', "--exp_dir", required=True)
    parser.add_argument('-k', '--kwargs', nargs='*', action=ParseKwargs,  default={})
    args = parser.parse_args()

    config_file = os.path.join(args.exp_dir, "config.json")
    config = Config(config_file, args.kwargs, mkdir=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset_reader = DatasetReader(config, device)
    model = get_model(config, dataset_reader, device)
    model.load_state_dict(torch.load(os.path.join(args.exp_dir, "best_model.pt")))

    batcher = Batcher(config, dataset_reader)
    dev_score = dev_eval(config, model, batcher, config.num_batches, {})
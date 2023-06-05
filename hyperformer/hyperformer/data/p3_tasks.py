"""Implements different tasks and defines the processors to convert each dataset
to a sequence to sequence format."""
from collections import OrderedDict

import abc
import datasets
import functools
import logging
import numpy as np
import torch
from hyperformer.metrics import metrics
from typing import Callable, Dict, Mapping, List

from hyperformer.data.utils import round_stsb_target, compute_task_max_decoding_length

logger = logging.getLogger(__name__)
import pdb
import numpy as np
from promptsource.templates import DatasetTemplates

class DatasetWithTemplate(torch.utils.data.dataset.Dataset):
    def __init__(self, dataset, list_templates, semi_supervised_ratio, name, num_layers, eval_all_templates):
        super().__init__()
        self.dataset = dataset
        self.templates = list_templates
        self.semi_supervised_ratio = semi_supervised_ratio
        self.name = name
        self.num_layers = num_layers
        self.eval_all_templates = eval_all_templates
        self.hash_lbls = np.random.randint(8, size=(len(self.dataset), self.num_layers))

    def __len__(self):
        if self.eval_all_templates:
            return len(self.dataset) * len(self.templates)
        return len(self.dataset)

    def seq2seq_format(self, key, src_strs: List[str], tgt_strs: List[str], add_task_label: bool = True):
        # need to hard code number of adapters and all routers in case of GLUE
        if add_task_label:
            return {"src_texts": ' '.join(src_strs),
                    "tgt_texts": ' '.join(tgt_strs),
                    "orig_task": self.name,
                    "task": self.name,
                    "hash_lbl": self.hash_lbls[key]}
        else:
            return {"src_texts": ' '.join(src_strs),
                    "tgt_texts": ' '.join(tgt_strs),
                    "orig_task": self.name,
                    "task": 'None',
                    "hash_lbl": self.hash_lbls[key]} 

    def __getitem__(self, index):
        if self.eval_all_templates:
            template = self.templates[index % len(self.templates)]
            key = index // len(self.templates)
        else:
            template = np.random.choice(self.templates)
            key = index
        example = self.dataset[key]
        src_texts, tgt_texts = template.apply(example)
        if self.semi_supervised_ratio == 1:
            return self.seq2seq_format(key, [src_texts], [tgt_texts], True)
        else:
            p = np.random.random()
            if p < self.semi_supervised_ratio:
                add_task_label = True
            else:
                add_task_label = False
            return self.seq2seq_format(key, [src_texts], [tgt_texts], add_task_label)

class AbstractTaskDataset(abc.ABC):
    """Defines the abstract class for all the tasks.
    name: the name of the task.
    task_specific_config: specifies the special configuration needs
        to be passed to encoder when decoding each task. Since different
        tasks, have different output space, the maximum decoding length
        varies based on the tasks.
    preprocessor: a processor to convert the given dataset to the sequence
        to sequence format.
    metrics: specifies the metrics to evaluate the task based on them.
    split_to_data_split: since not all the time, different splits of the
        datasets are available, we define a mapping from the wanted split
        to the existing dataset splits.
    small_datasets_without_all_splits: List of strings, defines the name
        of all low-resource tasks in which not all train/test/validation
        splits are available.
    large_data_without_all_splits: List of strings, defines the name of
        all high-resource tasks in which not all train/test/validation
        splits are available.
    """
    name = NotImplemented
    task_specific_config: Dict = NotImplemented
    preprocessor: Callable = NotImplemented
    metrics: List[Callable] = NotImplemented
    split_to_data_split: Mapping[str, str] = \
        {"train": "train", "validation": "validation", "test": "test"}

    small_datasets_without_all_splits = ["cola", "wnli", "rte", "trec", "superglue-cb", "sick",
                                         "mrpc", "stsb", "imdb", "commonsense_qa", "superglue-boolq"]
    large_data_without_all_splits = ["yelp_polarity", "qqp", "qnli",
                                     "social_i_qa", "cosmos_qa", "winogrande", "hellaswag", "sst2"]

    def __init__(self, semi_supervised_ratio=1.0, seed=42, num_layers=60, eval_all_templates=False):
        self.seed = seed
        self.semi_supervised_ratio = semi_supervised_ratio
        self.num_layers = num_layers
        self.eval_all_templates = eval_all_templates

    def get_sampled_split(self, split: int, n_obs: int = None):
        # If the requested number of observation is more than dataset
        # size we reset it to the maximum available.
        split = self.split_to_data_split[split]
        dataset = self.load_dataset(split)
        total_size = len(dataset)
        n_obs = self.check_n_obs(n_obs, total_size)
        if n_obs is not None:
            split = split + "[:{}]".format(n_obs)
        return split

    def get_shuffled_sampled_split(self, split: int, n_obs: int = None):
        # Defines the random generator.
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        # If the requested number of observation is more than dataset
        # size we reset it to the maximum available.
        mapped_split = self.split_to_data_split[split]
        dataset = self.load_dataset(mapped_split)
        # shuffle the dataset and get the random samples.
        train_size = len(dataset)
        indices = torch.randperm(train_size, generator=generator).tolist()
        dataset = self.select_dataset_samples(indices, dataset, n_obs=n_obs)
        return dataset

    def check_n_obs(self, n_obs, total_size):
        if n_obs is not None and n_obs > total_size:
            n_obs = total_size
            logger.warning("n_obs is set to %s", n_obs)
        return n_obs

    def select_dataset_samples(self, indices, dataset, n_obs: int = None):
        """
        Given a dataset for the split, obtains the sample indices for this split
        and returns the subsampled dataset.
        :param indices: the selected indices.
        :param dataset: dataset corresponding to this split.
        :return: subsampled dataset.
        """
        n_obs = self.check_n_obs(n_obs, len(indices))
        indices = indices[:n_obs] if n_obs is not None else indices
        return dataset.select(indices)

    def load_dataset(self, split: int):
        return datasets.load_dataset(self.name, split=split, )

    def get_train_split_indices(self, split):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        mapped_split = self.split_to_data_split["train"]
        dataset = self.load_dataset(mapped_split)
        train_size = len(dataset)
        indices = torch.randperm(train_size, generator=generator).tolist()
        validation_size = 1000
        if split == "validation":
            return indices[:validation_size]
        else:
            return indices[validation_size:]

    def get_half_validation_indices(self, split):
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        mapped_split = self.split_to_data_split["validation"]
        dataset = self.load_dataset(mapped_split)
        validation_size = len(dataset)
        indices = torch.randperm(validation_size, generator=generator).tolist()
        if split == "validation":
            return indices[:(validation_size // 2)]
        else:
            return indices[validation_size // 2:]

class MRPCTaskDataset(AbstractTaskDataset):
    name = "mrpc"
    metrics = [metrics.f1_score_with_invalid, metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    templates = DatasetTemplates("glue", "mrpc")
    template_names = templates.all_template_names
    list_idx = []
    list_templates = []
    for idx, template_name in enumerate(template_names):
        if templates[template_name].metadata.original_task:
            list_idx.append(idx)
            list_templates.append(templates[template_name])
    label_list = []
    for template in list_templates:
        label_list.extend(template.answer_choices.split("|||"))
    print(f"The label list for mrpc is {label_list}")
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'mrpc',
                                     split=split, )

    def get_dataset(self, split, n_obs=None, add_prefix=True, split_validation_test=False):
        # For small datasets (n_samples < 10K) without test set, we divide validation set to
        # half, use one half as test set and one half as validation set.
        if split_validation_test and self.name in self.small_datasets_without_all_splits \
                and split != "train":
            mapped_split = self.split_to_data_split["validation"]
            dataset = self.load_dataset(split=mapped_split)
            indices = self.get_half_validation_indices(split)
            dataset = self.select_dataset_samples(indices, dataset, n_obs)
        # For larger datasets (n_samples > 10K), we divide training set into 1K as
        # validation and the rest as training set, keeping the original validation
        # set as the test set.
        elif split_validation_test and self.name in self.large_data_without_all_splits \
                and split != "test":
            dataset = self.load_dataset(split="train")
            indices = self.get_train_split_indices(split)
            dataset = self.select_dataset_samples(indices, dataset, n_obs)
        else:
            # TODO: later we can join these as one.
            if n_obs == -1:
                split = self.get_sampled_split(split, n_obs)
                dataset = self.load_dataset(split=split)
            else:
                # shuffles the data and samples it.
                dataset = self.get_shuffled_sampled_split(split, n_obs)
        # return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix),
        #                    remove_columns=dataset.column_names)
        return DatasetWithTemplate(dataset, self.list_templates, self.semi_supervised_ratio, self.name, self.num_layers, self.eval_all_templates)


class COLATaskDataset(AbstractTaskDataset):
    name = "cola"
    metrics = [metrics.matthews_corrcoef]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    templates = DatasetTemplates("glue", "cola")
    template_names = templates.all_template_names
    list_idx = []
    list_templates = []
    for idx, template_name in enumerate(template_names):
        if templates[template_name].metadata.original_task:
            list_idx.append(idx)
            list_templates.append(templates[template_name])
    label_list = []
    for template in list_templates:
        label_list.extend(template.answer_choices.split("|||"))
    print(f"The label list for cola is {label_list}")
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'cola',
                                     split=split, )

    def get_dataset(self, split, n_obs=None, add_prefix=True, split_validation_test=False):
        # For small datasets (n_samples < 10K) without test set, we divide validation set to
        # half, use one half as test set and one half as validation set.
        if split_validation_test and self.name in self.small_datasets_without_all_splits \
                and split != "train":
            mapped_split = self.split_to_data_split["validation"]
            dataset = self.load_dataset(split=mapped_split)
            indices = self.get_half_validation_indices(split)
            dataset = self.select_dataset_samples(indices, dataset, n_obs)
        # For larger datasets (n_samples > 10K), we divide training set into 1K as
        # validation and the rest as training set, keeping the original validation
        # set as the test set.
        elif split_validation_test and self.name in self.large_data_without_all_splits \
                and split != "test":
            dataset = self.load_dataset(split="train")
            indices = self.get_train_split_indices(split)
            dataset = self.select_dataset_samples(indices, dataset, n_obs)
        else:
            # TODO: later we can join these as one.
            if n_obs == -1:
                split = self.get_sampled_split(split, n_obs)
                dataset = self.load_dataset(split=split)
            else:
                # shuffles the data and samples it.
                dataset = self.get_shuffled_sampled_split(split, n_obs)
        # return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix),
        #                    remove_columns=dataset.column_names)
        return DatasetWithTemplate(dataset, self.list_templates, self.semi_supervised_ratio, self.name, self.num_layers, self.eval_all_templates)


class SST2TaskDataset(AbstractTaskDataset):
    name = "sst2"
    metrics = [metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    templates = DatasetTemplates("glue", "sst2")
    template_names = templates.all_template_names
    list_idx = []
    list_templates = []
    for idx, template_name in enumerate(template_names):
        if templates[template_name].metadata.original_task:
            list_idx.append(idx)
            list_templates.append(templates[template_name])
    label_list = []
    for template in list_templates:
        label_list.extend(template.answer_choices.split("|||"))
    print(f"The label list for sst2 is {label_list}")
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'sst2',
                                     split=split, )

    def get_dataset(self, split, n_obs=None, add_prefix=True, split_validation_test=False):
        # For small datasets (n_samples < 10K) without test set, we divide validation set to
        # half, use one half as test set and one half as validation set.
        if split_validation_test and self.name in self.small_datasets_without_all_splits \
                and split != "train":
            mapped_split = self.split_to_data_split["validation"]
            dataset = self.load_dataset(split=mapped_split)
            indices = self.get_half_validation_indices(split)
            dataset = self.select_dataset_samples(indices, dataset, n_obs)
        # For larger datasets (n_samples > 10K), we divide training set into 1K as
        # validation and the rest as training set, keeping the original validation
        # set as the test set.
        elif split_validation_test and self.name in self.large_data_without_all_splits \
                and split != "test":
            dataset = self.load_dataset(split="train")
            indices = self.get_train_split_indices(split)
            dataset = self.select_dataset_samples(indices, dataset, n_obs)
        else:
            # TODO: later we can join these as one.
            if n_obs == -1:
                split = self.get_sampled_split(split, n_obs)
                dataset = self.load_dataset(split=split)
            else:
                # shuffles the data and samples it.
                dataset = self.get_shuffled_sampled_split(split, n_obs)
        # return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix),
        #                    remove_columns=dataset.column_names)
        return DatasetWithTemplate(dataset, self.list_templates, self.semi_supervised_ratio, self.name, self.num_layers, self.eval_all_templates)



class STSBTaskDataset(AbstractTaskDataset):
    name = "stsb"
    label_list = [str(np.round(label, decimals=1)) for label in np.arange(0, 5.2, 0.2)]
    # task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    metrics = [metrics.pearson_corrcoef, metrics.spearman_corrcoef]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    templates = DatasetTemplates("glue", "stsb")
    template_names = templates.all_template_names
    list_idx = []
    list_templates = []
    for idx, template_name in enumerate(template_names):
        if templates[template_name].metadata.original_task:
            list_idx.append(idx)
            list_templates.append(templates[template_name])
    print(f"The label list for stsb is {label_list}")
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'stsb',
                                     split=split, )

    def get_dataset(self, split, n_obs=None, add_prefix=True, split_validation_test=False):
        # For small datasets (n_samples < 10K) without test set, we divide validation set to
        # half, use one half as test set and one half as validation set.
        if split_validation_test and self.name in self.small_datasets_without_all_splits \
                and split != "train":
            mapped_split = self.split_to_data_split["validation"]
            dataset = self.load_dataset(split=mapped_split)
            indices = self.get_half_validation_indices(split)
            dataset = self.select_dataset_samples(indices, dataset, n_obs)
        # For larger datasets (n_samples > 10K), we divide training set into 1K as
        # validation and the rest as training set, keeping the original validation
        # set as the test set.
        elif split_validation_test and self.name in self.large_data_without_all_splits \
                and split != "test":
            dataset = self.load_dataset(split="train")
            indices = self.get_train_split_indices(split)
            dataset = self.select_dataset_samples(indices, dataset, n_obs)
        else:
            # TODO: later we can join these as one.
            if n_obs == -1:
                split = self.get_sampled_split(split, n_obs)
                dataset = self.load_dataset(split=split)
            else:
                # shuffles the data and samples it.
                dataset = self.get_shuffled_sampled_split(split, n_obs)
        # return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix),
        #                    remove_columns=dataset.column_names)
        return DatasetWithTemplate(dataset, self.list_templates, self.semi_supervised_ratio, self.name, self.num_layers, self.eval_all_templates)



class QQPTaskDataset(AbstractTaskDataset):
    name = "qqp"
    metrics = [metrics.f1_score_with_invalid, metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    templates = DatasetTemplates("glue", "qqp")
    template_names = templates.all_template_names
    list_idx = []
    list_templates = []
    for idx, template_name in enumerate(template_names):
        if templates[template_name].metadata.original_task:
            list_idx.append(idx)
            list_templates.append(templates[template_name])
    label_list = []
    for template in list_templates:
        label_list.extend(template.answer_choices.split("|||"))
    print(f"The label list for qqp is {label_list}")
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qqp',
                                     split=split, )

    def get_dataset(self, split, n_obs=None, add_prefix=True, split_validation_test=False):
        # For small datasets (n_samples < 10K) without test set, we divide validation set to
        # half, use one half as test set and one half as validation set.
        if split_validation_test and self.name in self.small_datasets_without_all_splits \
                and split != "train":
            mapped_split = self.split_to_data_split["validation"]
            dataset = self.load_dataset(split=mapped_split)
            indices = self.get_half_validation_indices(split)
            dataset = self.select_dataset_samples(indices, dataset, n_obs)
        # For larger datasets (n_samples > 10K), we divide training set into 1K as
        # validation and the rest as training set, keeping the original validation
        # set as the test set.
        elif split_validation_test and self.name in self.large_data_without_all_splits \
                and split != "test":
            dataset = self.load_dataset(split="train")
            indices = self.get_train_split_indices(split)
            dataset = self.select_dataset_samples(indices, dataset, n_obs)
        else:
            # TODO: later we can join these as one.
            if n_obs == -1:
                split = self.get_sampled_split(split, n_obs)
                dataset = self.load_dataset(split=split)
            else:
                # shuffles the data and samples it.
                dataset = self.get_shuffled_sampled_split(split, n_obs)
        # return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix),
        #                    remove_columns=dataset.column_names)
        return DatasetWithTemplate(dataset, self.list_templates, self.semi_supervised_ratio, self.name, self.num_layers, self.eval_all_templates)



class MNLITaskDataset(AbstractTaskDataset):
    name = "mnli"
    split_to_data_split = {"train": "train",
                           "validation": "validation_mismatched",
                           "test": "validation_matched"}
    metrics = [metrics.accuracy]
    templates = DatasetTemplates("glue", "mnli")
    template_names = templates.all_template_names
    list_idx = []
    list_templates = []
    for idx, template_name in enumerate(template_names):
        if templates[template_name].metadata.original_task:
            list_idx.append(idx)
            list_templates.append(templates[template_name])
    label_list = []
    for template in list_templates:
        label_list.extend(template.answer_choices.split("|||"))
    print(f"The label list for mnli is {label_list}")
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}

    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'mnli', split=split, )

    def get_dataset(self, split, n_obs=None, add_prefix=True, split_validation_test=False):
        # For small datasets (n_samples < 10K) without test set, we divide validation set to
        # half, use one half as test set and one half as validation set.
        if split_validation_test and self.name in self.small_datasets_without_all_splits \
                and split != "train":
            mapped_split = self.split_to_data_split["validation"]
            dataset = self.load_dataset(split=mapped_split)
            indices = self.get_half_validation_indices(split)
            dataset = self.select_dataset_samples(indices, dataset, n_obs)
        # For larger datasets (n_samples > 10K), we divide training set into 1K as
        # validation and the rest as training set, keeping the original validation
        # set as the test set.
        elif split_validation_test and self.name in self.large_data_without_all_splits \
                and split != "test":
            dataset = self.load_dataset(split="train")
            indices = self.get_train_split_indices(split)
            dataset = self.select_dataset_samples(indices, dataset, n_obs)
        else:
            # TODO: later we can join these as one.
            if n_obs == -1:
                split = self.get_sampled_split(split, n_obs)
                dataset = self.load_dataset(split=split)
            else:
                # shuffles the data and samples it.
                dataset = self.get_shuffled_sampled_split(split, n_obs)
        # return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix),
        #                    remove_columns=dataset.column_names)
        return DatasetWithTemplate(dataset, self.list_templates, self.semi_supervised_ratio, self.name, self.num_layers, self.eval_all_templates)



class QNLITaskDataset(AbstractTaskDataset):
    name = "qnli"
    metrics = [metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    templates = DatasetTemplates("glue", "qnli")
    template_names = templates.all_template_names
    list_idx = []
    list_templates = []
    for idx, template_name in enumerate(template_names):
        if templates[template_name].metadata.original_task:
            list_idx.append(idx)
            list_templates.append(templates[template_name])
    label_list = []
    for template in list_templates:
        label_list.extend(template.answer_choices.split("|||"))
    print(f"The label list for qnli is {label_list}")
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'qnli', split=split, )

    def get_dataset(self, split, n_obs=None, add_prefix=True, split_validation_test=False):
        # For small datasets (n_samples < 10K) without test set, we divide validation set to
        # half, use one half as test set and one half as validation set.
        if split_validation_test and self.name in self.small_datasets_without_all_splits \
                and split != "train":
            mapped_split = self.split_to_data_split["validation"]
            dataset = self.load_dataset(split=mapped_split)
            indices = self.get_half_validation_indices(split)
            dataset = self.select_dataset_samples(indices, dataset, n_obs)
        # For larger datasets (n_samples > 10K), we divide training set into 1K as
        # validation and the rest as training set, keeping the original validation
        # set as the test set.
        elif split_validation_test and self.name in self.large_data_without_all_splits \
                and split != "test":
            dataset = self.load_dataset(split="train")
            indices = self.get_train_split_indices(split)
            dataset = self.select_dataset_samples(indices, dataset, n_obs)
        else:
            # TODO: later we can join these as one.
            if n_obs == -1:
                split = self.get_sampled_split(split, n_obs)
                dataset = self.load_dataset(split=split)
            else:
                # shuffles the data and samples it.
                dataset = self.get_shuffled_sampled_split(split, n_obs)
        # return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix),
        #                    remove_columns=dataset.column_names)
        return DatasetWithTemplate(dataset, self.list_templates, self.semi_supervised_ratio, self.name, self.num_layers, self.eval_all_templates)



class RTETaskDataset(AbstractTaskDataset):
    name = "rte"
    metrics = [metrics.accuracy]
    split_to_data_split = {"train": "train",
                           "validation": "validation",
                           "test": "validation"}
    templates = DatasetTemplates("glue", "rte")
    template_names = templates.all_template_names
    list_idx = []
    list_templates = []
    for idx, template_name in enumerate(template_names):
        if templates[template_name].metadata.original_task:
            list_idx.append(idx)
            list_templates.append(templates[template_name])
    label_list = []
    for template in list_templates:
        label_list.extend(template.answer_choices.split("|||"))
    print(f"The label list for rte is {label_list}")
    task_specific_config = {'max_length': compute_task_max_decoding_length(label_list)}
    def load_dataset(self, split):
        return datasets.load_dataset('glue', 'rte',
                                     split=split, )
    def get_dataset(self, split, n_obs=None, add_prefix=True, split_validation_test=False):
        # For small datasets (n_samples < 10K) without test set, we divide validation set to
        # half, use one half as test set and one half as validation set.
        if split_validation_test and self.name in self.small_datasets_without_all_splits \
                and split != "train":
            mapped_split = self.split_to_data_split["validation"]
            dataset = self.load_dataset(split=mapped_split)
            indices = self.get_half_validation_indices(split)
            dataset = self.select_dataset_samples(indices, dataset, n_obs)
        # For larger datasets (n_samples > 10K), we divide training set into 1K as
        # validation and the rest as training set, keeping the original validation
        # set as the test set.
        elif split_validation_test and self.name in self.large_data_without_all_splits \
                and split != "test":
            dataset = self.load_dataset(split="train")
            indices = self.get_train_split_indices(split)
            dataset = self.select_dataset_samples(indices, dataset, n_obs)
        else:
            # TODO: later we can join these as one.
            if n_obs == -1:
                split = self.get_sampled_split(split, n_obs)
                dataset = self.load_dataset(split=split)
            else:
                # shuffles the data and samples it.
                dataset = self.get_shuffled_sampled_split(split, n_obs)
        # return dataset.map(functools.partial(self.preprocessor, add_prefix=add_prefix),
        #                    remove_columns=dataset.column_names)
        return DatasetWithTemplate(dataset, self.list_templates, self.semi_supervised_ratio, self.name, self.num_layers, self.eval_all_templates)


TASK_MAPPING = OrderedDict([
    ('mrpc', MRPCTaskDataset),
    ('cola', COLATaskDataset),
    ('sst2', SST2TaskDataset),
    ('stsb', STSBTaskDataset),
    ('qqp', QQPTaskDataset),
    ('mnli', MNLITaskDataset),
    ('qnli', QNLITaskDataset),
    ('rte', RTETaskDataset)]
)

class AutoTask:
    @classmethod
    def get(self, task_name, semi_supervised_ratio=1.0, seed=42, num_layers=60, eval_all_templates=False):
        if task_name in TASK_MAPPING:
            return TASK_MAPPING[task_name](semi_supervised_ratio, seed, num_layers, eval_all_templates)
        raise ValueError(
            "Unrecognized task {} for AutoTask Model: {}.\n"
            "Task name should be one of {}.".format(
                ", ".join(c for c in TASK_MAPPING.keys())
            )
        )

if __name__ == "__main__":
    dataset_class = AutoTask
    import pdb; pdb.set_trace()
    dataset_object = dataset_class.get('rte')
    dataset = dataset_object.get_dataset(split='validation', add_prefix=False)


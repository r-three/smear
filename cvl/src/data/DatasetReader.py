import ipdb
import torch
import torch.nn.functional as F
import random
from PIL import Image
import pickle
import os
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop, RandomHorizontalFlip
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC
from torchvision import datasets
from torch.utils.data import Dataset
import numpy as np 

class Reader(object):
    '''
    Reader reads images and text 
    '''

    def __init__(self, dataset, config, device):
        self.config = config
        self.dataset = dataset

    def read_dataset(self, split):
        '''
        Read the original dataset

        :param split: split of data
        '''
        def _convert_image_to_rgb(image):
            return image.convert("RGB")
        if self.config.model == "clip":
            _train_preprocess =  Compose([
                RandomResizedCrop(224),
                RandomHorizontalFlip(),
                _convert_image_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
            _test_preprocess =  Compose([
                Resize(256),
                CenterCrop(224),
                _convert_image_to_rgb,
                ToTensor(),
                Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
            ])
        else:
            _train_preprocess =  Compose([
                RandomResizedCrop(224),
                RandomHorizontalFlip(),
                _convert_image_to_rgb,
                ToTensor(),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
            _test_preprocess =  Compose([
                Resize(256),
                CenterCrop(224),
                _convert_image_to_rgb,
                ToTensor(),
                Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        if self.dataset == 'DomainNet':
            domain_index_dict = {"clipart":0, "infograph":1, "painting":2, "quickdraw":3, "real":4 , "sketch":5}                                
            class DomainNetDataset(Dataset):
                def __init__(self, config, input_file_list, root_dir, transform, train=True):
                    self.root_dir = root_dir
                    self.transform = transform
                    self.annotations = []
                    self.domain_lbls = []
                    if config.semi_supervised_ratio != 0:
                        self.modified_domain_lbls = []
                    self.config = config
                    if type(input_file_list) == type([]):
                        for index, input_file in enumerate(input_file_list):
                            with open(self.root_dir + input_file, 'r') as f:
                                examples = f.readlines()
                                if self.config.create_unbalance and train:
                                    if index == 0:
                                        examples = examples[0:1000]
                                    if index == 1:
                                        examples = examples[0:2000]
                                    if index == 2:
                                        examples = examples[0:10000]
                                    if index == 5:
                                        examples = examples[0:10000]

                                self.annotations.extend(examples)
                                self.domain_lbls.extend([index for _ in range(len(examples))])
                    else:
                        with open(self.root_dir + input_file_list, 'r') as f:
                            examples = f.readlines()
                            self.annotations.extend(examples)

                    # fixing random adapters for hash routing
                    self.hash_lbls = []
                    for index in range(len(self.annotations)):
                        self.hash_lbls.append(np.random.randint(config.num_adapters, size=config.all_routers))
                             
                    if config.semi_supervised_ratio != 1:
                        self.modified_domain_lbls = []
                        for i in range(len(self.domain_lbls)):
                            p = np.random.random()
                            if p < config.semi_supervised_ratio:
                                self.modified_domain_lbls.append(self.domain_lbls[i])
                            else:
                                self.modified_domain_lbls.append(-100)

                def __len__(self):
                    return len(self.annotations)

                def __getitem__(self, idx):
                    
                    annot = self.annotations[idx].strip().split()
                    image_path = self.root_dir + annot[0]
                    lbl = int(annot[1])
                    if len(self.domain_lbls) == 0:
                        domain_lbl = int(annot[2])
                    else:
                        domain_lbl = self.domain_lbls[idx]
                    image = self.transform(Image.open(image_path))

                    dict_input = {'idx': idx, 'image': image}
                    if self.config.semi_supervised_ratio != 1:
                        dict_output = {'lbl': lbl, 'domain_lbl': self.modified_domain_lbls[idx], 'orig_domain_lbl': domain_lbl, 'hash_lbl': self.hash_lbls[idx]}
                    else:
                        dict_output = {'lbl': lbl, 'domain_lbl': domain_lbl, 'orig_domain_lbl': domain_lbl, 'hash_lbl': self.hash_lbls[idx]}
                    dict_data = {"input": dict_input, "output": dict_output}
                    return dict_data

            if split == 'train' and self.config.train_size:
                input_file_list = f'Domainnet_train_{self.config.train_size}k.txt'
            else:
                if split == "test":
                    input_file_list = [f'{domain_name}_{split}.txt' for domain_name in domain_index_dict]
                else:
                    input_file_list = [f'{domain_name}_{split}_fold.txt' for domain_name in domain_index_dict]
            if split == 'train':
                dataset = DomainNetDataset(self.config, input_file_list, './data/DomainNet/', _train_preprocess)
            else:
                dataset = DomainNetDataset(self.config, input_file_list, './data/DomainNet/', _test_preprocess, train=False)
            print(f'Length of {split} data is {len(dataset)}')
            return dataset

    def get_num_lbl(self):
        if self.dataset == 'DomainNet':
            return 345

class ShapesReader(object):
    '''
    ShapesReader reads Shapes dataset
    '''

    def __init__(self, config):
        self.config = config
        self.dev_data = None
        self.images_lbl = None

    def get_num_lbl(self):
        '''
        Number of lbls in dataset

        :return:
        '''
        return 8

    def create_data(self, images_lbl):
        data = []
        for idx, (image,lbl) in enumerate(images_lbl):
            dict_input = {'idx': idx, 'image': image}
            dict_output = {'lbl': lbl}
            dict_data = {"input": dict_input, "output": dict_output}
            data.append(dict_data)
        return data

    def read_dataset(self, split):
        '''
        Read the original dataset

        :param split: split of data
        '''
        if split == 'dev' and self.dev_data != None:
            return self.dev_data
        data = []
        idx = 0
        _preprocess = Compose([
            ToTensor(),
        ])

        for letter in range(8):
            for position in range(8):
                for letter_color in range(8):
                    for bc_color in range(8):
                        idx+=1
                        image_path = 'image' + '_' + str(letter) + '_' + str(position) + '_' + str(letter_color) + '_' + str(bc_color) +  '_.png'
                        image_path = 'data/Shapes/' + image_path
                        im = Image.open(image_path)
                        image = _preprocess(im)
                        latent = np.zeros((4, 8))
                        latent[0,letter] = 1
                        latent[1, position] = 1
                        latent[2, letter_color] = 1
                        latent[3, bc_color] = 1
                        latent = torch.tensor(latent).float()

                        monolithic_latent = np.zeros((4, 8))
                        monolithic_latent[0,0] = 1
                        monolithic_latent[1, 0] = 1
                        monolithic_latent[2, 0] = 1
                        monolithic_latent[3, 0] = 1
                        monolithic_latent = torch.tensor(monolithic_latent).float()

                        hash_latent = F.one_hot(torch.randint(8, (4,)), num_classes=8).float()
                        if self.config.semi_supervised_ratio == 1:
                            modified_latent = torch.max(latent, dim=-1)[1]
                        else:
                            p = np.random.random()
                            if p < self.config.semi_supervised_ratio:
                                modified_latent = torch.max(latent, dim=-1)[1]
                            else:
                                modified_latent = torch.ones(self.config.latent_dim, dtype=int)*-100
                        dict_input = {'idx': idx, 'image': image}
                        dict_output = {'latent': latent, 'hash_latent': hash_latent, 'monolithic_latent': monolithic_latent, 'modified_latent': modified_latent}
                        dict_data = {"input": dict_input, "output": dict_output}
                        data.append(dict_data)
                        
        self.dev_data = data
        return data


class DatasetReader(object):
    '''
    DatasetReader is responsible for reading dataset
    '''
    def __init__(self, config, device):
        '''
        :param config:
        '''
        if config.dataset == "DomainNet":
            self.dataset_reader = Reader("DomainNet", config, device)
        elif config.dataset == "Shapes":
            self.dataset_reader = ShapesReader(config)
        else:
            raise ValueError("Invalid Dataset name")

    def get_preprocesser(self):
        '''
        Get preprocessor attached to dataset

        :return:
        '''
        return self.dataset_reader.preprocess

    def get_tokenizer(self):
        '''
        Get tokenizer attached to dataset

        :return:
        '''
        return self.dataset_reader.tokenizer

    def read_dataset(self, split):
        '''
        Read dataset based on split

        :param split:
        :return:
        '''
        return self.dataset_reader.read_dataset(split)

    def prepare_batch(self, batch):
        '''
        '''
        return self.dataset_reader.prepare_batch(batch)

    def get_num_lbl(self):
        return self.dataset_reader.get_num_lbl()
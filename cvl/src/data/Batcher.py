import torch
from torch.utils import data
from src.data.Dataset import Dataset


class Batcher(object):
    '''
    Batcher is responsible for returning batches of data
    '''
    def __init__(self, config, dataset_reader):
        '''
        :param config:
        '''
        self.config = config
        self.dataset_reader = dataset_reader

        self.train_loader = None
        self.dev_loader = None
        self.test_loader = None
        self.eval_train_loader = None


    def get_dataset_reader(self):
        '''
        Get dataset reader

        :return:
        '''
        return self.dataset_reader

    def _init_train(self):
        '''
        Initialize loader for train data
        '''
        train_data = self.dataset_reader.read_dataset("train")
        # if isinstance(train_data, list):
        #     self.data_len = len(train_data)
        #     self.train_loader = data.DataLoader(Dataset(train_data), batch_size=self.config.batch_size, shuffle=True)
        #     self.eval_train_loader = data.DataLoader(Dataset(train_data), batch_size=self.config.eval_batch_size, shuffle=False)
        # else:
        self.train_loader = data.DataLoader(train_data, batch_size=self.config.batch_size, shuffle=True, num_workers=20, pin_memory=False)
        self.eval_train_loader = data.DataLoader(train_data, batch_size=self.config.eval_batch_size, shuffle=False)

    def _init_dev(self):
        '''
        Initialize generators for dev data
        '''

        dev_data = self.dataset_reader.read_dataset("valid")
        # if isinstance(dev_data, list):
        #     self.dev_loader = data.DataLoader(Dataset(dev_data), batch_size=self.config.eval_batch_size, shuffle=False)
        # else:
        self.dev_loader = data.DataLoader(dev_data, batch_size=self.config.eval_batch_size, shuffle=False, num_workers=20, pin_memory=False)


    def _init_test(self):
        '''
        Initialize generators for test data
        '''

        test_data = self.dataset_reader.read_dataset("test")
        # if isinstance(test_data, list):
        #     self.test_loader = data.DataLoader(Dataset(test_data), batch_size=self.config.eval_batch_size, shuffle=False)
        # else:
        self.test_loader = data.DataLoader(test_data, batch_size=self.config.eval_batch_size, shuffle=False, num_workers=20, pin_memory=False)

    def get_train_batch(self):
        '''
        Yield regular train batches

        :return:
        '''
        if self.train_loader is None:
            self._init_train()

        while True:
            for x in self.train_loader:
                yield x

    def get_eval_train_batch(self):
        '''
        Yield regular train batches

        :return:
        '''
        if self.eval_train_loader is None:
            self._init_train()

        for x in self.eval_train_loader:
            yield x

    def get_dev_batch(self):
        '''
        Yield dev batches

        :return:
        '''
        if self.dev_loader is None:
            self._init_dev()

        for x in self.dev_loader:
            yield x


    def get_test_batch(self):
        '''
        Yield test batches

        :return:
        '''
        if self.test_loader is None:
            self._init_test()

        for x in self.test_loader:
            yield x

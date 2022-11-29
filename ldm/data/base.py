import math
from abc import abstractmethod

import torch
from torch.utils.data import Dataset, ConcatDataset, ChainDataset, IterableDataset
import os
import numpy as np
import cv2

class Txt2ImgIterableBaseDataset(IterableDataset):
    '''
    Define an interface to make the IterableDatasets for text2img data chainable
    '''
    def __init__(self, file_path: str, rank, world_size):
        super().__init__()
        self.file_path = file_path
        self.folder_list = []
        self.file_list = []
        self.txt_list = []
        self.info = self._get_file_info(file_path)
        self.start = self.info['start']
        self.end = self.info['end']
        self.rank = rank

        self.world_size = world_size
        # self.per_worker = int(math.floor((self.end - self.start) / float(self.world_size)))
        # self.iter_start = self.start + self.rank * self.per_worker
        # self.iter_end = min(self.iter_start + self.per_worker, self.end)
        # self.num_records = self.iter_end - self.iter_start
        # self.valid_ids = [i for i in range(self.iter_end)]
        self.num_records = self.end - self.start
        self.valid_ids = [i for i in range(self.end)]

        print(f'{self.__class__.__name__} dataset contains {self.__len__()} examples.')

    def __len__(self):
        # return self.iter_end - self.iter_start
        return self.end - self.start

    def __iter__(self):
        sample_iterator = self._sample_generator(self.start, self.end)
        # sample_iterator = self._sample_generator(self.iter_start, self.iter_end)
        return sample_iterator

    def _sample_generator(self, start, end):
        for idx in range(start, end):
            file_name = self.file_list[idx]
            txt_name = self.txt_list[idx]
            f_ = open(txt_name, 'r')
            txt_ = f_.read()
            f_.close()
            image = cv2.imdecode(np.fromfile(file_name, dtype=np.uint8), 1)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = torch.from_numpy(image) / 255
            yield {"caption": txt_, "image":image}


    def _get_file_info(self, file_path):
        info = \
        {
            "start": 1,
            "end": 0,
        }
        self.folder_list = [file_path + i for i in os.listdir(file_path) if '.' not in i]
        for folder in self.folder_list:
            files = [folder + '/' + i for i in os.listdir(folder) if 'jpg' in i]
            txts = [k.replace('jpg', 'txt') for k in files]
            self.file_list.extend(files)
            self.txt_list.extend(txts)
        info['end'] = len(self.file_list)
        # with open(file_path, 'r') as fin:
        #     for _ in enumerate(fin):
        #         info['end'] += 1
        # self.txt_list = [k.replace('jpg', 'txt') for k in self.file_list]
        return info
#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import time
import pickle
import os
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from util import list2tuple, tuple2list, flatten, flatten_query_and_convert_structure_to_idx
from collections import defaultdict
from constants import query_structure2idx


class TestDataset(Dataset):
    def __init__(self, queries, nentity, nrelation):
        """
        :param queries: list[(query, query_structure_idx)]
        """
        self.len = len(queries)
        self.queries = queries
        self.nentity = nentity
        self.nrelation = nrelation

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        query = self.queries[idx][0]
        query_structure_idx = self.queries[idx][1]
        negative_sample = torch.LongTensor(range(self.nentity))
        return negative_sample, flatten(query), query, query_structure_idx
    
    @staticmethod
    def collate_fn(data):
        negative_sample = torch.stack([_[0] for _ in data], dim=0)
        query = np.array([_[1] for _ in data])
        query_unflatten = [_[2] for _ in data]  # don't make it np.array. keep it as list of tuples
        query_structure_idx = np.array([_[3] for _ in data])
        return negative_sample, query, query_unflatten, query_structure_idx


class TrainDataset(Dataset):
    def __init__(self, queries, nentity, nrelation, negative_sample_size, answer):
        """
        :param queries: list[(query, query_structure_idx)]
        """
        self.len = len(queries)
        self.queries = queries
        self.nentity = nentity
        self.nrelation = nrelation
        self.negative_sample_size = negative_sample_size
        self.count = self.count_frequency(queries, answer)
        self.answer = answer

    def __len__(self):
        return self.len
    
    def __getitem__(self, idx):
        query = self.queries[idx][0]
        query_structure_idx = self.queries[idx][1]
        tail = np.random.choice(list(self.answer[query]))
        subsampling_weight = self.count[query] 
        subsampling_weight = torch.sqrt(1 / torch.Tensor([subsampling_weight]))

        # negative_sample_list = []
        # negative_sample_size = 0
        # while negative_sample_size < self.negative_sample_size:
        #     negative_sample = np.random.randint(self.nentity, size=self.negative_sample_size*2)
        #     mask = np.in1d(
        #         negative_sample,
        #         self.answer[query],
        #         assume_unique=True,
        #         invert=True
        #     )
        #     negative_sample = negative_sample[mask]
        #     negative_sample_list.append(negative_sample)
        #     negative_sample_size += negative_sample.size
        # negative_sample = np.concatenate(negative_sample_list)[:self.negative_sample_size]
        # negative_sample = torch.from_numpy(negative_sample).type(torch.LongTensor)

        # the above sampling is too slow but not significant performance gain
        # Shirley
        negative_sample = torch.randint(self.nentity, (self.negative_sample_size,))
        positive_sample = torch.LongTensor([tail])
        return positive_sample, negative_sample, subsampling_weight, flatten(query), query_structure_idx
    
    @staticmethod
    def collate_fn(data):
        positive_sample = torch.cat([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        subsample_weight = torch.cat([_[2] for _ in data], dim=0)
        query = np.array([_[3] for _ in data])  # can't convert to tensor due to the varying length
        query_structure_idx = np.array([_[4] for _ in data])
        return positive_sample, negative_sample, subsample_weight, query, query_structure_idx
    
    @staticmethod
    def count_frequency(queries, answer, start=4):
        count = {}
        for query, qtype in queries:
            count[query] = start + len(answer[query])
        return count


class SingledirectionalOneShotIterator(object):
    def __init__(self, dataloader):
        self.iterator = self.one_shot_iterator(dataloader)
        self.step = 0
        
    def __next__(self):
        self.step += 1
        data = next(self.iterator)
        return data
    
    @staticmethod
    def one_shot_iterator(dataloader):
        while True:
            for data in dataloader:
                yield data


def filter_by_tasks(queries, name_query_dict, tasks, evaluate_union):
    """
    remove queries not in tasks.
    """
    all_query_names = set(name_query_dict.keys())
    for name in all_query_names:
        if 'u' in name:
            name, name_evaluate_union = name.split('-')
        else:
            name_evaluate_union = evaluate_union
        if name not in tasks or name_evaluate_union != evaluate_union:
            query_structure = name_query_dict[name if 'u' not in name else '-'.join([name, name_evaluate_union])]
            if query_structure in queries:
                del queries[query_structure]
    return queries


def load_data_from_pickle(args, query_name_dict, tasks):
    """
    Load queries/answers and remove queries not in tasks.
    To save time, only load corresponding queries and answers
        when flags like args.do_train, args.do_valid, args.do_test are True.
        Otherwise return None.
    :param query_name_dict: all possible query types across models. type dict{query_str:query_name}
    :param tasks: task to use
    """
    # To save time, only load corresponding queries and answers
    #     when flags like args.do_train, args.do_valid, args.do_test are True.
    #     Otherwise return None.
    train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers, \
        test_queries, test_hard_answers, test_easy_answers = None, None, None, None, None, None, None, None

    print("loading data")
    time0 = time.time()

    name_query_dict = {value: key for key, value in query_name_dict.items()}  # {'1p': ('e',('r',)), ...}

    if args.do_train:
        train_queries = pickle.load(open(os.path.join(args.data_path, "train-queries.pkl"), 'rb'))
        train_answers = pickle.load(open(os.path.join(args.data_path, "train-answers.pkl"), 'rb'))
        train_queries = filter_by_tasks(train_queries, name_query_dict, tasks, args.evaluate_union)
    if args.do_valid:
        valid_queries = pickle.load(open(os.path.join(args.data_path, "valid-queries.pkl"), 'rb'))
        valid_hard_answers = pickle.load(open(os.path.join(args.data_path, "valid-hard-answers.pkl"), 'rb'))
        valid_easy_answers = pickle.load(open(os.path.join(args.data_path, "valid-easy-answers.pkl"), 'rb'))
        valid_queries = filter_by_tasks(valid_queries, name_query_dict, tasks, args.evaluate_union)
    if args.do_test:
        test_queries = pickle.load(open(os.path.join(args.data_path, "test-queries.pkl"), 'rb'))
        test_hard_answers = pickle.load(open(os.path.join(args.data_path, "test-hard-answers.pkl"), 'rb'))
        test_easy_answers = pickle.load(open(os.path.join(args.data_path, "test-easy-answers.pkl"), 'rb'))
        test_queries = filter_by_tasks(test_queries, name_query_dict, tasks, args.evaluate_union)

    print(f'Loading data uses time: {time.time()-time0}')

    return train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers, \
           test_queries, test_hard_answers, test_easy_answers


def load_data(args, query_name_dict, tasks):
    # only generate it when necessary
    train_path_iterator, train_other_iterator, valid_dataloader, test_dataloader = None, None, None, None

    train_queries, train_answers, valid_queries, valid_hard_answers, valid_easy_answers, \
        test_queries, test_hard_answers, test_easy_answers = load_data_from_pickle(args, query_name_dict, tasks)

    if args.do_train:
        print('Training query info:')
        for query_structure in train_queries:
            print(query_name_dict[query_structure] + ": " + str(len(train_queries[query_structure])))
        train_path_queries = defaultdict(set)
        train_other_queries = defaultdict(set)
        path_list = ['1p', '2p', '3p']
        for query_structure in train_queries:
            if query_name_dict[query_structure] in path_list:
                train_path_queries[query_structure] = train_queries[query_structure]
            else:
                train_other_queries[query_structure] = train_queries[query_structure]
        train_path_queries = flatten_query_and_convert_structure_to_idx(train_path_queries, query_structure2idx)
        train_path_iterator = SingledirectionalOneShotIterator(DataLoader(
            TrainDataset(train_path_queries, args.nentity, args.nrelation, args.negative_sample_size, train_answers),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.cpu_num,
            collate_fn=TrainDataset.collate_fn
        ))
        if len(train_other_queries) > 0:
            train_other_queries = flatten_query_and_convert_structure_to_idx(train_other_queries, query_structure2idx)
            train_other_iterator = SingledirectionalOneShotIterator(DataLoader(
                TrainDataset(train_other_queries, args.nentity, args.nrelation, args.negative_sample_size, train_answers),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.cpu_num,
                collate_fn=TrainDataset.collate_fn
            ))
        else:
            train_other_iterator = None

    if args.do_valid:
        print('Validation query info:')
        for query_structure in valid_queries:
            print(query_name_dict[query_structure] + ": " + str(len(valid_queries[query_structure])))
        valid_queries = flatten_query_and_convert_structure_to_idx(valid_queries, query_structure2idx)
        valid_dataloader = DataLoader(
            TestDataset(
                valid_queries,
                args.nentity,
                args.nrelation,
            ),
            batch_size=args.test_batch_size,
            num_workers=args.cpu_num,
            collate_fn=TestDataset.collate_fn
        )

    if args.do_test:
        print('Test query info:')
        for query_structure in test_queries:
            print(query_name_dict[query_structure] + ": " + str(len(test_queries[query_structure])))
        test_queries = flatten_query_and_convert_structure_to_idx(test_queries, query_structure2idx)
        test_dataloader = DataLoader(
            TestDataset(
                test_queries,
                args.nentity,
                args.nrelation,
            ),
            batch_size=args.test_batch_size,
            num_workers=args.cpu_num,
            collate_fn=TestDataset.collate_fn
        )

    return train_path_iterator, train_other_iterator, valid_dataloader, test_dataloader, \
           valid_hard_answers, valid_easy_answers, test_hard_answers, test_easy_answers  # keep answers for evaluation

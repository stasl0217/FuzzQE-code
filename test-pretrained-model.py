import torch
from os.path import join
from fuzzyreasoning import KGFuzzyReasoning
from dataloader import load_data_from_pickle, load_data
from constants import query_name_dict, query_structure_list, query_structure2idx
from collections import defaultdict
from main import parse_args
from util import evaluate, read_num_entity_relation_from_file, eval_tuple, wandb_initialize
import collections
import copy
from constants import query_structure2idx, query_name_dict, query_structure_list
from dataloader import TestDataset
from torch.utils.data import DataLoader
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from util import log_metrics
import torch.nn.functional as F
import os
from investigation_helper import set_GPU_id
from investigation_helper import wandb_log_metrics, prepare_new_attributes
import sys




def test_step(model, easy_answers, hard_answers, args, test_dataloader, query_name_dict, verbose=False):
    model.eval()

    device = model.device

    step = 0
    total_steps = len(test_dataloader)
    logs = collections.defaultdict(list)

    with torch.no_grad():
        for negative_sample, queries, queries_unflatten, query_structure_idxs in test_dataloader:
            # example: query_structures: [('e', ('r',))].  queries: [[1804,4]]. queries_unflatten: [(1804, (4,)]
            if args.cuda:
                negative_sample = negative_sample.to(device)

            # nn.DataParallel helper
            batch_size = len(negative_sample)
            slice_idxs = torch.arange(0, batch_size).view((batch_size, 1))

            _, negative_logit, _, idxs = model(
                None,
                negative_sample,
                None,
                queries,  # np.array([queries]), won't be split when using multiple GPUs
                query_structure_idxs,
                slice_idxs,  # to help track batch_queries and query_structures when using multiple GPUs
                inference=True
            )

            idxs_np = idxs.detach().cpu().numpy()
            # if not converted to numpy, idxs_np will be considered scalar when test_batch_size=1
            # queries_unflatten = queries_unflatten[idxs_np]
            query_structure_idxs = query_structure_idxs[idxs_np]
            queries_unflatten = [queries_unflatten[i] for i in idxs]

            #
            # query_structures = [query_structures[i] for i in idxs]
            argsort = torch.argsort(negative_logit, dim=1, descending=True)
            ranking = argsort.clone().to(torch.float)

            # rank all entities
            # If it is the same shape with test_batch_size, reuse batch_entity_range without creating a new one
            if len(argsort) == args.test_batch_size:
                # ranking = ranking.scatter_(1, argsort, model.module.batch_entity_range)  # achieve the ranking of all entities
                ranking = ranking.scatter_(1, argsort, model.batch_entity_range)  # achieve the ranking of all entities
            else:  # otherwise, create a new torch Tensor for batch_entity_range
                ranking = ranking.scatter_(
                    1,
                    argsort,
                    torch.arange(model.nentity).to(torch.float).repeat(argsort.shape[0], 1).to(device)
                    # torch.arange(model.module.nentity).to(torch.float).repeat(argsort.shape[0], 1).to(device)
                )

            for idx, (i, query, query_structure_idx) in enumerate(zip(argsort[:, 0], queries_unflatten, query_structure_idxs)):
                # convert query from np.ndarray to nested tuple
                query_key = tuple(query)
                query_structure = query_structure_list[query_structure_idx]

                hard_answer = hard_answers[query_key]
                easy_answer = easy_answers[query_key]
                num_hard = len(hard_answer)
                num_easy = len(easy_answer)
                assert len(hard_answer.intersection(easy_answer)) == 0
                cur_ranking = ranking[idx, list(easy_answer) + list(hard_answer)]
                cur_ranking, indices = torch.sort(cur_ranking)
                
                masks = indices >= num_easy
                if args.cuda:
                    answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
                else:
                    answer_list = torch.arange(num_hard + num_easy).to(torch.float)
                cur_ranking = cur_ranking - answer_list + 1 # filtered setting
                cur_ranking = cur_ranking[masks]  # only take indices that belong to the hard answers
                
                if verbose:
                    print(answer_list)
                    print('ranking', cur_ranking)

                mrr = torch.mean(1./cur_ranking).item()
                h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
                h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
                h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()

                logs[query_structure].append({
                    'MRR': mrr,
                    'HITS1': h1,
                    'HITS3': h3,
                    'HITS10': h10,
                    'num_hard_answer': num_hard,
                })

            if step % args.test_log_steps == 0:
                print('Evaluating the model... (%d/%d)' % (step, total_steps))

            step += 1

    metrics = collections.defaultdict(lambda: collections.defaultdict(int))
    for query_structure in logs:
        for metric in logs[query_structure][0].keys():
            if metric in ['num_hard_answer']:
                continue
            metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]])/len(logs[query_structure])
        metrics[query_structure]['num_queries'] = len(logs[query_structure])
        
    return metrics


model_dir = './trained_models'
set_GPU_id(gpu_id='7')
run = sys.argv[1]

logic = 'product'
model_path = join(model_dir, f'{run}.pt')
model = torch.load(model_path)


arg_str = f'--do_test --cuda --data_path data/NELL-betae --test_batch_size 2 --logic {logic}'

arg_str = arg_str.split()
args = parse_args(arg_str)
args.nentity = model.nentity
model.conjunction_net.use_attention = args.use_attention
prepare_new_attributes(model)



train_path_iterator, train_other_iterator, valid_dataloader, test_dataloader,\
    valid_hard_answers, valid_easy_answers, \
    test_hard_answers, test_easy_answers = load_data(args, query_name_dict, args.tasks)


data_dir = args.data_path
rel_id2str = pickle.load(open(join(data_dir, 'id2rel.pkl'), 'rb'))



metrics = test_step(model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict)
print(metrics)

a = wandb_log_metrics(metrics, args)

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
import os
from main import parse_args
import pandas as pd
from os.path import join
import pickle
import torch
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
from util import list2tuple, tuple2list, flatten, flatten_query_and_convert_structure_to_idx
from dataloader import SingledirectionalOneShotIterator, TrainDataset

def relPCA(model):
    rel_base0 = model.projection_net.rel_base.detach()
    n_base, d1, d2 = rel_base0.size()
    rel_base = rel_base0.view(n_base, d1*d2).cpu().numpy()
    scaler = MinMaxScaler()
    data_rescaled = scaler.fit_transform(rel_base.transpose())


    pca = PCA(n_components = 0.95)
    pca.fit(data_rescaled)
    reduced = pca.transform(data_rescaled)
    print(reduced.shape)


def set_GPU_id(gpu_id):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id  # specify which GPU(s) to be used


def print_one_sigmoid(regularizer):
    print(regularizer.weight)
    print(regularizer.bias)


def print_all_sigmoid(model):
    print_one_sigmoid(model.entity_regularizer)
    print_one_sigmoid(model.projection_net.regularizer)
    print_one_sigmoid(model.negation_net.regularizer)
    print_one_sigmoid(model.disjunction_net.regularizer)
    print_one_sigmoid(model.conjunction_net.regularizer)


def get_args(arg_str, model):
    arg_str = arg_str.split()
    args = parse_args(arg_str)
    args.nentity = model.nentity
    model.conjunction_net.use_attention = args.use_attention
    print(args)
    return args

def get_answers_from_train(data_dir):
    """Get answers from train.txt"""
    train_df = pd.read_csv(open(join(data_dir, 'train.txt')), sep='\t')
    hr2t = {}
    for index,row in train_df.iterrows():
        h,r,t = row[0], row[1], row[2]
        if (h,r) not in hr2t:
            hr2t[(h,r)] = set()
        hr2t[(h,r)].add(t)
    return hr2t

def get_answers_from_full(data_dir):
    """Get answers from train.txt"""
    train_df = pd.read_csv(open(join(data_dir, 'train.txt')), sep='\t')
    val_df = pd.read_csv(open(join(data_dir, 'valid.txt')), sep='\t')
    test_df = pd.read_csv(open(join(data_dir, 'test.txt')), sep='\t')
    hr2t = {}
    for index,row in train_df.iterrows():
        h,r,t = row[0], row[1], row[2]
        if (h,r) not in hr2t:
            hr2t[(h,r)] = set()
        hr2t[(h,r)].add(t)
    for index,row in val_df.iterrows():
        h,r,t = row[0], row[1], row[2]
        if (h,r) not in hr2t:
            hr2t[(h,r)] = set()
        hr2t[(h,r)].add(t)
    for index,row in test_df.iterrows():
        h,r,t = row[0], row[1], row[2]
        if (h,r) not in hr2t:
            hr2t[(h,r)] = set()
        hr2t[(h,r)].add(t)
    return hr2t_full

def get_ent_rel_labels(data_dir):
    rel_id2str = pickle.load(open(join(data_dir, 'id2rel.pkl'), 'rb'))

    if 'FB15k-237' in data_dir:
        # use the entity label pulled from Wikidata
        # since the original entity str in FB15k is FreeBase ID and not readable
        entity_df = pd.read_csv(open(join(data_dir, 'entity_label.tsv')), sep='\t')
        ent_id2str = pd.Series(entity_df['wiki'].values, index=entity_df['id']).to_dict()
        print(entity_df.head(5))
    else:  # e.g. NELL
        ent_id2str = pickle.load(open(join(data_dir, 'id2ent.pkl'), 'rb'))


    return ent_id2str, rel_id2str



def get_projected_t_vec(model, h, rs):
    """rs is a list of relations"""

    h_vec = model.entity_regularizer(
        torch.index_select(
            model.entity_embedding,
            dim=0,
            index=torch.LongTensor([h]).to(model.device)
        )
    )
    rid =torch.LongTensor(rs)
    
    t_vec = model.projection_net(h_vec, rid)
    return t_vec

def get_negated_vec(model, v):
    neg_v = model.negation_net(v)
    return neg_v


def get_conjunction_vec(model, v1, v2):
    stack = torch.stack([v1, v2])
    conj = model.conjunction_net(stack)
    return conj


def get_disjunction_vec(v1, v2):
    stack = torch.stack([v1, v2])
    union = model.disjunction_net(stack)
    return union

def plot_vec(v, firstk=20):
    """
    vec.shape: [1, dim]
    """
    plt.figure()
    vec = v[0,:firstk].detach().cpu().numpy()
    ind = np.arange(len(vec))
    plt.bar(ind, vec)


def score_distribution_by_vec(model, vec):
    model.eval()

    device = model.device

    step = 0
    logs = collections.defaultdict(list)

    with torch.no_grad():
        negative_samples = torch.arange(0, model.nentity).to(model.device)  # all entities
        negative_embedding = model.entity_regularizer(
                    torch.index_select(
                        model.entity_embedding,
                        dim=0,
                        index=negative_samples.view(-1)
                    ).view(
                        1,
                        model.nentity,
                        -1
                    )
        )
        
        scores = model.cal_logit_fuzzy(negative_embedding, vec)
        return scores
    
def plot_vec_wide(v, firstk=20, height=5, width=20):
    """
    vec.shape: [1, dim]
    """
    plt.figure(figsize=(width, height))
    vec = v[0,:firstk].detach().cpu().numpy()
    ind = np.arange(len(vec))
    plt.bar(ind, vec)
    
def plot_vec_wide_with_color(v, highlight, flip=False, start=0, length=20, height=5, width=20):
    """
    vec.shape: [1, dim]
    highlight: set of index set(3, 8, ...), otherwise blue
    """

    colors = ['b' for i in range(v.shape[1])]
    for i in highlight:
        colors[i] = 'r'
    if flip:
        colors = ['r' for i in range(v.shape[1])]
        for i in highlight:
            colors[i] = 'b'
    
    plt.figure(figsize=(width, height))
    vec = v[0,start:start+length].detach().cpu().numpy()
    ind = np.arange(len(vec))
    plt.bar(ind, vec, color=colors[start:start+length])


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


def wandb_log_metrics(metrics, args):
    run = wandb_initialize(vars(args))

    average_metrics = defaultdict(float)
    average_pos_metrics = defaultdict(float)
    average_neg_metrics = defaultdict(float)
    all_metrics = defaultdict(float)

    num_query_structures = 0
    num_pos_query_structures = 0
    num_neg_query_structures = 0

    num_queries = 0
    mode="Test"
    step=0
    for query_structure in metrics:
        log_metrics(mode + " " + query_name_dict[query_structure], step, metrics[query_structure])
        for metric in metrics[query_structure]:
            query_name = query_name_dict[query_structure]  # e.g. 1p
            all_metrics["_".join([query_name, metric])] = metrics[query_structure][metric]
            if metric != 'num_queries':
                average_metrics[metric] += metrics[query_structure][metric]
                if 'n' in query_name:
                    average_neg_metrics[metric] += metrics[query_structure][metric]
                else:
                    average_pos_metrics[metric] += metrics[query_structure][metric]
        num_queries += metrics[query_structure]['num_queries']
        num_query_structures += 1
        if 'n' in query_name:
            num_neg_query_structures += 1
        else:
            num_pos_query_structures += 1

    for metric in average_pos_metrics:
        average_pos_metrics[metric] /= num_pos_query_structures
        # writer.add_scalar("_".join([mode, 'average', metric]), average_metrics[metric], step)
        all_metrics["_".join(["average_pos", metric])] = average_pos_metrics[metric]

    for metric in average_neg_metrics:
        average_neg_metrics[metric] /= num_neg_query_structures
        # writer.add_scalar("_".join([mode, 'average', metric]), average_metrics[metric], step)
        all_metrics["_".join(["average_neg", metric])] = average_neg_metrics[metric]

    for metric in average_metrics:
        average_metrics[metric] /= num_query_structures
        all_metrics["_".join(["average", metric])] = average_metrics[metric]
    log_metrics('%s average' % mode, step, average_metrics)
    log_metrics('%s average_pos' % mode, step, average_pos_metrics)
    log_metrics('%s average_neg' % mode, step, average_neg_metrics)
    
    log_metrics('%s average' % mode, step, all_metrics)

def get_query_structure_by_type_name(type_name):
    """1p -> (e, (r,))"""
    for qstructure, qname in query_name_dict.items():  # query_name_dict: imported from constants
        if qname == type_name:
            return qstructure


def get_type_idx_by_name(type_name):
    """type_name: e.g. 1p"""
    
    for qstructure, qname in query_name_dict.items():  # query_name_dict: imported from constants
        if qname == type_name:
            type_structure = qstructure
            
    return query_structure2idx[type_structure]  # query_structure2idx: imported from constants

def get_queries_by_type_id(all_queries, type_idx):
    """all queries: list[tuple(nested_query, type_idx)]"""
    keep = [q for q in all_queries if q[1] == type_idx]
    return keep


def get_queries_by_type_name(all_queries, type_name):
    """all queries: list[tuple(nested_query, type_idx)]"""
    type_idx = get_type_idx_by_name(type_name)
    keep = get_queries_by_type_id(all_queries, type_idx)
    return keep



def get_sub_dataloader(full_test_dataset, type_name, args):
    full_queries = full_test_dataset.queries
    sub_queries = get_queries_by_type_name(full_queries, type_name)
    print('number of queries:', len(sub_queries))
    sub_test_dataset =  TestDataset(
        sub_queries,
        full_test_dataset.nentity,
        full_test_dataset.nrelation
    )
    sub_test_dataloader = sub_test_dataloader = DataLoader(
        sub_test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.cpu_num,
        collate_fn=TestDataset.collate_fn
    )
    return sub_queries, sub_test_dataloader

def test_on_a_type(model, full_test_dataset, args, test_easy_answers, test_hard_answers, type_name):
    sub_queries, sub_test_dataloader = get_sub_dataloader(full_test_dataset, type_name, args)
    metrics = test_step(model, test_easy_answers, test_hard_answers, args, sub_test_dataloader, query_name_dict)
    return sub_queries, metrics


def get_dataloader_for_one_query(full_test_dataset, args, q):
    oneq = [q]
    sub_test_dataset =  TestDataset(
        oneq,
        full_test_dataset.nentity,
        full_test_dataset.nrelation
    )
    sub_test_dataloader =  DataLoader(
        sub_test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.cpu_num,
        collate_fn=TestDataset.collate_fn
    )
    return sub_test_dataloader

def test_on_one_query(full_test_dataset, args, q):
    oneq = [q]
    sub_test_dataset =  TestDataset(
        oneq,
        full_test_dataset.nentity,
        full_test_dataset.nrelation
    )
    sub_test_dataloader =  DataLoader(
        sub_test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.cpu_num,
        collate_fn=TestDataset.collate_fn
    )
    metrics = test_step(model, test_easy_answers, test_hard_answers, args, sub_test_dataloader, query_name_dict, verbose=True)
    return metrics


def prepare_new_attributes(model, margin_type=None):
    """Prepare value for the new attributes in the model"""

    def set_dual_for_regularizer(regularizer):
        if not hasattr(regularizer, 'dual'):
            regularizer.dual = False

    def set_dual_as_false(model):
        model.entity_regularizer.dual = False
        model.projection_net.dual = False
        model.projection_net.regularizer.dual = False

    set_dual_as_false(model)
    model.counter_for_neg = False

    if not hasattr(model, 'margin_type'):
        if margin_type is not None:
            model.margin_type = margin_type
        else:
            print('Model margin type not recorded. Set as "logsigmoid" by default. Please double check if it\'s consistent with the model.')
            model.margin_type = 'logsigmoid'

    if not hasattr(model, 'no_anchor_reg'):
        model.no_anchor_reg = False

    if not hasattr(model, 'simplE'):
        model.simplE = False

    set_dual_for_regularizer(model.entity_regularizer)
    set_dual_for_regularizer(model.projection_net.regularizer)
    set_dual_for_regularizer(model.conjunction_net.regularizer)
    set_dual_for_regularizer(model.disjunction_net.regularizer)
            


################## Translate ##############
def translate(ent_id2str, rel_id2str, x, structure):
    def translate_one(x, structure):
        """Get into nested list and translate one by one"""
        if isinstance(x, int):
            if x >= 0 and structure in ('e', 'r'):
                if structure == 'e':
                    if x in ent_id2str:
                        return ent_id2str[x]
                    else:
                        return 'missed_x'
                elif structure == 'r':
                    return rel_id2str[x]
            else:
                return structure
        else:
            return type(x)(map(translate_one, x, structure))
    
    return translate_one(x, structure)
        
    
def translate_queries(ent_id2str, rel_id2str, qs):
    results = []
    for q in qs:
        query, structure_idx = q[0], q[1]
        res = translate(ent_id2str, rel_id2str, query, query_structure_list[structure_idx])
        results.append(res)
    return results

def translate_answers(ent_id2str, ents, with_id=False):
    labels = []
    for e in ents:
        e0 = int(e)  # incase of torch tensor
        if e0 in ent_id2str:
            if not with_id:
                labels.append(ent_id2str[e0])
            else:
                labels.append((ent_id2str[e0], e0))
        else:
            labels.append(e0)
    return labels

def translate_queries_with_hard_answers(ent_id2str, rel_id2str, test_hard_answers, qs):
    query_results = []
    answer_results = []
    results = []
    for q in qs:
        query, structure_idx = q[0], q[1]
        q_res = translate(ent_id2str, rel_id2str, query, query_structure_list[structure_idx])
        
        answers_in_id = test_hard_answers[query]
        a_res = translate_answers(ent_id2str, answers_in_id)
        
        query_results.append(q_res)
        answer_results.append(a_res)
        results.append((q_res, a_res))
    return results




############# Find top ranked answers
def rank_all_entities_for_one_query(model, full_test_dataset, args,  q):
    """
    q: tuple(nested_query, query_structure_idx), e.g. (((967, (35,)), (8734, (351, -2))), 7)
    """
    # easy_answers, hard_answers, args, test_dataloader, query_name_dict
    model.eval()
    device = model.device
    test_dataloader = get_dataloader_for_one_query(full_test_dataset, args, q)
    
    logs = collections.defaultdict(list)
    with torch.no_grad():
        for negative_sample, queries, queries_unflatten, query_structure_idxs in test_dataloader:
            # example: query_structures: [('e', ('r',))].  queries: [[1804,4]]. queries_unflatten: [(1804, (4,)]
            negative_sample = negative_sample.to(device)
            batch_size = len(negative_sample)
            slice_idxs = torch.arange(0, batch_size).view((batch_size, 1))
            _, negative_logit, _, idxs = model(
                None,
                negative_sample,
                None,
                queries,  # np.array([queries]), won't be split when using multiple GPUs
                query_structure_idxs,
                slice_idxs  # to help track batch_queries and query_structures when using multiple GPUs
            )
            idxs_np = idxs.detach().cpu().numpy()
            query_structure_idxs = query_structure_idxs[idxs_np]
            queries_unflatten = [queries_unflatten[i] for i in idxs]

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
                
            return ranking
        

        
def find_top_ranked_entities(model, full_test_dataset, args, q, topk=10):
    ranking = rank_all_entities_for_one_query(model, full_test_dataset, args, q)
    top_entities = (ranking <= topk).nonzero()  # not sorted among top k
    return ranking, top_entities

def translate_top_k(ent_id2str, topk, ranking, easy_answer, hard_answer):
    res = translate_answers(ent_id2str, topk[:, 1], with_id=True)
    df = pd.DataFrame(res, columns=['entity', 'id'])
    df['rank'] = df['id'].apply(lambda x: int(ranking[0,x]))
    df['easy_answer'] = df['id'].apply(lambda x: x in easy_answer)
    df['hard_answer'] = df['id'].apply(lambda x: x in hard_answer)
    return df.sort_values(by='rank')


def translate_entity_and_ranking(ent_id2str, eids, ranking, with_id=False):
    res = []
    for e, r in zip(eids, ranking):
        e0 = int(e)  # incase of torch tensor
        if e0 in ent_id2str:
            if not with_id:
                res.append((ent_id2str[e0], int(r)))
            else:
                res.append((ent_id2str[e0], e0, int(r)))
        else:
            res.append(e0)
    return res

def get_metrics(cur_ranking):
    mrr = torch.mean(1./cur_ranking).item()
    h1 = torch.mean((cur_ranking <= 1).to(torch.float)).item()
    h3 = torch.mean((cur_ranking <= 3).to(torch.float)).item()
    h10 = torch.mean((cur_ranking <= 10).to(torch.float)).item()
    
    return mrr, h1, h3, h10



################ Find ranking for hard answers
def find_ranking_for_hard_answers(test_easy_answers, test_hard_answers, args, q, ranking):
    """
    q: tuple(nested_query, query_structure_idx), e.g. (((967, (35,)), (8734, (351, -2))), 7)
    """
    query_key = q[0]
    easy_answer = test_easy_answers[query_key]
    hard_answer = test_hard_answers[query_key]
    num_easy = len(easy_answer)
    num_hard = len(hard_answer)
    
    answers = torch.Tensor(list(easy_answer) + list(hard_answer)).type(torch.LongTensor)
    cur_ranking = ranking[0, answers]  # ranking of easy and hard answers
    
    unfiltered_ranking = cur_ranking.clone()
    
    cur_ranking, indices = torch.sort(cur_ranking)
    
    
    masks = (indices >= num_easy)
    if args.cuda:
        answer_list = torch.arange(num_hard + num_easy).to(torch.float).cuda()
    else:
        answer_list = torch.arange(num_hard + num_easy).to(torch.float)
    cur_ranking = cur_ranking - answer_list + 1 # filtered setting
    cur_ranking = cur_ranking[masks]  # only take indices that belong to the hard answers
        
    return easy_answer, hard_answer, answers, indices, cur_ranking, unfiltered_ranking

def answer_ranking_df(ent_id2str, entities, ranking, easy_answer, hard_answer):
    res = translate_entity_and_ranking(ent_id2str, entities, ranking, with_id=True)
    df = pd.DataFrame(res, columns=['entity', 'id', 'ranking'])
    df['easy_answer'] = df['id'].apply(lambda x: x in easy_answer)
    df['hard_answer'] = df['id'].apply(lambda x: x in hard_answer)
    return df



####################### Check training loss #################################
def train_step_full(model, train_iterator, args, step):
    """
    Adapted for multiple GPUs
    """
    # device = model.module.device
    device = model.device
    model.eval()
    
    with torch.no_grad():

        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structure_idxs = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.to(device)
            negative_sample = negative_sample.to(device)
            subsampling_weight = subsampling_weight.to(device)
            # no need to move query_structure_idxs to GPU

        # nn.DataParallel helper
        batch_size = len(positive_sample)
        slice_idxs = torch.arange(0, batch_size).view((batch_size, 1))

        positive_score, negative_score, subsampling_weight, _ = model(
            positive_sample,
            negative_sample,
            subsampling_weight,
            batch_queries,  # np.array([queries]), won't be split when using multiple GPUs
            query_structure_idxs,  # torch.LongTensor
            slice_idxs  # to help track batch_queries and query_structures when using multiple GPUs
        )
        if args.margin_type == 'logsigmoid':
            # the loss of BetaE and RotatE
            positive_dist = 1-positive_score
            negative_dist = 1-negative_score
            positive_unweighted_loss = -F.logsigmoid((model.gamma - positive_dist)*model.gamma_coff).squeeze(dim=1)
            negative_unweighted_loss = -F.logsigmoid((negative_dist - model.gamma)*model.gamma_coff).mean(dim=1)
            positive_sample_loss = (subsampling_weight * positive_unweighted_loss).sum()
            negative_sample_loss = (subsampling_weight * negative_unweighted_loss).sum()
            positive_sample_loss /= subsampling_weight.sum()
            negative_sample_loss /= subsampling_weight.sum()
            loss = (positive_sample_loss + negative_sample_loss) / 2
            log = {
                'positive_sample_loss': positive_sample_loss.item(),
                'negative_sample_loss': negative_sample_loss.item(),
                'loss': loss.item(),
            }
        elif args.margin_type == 'logsigmoid_avg':
            # use with cos_digits
            positive_dist = 1-positive_score
            negative_dist = 1-negative_score
            positive_unweighted_loss = -torch.mean(F.logsigmoid((model.gamma - positive_dist)*model.gamma_coff), dim=-1).squeeze(dim=1)
            negative_unweighted_loss = -torch.mean(F.logsigmoid((negative_dist - model.gamma)*model.gamma_coff), dim=-1).mean(dim=1)
            positive_sample_loss = (subsampling_weight * positive_unweighted_loss).sum()
            negative_sample_loss = (subsampling_weight * negative_unweighted_loss).sum()
            positive_sample_loss /= subsampling_weight.sum()
            negative_sample_loss /= subsampling_weight.sum()
            loss = (positive_sample_loss + negative_sample_loss) / 2
            log = {
                'positive_sample_loss': positive_sample_loss.item(),
                'negative_sample_loss': negative_sample_loss.item(),
                'loss': loss.item(),
            }
        elif args.margin_type == 'logsigmoid_bpr':
            # gamma as margin
            diff = -F.logsigmoid(model.gamma_coff*(positive_score - negative_score))
            # diff = torch.mean(-F.logsigmoid(model.gamma_coff*(positive_score - negative_score)), dim=-1)
            unweighted_sample_loss = torch.mean(diff, dim=-1)
            loss = (subsampling_weight * unweighted_sample_loss).sum()
            loss /= subsampling_weight.sum()
            log = {
                'loss': loss.item(),
            }
        elif args.margin_type == 'logsigmoid_bpr_digits':
            # positive_score: shape [batch_size, 1, dim]  (not aggregated yet)
            # negative_score: shape [batch_size, neg_per_pos, dim]
            diff = -F.logsigmoid(model.gamma_coff*(torch.mean(positive_score - negative_score, dim=-1)))
            # diff = torch.mean(-F.logsigmoid(model.gamma_coff*(positive_score - negative_score)), dim=-1)
            unweighted_sample_loss = torch.mean(diff, dim=-1)
            loss = (subsampling_weight * unweighted_sample_loss).sum()
            loss /= subsampling_weight.sum()
            log = {
                'loss': loss.item(),
            }
        elif args.margin_type == 'softmax':
            # positive_score shape [batch_size, 1]
            #TODO: multi positives, same negative sample for the batch
            criterion = nn.CrossEntropyLoss(reduction='none')  # keep loss for each sample
            scores = torch.cat([positive_score, negative_score], dim=1)*model.softmax_weight  # [batch_size, 1+negative_sample_size]
            target = torch.zeros((positive_score.shape[0],), dtype=torch.long).to(device)
            loss = (criterion(scores, target) * subsampling_weight).sum()
            loss /= subsampling_weight.sum()
            log = {'loss': loss.item()}
        elif args.margin_type == 'bpr':
            # gamma as margin
            diff = torch.relu(model.gamma + negative_score -positive_score)  # relu or softplus
            unweighted_sample_loss = torch.mean(diff, dim=-1)
            loss = (subsampling_weight * unweighted_sample_loss).sum()
            loss /= subsampling_weight.sum()
            log = {
                'loss': loss.item(),
            }

    return positive_score, negative_score, log

def train_step(model, train_iterator, args, step):
    """
    Adapted for multiple GPUs
    """
    # device = model.module.device
    device = model.device
    model.eval()
    
    with torch.no_grad():

        positive_sample, negative_sample, subsampling_weight, batch_queries, query_structure_idxs = next(train_iterator)
 
        if args.cuda:
            positive_sample = positive_sample.to(device)
            negative_sample = negative_sample.to(device)
            subsampling_weight = subsampling_weight.to(device)
            # no need to move query_structure_idxs to GPU

        # nn.DataParallel helper
        batch_size = len(positive_sample)
        slice_idxs = torch.arange(0, batch_size).view((batch_size, 1))

        positive_score, negative_score, subsampling_weight, _ = model(
            positive_sample,
            negative_sample,
            subsampling_weight,
            batch_queries,  # np.array([queries]), won't be split when using multiple GPUs
            query_structure_idxs,  # torch.LongTensor
            slice_idxs  # to help track batch_queries and query_structures when using multiple GPUs
        )
        if args.margin_type == 'logsigmoid':
            # the loss of BetaE and RotatE
            positive_dist = 1-positive_score
            negative_dist = 1-negative_score
            positive_unweighted_loss = -F.logsigmoid((model.gamma - positive_dist)*model.gamma_coff).squeeze(dim=1)
            negative_unweighted_loss = -F.logsigmoid((negative_dist - model.gamma)*model.gamma_coff).mean(dim=1)
            positive_sample_loss = (subsampling_weight * positive_unweighted_loss).sum()
            negative_sample_loss = (subsampling_weight * negative_unweighted_loss).sum()
            positive_sample_loss /= subsampling_weight.sum()
            negative_sample_loss /= subsampling_weight.sum()
            loss = (positive_sample_loss + negative_sample_loss) / 2
            log = {
                'positive_sample_loss': positive_sample_loss.item(),
                'negative_sample_loss': negative_sample_loss.item(),
                'loss': loss.item(),
            }
    return positive_score, negative_score, log

def get_query_structure_by_type_name(type_name):
    """1p -> (e, (r,))"""
    for qstructure, qname in query_name_dict.items():  # query_name_dict: imported from constants
        if qname == type_name:
            return qstructure

def get_sub_train_iterator(full_train_queries, full_train_answers, type_name, args):
    sub_queries = defaultdict(set)
    structure = get_query_structure_by_type_name(type_name)
    sub_queries[structure] = full_train_queries[structure]  # take queries of a type
    sub_queries = flatten_query_and_convert_structure_to_idx(sub_queries, query_structure2idx)
    n_queries = len(sub_queries)
    # print('type name', type_name)
    # print('query number', len(sub_queries))
    if n_queries > 0:
        if type_name in ('1p', '2p', '3p'):
            train_iterator = SingledirectionalOneShotIterator(DataLoader(
                TrainDataset(sub_queries, args.nentity, args.nrelation, args.negative_sample_size, full_train_answers),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.cpu_num,
                collate_fn=TrainDataset.collate_fn
            ))
        else:
            train_iterator = SingledirectionalOneShotIterator(DataLoader(
                TrainDataset(sub_queries, args.nentity, args.nrelation, args.negative_sample_size, full_train_answers),
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.cpu_num,
                collate_fn=TrainDataset.collate_fn
            ))
        return train_iterator
    else:
        return None

def compute_loss_by_type(model, full_train_queries, full_train_answers, type_name, args):
    sub_train_iterator = get_sub_train_iterator(full_train_queries, full_train_answers, type_name=type_name, args=args)
    if sub_train_iterator is not None:  # None if the type is the present in training queries
        positive_score, negative_score, log = train_step(model, sub_train_iterator, args, step=0)
        return positive_score, negative_score, log
    else:
        return None, None, None

def loss_of_all_types(model, full_train_queries, full_train_answers, args):
    types = [s for s in query_name_dict.values() if 'DM' not in s]
    logs = {}  # {type: log}
    for type_name in types:  # e.g. '1p'
        positive_score, negative_score, log = compute_loss_by_type(model, full_train_queries, full_train_answers, type_name, args)
        logs[type_name] = log
    log_df = pd.DataFrame(logs)
    return log_df


############# entity, relation embedding #############
def get_rel(model, rid):
    """Get relation embedding"""
    therid = [rid]
    projection = model.projection_net  # Projection() object
    r_trans = torch.einsum('br,rio->bio', projection.rel_att[therid], projection.rel_base)
    r_bias = projection.rel_bias[therid]
    return r_trans, r_bias

def get_ent(model, eid, return_raw=False):
    """Get entity embedding"""
    raw = model.entity_embedding[eid]
    constrained = model.entity_regularizer(raw)
    if return_raw:
        return raw, constrained
    return constrained


def slice_rel(model, rid):
    with torch.no_grad():
        """slice the first 10 element from the relation embeddings and check"""
        r_trans, r_bias = get_rel(model, rid)
        return r_trans[0][0][:10], r_bias[0][:10]

def slice_ent(model, eid):
    with torch.no_grad():
        e_emb = get_ent(model, eid)
        return e_emb[:10]

def slice_vec(vec):
    if len(vec) > 1:
        return vec[:10]
    elif len(vec[0]) > 1:
        return vec[0][:10]


def loss_from_triple(model, h,r,t):
    """
    h,r: query; t: answer
    """
    projected_t = get_projected_t_vec(model, h=h, rs=[r])
    true_t = get_ent(model, eid=t)
    score = model.cal_logit_fuzzy(true_t, projected_t)
    loss = loss_from_score(score, is_positive=True)
    return loss


def loss_from_score(model, score, is_positive=True):
    """Unweighted loss. Default type: logsigmoid"""
    dist = 1-score
    if is_positive:
        loss = -F.logsigmoid((model.gamma - dist)*model.gamma_coff)
    else:
        loss = -F.logsigmoid((dist - model.gamma)*model.gamma_coff)
    return loss

def get_train_queries_answers(data_dir):
    train_queries = pickle.load(open(os.path.join(data_dir, "train-queries.pkl"), 'rb'))
    train_answers = pickle.load(open(os.path.join(data_dir, "train-answers.pkl"), 'rb'))
    return train_queries, train_answers


def get_translated_sample_score_df(model, test_queries, test_hard_answers, type_name='pi'):
    sub_train_iterator = get_sub_train_iterator(test_queries, test_hard_answers, type_name=type_name, args=args)
    positive_score, negative_score, log, batch_queries, positive_sample, negative_sample = train_step1(model, sub_train_iterator, args, step=0, return_queries=True)
    if type_name == 'pi':
        flattened_structure = ('e', 'r', 'r', 'e', 'r')
    
    translated_queries = translate_flattened_queries(batch_queries, flattened_structure)
    translated_answers = translate_answers(ent_id2str, positive_sample, with_id=False)
    query_df = pd.DataFrame(translated_queries, columns=['e','r','r','e','r'])
    query_df['answer'] = translated_answers
    query_df['score'] = list(positive_score.squeeze().cpu().numpy())

    query_df_with_ids = query_df.copy()
    query_df_with_ids['query'] = list(batch_queries)
    query_df_with_ids['answer_id'] = list(positive_sample.cpu().numpy())
    return query_df, query_df_with_ids


def scoring_triple(model, h, r, t):
    """
    h,r: query; t: answer
    """
    projected_t = get_projected_t_vec(model, h=h, rs=[r])
    true_t = get_ent(model, eid=t)
    score = model.cal_logit_fuzzy(true_t, projected_t)
    return score




def get_all_rids(q, qstructure_idx):
    """
    q: e.g. (((711, (139, 59)), (1295, (59,)))
    """
    if qstructure_idx == 0:  # 1p
        entities = [q[0]]
        relations = [q[1][0]]
    elif qstructure_idx == 1:  # 2p
        entities = [q[0]]
        relations = [q[1][0], q[1][1]]
    elif qstructure_idx == 2:  # 3p
        entities = [q[0]]
        relations = [q[1][0], q[1][1], q[1][2]]
    elif qstructure_idx == 3:  # 2i
        entities = [q[0][0], q[1][0]]
        relations = [q[0][1][0], q[1][1][0]]
    elif qstructure_idx == 4:  # 3i
        entities = [q[0][0], q[1][0], q[2][0]]
        relations = [q[0][1][0], q[1][1][0], q[2][1][0]]
    elif qstructure_idx == 5:  # ip
        entities = [q[0][0][0], q[0][1][0]]
        relations = [q[0][0][1][0], q[0][1][1][0], q[1][0]]
    elif qstructure_idx == 6:  # pi
        entities = [q[0][0], q[1][0]]
        relations = [q[0][1][0], q[0][1][1], q[1][1][0]]
    else:
        print(f'type {qstructure_idx} not defined relation set!')
    return entities, relations

def filter_queries(queries, badrelations, badentities):
    """
    :param queries: list[tuple(q, q_structure_idx)]
    """
    print(badrelations)
    def q_is_valid(badrelations, q, qstructure_idx):
        entities, relations = get_all_rids(q, qstructure_idx)
        for r in relations:
            if r in badrelations:
                return False
        for e in entities:
            if e in badentities:
                return False
        return True

    ok_queries = []
    for qtuple in queries:
        q, qstructure_idx = qtuple[0], qtuple[1]
        if q_is_valid(badrelations, q, qstructure_idx):
            ok_queries.append(qtuple)
    return ok_queries



def make_test_dataset_and_dataloader(queries, args):
    sub_test_dataset =  TestDataset(
        queries,
        args.nentity,
        args.nrelation
    )

    sub_test_dataloader = DataLoader(
        sub_test_dataset,
        batch_size=args.test_batch_size,
        num_workers=args.cpu_num,
        collate_fn=TestDataset.collate_fn
    )
    return sub_test_dataset, sub_test_dataloader


# obsolete
def add2p():
    q1p = train_queries[('e',('r',))]
    print(len(q1p))
    new_2p_queries = set()
    new_2p_ans = {}  # {query: set(ans)}
    for q in q1p:
        h = q[0]
        r = q[1][0]
        if r % 2 == 0:  # r is even
            q2p = (h, (r, r+1))
        else:
            q2p = (h, (r, r-1))
        ans = h

        new_2p_queries.add(q2p)
        if q2p in new_2p_ans:
            new_2p_ans[q2p].add(ans)
        else:
            new_2p_ans[q2p] = set([ans])
    train_queries[('e', ('r', 'r'))] = set.union(train_queries[('e', ('r', 'r'))], new_2p_queries)
    train_answers.update(new_2p_ans)
    with open(join(data_dir, 'train-queries-new.pkl'), 'wb') as f:
        pickle.dump(train_queries, f)
    with open(join(data_dir, 'train-answers-new.pkl'), 'wb') as f:
        pickle.dump(train_answers, f)



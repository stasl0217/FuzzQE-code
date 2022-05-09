
#!/usr/bin/python3

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import wandb
import random
import torch.nn as nn
from os.path import join
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from models import KGReasoning
from fuzzyreasoning import KGFuzzyReasoning
from dataloader import TestDataset, TrainDataset, SingledirectionalOneShotIterator
# from tensorboardX import SummaryWriter
import time
import pickle
from util import *
from dataloader import load_data
from constants import *


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description='Training and Testing Knowledge Graph Embedding Models',
        usage='train.py [<args>] [-h | --help]'
    )

    parser.add_argument('--cuda', action='store_true', help='use GPU')

    parser.add_argument('--do_train', action='store_true', help="do train")
    parser.add_argument('--do_valid', action='store_true', help="do valid")
    parser.add_argument('--do_test', action='store_true', help="do test")

    parser.add_argument('--data_path', type=str, default=None, help="KG data path")
    parser.add_argument('-n', '--negative_sample_size', default=128, type=int, help="negative entities sampled per query")
    parser.add_argument('-d', '--hidden_dim', default=500, type=int, help="embedding dimension")
    parser.add_argument('-g', '--gamma', default=0.5, type=float, help="margin in the loss")
    parser.add_argument('-b', '--batch_size', default=1024, type=int, help="batch size of queries")
    parser.add_argument('--test_batch_size', default=1, type=int, help='valid/test batch size')
    parser.add_argument('-lr', '--learning_rate', default=0.0001, type=float)
    parser.add_argument('-cpu', '--cpu_num', default=10, type=int, help="used to speed up torch.dataloader")
    parser.add_argument('-save', '--save_path', default='./trained_models', type=str, help="no need to set manually, will configure automatically")
    parser.add_argument('--max_steps', default=100000, type=int, help="maximum iterations to train")
    parser.add_argument('--warm_up_steps', default=None, type=int, help="no need to set manually, will configure automatically")
    parser.add_argument('--valid_steps', default=10000, type=int, help="evaluate validation queries every xx steps")
    parser.add_argument('--log_steps', default=100, type=int, help='train log every xx steps')
    parser.add_argument('--test_log_steps', default=1000, type=int, help='valid/test log every xx steps')

    parser.add_argument('--nentity', type=int, default=0, help='DO NOT MANUALLY SET')
    parser.add_argument('--nrelation', type=int, default=0, help='DO NOT MANUALLY SET')

    parser.add_argument('--geo', default='fuzzy', type=str, choices=['vec', 'box', 'beta', 'fuzzy'], help='the reasoning model, vec for GQE, box for Query2box, beta for BetaE')
    parser.add_argument('--print_on_screen', action='store_true')

    parser.add_argument('--tasks', default='1p.2p.3p.2i.3i.ip.pi.2in.3in.inp.pin.pni.2u.up', type=str, help="tasks connected by dot, refer to the BetaE paper for detailed meaning and structure of each task")
    parser.add_argument('--seed', default=0, type=int, help="random seed")
    parser.add_argument('-betam', '--beta_mode', default="(1600,2)", type=str, help='(hidden_dim,num_layer) for BetaE relational projection')
    parser.add_argument('-boxm', '--box_mode', default="(none,0.02)", type=str, help='(offset activation,center_reg) for Query2box, center_reg balances the in_box dist and out_box dist')
    parser.add_argument('--prefix', default=None, type=str, help='prefix of the log path')
    parser.add_argument('--checkpoint_path', default=None, type=str, help='path for loading the checkpoints')
    parser.add_argument('-evu', '--evaluate_union', default="DNF", type=str, choices=['DNF', 'DM'], help='the way to evaluate union queries, transform it to disjunctive normal form (DNF) or use the De Morgan\'s laws (DM)')

    # fuzzy logic
    parser.add_argument('--logic', default='godel', type=str, choices=['luka', 'godel', 'product', 'godel_gumbel'],
                        help='fuzzy logic type')


    # regularizer
    parser.add_argument('--regularizer', default='sigmoid', type=str,
                        choices=['01', 'vector_softmax', 'matrix_softmax', 'matrix_L1', 'matrix_sigmoid_L1','sigmoid', 'vector_sigmoid_L1'],
                        help='ways to regularize parameters')  # By default, this regularizer applies to both entities and queries

    parser.add_argument('--e_regularizer', default='same', type=str,
                        choices=['same', '01', 'vector_softmax', 'matrix_softmax', 'matrix_L1', 'matrix_sigmoid_L1','sigmoid', 'vector_sigmoid_L1'],
                        help='set regularizer for entities, different from queries')  # if 'same' (default), just use args.regularizer
    parser.add_argument('--entity_ln_before_reg', action="store_true", help='apply layer normalization before applying regularizer to entities')


    parser.add_argument('--gamma_coff', default=20, type=float, help='coefficient for gamma')
    parser.add_argument('-k', '--prob_dim', default=8, type=int, help="for matrix_softmax and matrix_L1. dims per prob vector")
    parser.add_argument('--godel_gumbel_beta', default=0.01, type=int, help="Gumbel beta for min/max computation when logic=godel_gumbel")
    parser.add_argument('--loss_type', default='cos', type=str,
                        choices=['cos', 
                                 'cos_digits', 'L1_cos_digits', 'dot_layernorm_digits',
                                 'dot', 'weighted_dot',
                                 'soft_min_digits',
                                 'kl', 'entropy',
                                 'discrete_cos', 'discrete_prob', 'discrete_gumbel', 'gumbel_softmax',
                                 'fuzzy_containment', 'weighted_fuzzy_containment',
                                 'entity_multinomial_dot',  # use with sigmoid regularizer. L1 noramlize entity before computing score
                                 'normalized_entity_dot'  # normalize entity when no grad. use with 0/1 regularizer for entity, and sigmoid for query
                                 ], help="loss type")
    parser.add_argument(
        '--margin_type', default='logsigmoid_bpr', type=str, 
        choices=[
            'logsigmoid', 'logsigmoid_bpr', 'logsigmoid_bpr_digits', 'bpr_digits', 'logsigmoid_avg', 'bpr', 'softmax', 'nll'
        ], 
        help='ways to implement margin'
    )
    parser.add_argument(
        '--with_counter', action="store_true", help="add neg q into negative samples"
    )
    parser.add_argument('--gpu_ids', default='0', type=str)
    parser.add_argument('--continue_train', default=None, type=str, help='run name to load and continue training')

    # gumbel softmax
    parser.add_argument('--gumbel_temperature', default=1, type=float,
                        help="Gumbel temperature for gumbel softmax")
    parser.add_argument('--gumbel_attention', default='none', type=str, choices=['none', 'plain', 'query_dependent'], help="Add distribution-wise attention")
    parser.add_argument('--query_unnorm', action="store_true")
    parser.add_argument('--simplE', action="store_true", help="Use different head and tail embeddings for entities")

    # conjunction
    parser.add_argument('--use_attention', action='store_true', help='use attention for conjunction')

    # relation as a transformation
    parser.add_argument('--projection_type', default='rtransform', type=str, choices=['mlp', 'rtransform', 'transe'])
    parser.add_argument('--num_rel_base', default=50, type=int)

    # lr scheduler
    # original is BetaE original
    parser.add_argument('--lr_scheduler', default='annealing', type=str, choices=['none', 'original', 'step', 'annealing', 'plateau', 'onecycle'])
    parser.add_argument('--optimizer', default='Adam', type=str, choices=['Adam', 'AdamW'])
    parser.add_argument('--L2_reg', default=0, type=float)
    parser.add_argument('--N3_regularization', action='store_true', help='nuclear 3-norm regularization. L2_reg as coefficient. not using weight decay.')

    parser.add_argument('--in_batch_negative', action='store_true', help='use in-batch negatives')

    parser.add_argument('--load_pretrained', action='store_true', help='load pretrained embeddings. dimension=1000. only for NELL')

    parser.add_argument('--no_anchor_reg', action='store_true', help='no anchor entity regularizer')
    parser.add_argument('--share_relation_bias', action='store_true', help='share relation bias')


    return parser.parse_args(args)


def main(args):
    run = wandb_initialize(vars(args))
    run.save()
    print(f'Wandb run name: {run.name}')
    
    # cuda settings
    if args.cuda:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids  # specify which GPU(s) to be used
        args.batch_size = args.batch_size * torch.cuda.device_count()  # adjust batch size
        print(f'Cuda device count:{torch.cuda.device_count()}')
        wandb.log({'num_gpu': torch.cuda.device_count()})
    device = torch.device('cuda' if args.cuda else 'cpu')

    set_global_seed(args.seed)
    tasks = args.tasks.split('.')
    for task in tasks:
        if 'n' in task and args.geo in ['box', 'vec']:
            assert False, "Q2B and GQE cannot handle queries with negation"
    if args.evaluate_union == 'DM':
        assert args.geo == 'beta', "only BetaE supports modeling union using De Morgan's Laws"

    # Model save path
    args.save_path = join('./trained_models')
    print(f'Overwrite model save path. Save model and log to folder {args.save_path}')
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # logger
    set_logger(args)

    nentity, nrelation = read_num_entity_relation_from_file(args.data_path)
    args.nentity, args.nrelation = nentity, nrelation
    wandb.log({'nentity': nentity, 'nrelation': nrelation})

    train_path_iterator, train_other_iterator, valid_dataloader, test_dataloader,\
        valid_hard_answers, valid_easy_answers, \
        test_hard_answers, test_easy_answers = load_data(args, query_name_dict, tasks)
    
    if len(tasks) == 1:  # 1p only
        # load full test data for testing
        full_tasks = '1p.2p.2i.2in'.split('.')
        _, _, _, test_dataloader,\
        _, _, \
        test_hard_answers, test_easy_answers = load_data(args, query_name_dict, full_tasks)
     

    # Fuzzy only. This repo does not support other geo like BetaE, box anymore
    if args.geo == 'fuzzy':
        model = KGFuzzyReasoning(
            nentity=nentity,
            nrelation=nrelation,
            hidden_dim=args.hidden_dim,
            gamma=args.gamma,
            geo=args.geo,
            use_cuda=args.cuda,
            box_mode=eval_tuple(args.box_mode),
            beta_mode=eval_tuple(args.beta_mode),
            test_batch_size=args.test_batch_size,
            query_name_dict=query_name_dict,
            logic_type=args.logic,
            gamma_coff=args.gamma_coff,
            regularizer_setting={
                'type': args.regularizer,  # for query
                'e_reg_type': args.regularizer if args.e_regularizer == 'same' else args.e_regularizer,
                'prob_dim': args.prob_dim,  # for matrix softmax
                'dual': True if args.loss_type == 'weighted_fuzzy_containment' else False,
                'e_layernorm': args.entity_ln_before_reg  # apply Layer Norm before next step's regularizer
            },
            loss_type=args.loss_type,
            margin_type=args.margin_type,
            device=device,
            godel_gumbel_beta=args.godel_gumbel_beta,
            gumbel_temperature=args.gumbel_temperature,
            projection_type=args.projection_type,
            args=args
        )
    else:
        model = KGReasoning(
            nentity=nentity,
            nrelation=nrelation,
            hidden_dim=args.hidden_dim,
            gamma=args.gamma,
            geo=args.geo,
            use_cuda=args.cuda,
            box_mode=eval_tuple(args.box_mode),
            beta_mode=eval_tuple(args.beta_mode),
            test_batch_size=args.test_batch_size,
            query_name_dict=query_name_dict
        )
    model = model.to(device)
    # model = nn.DataParallel(model)  # make it parallel
    wandb.watch(model)
    print_parameters(model)

    # set lr and optimizer
    if args.do_train:
        current_learning_rate = args.learning_rate
        if args.optimizer == 'AdamW':  # use together with lr_scheduler none
            if args.L2_reg > 0:
                weight_decay = args.L2_reg
            else:
                weight_decay = 1e-2
            print(f'AdamW weight decay: {weight_decay}')
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, list(model.parameters())), 
                lr=args.learning_rate,
                eps=1e-06,
                weight_decay=weight_decay
            )
        else:
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, list(model.parameters())),
                lr=current_learning_rate,
                weight_decay=args.L2_reg  # L2 regularization
            )

        if args.lr_scheduler == 'original':
            warm_up_steps = args.max_steps // 2  # reduce lr when reaching warm up steps
        elif args.lr_scheduler == 'step':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50000, gamma=0.2)
        elif args.lr_scheduler == 'annealing':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_steps, eta_min=0, last_epoch=-1)
        elif args.lr_scheduler == 'plateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=args.valid_steps*2,
                verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0,
                min_lr=0.0001, eps=1e-07
            )
        elif args.lr_scheduler == 'onecycle':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, pct_start=0.05, anneal_strategy='linear', final_div_factor=10,\
                        max_lr = 5e-4, total_steps = args.batch_size * args.max_steps + 1)

    if args.continue_train is not None:
        saved_run_name = args.continue_train
        model_path = join(args.save_path, saved_run_name+'.pt')
        model = torch.load(model_path)


    init_step = 0
    step = init_step

    if args.do_train:
        print('=== Start training ===')
        time0 = time.time()

        training_logs = []
        # #Training Loop

        last_best_metric = None
        last_best_step = 0
        early_stop_metric = 'average_MRR'
        patience = args.valid_steps * 5

        for step in range(init_step, args.max_steps):
            if step == 2*args.max_steps//3:
                args.valid_steps *= 4

            # if args.loss_type == 'gumbel_softmax' and step % 5000 == 0:
            #     # gumbel temperature annealing; temperature = max(0.5, exp(-rt)), r={1e-5, 1e-4}, t=step
            #     model.gumbel_temperature = max(0.5, math.exp(-1e-4*step))

            # log = model.module.train_step(model, optimizer, train_path_iterator, args, step)
            log = model.train_step(model, optimizer, train_path_iterator, args, step)
            if train_other_iterator is not None:
                log = model.train_step(model, optimizer, train_other_iterator, args, step)
                # log = model.train_step(model, optimizer, train_path_iterator, args, step)

            training_logs.append(log)

            # update learning rate
            if args.lr_scheduler != 'none':  # do not change lr if 'none'
                if args.lr_scheduler == 'original':  # BetaE original
                    if step >= warm_up_steps:
                        current_learning_rate = current_learning_rate / 5
                        warm_up_steps = warm_up_steps * 1.5
                        optimizer = torch.optim.Adam(
                            filter(lambda p: p.requires_grad, model.parameters()),
                            lr=current_learning_rate
                        )  # new optimizer

                elif args.lr_scheduler in ('step', 'annealing', 'plateau', 'onecycle'):
                    if args.lr_scheduler == 'plateau':
                        scheduler.step(log['loss'])
                    else:
                        scheduler.step()


            if step % args.valid_steps == 0 and step > 0:
                if args.do_valid:
                    print('Evaluating on Valid Dataset...')
                    valid_all_metrics = evaluate(model, valid_easy_answers, valid_hard_answers, args, valid_dataloader, query_name_dict, 'Valid', step)

                if args.do_test:
                    print('Evaluating on Test Dataset...')
                    time1 = time.time()
                    test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step)
                    print(test_all_metrics)
                    print(f'Finished testing. Testing used time {time.time()-time1:.2f}')

                    # if last_best_metric is None:
                    #     last_best_metric = test_all_metrics.copy()

                    # early stop
                    #TODO: change it to valid_all_metrics
                    if last_best_metric is None or test_all_metrics[early_stop_metric] > last_best_metric[early_stop_metric]:
                        last_best_metric = test_all_metrics.copy()
                        last_best_step = step
                        # save
                        if args.geo == 'fuzzy':
                            save_path = os.path.join(args.save_path, f'{run.name}.pt')
                            torch.save(model, save_path)
                        else:  # baseline models. can only save model.state_dict
                            save_dir = join(args.save_path, 'baselines', args.geo, str(run.name))
                            if not os.path.exists(save_dir):
                                os.makedirs(save_dir)
                            save_variable_list = {
                                'step': step,
                                'current_learning_rate': current_learning_rate,
                                'warm_up_steps': warm_up_steps
                            }
                            save_model(model, optimizer, save_variable_list, save_dir, args)
                    elif step > last_best_step + patience:
                        # early stop
                        break


            if step % args.log_steps == 0:
                metrics = {}
                for metric in training_logs[0].keys():
                    metrics[metric] = sum([log[metric] for log in training_logs])/len(training_logs)

                log_metrics('Training average', step, metrics)
                training_logs = []

                print(f'Time to train {args.log_steps} step: {time.time() - time0:.2f}')

                # # debug parameter change
                # if args.projection_type == 'mlp':
                #     wandb.log({
                #         'projection_layer00': model.projection_net.layer0.weight[0,0]
                #     })
                # if args.regularizer == 'sigmoid':
                #     wandb.log({
                #         'conjunction_regularizer': model.conjunction_net.regularizer.weight[0]
                #     })

                cur_lr = optimizer.param_groups[0]['lr']
                # print('Change learning_rate to %f at step %d' % (current_learning_rate, step))
                wandb.log({'current_lr': cur_lr})





    try:
        print(step)
    except:
        step = 0

    if args.do_test:
        logging.info('Evaluating on Test Dataset...')
        test_all_metrics = evaluate(model, test_easy_answers, test_hard_answers, args, test_dataloader, query_name_dict, 'Test', step)

    logging.info("Training finished!!")

if __name__ == '__main__':
    main(parse_args())

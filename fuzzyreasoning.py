import torch

from models import *
import wandb
from constants import query_structure_list, query_structure2idx
from util import get_regularizer
from operations import Projection, Conjunction, Disjunction, Negation
from gumbel import gumbel_softmax
import torch.nn.functional as F

class KGFuzzyReasoning(KGReasoning):
    def __init__(
        self, nentity, nrelation, hidden_dim, gamma,
        geo, test_batch_size=1,
        box_mode=None, use_cuda=False,
        query_name_dict=None, beta_mode=None,
        logic_type='product',
        regularizer_setting=None,
        gamma_coff=20,
        loss_type='cos',
        margin_type='logsigmoid',
        device=None,
        godel_gumbel_beta=0.01,
        gumbel_temperature=1,
        projection_type='mlp',
        args=None

    ):
        super(KGFuzzyReasoning, self).__init__(nentity, nrelation, hidden_dim, gamma,
                                               geo, test_batch_size,
                                               box_mode, use_cuda,
                                               query_name_dict, beta_mode)

        self.device = device

        # embedding
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0

        self.batch_entity_range = torch.arange(nentity).to(torch.float).repeat(test_batch_size,1).to(self.device)

        self.entity_dim = hidden_dim

        self.no_anchor_reg = args.no_anchor_reg


        if args.load_pretrained == True:
            with open('./trained_models/NELL-entity-emb.pt', 'rb') as f:
                # use pretrained embeddings to initialize and speed up training
                entity_embs = pickle.load(f)
            self.entity_embedding = nn.Parameter(entity_embs)
        else:
            self.entity_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            if self.no_anchor_reg:
                nn.init.xavier_uniform_(self.entity_embedding)
            else:
                # embedding definition
                # embedding initialization
                nn.init.uniform_(tensor=self.entity_embedding, a=0, b=1)
                

        self.simplE = args.simplE
        if args.simplE:  # use separate head and tail embeddings for entities
            self.entity_head_embedding = nn.Parameter(torch.zeros(nentity, self.entity_dim))
            nn.init.uniform_(tensor=self.entity_embedding, a=0, b=1)

        


        # loss
        self.gamma_coff = gamma_coff
        self.loss_type = loss_type
        self.margin_type = margin_type
        if self.loss_type == 'weighted_dot':
            self.dim_weight = nn.Parameter(torch.ones((self.entity_dim,)))
            self.dim_weight_softmax = nn.Softmax(dim=-1)

        if margin_type == 'softmax':
            self.softmax_weight = torch.Tensor([10]).to(device)

        # regularizer: how to turn elements into 0,1
        self.entity_regularizer = get_regularizer(regularizer_setting, self.entity_dim, neg_input_possible=True, entity=True)

        wandb.log({'loss_type': loss_type})

        self.godel_gumbel_beta = godel_gumbel_beta

        # intersection and projectizAaz<>on
        projection_dim, num_layers = beta_mode
        self.projection_net = Projection(
            nrelation,
            self.entity_dim,
            logic_type,
            regularizer_setting,
            self.relation_dim,
            projection_dim,
            num_layers,
            projection_type,
            num_rel_base=args.num_rel_base
        )

        self.conjunction_net = Conjunction(self.entity_dim, logic_type, regularizer_setting, use_attention=args.use_attention, godel_gumbel_beta=godel_gumbel_beta)
        self.disjunction_net = Disjunction(self.entity_dim, logic_type, regularizer_setting, godel_gumbel_beta=godel_gumbel_beta)
        self.negation_net = Negation(self.entity_dim, logic_type, regularizer_setting)

        # gumbel softmax
        self.gumbel_temperature = gumbel_temperature  # used if loss_type == 'gumbel_softmax'
        self.gumbel_attention = args.gumbel_attention if args.gumbel_attention != 'none' else None  # None or 'plain' or 'query_dependent'
        if self.loss_type == 'gumbel_softmax' and args.gumbel_attention:
            self.n_distribution = self.entity_regularizer.get_num_distributions()
            self.distribution_weights = nn.Parameter(torch.ones(self.n_distribution))
            if args.gumbel_attention == 'query_dependent':
                self.attention_layer = nn.Linear(self.entity_dim, self.n_distribution)
        self.gumbel_query_unnorm = args.query_unnorm

        self.in_batch_negative = args.in_batch_negative

        if self.loss_type == 'dot_layernorm_digits':
            self.entity_ln = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)
            self.query_ln = nn.LayerNorm(self.hidden_dim, elementwise_affine=False)

        self.counter_for_neg = args.with_counter  # add \neg q to negative samples

        self.margin_type = args.margin_type

        



    def forward(
            self,
            positive_sample,
            negative_sample,
            subsampling_weight,
            batch_queries_full,
            query_structure_idxs_full,
            idxs,
            inference=False  # for discrete, use soft for training and hard for inference
    ):
        """
        :param batch_queries_full: np.array[queries], e.g. array[array[8140,0], array[7269, 12, 13]]
        :param query_structures_idxs: np.array[query_structure_idx], e.g. array[0 3]
        """

        # batch_queries_full is numpy and wasn't split when using multiple GPUs
        if len(idxs) != len(batch_queries_full):  # multiple GPUs
            min_id, max_id = idxs[0], idxs[-1]
            batch_queries = batch_queries_full[min_id:max_id+1]
            query_structure_idxs = query_structure_idxs_full[min_id:max_id+1]
        else:
            batch_queries, query_structure_idxs = batch_queries_full, query_structure_idxs_full
        # print('query_structure_idxs', query_structure_idxs)

        # aggregate by query structure
        # i_qs: index for query structures
        sample_idx_list = [(query_structure_idxs == i_qs) for i_qs in range(len(query_structure_list))]
        batch_idxs_dict = {
            query_structure_list[i]: sample_idx.nonzero()
            for i, sample_idx in enumerate(sample_idx_list)
            if np.any(sample_idx)
        }

        batch_queries_dict = {
            query_structure: torch.LongTensor(np.stack(batch_queries[sample_idxs])).to(self.device)
            for query_structure, sample_idxs in batch_idxs_dict.items()
        }

        # all query embeddings
        # concatenate vectors
        all_idxs = np.concatenate([batch_idxs_dict[query_structure] for query_structure in batch_queries_dict], axis=None)
        all_embeddings = torch.cat(
            [
                self.embed_query_fuzzy(
                    batch_queries_dict[query_structure],
                    query_structure,
                    idx=0
                )[0]
                for query_structure in batch_queries_dict
            ],
            dim=0
        ).unsqueeze(1)

        all_idxs = torch.from_numpy(all_idxs).to(negative_sample.device)

        #
        # if len(all_embeddings) > 0:
        #     all_embeddings = torch.cat(all_embeddings, dim=0).unsqueeze(1)

        if subsampling_weight is not None:
            subsampling_weight = subsampling_weight[all_idxs]

        if positive_sample is not None:
            if len(all_embeddings) > 0:
                # positive samples for non-union queries in this batch
                positive_sample_regular = positive_sample[all_idxs]
                if self.loss_type.startswith('discrete'):
                    # soft discretization
                    # use steep sigmoid to make entries closer to 0,1
                    positive_embedding = self.entity_regularizer.soft_discretize(
                        torch.index_select(
                            self.entity_embedding,
                            dim=0,
                            index=positive_sample_regular
                        ).unsqueeze(1)
                    )
                else:
                    positive_embedding = self.entity_regularizer(
                        torch.index_select(
                            self.entity_embedding,
                            dim=0,
                            index=positive_sample_regular
                        ).unsqueeze(1)
                    )

                positive_score = self.cal_logit_fuzzy(positive_embedding, all_embeddings, inference=inference)
            else:
                positive_score = torch.Tensor([]).to(self.device)

        else:
            positive_score = None

        if negative_sample is None:
            negative_score = None
        else:
            if len(all_embeddings) > 0:
                negative_sample_regular = negative_sample[all_idxs]

                batch_size, negative_size = negative_sample_regular.shape
                if self.loss_type.startswith('discrete'):
                    # soft discretization
                    # use steep sigmoid to make entries closer to 0,1
                        negative_embedding = self.entity_regularizer.soft_discretize(
                            torch.index_select(
                                self.entity_embedding,
                                dim=0,
                                index=negative_sample_regular.view(-1)
                            ).view(
                                batch_size,
                                negative_size,
                                -1
                            )
                        )
                else:
                    negative_embedding = self.entity_regularizer(
                        torch.index_select(
                            self.entity_embedding,
                            dim=0,
                            index=negative_sample_regular.view(-1)
                        ).view(
                            batch_size,
                            negative_size,
                            -1
                        )
                    )
                # random negative samples
                negative_score = self.cal_logit_fuzzy(negative_embedding, all_embeddings, inference=inference)
            else:
                negative_score = torch.Tensor([]).to(self.entity_embedding.device)
            
        if self.counter_for_neg and (not inference):
            # add \neg q as a negative sample into training
            emphasize = 16
            neg_q_embeddings = self.negation_net(all_embeddings)
            negative_score_2 = self.cal_logit_fuzzy(positive_embedding, neg_q_embeddings, inference=inference)  # [batch_size, 1]
            negative_score_2 = negative_score_2.expand(-1, emphasize)
            negative_score = torch.cat((negative_score, negative_score_2), dim=1)
            return positive_score, negative_score, subsampling_weight, all_idxs
        else:
            return positive_score, negative_score, subsampling_weight, all_idxs

    def embed_query_fuzzy(self, queries, query_structure, idx):
        """
        :param query_structure: e.g. ((('e', ('r',)), ('e', ('r',)), ('u',)), ('r',))
        :param queries: Tensor. shape [batch_size, M],
            where M is the number of elements in query_structure (6 in the above examples)
        :param idx: which column to start in tensor queries
        """
        all_relation_flag = True
        for ele in query_structure[-1]:
            # whether the current query tree has merged to one branch
            # and only need to do relation traversal,
            # e.g., path queries or conjunctive queries after the intersection
            if ele not in ['r', 'n']:
                all_relation_flag = False
                break
        if all_relation_flag:  # only relation traversal
            if query_structure[0] == 'e':
                if self.simplE:
                    # use head embeddings
                    embedding = self.entity_regularizer(
                        torch.index_select(self.entity_head_embedding, dim=0, index=queries[:, idx])
                    )
                else:
                    if self.no_anchor_reg:
                        # entity embedding
                        embedding = torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                        
                    else:
                        # entity embedding
                        embedding = self.entity_regularizer(
                            torch.index_select(self.entity_embedding, dim=0, index=queries[:, idx])
                        )

                idx += 1  # move to next element (next column in queries)
            else:
                # recursion
                embedding, idx = self.embed_query_fuzzy(queries, query_structure[0], idx)

            for i in range(len(query_structure[-1])):  # query_structure[-1]: ('r', 'n', 'r', ...'r')
                if query_structure[-1][i] == 'n':  # negation
                    assert (queries[:, idx] == -2).all()
                    # embedding = self.fuzzy_logic.negation(embedding)
                    embedding = self.negation_net(embedding)
                else:
                    rel_indices = queries[:,idx]
                    # embedding = self.fuzzy_logic.projection(embedding, r_embedding)
                    embedding = self.projection_net(embedding, rel_indices)
                idx += 1
        else:
            subtree_embedding_list = []
            if 'u' in query_structure[-1]:  # last one is ('u')
                # aggregation by disjunction (union)
                num_subtrees = len(query_structure) - 1  # last one is 'u'
                # agg_net = self.fuzzy_logic.disjunction
                agg_net = self.disjunction_net
            else:
                # aggregation by conjunction (intersection)
                num_subtrees = len(query_structure)
                agg_net = self.conjunction_net

            for i in range(num_subtrees):
                subtree_embedding, idx = self.embed_query_fuzzy(queries, query_structure[i], idx)
                subtree_embedding_list.append(subtree_embedding)

            embedding = agg_net(torch.stack(subtree_embedding_list))

            if 'u' in query_structure[-1]:  # move to next
                idx += 1

        return embedding, idx

    def get_distribution_attention(self, query_embedding=None):
        # for gumbel softmax
        softmax = nn.Softmax(dim=-1)

        if self.gumbel_attention == 'plain':
            return softmax(self.distribution_weights)
        elif self.gumbel_attention == 'query_dependent':
            distribution_attention = softmax(self.attention_layer(query_embedding))
            return distribution_attention



    def cal_logit_fuzzy(self, entity_embedding, query_embedding, inference=False):
        """
        define scoring function for loss
        :param entity_embedding: shape [batch_size, 1, dim] (positive), [batch_size, num_neg, dim] (negative)
        :param query_embedding:　shape [batch_size, 1, dim]
        :param inference: for discrete case, use soft for training and hard for inference
        :return score: shape [batch_size, 1] for positive, [batch_size, num_neg] for negative
        """
        cos = nn.CosineSimilarity(dim=-1)
        if self.loss_type == 'gumbel_softmax':  # regularizer must start with 'matrix'
            # entity embedding has been normalized
            # query embedding has been normalized as summing up to 1 if it's out of projection
            # not necessarily summing up to 1 if out of logic operations

            if self.gumbel_query_unnorm:
                query_normalized = query_embedding
            else:
                query_normalized = self.entity_regularizer.L1_normalize(query_embedding)  # vector shape

            # query_normalized = query_embedding # vector shape
            if inference:
                # hard discrete 
                entity_one_hot = self.entity_regularizer.hard_discretize(entity_embedding)  # vector shape
            else:
                # convert entity to one-hot vector using gumbel
                entity_one_hot = self.entity_regularizer.soft_discretize(entity_embedding, self.gumbel_temperature)

            if self.gumbel_attention:
                entity_one_hot = self.entity_regularizer.reshape_to_matrix(entity_one_hot)
                query_normalized = self.entity_regularizer.reshape_to_matrix(query_normalized)
                score = cos(entity_one_hot, query_normalized)
                distribution_attention = self.get_distribution_attention(query_embedding)
                score = torch.sum(score * distribution_attention, dim=-1)
            else:
                # equivalent to torch.sum(entity_one_hot, query_normalized)/constant
                #    since ||entity_one_hot|| is the same for all entities
                score = cos(entity_one_hot, query_normalized)
            return score

        if self.loss_type == 'dot':
            # score = torch.sum(entity_embedding * query_embedding, dim=-1) / math.sqrt(self.entity_dim)  # dot product
            score = torch.sum(entity_embedding * query_embedding, dim=-1)  # dot product
        elif self.loss_type == 'weighted_dot':
            dim_weights = self.dim_weight_softmax(self.dim_weight)
            score = torch.sum(entity_embedding * query_embedding * dim_weights, dim=-1)
        elif self.loss_type.startswith('discrete'):
            # entity embedding should have been discretized

            if self.loss_type == 'discrete_cos':
                cos = nn.CosineSimilarity(dim=-1)
                score = cos(entity_embedding, query_embedding)
                # inference only
                # thres = 0.7
                # entity_embedding[entity_embedding >= thres] = 1
                # entity_embedding[entity_embedding < thres] = 0

            elif self.loss_type == 'discrete_prob':
                # In discrete representation, entities are considered entry value 0 or 1
                # entity_embedding should have been discretized

                # For the qth query
                # unlike other score computation, this score is not aggregated for each sample
                score = entity_embedding * query_embedding + (1-entity_embedding) * (1-query_embedding)

        elif self.loss_type == 'entropy':
            query_embedding = self.entity_regularizer.L1_normalize(query_embedding)  # vector shape

            # score = torch.mean(query_embedding * torch.log(entity_embedding+eps), dim=-1)

            # JSD
            m = torch.log2((query_embedding + entity_embedding) / 2 + 1e-9)
            dist = F.kl_div(m, query_embedding.expand(m.shape), reduction="none") \
                   + F.kl_div(m, entity_embedding,  reduction="none")
            num_distributions = self.entity_regularizer.get_num_distributions()  # entity_dim // k
            dist = 0.5 * torch.sum(dist, dim=-1) / num_distributions
            score = 1 - dist

        elif self.loss_type == 'fuzzy_containment':
            # for Godel only
            # use with sigmoid regularizer
            # L1
            score = entity_embedding - torch.relu(entity_embedding - query_embedding)
            score = torch.max(score, dim=-1) 
            # / torch.sum(entity_embedding, dim=-1)

        elif self.loss_type == 'weighted_fuzzy_containment':
            # for Godel only, use with sigmoid regularizer
            entity_vals, entity_val_weights = torch.chunk(entity_embedding, 2, dim=-1)
            query_vals, query_val_weights = torch.chunk(query_embedding, 2, dim=-1)
            val_weights = F.softmax(entity_val_weights * query_val_weights, dim=-1)

            score = entity_vals - torch.relu(entity_vals - query_vals)  # containment score
            score = torch.sum(score * val_weights, dim=-1) / torch.sum(entity_vals * val_weights, dim=-1)

        elif self.loss_type == 'cos_digits':  # use with logsigmoid_bpr_digits
            if not inference:
                entity_embedding = F.normalize(entity_embedding, p=2, dim=-1)
                query_embedding = F.normalize(query_embedding, p=2, dim=-1)
                score_digits = (entity_embedding * query_embedding) * self.entity_dim
                # score_digits = score_digits / norm.unsqueeze(2) * self.entity_dim
                return score_digits  # no aggregation, [batch_size, 1 or num_neg, dim]

            # # use cos for inference
            cos = nn.CosineSimilarity(dim=-1)
            score = cos(entity_embedding, query_embedding)

        elif self.loss_type == 'dot_layernorm_digits':  # use with logsigmoid_bpr_digits
            entity_embedding = self.entity_ln(entity_embedding)
            query_embedding = self.query_ln(query_embedding)
            score_digits = (entity_embedding * query_embedding)
            # score_digits = score_digits / norm.unsqueeze(2) * self.entity_dim
            if not inference:
                return score_digits  # no aggregation, [batch_size, 1 or num_neg, dim]
            # inference
            return torch.mean(score_digits, dim=-1)


        elif self.loss_type == 'L1_cos_digits':  # use with logsigmoid_avg
            entity_embedding = F.normalize(entity_embedding, p=1, dim=-1)
            query_embedding = F.normalize(query_embedding, p=1, dim=-1)
            score_digits = (entity_embedding * query_embedding) * self.entity_dim
            # score_digits = score_digits / norm.unsqueeze(2) * self.entity_dim
            if not inference:
                return score_digits  # no aggregation, [batch_size, 1 or num_neg, dim]
            # inference
            return torch.mean(score_digits, dim=-1)


        elif self.loss_type == 'soft_min_digits':
            # use with godel logic
            # entity_embedding = F.normalize(entity_embedding, p=2, dim=-1)
            # query_embedding = F.normalize(query_embedding, p=2, dim=-1)
            entity_embedding, query_embedding = torch.broadcast_tensors(entity_embedding, query_embedding)
            compare = torch.stack((entity_embedding, query_embedding))
            # a smooth way to compute min
            score_digits = -self.godel_gumbel_beta * torch.logsumexp(
                -compare / self.godel_gumbel_beta, 0
            )
            if not inference:
                return score_digits  # no aggregation, [batch_size, 1 or num_neg, dim]
            # inference, aggregated
            score = torch.mean(score_digits, dim=-1)
            # score = torch.logsumexp(-score_digits, dim=-1)

        elif self.loss_type == 'entity_multinomial_dot':
            entity_embedding = F.normalize(entity_embedding, p=1, dim=-1)
            score = torch.sum(entity_embedding * query_embedding, dim=-1)

        elif self.loss_type == 'normalized_entity_dot':
            score = torch.sum(entity_embedding * query_embedding, dim=-1)

        else:  # cos by default
            cos = nn.CosineSimilarity(dim=-1)
            score = cos(entity_embedding, query_embedding)
        return score

    @staticmethod
    def compute_loss(model, positive_score, negative_score, subsampling_weight):
        if model.margin_type == 'logsigmoid':
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
        elif model.margin_type == 'logsigmoid_avg':
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
        elif model.margin_type == 'softmax':
            # positive_score shape [batch_size, 1]
            criterion = nn.CrossEntropyLoss(reduction='none')  # keep loss for each sample
            if model.loss_type != 'discrete_prob':
                softmax_weight = 10
                scores = torch.cat([positive_score, negative_score], dim=1)*softmax_weight  # [batch_size, 1+negative_sample_size]
            else:
                # score: log(prob)
                # softmax=exp(x1)/(exp(x1)+...exp(xn))=exp(x1+exp_shift)/(exp(x1+exp_shift)+...+exp(xn+exp_shift))
                # otherwise the log scores are too small and the results are all zero
                exp_shift, _ = torch.max(positive_score, dim=-1)
                exp_shift = torch.unsqueeze(exp_shift, 1)
                positive_score = positive_score - exp_shift  # still in log scale
                negative_score = negative_score - exp_shift

                # debug only
                positive_score_real = torch.exp(positive_score)
                negative_score_real = torch.exp(negative_score)

                scores = torch.cat([positive_score, negative_score], dim=1)

            target = torch.zeros((positive_score.shape[0],), dtype=torch.long).to(device)
            loss = (criterion(scores, target) * subsampling_weight).sum()  # CrossEntropyLoss includes softmax
            loss /= subsampling_weight.sum()
            log = {'loss': loss.item()}
        elif model.margin_type == 'bpr':
            # gamma as margin
            diff = torch.relu(model.gamma + negative_score -positive_score)  # relu or softplus
            unweighted_sample_loss = torch.mean(diff, dim=-1)
            loss = (subsampling_weight * unweighted_sample_loss).sum()
            loss /= subsampling_weight.sum()
            log = {
                'loss': loss.item(),
            }
        elif model.margin_type == 'bpr_digits':
            # positive_score: shape [batch_size, 1, dim]  (not aggregated yet)
            # negative_score: shape [batch_size, neg_per_pos, dim]
            # gamma as margin
            diff = torch.mean(torch.relu(model.gamma + negative_score -positive_score), dim=-1)  # relu or softplus
            unweighted_sample_loss = torch.mean(diff, dim=-1)
            loss = (subsampling_weight * unweighted_sample_loss).sum()
            loss /= subsampling_weight.sum()
            log = {
                'loss': loss.item(),
            }
        elif model.margin_type == 'logsigmoid_bpr_digits':
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
        elif model.margin_type == 'logsigmoid_bpr':
            # gamma as margin
            diff = -F.logsigmoid(model.gamma_coff*(positive_score - negative_score))
            # diff = torch.mean(-F.logsigmoid(model.gamma_coff*(positive_score - negative_score)), dim=-1)
            unweighted_sample_loss = torch.mean(diff, dim=-1)
            loss = (subsampling_weight * unweighted_sample_loss).sum()
            loss /= subsampling_weight.sum()
            log = {
                'loss': loss.item(),
            }

        elif model.margin_type == 'nll':  # negative log likelihood. used together with discrete_prob
            if model.loss_type == 'discrete_prob':
                # positive_score: shape [batch_size, 1, dim]  (not aggregated yet)
                # negative_score: shape [batch_size, neg_per_pos, dim]

                eps = 1e-4  # avoid torch.log(zero)
                log_positive_score = torch.log(positive_score+eps)
                log_negative_score = torch.log(1-negative_score+eps)  # flip for negative samples

                # negative log likelihood
                # use torch.mean instead of torch.sum to divide by a constant (dim)
                positive_sample_loss = - torch.mean(log_positive_score, dim=-1).squeeze(dim=1)
                negative_sample_loss = - torch.mean(log_negative_score, dim=-1).mean(dim=1)
                # positive_sample_loss = -positive_score.squeeze(dim=1)
                # negative_sample_loss = -torch.log(1-torch.exp(negative_score)+eps).mean(dim=1)

                positive_sample_loss = (subsampling_weight * positive_sample_loss).sum()
                negative_sample_loss = (subsampling_weight * negative_sample_loss).sum()
                positive_sample_loss /= subsampling_weight.sum()
                negative_sample_loss /= subsampling_weight.sum()
                loss = (positive_sample_loss + negative_sample_loss)
                log = {
                    'positive_sample_loss': positive_sample_loss.item(),
                    'negative_sample_loss': negative_sample_loss.item(),
                    'loss': loss.item(),
                }
            elif model.loss_type == 'entropy':
                # version 1
                # positive_sample_loss = - positive_score.squeeze(dim=-1)
                # negative_sample_loss = negative_score.mean(dim=-1)
                # positive_sample_loss = (subsampling_weight * positive_sample_loss).sum()
                # negative_sample_loss = (subsampling_weight * negative_sample_loss).sum()
                # positive_sample_loss /= subsampling_weight.sum()
                # negative_sample_loss /= subsampling_weight.sum()
                # loss = (positive_sample_loss + negative_sample_loss)
                # log = {
                #     'positive_sample_loss': positive_sample_loss.item(),
                #     'negative_sample_loss': negative_sample_loss.item(),
                #     'loss': loss.item(),
                # }

                # # # version 2
                positive_score = positive_score.squeeze(dim=-1)
                negative_score = negative_score.mean(dim=-1)
                diff = torch.relu(model.gamma + negative_score-positive_score)
                unweighted_sample_loss = torch.mean(diff, dim=-1)
                loss = (subsampling_weight * unweighted_sample_loss).sum()
                loss /= subsampling_weight.sum()
                log = {
                    'loss': loss.item(),
                }
        
        return loss, log


    @staticmethod
    def train_step(model, optimizer, train_iterator, args, step):
        """
        Adapted for multiple GPUs
        """
        # device = model.module.device
        device = model.device

        model.train()
        optimizer.zero_grad()

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
            slice_idxs,  # to help track batch_queries and query_structures when using multiple GPUs
            inference=False
        )
        loss, log = KGFuzzyReasoning.compute_loss(model, positive_score, negative_score, subsampling_weight)

        loss.backward()
        optimizer.step()

        if model.loss_type == 'normalized_entity_dot':
            with torch.no_grad():
                # normalize entity embeddings
                normalized = nn.Parameter(torch.clamp(model.entity_embedding, 0, 1))
                # F1 normalize
                model.entity_embedding = nn.Parameter(F.normalize(normalized, p=1, dim=-1))



        return log

    @staticmethod
    def test_step(model, easy_answers, hard_answers, args, test_dataloader, query_name_dict, save_result=False, save_str="", save_empty=False):
        model.eval()

        # device = model.module.device
        device = model.device

        step = 0
        total_steps = len(test_dataloader)
        logs = collections.defaultdict(list)

        with torch.no_grad():
            for negative_sample, queries, queries_unflatten, query_structure_idxs in tqdm(test_dataloader, disable=not args.print_on_screen):
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

                if model.loss_type == 'discrete_prob':
                    # negative_logit shape[batch_size, num_entity, dim], not aggregated yet
                    eps = 1e-4
                    negative_logit = torch.sum(torch.log(negative_logit+eps), dim=-1)

                idxs_np = idxs.detach().cpu().numpy()
                # if not converted to numpy, idxs_np will be considered scalar when test_batch_size=1
                # queries_unflatten = queries_unflatten[idxs_np]
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
                    logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                step += 1

        metrics = collections.defaultdict(lambda: collections.defaultdict(int))
        for query_structure in logs:
            for metric in logs[query_structure][0].keys():
                if metric in ['num_hard_answer']:
                    continue
                metrics[query_structure][metric] = sum([log[metric] for log in logs[query_structure]])/len(logs[query_structure])
            metrics[query_structure]['num_queries'] = len(logs[query_structure])

        return metrics

def JSD(p, q):
    m = (p + q) / 2
    loss = F.kl_div(p.log(), m, reduction="mean") + F.kl_div(q.log(), m, reduction="mean")
    return 1 - (0.5 * loss)
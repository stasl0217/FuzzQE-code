import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# ================================
#
def sample_gumbel(shape, eps=1e-20):
    U = torch.rand(shape).cuda()  # from uniform distribution [0,1)
    return -Variable(torch.log(-torch.log(U + eps) + eps))


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.size())
    return F.softmax(y / temperature, dim=-1)


def gumbel_softmax(logits, temperature):
    """
    input: [*, n_class]
    return: [*, n_class] an one-hot vector
    """
    y = gumbel_softmax_sample(logits, temperature)
    shape = y.size()
    _, ind = y.max(dim=-1)
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    return (y_hard - y).detach() + y



# ===============================
# https://github.com/shaabhishek/gumbel-softmax-pytorch/blob/master/Gumbel-softmax%20visualization.ipynb

# def sample_gumbel(shape):
#     unif = torch.distributions.Uniform(0,1).sample(shape).cuda()
#     g = -torch.log(-torch.log(unif))
#     return g
#
#
# def gumbel_softmax(pi, temperature):
#     g = sample_gumbel(pi.size())
#     h = (g + torch.log(pi))/temperature
#     h_max = h.max(dim=1, keepdim=True)[0]
#     h = h - h_max
#     cache = torch.exp(h)
# #     print(pi, torch.log(pi), intmdt)
#     y = cache / cache.sum(dim=-1, keepdim=True)
#     return y



#==================
# https://github.com/yandexdataschool/gumbel_dpg/blob/master/gumbel.py

#
# class GumbelSigmoid:
#     """
#     A gumbel-sigmoid nonlinearity with gumbel(0,1) noize
#     In short, it's a function that mimics #[a>0] indicator where a is the logit
#
#     Explaination and motivation: https://arxiv.org/abs/1611.01144
#
#     Math:
#     Sigmoid is a softmax of two logits: a and 0
#     e^a / (e^a + e^0) = 1 / (1 + e^(0 - a)) = sigm(a)
#
#     Gumbel-sigmoid is a gumbel-softmax for same logits:
#     gumbel_sigm(a) = e^([a+gumbel1]/t) / [ e^([a+gumbel1]/t) + e^(gumbel2/t)]
#     where t is temperature, gumbel1 and gumbel2 are two samples from gumbel noize: -log(-log(uniform(0,1)))
#     gumbel_sigm(a) = 1 / ( 1 +  e^(gumbel2/t - [a+gumbel1]/t) = 1 / ( 1+ e^(-[a + gumbel1 - gumbel2]/t)
#     gumbel_sigm(a) = sigm([a+gumbel1-gumbel2]/t)
#
#     For computation reasons:
#     gumbel1-gumbel2 = -log(-log(uniform1(0,1)) +log(-log(uniform2(0,1)) = -log( log(uniform2(0,1)) / log(uniform1(0,1)) )
#     gumbel_sigm(a) = sigm([a-log(log(uniform2(0,1))/log(uniform1(0,1))]/t)
#
#
#     :param t: temperature of sampling. Lower means more spike-like sampling. Can be symbolic.
#     :param eps: a small number used for numerical stability
#     :returns: a callable that can (and should) be used as a nonlinearity
#
#     """
#
#     def __init__(self,
#                  t=0.1,
#                  eps=1e-20):
#         assert t != 0
#         self.temperature = t
#         self.eps = eps
#         self._srng = RandomStreams(get_rng().randint(1, 2147462579))
#
#     def __call__(self, logits):
#         """computes a gumbel softmax sample"""
#
#         # sample from Gumbel(0, 1)
#         uniform1 = self._srng.uniform(logits.shape, low=0, high=1)
#         uniform2 = self._srng.uniform(logits.shape, low=0, high=1)
#
#         noise = -T.log(T.log(uniform2 + self.eps) / T.log(uniform1 + self.eps) + self.eps)
#
#         # draw a sample from the Gumbel-Sigmoid distribution
#         return T.nnet.sigmoid((logits + noise) / self.temperature)
#
#
# def hard_sigm(logits):
#     """computes a hard indicator function. Not differentiable"""
#     return T.switch(T.gt(logits, 0), 1, 0)
#
#
# class GumbelSigmoidLayer(Layer):
#     """
#     lasagne.layers.GumbelSigmoidLayer(incoming,**kwargs)
#     A layer that just applies a GumbelSigmoid nonlinearity.
#     In short, it's a function that mimics #[a>0] indicator where a is the logit
#
#     Explaination and motivation: https://arxiv.org/abs/1611.01144
#
#     Math:
#     Sigmoid is a softmax of two logits: a and 0
#     e^a / (e^a + e^0) = 1 / (1 + e^(0 - a)) = sigm(a)
#
#     Gumbel-sigmoid is a gumbel-softmax for same logits:
#     gumbel_sigm(a) = e^([a+gumbel1]/t) / [ e^([a+gumbel1]/t) + e^(gumbel2/t)]
#     where t is temperature, gumbel1 and gumbel2 are two samples from gumbel noize: -log(-log(uniform(0,1)))
#     gumbel_sigm(a) = 1 / ( 1 +  e^(gumbel2/t - [a+gumbel1]/t) = 1 / ( 1+ e^(-[a + gumbel1 - gumbel2]/t)
#     gumbel_sigm(a) = sigm([a+gumbel1-gumbel2]/t)
#
#     For computation reasons:
#     gumbel1-gumbel2 = -log(-log(uniform1(0,1)) +log(-log(uniform2(0,1)) = -log( log(uniform2(0,1)) / log(uniform1(0,1)) )
#     gumbel_sigm(a) = sigm([a-log(log(uniform2(0,1))/log(uniform1(0,1))]/t)
#
#     Parameters
#     ----------
#     incoming : a :class:`Layer` instance or a tuple
#         The layer feeding into this layer, or the expected input shape
#     t: temperature of sampling. Lower means more spike-like sampling. Can be symbolic (e.g. shared)
#     eps: a small number used for numerical stability
#     """
#
#     def __init__(self, incoming, t=0.1, eps=1e-20, **kwargs):
#         super(GumbelSigmoidLayer, self).__init__(incoming, **kwargs)
#         self.gumbel_sigm = GumbelSigmoid(t=t, eps=eps)
#
#     def get_output_for(self, input, hard_max=False, **kwargs):
#         if hard_max:
#             return hard_sigm(input)
#         else:
#             return self.gumbel_sigm(input)
from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from Models.stochastic_inits import init_stochastic_linear
from Utils.common import list_mult

# -------------------------------------------------------------------------------------------
#  Stochastic layer 随机网络层 层的参数为随机的：
#  可以进一步创建随机线性层StochasticLinear(StochasticLayer)
# -------------------------------------------------------------------------------------------
class StochasticLayer(nn.Module):
    # base class of stochastic layers with re-parametrization
    # self.init  and self.operation should be filled by derived classes

    """
    记录参数总个数weights_count「权重w 偏置b」
    权重w：mean: w_mu & log_var: w_log_var
    偏置b：mean: b_mu & log_var: b_log_var
    """
    def create_stochastic_layer(self, weights_shape, bias_size, log_var_init):
        # 确定网络参数【权重偏差】数量
        # create the layer parameters
        # values initialization is done later
        self.weights_shape = weights_shape
        # 计算权重个数
        self.weights_count = list_mult(weights_shape)
        # 如果有偏置，则增加偏置数量
        if bias_size is not None:
            self.weights_count += bias_size
        # 对网络中每一个参数 w 设置对应的 权重 均值以及参数噪声
        self.w_mu = get_param(weights_shape)
        self.w_log_var = get_param(weights_shape)
        self.w = {'mean': self.w_mu, 'log_var': self.w_log_var}
        # 网络参数数量 偏差 均值以及噪声
        if bias_size is not None:
            self.b_mu = get_param(bias_size)
            self.b_log_var = get_param(bias_size)
            self.b = {'mean': self.b_mu, 'log_var': self.b_log_var}

    """
    计算 网络层输出 layer_out  先算均值out_mean，再计算方差
    """
    def forward(self, x):

        # Layer computations (based on "Variational Dropout and the Local
        # Reparameterization Trick", Kingma et.al 2015)
        # self.operation should be linear or conv

        # 如果使用 bias 则计算产生偏置 b 的分布 ：均值 bias_mean 以及方差 b_var
        if self.use_bias:
            b_var = torch.exp(self.b_log_var)
            bias_mean = self.b['mean']
        else:
            b_var = None
            bias_mean = None

        # 定义的线性函数 operation out_mean = w*x + bias
        out_mean = self.operation(x, self.w['mean'], bias=bias_mean)

        eps_std = self.eps_std
        # 如果方差为0
        if eps_std == 0.0:
            layer_out = out_mean
        # 若方差不为零，则添加噪声
        else:
            w_var = torch.exp(self.w_log_var)
            # 定义的线性函数 w_var * x.pow(2) + b_var
            out_var = self.operation(x.pow(2), w_var, bias=b_var)

            # Draw Gaussian random noise, N(0, eps_std) in the size of the
            # layer output: normal
            # 为网络层每一个参数都建立一个零均值，方差为eps_std 的高斯分布
            noise = out_mean.data.new(out_mean.size()).normal_(0, eps_std)
            # noise = eps_std * torch.randn_like(out_mean, requires_grad=False)

            # out_var = F.relu(out_var) # to avoid nan due to numerical errors
            # 输出带随机噪声的输出
            layer_out = out_mean + noise * torch.sqrt(out_var)

        return layer_out

    # 画高斯随机噪声时用到的参数方差, N(0, eps_std)
    def set_eps_std(self, eps_std):
        old_eps_std = self.eps_std
        self.eps_std = eps_std
        return old_eps_std

# -------------------------------------------------------------------------------------------
#  Stochastic linear layer
#  创建随机化线性层
# -------------------------------------------------------------------------------------------
class StochasticLinear(StochasticLayer):

    def __init__(self, in_dim, out_dim, log_var_init, use_bias=True):
        super(StochasticLinear, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        weights_size = (out_dim, in_dim)
        self.use_bias = use_bias
        if use_bias:
            bias_size = out_dim
        else:
            bias_size = None
        self.create_stochastic_layer(weights_size, bias_size, log_var_init)
        # 对随机线性化层初始化
        # prm.log_var_init参数：log_var_init['mean'], log_var_init['std']
        init_stochastic_linear(self, log_var_init)
        self.eps_std = 1.0


    def __str__(self):
        return 'StochasticLinear({0} -> {1})'.format(self.in_dim, self.out_dim)

    def operation(self, x, weight, bias):
        return F.linear(x, weight, bias)

# -------------------------------------------------------------------------------------------
#  Auxilary functions
# -------------------------------------------------------------------------------------------
def make_pair(x):
    if isinstance(x, int):
        return (x, x)
    else:
        return x

# def get_randn_param(shape, mean, std):
#     return nn.Parameter(randn_gpu(shape, mean, std))


# def get_randn_param(shape, mean, std):
#     if isinstance(shape, int):
#         shape = (shape,)
#     return nn.Parameter(torch.FloatTensor(*shape).normal_(mean, std))


def get_param(shape):
    # create a parameter
    if isinstance(shape, int):
        shape = (shape,)
    return nn.Parameter(torch.empty(*shape))
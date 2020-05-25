#
# the code is inspired by: https://github.com/katerakelly/pytorch-maml

from __future__ import absolute_import, division, print_function

import torch.nn as nn
import torch.nn.functional as F

from Models.layer_inits import init_layers
from Models.stochastic_layers import StochasticLinear, StochasticLayer
# from Utils import data_gen
from Utils.common import list_mult


# -------------------------------------------------------------------------------------------
# Auxiliary functions
# -------------------------------------------------------------------------------------------

def count_weights(model):
    # note: don't counts batch-norm parameters
    count = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            count += list_mult(m.weight.shape)
            if hasattr(m, 'bias'):
                count += list_mult(m.bias.shape)
        elif isinstance(m, StochasticLayer):
            count += m.weights_count
    return count


#  -------------------------------------------------------------------------------------------
#  Main function
#  -------------------------------------------------------------------------------------------
def get_model(device,
              log_var_init,
              input_size,
              output_size):
    # Define default layers functions
    def linear_layer(input_size, output_size, use_bias=True):
        return StochasticLinear(input_size,
                                output_size,
                                log_var_init,
                                use_bias)

    model = FcNet3(linear_layer,
                   input_size,
                   output_size)

    # Move model to device (GPU\CPU):
    model.to(device)
    # DEBUG check: [(x[0], x[1].device) for x in model.named_parameters()]

    # init model:
    init_layers(model, log_var_init)

    model.weights_count = count_weights(model)

    # # For debug: set the STD of epsilon variable for re-parametrization trick (default=1.0)
    # if hasattr(prm, 'override_eps_std'):
    #     model.set_eps_std(prm.override_eps_std)  # debug
    return model


#  -------------------------------------------------------------------------------------------
#   Base class for all stochastic models
# -------------------------------------------------------------------------------------------
class general_model(nn.Module):
    def __init__(self):
        super(general_model, self).__init__()

    def set_eps_std(self, eps_std):
        old_eps_std = None
        for m in self.modules():
            if isinstance(m, StochasticLayer):
                old_eps_std = m.set_eps_std(eps_std)
        return old_eps_std

    def _init_weights(self, log_var_init):
        init_layers(self, log_var_init)


# -------------------------------------------------------------------------------------------
# Models collection
# -------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------
#  3-hidden-layer Fully-Connected Net
# -------------------------------------------------------------------------------------------
class FcNet3(general_model):
    def __init__(self,
                 linear_layer,
                 input_size,
                 output_size):
        super(FcNet3, self).__init__()
        self.layers_names = ('FC1', 'FC2', 'FC_out')

        input_size = input_size
        output_size = output_size
        # input_size = input_shape[0] * input_shape[1] * input_shape[2]

        self.input_size = input_size
        n_hidden1 = 100
        n_hidden2 = 100
        self.fc1 = linear_layer(input_size, n_hidden1)
        self.fc2 = linear_layer(n_hidden1, n_hidden2)
        self.fc_out = linear_layer(n_hidden2, output_size)

        # self._init_weights(log_var_init)  # Initialize weights

    def forward(self, x):
        x = x.view(-1, self.input_size)  # flatten image
        x = F.elu(self.fc1(x))
        x = F.elu(self.fc2(x))
        x = self.fc_out(x)
        # scale = torch.exp(torch.clamp(params['sigma'], min=self.min_log_std))
        # return Independent(Normal(loc=mu, scale=scale), 1)
        return x

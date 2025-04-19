import torch
from torch import nn
import numpy as np
from collections import OrderedDict
import math


class BatchLinear(nn.Linear):
    '''
    This is a linear transformation implemented manually. It also allows maually input parameters.
    for initialization, (in_features, out_features) needs to be provided.
    weight is of shape (out_features*in_features)
    bias is of shape (out_features)

    '''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = torch.matmul(input, weight.permute(*[i for i in range(len(weight.shape) - 2)], -1, -2))

        if not bias == None:
            output += bias.unsqueeze(-2)

        return output


class MLP_liststyle(nn.Module):
    '''
    MLP with different activation functions
    '''

    def __init__(self, in_features, out_features, hidden_features_list,
                 outermost_linear=True, nonlinearity='relu', weight_init=None, output_mode='single',
                 premap_mode=None, bias=True, **kwargs):
        super().__init__()
        self.premap_mode = premap_mode
        if not self.premap_mode == None:
            self.premap_layer = FeatureMapping(in_features, mode=premap_mode, **kwargs)
            in_features = self.premap_layer.dim  # update the nf in features

        self.first_layer_init = None
        self.output_mode = output_mode
        num_hidden_layers = len(hidden_features_list) - 1
        # Dictionary that maps nonlinearity name to the respective function, initialization, and, if applicable,
        # special first-layer initialization scheme
        # different layers has different initialization schemes:
        nls_and_inits = {'sine': (Sine(), sine_init, first_layer_sine_init),
                         # act name, init func, first layer init func
                         'relu': (nn.ReLU(inplace=True), init_weights_normal, None),
                         'sigmoid': (nn.Sigmoid(), init_weights_xavier, None),
                         'tanh': (nn.Tanh(), init_weights_xavier, None),
                         'selu': (nn.SELU(inplace=True), init_weights_selu, None),
                         'softplus': (nn.Softplus(), init_weights_normal, None),
                         'elu': (nn.ELU(inplace=True), init_weights_elu, None),
                         'swish': (Swish(), init_weights_xavier, None),
                         }

        nl, nl_weight_init, first_layer_init = nls_and_inits[nonlinearity]

        if weight_init is not None:  # Overwrite weight init if passed
            self.weight_init = weight_init
        else:
            self.weight_init = nl_weight_init  # those are default init funcs

        self.net = []

        # append the first layer
        self.net.append(nn.Sequential(
            BatchLinear(in_features, hidden_features_list[0], bias=bias), nl
        ))

        # append the hidden layers
        for i in range(num_hidden_layers):
            self.net.append(nn.Sequential(
                BatchLinear(hidden_features_list[i], hidden_features_list[i + 1], bias=bias), nl
            ))

        # append the last layer
        if outermost_linear:
            self.net.append(nn.Sequential(BatchLinear(hidden_features_list[-1], out_features, bias=bias)))
        else:
            self.net.append(nn.Sequential(
                BatchLinear(hidden_features_list[-1], out_features, bias=bias), nl
            ))

        # put them as a meta sequence
        self.net = nn.Sequential(*self.net)

        if self.weight_init is not None:
            self.net.apply(self.weight_init)

        if first_layer_init is not None:  # Apply special initialization to first layer, if applicable.
            self.net[0].apply(first_layer_init)

    def forward(self, x):
        # propagate through the nf net, implementing SIREN
        if not self.premap_mode == None:
            x = self.premap_layer(x)

        if self.output_mode == 'single':
            output = self.net(x)
            return output

        elif self.output_mode == 'double':
            x = x.clone().detach().requires_grad_(True)
            output = self.net(x)
            return output, x


########################
# define all activation functions that are not in pytorch library.
########################
class Swish(nn.Module):
    def __init__(self):
        super().__init__()
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        return x * self.Sigmoid(x)


class Sine(nn.Module):
    def __init__(self, w0=30):
        self.w0 = w0
        super().__init__()

    def forward(self, input):
        return torch.sin(self.w0 * input)


########################
# Initialization methods
########################
def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    # grab from upstream pytorch branch and paste here for now
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def init_weights_trunc_normal(m):
    # For PINNet, Raissi et al. 2019
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            fan_in = m.weight.size(1)
            fan_out = m.weight.size(0)
            std = math.sqrt(2.0 / float(fan_in + fan_out))
            mean = 0.
            # initialize with the same behavior as tf.truncated_normal
            # "The generated values follow a normal distribution with specified mean and
            # standard deviation, except that values whose magnitude is more than 2
            # standard deviations from the mean are dropped and re-picked."
            _no_grad_trunc_normal_(m.weight, mean, std, -2 * std, 2 * std)


def init_weights_uniform(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.uniform_(m.weight, -1 / num_input, 1 / num_input)


def init_weights_uniform_mfn(m, weight_scale=1.):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.uniform_(m.weight, -math.sqrt(weight_scale / num_input), math.sqrt(weight_scale / num_input))


def init_weights_uniform_siren_scale(m, scale=1e-2):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.uniform_(m.weight, -math.sqrt(6 / num_input) * scale, math.sqrt(6 / num_input) * scale)


def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            # print('applying normal init')
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1 / math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277) / math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def init_weights_const(m, val=0.):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.constant_(m.weight, val)


def sine_init(m, w0=30):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            # print('applying normal init')
            num_input = m.weight.size(-1)
            # See supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-np.sqrt(6 / num_input) / w0, np.sqrt(6 / num_input) / w0)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of factor 30
            m.weight.uniform_(-1 / num_input, 1 / num_input)


def init_bias_uniform(m):
    with torch.no_grad():
        if hasattr(m, 'bias'):
            num_input = m.weights.size(-1)
            m.bias.uniform_(-1 / num_input, 1 / num_input)


def init_bias_uniform_sqrt(m):
    with torch.no_grad():
        if hasattr(m, 'bias'):
            num_input = m.weights.size(-1)
            m.bias.uniform_(-1 / math.sqrt(num_input), 1 / math.sqrt(num_input))

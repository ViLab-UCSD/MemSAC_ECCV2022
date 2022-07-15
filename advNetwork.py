import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable

class AdversarialNetwork(nn.Module):
    def __init__(self, in_feature, hidden_size):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()
        self.apply(init_weights)
        self.iter_num = 0
        self.alpha = 10
        self.low = 0.0
        self.high = 1.0
        self.max_iter = 10000.0

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha, self.max_iter)
        x = x * 1.0
        x.register_hook(grl_hook(coeff))
        x = self.ad_layer1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = self.sigmoid(y)
        return y

    def output_num(self):
        return 1
    def get_parameters(self):
        return [{"params":self.parameters(), "lr_mult":10, 'decay_mult':2}]


class AdversarialLayer(torch.autograd.Function):
    iter_num = 0
    max_iter = 12500
    @staticmethod
    def forward(ctx, input):
        #self.iter_num += 1
#         ctx.save_for_backward(iter_num, max_iter)
        AdversarialLayer.iter_num += 1
        return input * 1.0

    @staticmethod
    def backward(ctx, gradOutput):
        alpha = 10
        low = 0.0
        high = 1.0
        lamb = 2.0
        iter_num, max_iter = AdversarialLayer.iter_num, AdversarialLayer.max_iter 
        # print('iter_num {}'.format(iter_num))
        coeff = np.float(lamb * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)
        return -coeff * gradOutput

# class AdversarialLayer(torch.autograd.Function):
#     def __init__(self, max_iter):
#         self.iter_num = 0
#         self.alpha = 10
#         self.low = 0.0
#         self.high = 1.0
#         self.max_iter = max_iter
#         self.lamb = 2.0
  
   
#     def forward(self, input):
#         self.iter_num += 1
#         return input * 1.0

   
#     def backward(self, gradOutput):
#         coeff = np.float(self.lamb * (self.high - self.low) / (1.0 + np.exp(-self.alpha*self.iter_num / self.max_iter)) - (self.high - self.low) + self.low)
#         return -coeff * gradOutput

class RMANLayer(torch.autograd.Function):
    def __init__(self, input_dim_list=[], output_dim=1024):
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [Variable(torch.randn(input_dim_list[i], output_dim)) for i in xrange(self.input_num)]
        for val in self.random_matrix:
            val.requires_grad = False
    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in xrange(self.input_num)]
        return_list[0] = return_list[0] / float(self.output_dim)
        return return_list
    def cuda(self):
        self.random_matrix = [val.cuda() for val in self.random_matrix]

class SilenceLayer(torch.autograd.Function):
    def __init__(self):
        pass
    def forward(self, input):
        return input

    def backward(self, gradOutput):
        return 0 * gradOutput


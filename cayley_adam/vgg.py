import torch
import torch.nn.functional as F
from utils import conv_params, linear_params, bnparams, bnstats, \
        flatten_params, flatten_stats
import numpy as np
   
    
cfg = {
        '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        '13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        '16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
        '19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    }

        
def vgg(depth, width, num_classes):
    assert depth in [11, 13, 16, 19]
    depth_str = str(int(depth))
    _cfg = cfg[depth_str]
    
    
    def gen_feature_params():
        in_channels = 3
        dic = {}
        for i in range(len(_cfg)):
            if not _cfg[i] == 'M':
                dic['conv{0}'.format(i)] = conv_params(in_channels, _cfg[i], 3)
                dic['bn{0}'.format(i)] = bnparams(_cfg[i])
                in_channels = _cfg[i]
                
        return dic
        
        
    def gen_feature_stats():
        dic = {}
        for i in range(len(_cfg)):
            if not _cfg[i] == 'M':
                dic['bn{0}'.format(i)] = bnstats(_cfg[i])
        return dic
        
       
    def gen_classifier_params():
        return {
            'fc1': linear_params(512, 4096),
            'fc2': linear_params(4096, 4096),
            'fc3': linear_params(4096, num_classes),
            }
        

    def feature(input, params, stats, mode):
        out = input
        for i in range(len(_cfg)):
            if _cfg[i] == 'M':
                out = F.max_pool2d(out, 2, 2, 0)
            else:
                out = F.conv2d(out, params['conv{0}'.format(i)], padding=1)
                out = activation(out, params, stats, 'bn{0}'.format(i), mode)
                
        return out
                
                
    def activation(x, params, stats, base, mode):
        return F.relu(F.batch_norm(x, weight=params[base + '.weight'],
                                   bias=params[base + '.bias'],
                                   running_mean=stats[base + '.running_mean'],
                                   running_var=stats[base + '.running_var'],
                                   training=mode, momentum=0.1, eps=1e-5), inplace=True)
            
    
    def classifier(input, params, num_classes, mode):
        out = F.relu(F.linear(input, params['fc1.weight'], params['fc1.bias']),
                              inplace=False)       
#         out = F.dropout(out, p=0.3, training=mode)
        out = F.relu(F.linear(out, params['fc2.weight'], params['fc2.bias']),
                              inplace=False)
#         out = F.dropout(out, p=0.3, training=mode)
        out = F.linear(out, params['fc3.weight'], params['fc3.bias'])
    
        return out
    
    
    params = {**gen_feature_params(), **gen_classifier_params()}
    stats = gen_feature_stats()
    

    flat_params = flatten_params(params)
    flat_stats = flatten_stats(stats)
    
    
    def f(input, params, stats, mode):
        out = feature(input, params, stats, mode)
        out = out.view(-1, np.prod(out.size()[1:])).contiguous()
        out = classifier(out, params, num_classes, mode)
        
        return out
        
    
    return f, flat_params, flat_stats

import torch
import torch.nn.functional as F
from utils import conv_params, linear_params, bnparams, bnstats, \
        flatten_params, flatten_stats


def resnet(depth, width, num_classes):
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = (depth - 4) // 6
    widths = torch.Tensor([16, 32, 64]).mul(width).int()

    def gen_block_params(ni, no):
        return {
            'conv0': conv_params(ni, no, 3),
            'conv1': conv_params(no, no, 3),
            'bn0': bnparams(ni),
            'bn1': bnparams(no),
            'convdim': conv_params(ni, no, 1) if ni != no else None,
        }

    def gen_group_params(ni, no, count):
        return {'block%d' % i: gen_block_params(ni if i == 0 else no, no)
                for i in range(count)}

    def gen_group_stats(ni, no, count):
        return {'block%d' % i: {'bn0': bnstats(ni if i == 0 else no), 'bn1': bnstats(no)}
                for i in range(count)}

    params = {
        'conv0': conv_params(3,16,3),
        'group0': gen_group_params(16, widths[0], n),
        'group1': gen_group_params(widths[0], widths[1], n),
        'group2': gen_group_params(widths[1], widths[2], n),
        'bn': bnparams(widths[2]),
        'fc': linear_params(widths[2], num_classes),
    }

    stats = {
        'group0': gen_group_stats(16, widths[0], n),
        'group1': gen_group_stats(widths[0], widths[1], n),
        'group2': gen_group_stats(widths[1], widths[2], n),
        'bn': bnstats(widths[2]),
    }

    flat_params = flatten_params(params)
    flat_stats = flatten_stats(stats)

    def activation(x, params, stats, base, mode):
        return F.relu(F.batch_norm(x, weight=params[base + '.weight'],
                                   bias=params[base + '.bias'],
                                   running_mean=stats[base + '.running_mean'],
                                   running_var=stats[base + '.running_var'],
                                   training=mode, momentum=0.1, eps=1e-5), inplace=True)

    def block(x, params, stats, base, mode, stride):
        o1 = activation(x, params, stats, base + '.bn0', mode)
        y = F.conv2d(o1, params[base + '.conv0'], stride=stride, padding=1)
        o2 = activation(y, params, stats, base + '.bn1', mode)
        o2d = F.dropout(o2, p=0.3, training=mode)
        z = F.conv2d(o2d, params[base + '.conv1'], stride=1, padding=1)
        if base + '.convdim' in params:
            return z + F.conv2d(o1, params[base + '.convdim'], stride=stride)
        else:
            return z + x

    def group(o, params, stats, base, mode, stride):
        for i in range(n):
            o = block(o, params, stats, '%s.block%d' % (base,i), mode, stride if i == 0 else 1)
        return o

    def f(input, params, stats, mode):
        assert input.get_device() == params['conv0'].get_device()
        x = F.conv2d(input, params['conv0'], padding=1)
        g0 = group(x, params, stats, 'group0', mode, 1)
        g1 = group(g0, params, stats, 'group1', mode, 2)
        g2 = group(g1, params, stats, 'group2', mode, 2)
        o = activation(g2, params, stats, 'bn', mode)
        o = F.avg_pool2d(o, 8, 1, 0)
        o = o.view(o.size(0), -1)
        o = F.linear(o, params['fc.weight'], params['fc.bias'])
        return o

    return f, flat_params, flat_stats
 

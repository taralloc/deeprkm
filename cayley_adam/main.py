"""
    PyTorch training code for Wide Residual Networks:
    http://arxiv.org/abs/1605.07146

    2019 Jun Li
"""

import argparse
import os
import json
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torch.optim
import torch.utils.data
import cvtransforms as T
import torchvision.datasets as datasets
from torch.autograd import Variable
import torch.nn.functional as F
import torchnet as tnt
from torchnet.engine import Engine
from utils import cast, data_parallel
import torch.backends.cudnn as cudnn
from resnet import resnet
from vgg import vgg

import grassmann_optimizer
import stiefel_optimizer
from gutils import unit, qr_retraction

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Wide Residual Networks')
# Model options
parser.add_argument('--model', default='resnet', type=str)
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--width', default=1, type=float)
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--dataroot', default='/scratch/liju2/data/', type=str)
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--groups', default=1, type=int)
parser.add_argument('--nthread', default=4, type=int)

# Training options
parser.add_argument('--batchSize', default=128, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--lrg', default=0.1, type=float)
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--weightDecay', default=0.0005, type=float)
parser.add_argument('--bnDecay', default=0, type=float)
parser.add_argument('--omega', default=0.1, type=float)
parser.add_argument('--grad_clip', default=0.1, type=float)
parser.add_argument('--epoch_step', default='[60,120,160]', type=str,
                    help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--resume', default='', type=str)
parser.add_argument('--optim_method', default='SGD', type=str)
parser.add_argument('--randomcrop_pad', default=4, type=float)

# Device options
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--save', default='', type=str,
                    help='save parameters and logs in this folder')
parser.add_argument('--ngpu', default=1, type=int,
                    help='number of GPUs to use for training')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')

def create_dataset(opt, mode):
    if opt.dataset == 'CIFAR10':
        mean = [125.3, 123.0, 113.9]
        std = [63.0, 62.1, 66.7]
    elif opt.dataset =='CIFAR100':
        mean = [129.3, 124.1, 112.4]
        std = [68.2, 65.4, 70.4]
    else:
        mean = [0, 0, 0]
        std = [1.0, 1.0, 1.0]


    convert = tnt.transform.compose([
        lambda x: x.astype(np.float32),
        T.Normalize(mean, std),
        lambda x: x.transpose(2,0,1).astype(np.float32),
        torch.from_numpy,
    ])

    train_transform = tnt.transform.compose([
        T.RandomHorizontalFlip(),
        T.Pad(opt.randomcrop_pad, cv2.BORDER_REFLECT),
        T.RandomCrop(32),
        convert,
    ])

    ds = getattr(datasets, opt.dataset)(opt.dataroot, train=mode, download=True)
    smode = 'train' if mode else 'test'
    ds = tnt.dataset.TensorDataset([getattr(ds, smode + '_data'),
                                    getattr(ds, smode + '_labels')])
    return ds.transform({0: train_transform if mode else convert})


def main():
    opt = parser.parse_args()
    print('parsed options:', vars(opt))
    epoch_step = json.loads(opt.epoch_step)
    num_classes = 100 if opt.dataset == 'CIFAR100' else 10

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id
    # to prevent opencv from initializing CUDA in workers
    torch.randn(8).cuda()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    def create_iterator(mode):
        ds = create_dataset(opt, mode)
        return ds.parallel(batch_size=opt.batchSize, shuffle=mode,
                           num_workers=opt.nthread, pin_memory=True)

    train_loader = create_iterator(True)
    test_loader = create_iterator(False)
    
    if opt.model == 'resnet':
        model = resnet
    elif opt.model == 'vgg':
        model = vgg

    f, params, stats = model(opt.depth, opt.width, num_classes)

    key_g = []
    if opt.optim_method in ['SGDG', 'AdamG', 'Cayley_SGD', 'Cayley_Adam'] :
        param_g = []
        param_e0 = []
        param_e1 = []

        for key, value in params.items():
            if 'conv' in key and value.size()[0] <= np.prod(value.size()[1:]):
                param_g.append(value)
                key_g.append(key)
                if opt.optim_method in ['SGDG', 'AdamG']:
                    # initlize to scale 1
                    unitp, _ = unit(value.data.view(value.size(0), -1)) 
                    value.data.copy_(unitp.view(value.size()))
                elif opt.optim_method in ['Cayley_SGD', 'Cayley_Adam']:
                    # initlize to orthogonal matrix
                    q = qr_retraction(value.data.view(value.size(0), -1)) 
                    value.data.copy_(q.view(value.size()))               
            elif 'bn' in key or 'bias' in key:
                param_e0.append(value)
            else:
                param_e1.append(value)

    def create_optimizer(opt, lr, lrg):
        print('creating optimizer with lr = ', lr, ' lrg = ', lrg)
        if opt.optim_method == 'SGD':
            return torch.optim.SGD(params.values(), lr, 0.9, weight_decay=opt.weightDecay)

        elif opt.optim_method == 'SGDG':
            dict_g = {'params':param_g,'lr':lrg,'momentum':0.9,'grassmann':True, 'omega':opt.omega, 'grad_clip':opt.grad_clip}
            dict_e0 = {'params':param_e0,'lr':lr,'momentum':0.9,'grassmann':False,'weight_decay':opt.bnDecay,'nesterov':True}
            dict_e1 = {'params':param_e1,'lr':lr,'momentum':0.9,'grassmann':False,'weight_decay':opt.weightDecay,'nesterov':True}
            return grassmann_optimizer.SGDG([dict_g, dict_e0, dict_e1])

        elif opt.optim_method == 'AdamG':
            dict_g = {'params':param_g,'lr':lrg,'momentum':0.9,'grassmann':True, 'omega':opt.omega, 'grad_clip':opt.grad_clip}
            dict_e0 = {'params':param_e0,'lr':lr,'momentum':0.9,'grassmann':False,'weight_decay':opt.bnDecay,'nesterov':True}
            dict_e1 = {'params':param_e1,'lr':lr,'momentum':0.9,'grassmann':False,'weight_decay':opt.weightDecay,'nesterov':True}
            return grassmann_optimizer.AdamG([dict_g, dict_e0, dict_e1])
        
        elif opt.optim_method == 'Cayley_SGD':
            dict_g = {'params':param_g,'lr':lrg,'momentum':0.9,'stiefel':True}
            dict_e0 = {'params':param_e0,'lr':lr,'momentum':0.9,'stiefel':False,'weight_decay':opt.bnDecay,'nesterov':True}
            dict_e1 = {'params':param_e1,'lr':lr,'momentum':0.9,'stiefel':False,'weight_decay':opt.weightDecay,'nesterov':True}
            return stiefel_optimizer.SGDG([dict_g, dict_e0, dict_e1])
        
        elif opt.optim_method == 'Cayley_Adam':
            dict_g = {'params':param_g,'lr':lrg,'momentum':0.9,'stiefel':True}
            dict_e0 = {'params':param_e0,'lr':lr,'momentum':0.9,'stiefel':False,'weight_decay':opt.bnDecay,'nesterov':True}
            dict_e1 = {'params':param_e1,'lr':lr,'momentum':0.9,'stiefel':False,'weight_decay':opt.weightDecay,'nesterov':True}
            return stiefel_optimizer.AdamG([dict_g, dict_e0, dict_e1])

    optimizer = create_optimizer(opt, opt.lr, opt.lrg)

    epoch = 0
    if opt.resume != '':
        state_dict = torch.load(opt.resume)
        epoch = state_dict['epoch']
        params_tensors, stats = state_dict['params'], state_dict['stats']
        for k, v in list(params.items()):
            v.data.copy_(params_tensors[k])
        optimizer.load_state_dict(state_dict['optimizer'])

    print('\nParameters:')
    kmax = max(len(key) for key in params.keys())
    for i, (key, v) in enumerate(params.items()):
        print(str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.size())).ljust(23), torch.typename(v.data), end='')
        print(' on G(1,n)' if key in key_g else '')

    print('\nAdditional buffers:')
    kmax = max(len(key) for key in stats.keys())
    for i, (key, v) in enumerate(stats.items()):
        print(str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.size())).ljust(23), torch.typename(v))

    n_training_params = sum(p.numel() for p in params.values())
    n_parameters = sum(p.numel() for p in params.values()) + sum(p.numel() for p in stats.values())
    print('Total number of parameters:', n_parameters, '(%d)'%n_training_params)

    meter_loss = tnt.meter.AverageValueMeter()
    classacc = tnt.meter.ClassErrorMeter(accuracy=True)
    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')

    if not os.path.exists(opt.save):
        os.mkdir(opt.save)

    def h(sample):
        inputs = Variable(cast(sample[0], opt.dtype))
        targets = Variable(cast(sample[1], 'long'))
        y = data_parallel(f, inputs, params, stats, sample[2], np.arange(opt.ngpu))
        return F.cross_entropy(y, targets), y

    def log(t, state):
        torch.save(dict(params={k: v.data for k, v in list(params.items())},
                        stats=stats,
                        optimizer=state['optimizer'].state_dict(),
                        epoch=t['epoch']),
                   open(os.path.join(opt.save, 'model.pt7'), 'wb'))
        z = vars(opt).copy(); z.update(t)
        logname = os.path.join(opt.save, 'log.txt')
        with open(logname, 'a') as f:
            f.write('json_stats: ' + json.dumps(z) + '\n')
        print(z)

    def on_sample(state):
        state['sample'].append(state['train'])

    def on_forward(state):
        classacc.add(state['output'].data, torch.LongTensor(state['sample'][1]))
        meter_loss.add(state['loss'].data.item())

    def on_start(state):
        state['epoch'] = epoch

    def on_start_epoch(state):
        classacc.reset()
        meter_loss.reset()
        timer_train.reset()
        state['iterator'] = tqdm(train_loader)

        epoch = state['epoch'] + 1
        if epoch in epoch_step:
            power=sum(epoch>=i for i in epoch_step)
            lr = opt.lr*pow(opt.lr_decay_ratio, power)
            lrg = opt.lrg*pow(opt.lr_decay_ratio, power)
            state['optimizer'] = create_optimizer(opt, lr, lrg)

    def on_end_epoch(state):
        train_loss = meter_loss.value()
        train_acc = classacc.value()
        train_time = timer_train.value()
        meter_loss.reset()
        classacc.reset()
        timer_test.reset()

        engine.test(h, test_loader)

        test_acc = classacc.value()[0]
        print(log({
            "train_loss": train_loss[0],
            "train_acc": train_acc[0],
            "test_loss": meter_loss.value()[0],
            "test_acc": test_acc,
            "epoch": state['epoch'],
            "num_classes": num_classes,
            "n_parameters": n_parameters,
            "train_time": train_time,
            "test_time": timer_test.value(),
        }, state))
        print('==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m' % \
                (opt.save, state['epoch'], opt.epochs, test_acc))

    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_start'] = on_start
    engine.train(h, train_loader, opt.epochs, optimizer)


if __name__ == '__main__':
    main()

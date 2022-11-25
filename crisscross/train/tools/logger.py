# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""

# this script is taken from https://github.com/facebookresearch/AVID-CMA

import datetime
import sys
import torch
from torch import distributed as dist
from collections import deque

class Logger(object):
    def __init__(self, quiet=False, log_fn=None, rank=0, prefix=""):
        self.rank = rank if rank is not None else 0
        self.quiet = quiet
        self.log_fn = log_fn

        self.prefix = ""
        if prefix:
            self.prefix = prefix + ' | '

        self.file_pointers = []
        if self.rank == 0:
            if self.quiet:
                open(log_fn, 'a').close() # change to w --> a as to append during resume mode

    def add_line(self, content):
        if self.rank == 0:
            msg = self.prefix+content
            if self.quiet:
                fp = open(self.log_fn, 'a')
                fp.write(msg+'\n')
                fp.flush()
                fp.close()
            else:
                print(msg)
                sys.stdout.flush()


class ProgressMeter(object):
    def __init__(self, num_batches, meters, phase, epoch=None, logger=None, tb_writter=None):
        self.batches_per_epoch = num_batches
        self.batch_fmtstr = self._get_batch_fmtstr(epoch, num_batches)
        self.meters = meters
        self.phase = phase
        self.epoch = epoch
        self.logger = logger
        self.tb_writter = tb_writter

    def display(self, batch):
        step = self.epoch * self.batches_per_epoch + batch
        date = str(datetime.datetime.now())
        entries = ['{} | {} {}'.format(date, self.phase, self.batch_fmtstr.format(batch))]
        entries += [str(meter) for meter in self.meters]
        if self.logger is None:
            print('\t'.join(entries))
        else:
            self.logger.add_line('\t'.join(entries))

        if self.tb_writter is not None:
            for meter in self.meters:
                self.tb_writter.add_scalar('{}/{}'.format(self.phase, meter.name), meter.val, step)

    def _get_batch_fmtstr(self, epoch, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        epoch_str = '[{}]'.format(epoch) if epoch is not None else ''
        return epoch_str+'[' + fmt + '/' + fmt.format(num_batches) + ']'

    def synchronize_meters(self, cur_gpu):
        metrics = torch.tensor([m.avg for m in self.meters]).cuda(cur_gpu)
        metrics_gather = [torch.ones_like(metrics) for _ in range(dist.get_world_size())]
        dist.all_gather(metrics_gather, metrics)

        metrics = torch.stack(metrics_gather).float().mean(0).cpu().numpy()
        for meter, m in zip(self.meters, metrics):
            meter.avg = m

    
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', window_size=0):
        self.name = name
        self.fmt = fmt
        self.window_size = window_size
        self.reset()

    def reset(self):
        if self.window_size > 0:
            self.q = deque(maxlen=self.window_size)
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        if self.window_size > 0:
            self.q.append((val, n))
            self.count = sum([n for v, n in self.q])
            self.sum = sum([v * n for v, n in self.q])
        else:
            self.sum += val * n
            self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
  





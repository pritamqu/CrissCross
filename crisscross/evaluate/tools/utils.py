# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""

import os
import torch
from tools import paths
import numpy as np
import warnings
from datetime import datetime
import random


def resume_model(args, model, optimizer, lr_scheduler, amp, logger):
    
    start_epoch=0
    if args.resume:
        model_path = os.path.join(args.resume, "checkpoint.pth.tar")
        if os.path.isfile(model_path):
            logger.add_line("=> loading checkpoint '{}'".format(model_path))
            if args.gpu is None:
                checkpoint = torch.load(model_path)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(model_path, map_location=loc)
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            if amp is not None:
                amp.load_state_dict(checkpoint['amp'])
            
            logger.add_line("=> loaded checkpoint '{}' (epoch {})"
                  .format(model_path, checkpoint['epoch']))
        else:
            logger.add_line("=> no checkpoint found at '{}'".format(model_path))
            
    return model, optimizer, lr_scheduler, start_epoch, amp


def save_checkpoint(args, model, optimizer, lr_scheduler, epoch, amp, logger):
    # Save checkpoint
    model_path = os.path.join(args.ckpt_dir, "checkpoint.pth.tar")
    
    checkpoint = {'optimizer': optimizer.state_dict(), 
                'model': model.state_dict(), 
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch + 1}
    if amp is not None:
        checkpoint.update({'amp': amp.state_dict()})
    torch.save(checkpoint, model_path)
    print(f"Model saved to {model_path}")
    
    
def get_parent_dir(file, step=1):
    
    folder=file
    for k in range(step):
        folder = os.path.dirname(folder)
    return folder
    
def set_deterministic(seed):
    # seed by default is None 
    if seed is not None:
        print(f"Deterministic with seed = {seed}")
        random.seed(seed) 
        np.random.seed(seed) 
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True 
        torch.backends.cudnn.benchmark = False 
        warnings.warn('You have chosen to seed training. '
              'This will turn on the CUDNN deterministic setting, '
              'which can slow down your training considerably! '
              'You may see unexpected behavior when restarting '
              'from checkpoints.')

def sanity_check(args):
    
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:2"
    # for scinet drama
    if args.server == 'scinet':
        import cv2
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["OPENBLAS_NUM_THREADS"] = "1"
        cv2.setNumThreads(1)
    
    args.output_dir, args.data_dir = paths.my_paths(args.server, args.db)
        
    if args.job_id == '00':
        fmt = '%Y_%m_%d_%H_%M_%S'
        job_id = str(datetime.now().strftime(fmt))
        args.job_id = job_id
        
    args.config_dir = os.path.join(os.getcwd(), 'configs')
    args.config_file = os.path.join(os.getcwd(), 'configs', args.sub_dir, args.db, args.config_file + '.yaml')
    print('selected config file: ', args.config_file)
    
    if args.resume:
        args.job_id = args.resume
        args.resume = os.path.join(args.output_dir, args.parent_dir, args.sub_dir, args.resume, 'model')
        print('Resume path is: ', args.resume)
        
    args.output_dir = os.path.join(args.output_dir, args.parent_dir, args.sub_dir, args.job_id)
    args.ckpt_dir = os.path.join(args.output_dir, 'model')
    args.log_dir = os.path.join(args.output_dir, 'logs')
    
    
    return args 
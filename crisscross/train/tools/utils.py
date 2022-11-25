# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 20:47:48 2021

@author: pritam
"""
import os
import torch

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
    
# def scale_lr(batch_size, lr):
#     return lr * (batch_size / MIN_BATCH_SIZE)    
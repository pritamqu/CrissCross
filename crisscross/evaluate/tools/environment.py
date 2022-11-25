# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""

import os
import shutil
import torch
import numpy as np
import torch.distributed as dist
import datetime
from .logger import Logger

WANDB_ID = 'abcde' # change with your wandb account id.

def initialize_distributed_backend(args, ngpus_per_node):
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + args.gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        if args.rank == 0:
            print('Initialized dist mode')
    else:
        print('Not initialized dist mode')

    if args.rank == -1:
        args.rank = 0
    return args


def prep_environment(args, cfg):
    from torch.utils.tensorboard import SummaryWriter
    
    if args.rank == 0 and not args.resume:
        print(f'creating folders: \n{args.output_dir} \n{args.log_dir} \n{args.ckpt_dir}')
        args.log_dir = os.path.join(args.log_dir, cfg['name'])
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.log_dir, exist_ok=True)
        os.makedirs(args.ckpt_dir, exist_ok=True)
        shutil.copy2(args.config_file, args.log_dir)
        resume_id="auto"
    elif args.rank == 0 and args.resume:
        print(f'Resuming at folders: \n{args.output_dir} \n{args.log_dir} \n{args.ckpt_dir}')
        args.log_dir = os.path.join(args.log_dir, cfg['name'])
        resume_id=args.job_id
        
    log_fn = '{}/train.log'.format(args.log_dir)
    logger = Logger(quiet=args.quiet, log_fn=log_fn, rank=args.rank)
    logger.add_line(str(datetime.datetime.now()))
    if any(['SLURM' in env for env in list(os.environ.keys())]):
        logger.add_line("=" * 30 + "   SLURM   " + "=" * 30)
        for env in os.environ.keys():
            if 'SLURM' in env:
                logger.add_line('{:30}: {}'.format(env, os.environ[env]))

    logger.add_line("=" * 30 + "   Config   " + "=" * 30)
    def print_dict(d, ident=''):
        for k in d:
            if isinstance(d[k], dict):
                logger.add_line("{}{}".format(ident, k))
                print_dict(d[k], ident='  '+ident)
            else:
                logger.add_line("{}{}: {}".format(ident, k, str(d[k])))

    print_dict(cfg)
    logger.add_line("=" * 30 + "   Args   " + "=" * 30)
    for k in args.__dict__:
        logger.add_line('{:30} {}'.format(k, args.__dict__[k]))

    wandb_writter = None
    if cfg['progress']['wandb'] and args.rank == 0:
        import wandb
        wandb_writter = wandb.init(project=f"{args.parent_dir}-{args.sub_dir}", 
                                    entity=WANDB_ID,
                                    config={'cfg': cfg, 'args': vars(args)},
                                    dir=args.log_dir,
                                    id=args.job_id,
                                    name=cfg['name']+'_'+args.job_id,
                                    tags=[args.sub_dir, args.db, args.server],
                                    resume=resume_id,
                                    )
            
    tb_writter = None
    if cfg['progress']['log2tb'] and args.rank == 0:
        tb_writter = SummaryWriter(args.log_dir)
        
    return logger, tb_writter, wandb_writter


def distribute_model_to_cuda(models, args, batch_size, num_workers, ngpus_per_node):
    if ngpus_per_node == 0:
        return models, args, batch_size, num_workers
    squeeze = False
    if not isinstance(models, list):
        models = [models]
        squeeze = True

    for i in range(len(models)):
        if args.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                models[i].cuda(args.gpu)
                # we need to do this (find_unused_parameters=True) as we have stopgrad in siamese network.
                models[i] = torch.nn.parallel.DistributedDataParallel(models[i], device_ids=[args.gpu], find_unused_parameters=True)
                # models[i] = torch.nn.parallel.DistributedDataParallel(models[i], device_ids=[args.gpu])
            else:
                models[i].cuda()
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                # we need to do this (find_unused_parameters=True) as we have stopgrad in siamese network.
                models[i] = torch.nn.parallel.DistributedDataParallel(models[i], find_unused_parameters=True)
                # models[i] = torch.nn.parallel.DistributedDataParallel(models[i])
        elif args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            models[i] = models[i].cuda(args.gpu)
            # (technically not required, but easy fix while switching b/w debug mode)
            models[i] = torch.nn.DataParallel(models[i])
        else:
            # DataParallel will divide and allocate batch_size to all available GPUs
            models[i] = torch.nn.DataParallel(models[i]).cuda()

    if squeeze:
        models = models[0]

    if args.distributed and args.gpu is not None:
        # When using a single GPU per process and per
        # DistributedDataParallel, we need to divide the batch size
        # ourselves based on the total number of GPUs we have
        batch_size = int(batch_size / ngpus_per_node)
        num_workers = int((num_workers + ngpus_per_node - 1) / ngpus_per_node)

    return models, args, batch_size, num_workers

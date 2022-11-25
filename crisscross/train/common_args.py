import argparse
import os
import torch
import numpy as np
import torch
import random
import re 
import yaml
import shutil
import warnings
from datetime import datetime
from tools import paths
          

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


def get_args(mode='default'):
    
    # mode will take care specific arguments for specific cases
    
    parser = argparse.ArgumentParser()
    
    # some stuff
    parser.add_argument("--parent_dir", default="CrissCross", help="output folder name",)    
    parser.add_argument("--sub_dir", default="pretext", help="output folder name",)    
    parser.add_argument("--job_id", type=str, default='00', help="jobid=%j")
    parser.add_argument("--server", type=str, default="local", help="location of server", 
                        choices=["ingenuity", "vector", "local", "scinet"])
    parser.add_argument("--db", default="kinetics400", help="target db", 
                        choices=['kinetics400', 'audioset', 'kinetics_sound'])
    parser.add_argument('-c', '--config-file', type=str, help="config", default="crisscross")

    ## debug mode
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--debug_subset_size', type=int, default=2)
    
    ## dir stuff
    parser.add_argument('--data_dir', type=str, default='D:\\datasets\\Vision\\image')
    parser.add_argument("--output_dir", default="D:\\projects\\OUTPUTS", help="path where to save")
    parser.add_argument("--resume", default="", help="path where to resume")
    parser.add_argument('--log_dir', type=str, default=os.getenv('LOG'))
    parser.add_argument('--ckpt_dir', type=str, default=os.getenv('CHECKPOINT'))

    ## dist training stuff
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use., default to 0 while using 1 gpu')    
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--dist-url', default="env://", type=str, help='url used to set up distributed training, change to; "tcp://localhost:15475"')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    
    parser.add_argument("--checkpoint_path", default="", help="checkpoint_path for system restoration")
    parser.add_argument('--checkpoint_interval', default=3600, type=int, help='checkpoint_interval')
    
    args = parser.parse_args()
    args.mode = mode
    args = sanity_check(args)
    set_deterministic(args.seed)
    torch.backends.cudnn.benchmark = True 
       
    return args

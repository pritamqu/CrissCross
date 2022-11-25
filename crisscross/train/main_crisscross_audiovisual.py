import os
import time
import torch
import warnings
import torch.multiprocessing as mp
import yaml
from common_args import get_args
from datasets.augmentations import get_aud_aug, get_vid_aug
from datasets import get_dataset, dataloader, FetchSubset
from tools import environment as environ
from models import get_model
from optimizers import get_optimizer_av, CosineLRAV2
from tools import AverageMeter, ProgressMeter
# from checkpointing import create_or_restore_training_state, commit_state
from tools.utils import resume_model, save_checkpoint


def main(args):
    
    cfg = yaml.safe_load(open(args.config_file))
    print(args)
    print(cfg)
    
    if args.server == 'vector':
        cfg['progress']['log2tb']=False
    
    if args.debug:
        cfg, args = environ.set_debug_mode(cfg, args)

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    print(f'number of gpus per node {ngpus_per_node} - Rank {args.rank}')
    
    if args.multiprocessing_distributed:
        print('mp.spawn calling main_worker')
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args, cfg))
    else:
        print('direct calling main_worker')
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args, cfg)


def main_worker(gpu, ngpus_per_node, args, cfg):

    args.gpu = gpu
    
    # Setup environment
    args = environ.initialize_distributed_backend(args, ngpus_per_node)
    logger, tb_writter, wandb_writter = environ.prep_environment_ddp(args, cfg)   

    # define model
    model = get_model(cfg['model'])
    if args.distributed and cfg['sync_bn']:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda(args.gpu)
    
    # define optimizer
    optimizer = get_optimizer_av(name= cfg['optimizer']['name'],
                              model=model, lr=1e-3, # this is overwritten by lr-scheduler
                              # lr=cfg['optimizer']['lr']['base_lr'], 
                              momentum=cfg['optimizer']['momentum'], 
                              weight_decay=cfg['optimizer']['weight_decay'],
                              betas=cfg['optimizer']['betas'])
    
    ## applying larc to stabilize large batch training
    if cfg['optimizer']['apply_larc']:
        from optimizers import LARC
        optimizer = LARC(optimizer, 
                         trust_coefficient=cfg['optimizer']['larc_trust_coefficient'], 
                         clip=cfg['optimizer']['larc_clip'], 
                         eps=float(cfg['optimizer']['larc_eps']))
        
    
    # use apex for mixed precision training
    if cfg['apex']:
        amp = torch.cuda.amp.GradScaler() 
    else:
        amp=None
        
    # wrap in ddp
    model, args, cfg['dataset']['batch_size'], cfg['num_workers'] = environ.distribute_model_to_cuda(models=model, 
                                                                         args=args, 
                                                                         batch_size=cfg['dataset']['batch_size'], 
                                                                         num_workers=cfg['num_workers'], 
                                                                         ngpus_per_node=ngpus_per_node)
        
    logger.add_line(f"new batch size: {cfg['dataset']['batch_size']} - num_workers: {cfg['num_workers']}")


    # transformations
    vid_transformations = get_vid_aug(name=cfg['dataset']['vid_transform'],
                                    crop_size=cfg['dataset']['crop_size'],
                                    num_frames=cfg['dataset']['clip_duration']*cfg['dataset']['video_fps'],
                                    mode=cfg['dataset']['train']['aug_mode'],                                    
                                    aug_kwargs=cfg['dataset']['train']['vid_aug_kwargs'])

    aud_transformations = get_aud_aug(name=cfg['dataset']['aud_transform'],
                                    audio_fps=cfg['dataset']['audio_fps'],
                                    n_fft=cfg['dataset']['n_fft'],
                                    n_mels=cfg['dataset']['n_mels'],
                                    duration=cfg['dataset']['audio_clip_duration'],
                                    hop_length=cfg['dataset']['hop_length'],
                                    mode=cfg['dataset']['train']['aug_mode'],                                    
                                    aug_kwargs=cfg['dataset']['train']['aud_aug_kwargs'])

    # dataset
    train_dataset = get_dataset(root=args.data_dir,
                                dataset_kwargs=cfg['dataset'],
                                video_transform=vid_transformations, 
                                audio_transform=aud_transformations,
                                split='train')
           
    if args.debug:
        train_dataset = FetchSubset(train_dataset, cfg['dataset']['batch_size']*ngpus_per_node*args.debug_subset_size)
           
    # dataloader
    train_loader = dataloader.make_dataloader(dataset=train_dataset, 
                                              batch_size=cfg['dataset']['batch_size'],
                                              use_shuffle=cfg['dataset']['train']['use_shuffle'],
                                              drop_last=cfg['dataset']['train']['drop_last'],
                                              num_workers=cfg['num_workers'],
                                              distributed=args.distributed)
    
     
    # define lr scheduler
    if cfg['optimizer']['lr']['name'] == 'cosine':
        lr_scheduler = CosineLRAV2(optimizer=optimizer,
                                warmup_epochs=cfg['optimizer']['lr']['warmup_epochs'],
                                warmup_lr=cfg['optimizer']['lr']['warmup_lr'],
                                num_epochs=cfg['optimizer']['num_epochs'], 
                                iter_per_epoch=len(train_loader), 
                                vid_base_lr=cfg['optimizer']['lr']['vid_base_lr'],
                                vid_final_lr=cfg['optimizer']['lr']['vid_final_lr'],
                                aud_base_lr=cfg['optimizer']['lr']['aud_base_lr'],
                                aud_final_lr=cfg['optimizer']['lr']['aud_final_lr'],
                                constant_vid_predictor_lr=cfg['optimizer']['lr']['constant_vid_predictor_lr'],
                                constant_aud_predictor_lr=cfg['optimizer']['lr']['constant_aud_predictor_lr'],
                                vid_predictor_lr=cfg['optimizer']['lr']['vid_predictor_lr'],
                                aud_predictor_lr=cfg['optimizer']['lr']['aud_predictor_lr'],
                                )
        
    else:
        NotImplementedError(print(f"{cfg['optimizer']['lr']['name']} not implemented"))
        
       
    ## try loading from checkpoint
    ## manual resume
    model, optimizer, lr_scheduler, start_epoch, amp = resume_model(args, model, optimizer, lr_scheduler, amp, logger)
    ## manual resume + system resume   
    # model, optimizer, lr_scheduler, start_epoch, amp, rng = create_or_restore_training_state(args, model, optimizer, lr_scheduler, logger, amp)
         
    # Start training
    end_epoch = cfg['optimizer']['num_epochs']
    logger.add_line('='*30 + ' Training Started' + '='*30)    
                
    for epoch in range(start_epoch, end_epoch):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            
        train_one_epoch(args, model, optimizer, lr_scheduler, train_loader, logger, tb_writter, wandb_writter, epoch, cfg['progress']['print_freq'], amp)
        
        # Save checkpoint
        if args.rank==0:
            ## normal checkpoint
            save_checkpoint(args, model, optimizer, lr_scheduler, epoch, amp, logger)
            ## normal + preemption checkpoint
            logger.add_line('saving model')    
            # commit_state(args, model, optimizer, lr_scheduler, epoch, amp, rng, logger)
        
        # Save just the backbone for further use
        if args.rank==0 and (epoch+1==end_epoch):
            vid_model_path = os.path.join(args.ckpt_dir, f"vid_{args.sub_dir}_{args.db}_ep{epoch}.pth.tar")
            aud_model_path = os.path.join(args.ckpt_dir, f"aud_{args.sub_dir}_{args.db}_ep{epoch}.pth.tar")
            print(f"Models saved to \n{vid_model_path} \n{aud_model_path}")
            torch.save(model.module.video_backbone.state_dict(), vid_model_path)
            torch.save(model.module.audio_backbone.state_dict(), aud_model_path)     

                    
    # finish logging for this run
    if wandb_writter is not None:
        wandb_writter.finish()
    return
            
def train_one_epoch(args, model, optimizer, lr_scheduler, train_loader, logger, tb_writter, wandb_writter, epoch, print_freq, amp):
    
    model.train()
    logger.add_line('[Train] Epoch {}'.format(epoch))
    batch_time = AverageMeter('Time', ':6.3f', window_size=100)
    data_time = AverageMeter('Data', ':6.3f', window_size=100)
    loss_meter = AverageMeter('Loss', ':.3e')
    progress = ProgressMeter(len(train_loader), [batch_time, data_time, loss_meter],
                                          phase='pretext-iter', epoch=epoch, logger=logger, tb_writter=tb_writter)
    device = args.gpu if args.gpu is not None else 0
    end = time.time()
    for i, sample in enumerate(train_loader):
        
        # measure data loading time
        data_time.update(time.time() - end)
        frames1 = sample['frames'][:, 0, ::]
        frames2 = sample['frames'][:, 1, ::]
        specs1 = sample['audio'][:, 0, ::]
        specs2 = sample['audio'][:, 1, ::]
                
        model.zero_grad()
        
        with torch.cuda.amp.autocast():
            data_dict = model.forward(frames1.cuda(device, non_blocking=True), frames2.cuda(device, non_blocking=True), 
                specs1.cuda(device, non_blocking=True), specs2.cuda(device, non_blocking=True))

        loss = data_dict['loss'].mean()
        loss_meter.update(loss.item(), frames1.size(0))
        data_dict.update({'lr':lr_scheduler.get_lr()})
        data_dict.update({'loss':loss})
        
        if amp is not None:
            amp.scale(loss).backward()
            amp.step(optimizer)
            amp.update()
        else:
            loss.backward()
            optimizer.step()
            
        lr_scheduler.step()
        
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # print to terminal and tensorboard
        step = epoch * len(train_loader) + i
        if (i+1) % print_freq == 0 or i == 0 or i+1 == len(train_loader):
            progress.display(i+1)
            
            if tb_writter is not None:
                for lr in data_dict['lr'].keys():
                    tb_writter.add_scalar(f'Iter/LR_{lr}', data_dict['lr'][lr], step)
                for sl in data_dict['subloss'].keys():
                    tb_writter.add_scalar(f'Iter/Subloss_{sl}', data_dict['subloss'][sl], step)
                for meter in progress.meters:
                    tb_writter.add_scalar(f'Iter/{meter.name}', meter.val, step)
                
            if wandb_writter is not None:
                for lr in data_dict['lr'].keys():
                    wandb_writter.log({f'Iter/LR_{lr}': data_dict['lr'][lr], 'custom_step': step})
                for sl in data_dict['subloss'].keys():
                    wandb_writter.log({f'Iter/Subloss_{sl}': data_dict['subloss'][sl], 'custom_step': step})
                for meter in progress.meters:
                     wandb_writter.log({f'Iter/{meter.name}': meter.val, 'custom_step': step})
                     
                
    # Sync metrics across all GPUs and print final averages
    if args.distributed:
        progress.synchronize_meters(args.gpu)
                    
    if tb_writter is not None:
        tb_writter.add_scalar('Epoch/Epochs', epoch, epoch)     
        for meter in progress.meters:
            tb_writter.add_scalar('Epoch/{}'.format(meter.name), meter.avg, epoch)       
            
    if wandb_writter is not None:
        wandb_writter.log({'Epoch/Epochs': epoch, 'custom_step': epoch})
        for meter in progress.meters:
            wandb_writter.log({'Epoch/{}'.format(meter.name): meter.avg, 'custom_step': epoch})
            
    return


if __name__ == "__main__":
    
    args = get_args()
    if args.server =='local':
        args.debug=True   
    main(args=args)
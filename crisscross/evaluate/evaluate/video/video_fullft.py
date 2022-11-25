# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""

import os
import torch
import time
from evaluate import AverageMeter, ProgressMeter, accuracy, get_optimizer
from datasets.augmentations import get_vid_aug
from datasets import get_dataset, dataloader, fetch_subset        
from tools import environment as environ
from models import VideoClassifier, get_backbone, VideoFinetune
# from checkpointing import create_or_restore_training_state, commit_state
from tools.utils import resume_model, save_checkpoint

def finetune(args, cfg, backbone_state_dict, ngpus_per_node, logger, tb_writter, wandb_writter, dense=False):
    
    if args.debug:
        cfg, args = environ.set_debug_mode(cfg, args)
        
    # get backbone    
    net = get_backbone(cfg['model']['video_backbone'])(cfg['model']['video_backbone_args']['depth'])
    # load linear classifier
    classifier = VideoClassifier(**cfg['model']['classifier'])    
    
    if cfg['model']['video_backbone_args']['pool'] == 'adaptive_maxpool':
        net.pool = torch.nn.AdaptiveMaxPool3d((1, 1, 1))
    elif cfg['model']['video_backbone_args']['pool'] == 'adaptive_avgpool':
        net.pool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
    else:
        raise ValueError(f"{cfg['model']['video_backbone_args']['pool']} not available")
        
    model = VideoFinetune(net, classifier, feat_op=cfg['model']['video_backbone_args']['feat_op'])

    model.backbone.load_state_dict(backbone_state_dict, strict=True)

    if args.distributed and cfg['sync_bn']:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.cuda(args.gpu)
    model, _, cfg['dataset']['batch_size'], cfg['num_workers'] = environ.distribute_model_to_cuda(models=model, 
                                                                     args=args, 
                                                                     batch_size=cfg['dataset']['batch_size'], 
                                                                     num_workers=cfg['num_workers'], 
                                                                     ngpus_per_node=ngpus_per_node)

    # transformations
    train_transformations = get_vid_aug(name=cfg['dataset']['vid_transform'],
                                    crop_size=cfg['dataset']['crop_size'],
                                    num_frames=cfg['dataset']['clip_duration']*cfg['dataset']['video_fps'],
                                    mode=cfg['dataset']['train']['aug_mode'],                                    
                                    aug_kwargs=cfg['dataset']['train']['vid_aug_kwargs'])

    val_transformations = get_vid_aug(name=cfg['dataset']['vid_transform'],
                                    crop_size=cfg['dataset']['crop_size'],
                                    num_frames=cfg['dataset']['clip_duration']*cfg['dataset']['video_fps'],
                                    mode=cfg['dataset']['test']['aug_mode'],
                                    aug_kwargs=cfg['dataset']['test']['vid_aug_kwargs'])  

    # dataset
    train_dataset = get_dataset(root=args.data_dir,
                                dataset_kwargs=cfg['dataset'],
                                video_transform=train_transformations, 
                                split='train')

    if dense:
        val_dataset = get_dataset(root=args.data_dir,
                                dataset_kwargs=cfg['dataset'],
                                video_transform=val_transformations, 
                                split='test_dense')
    else:
        val_dataset = get_dataset(root=args.data_dir,
                                    dataset_kwargs=cfg['dataset'],
                                    video_transform=val_transformations, 
                                    split='test')        
            
    logger.add_line(f'Training dataset size: {len(train_dataset)} - Validation dataset size: {len(val_dataset)}')
            
    # dataloader
    train_loader = dataloader.make_dataloader(dataset=train_dataset, 
                                              batch_size=cfg['dataset']['batch_size'],
                                              use_shuffle=cfg['dataset']['train']['use_shuffle'],
                                              drop_last=cfg['dataset']['train']['drop_last'],
                                              num_workers=cfg['num_workers'],
                                              distributed=args.distributed)
    
    if dense:
        test_loader = dataloader.make_dataloader(dataset=val_dataset, 
                                                  batch_size=1,
                                                  use_shuffle=cfg['dataset']['test_dense']['use_shuffle'],
                                                  drop_last=cfg['dataset']['test_dense']['drop_last'],
                                                  num_workers=cfg['num_workers'],
                                                  distributed=args.distributed)
    else:
        test_loader = dataloader.make_dataloader(dataset=val_dataset, 
                                              batch_size=cfg['dataset']['batch_size'],
                                              use_shuffle=cfg['dataset']['test']['use_shuffle'],
                                              drop_last=cfg['dataset']['test']['drop_last'],
                                              num_workers=cfg['num_workers'],
                                              distributed=args.distributed)

    optimizer, lr_scheduler = get_optimizer(model.parameters(), cfg['optimizer'], logger)

    # use apex for mixed precision training
    if cfg['apex']:
        amp = torch.cuda.amp.GradScaler() 
    else:
        amp=None
    
    # model, optimizer, lr_scheduler, start_epoch, amp, rng = create_or_restore_training_state(args, model, optimizer, lr_scheduler, logger, amp)
    model, optimizer, lr_scheduler, start_epoch, amp = resume_model(args, model, optimizer, lr_scheduler, amp, logger)
    
    # Start training
    end_epoch = cfg['optimizer']['num_epochs']
    logger.add_line('='*30 + ' Training Started Finetune' + '='*30)
    
    best_top1=0
    for epoch in range(start_epoch, end_epoch):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            
        # train
        tr_top1, tr_top5 = run_phase('train', train_loader, model, optimizer, epoch, args, logger, tb_writter, wandb_writter, print_freq=cfg['progress']['print_freq'])
        lr_scheduler.step()
        # Note make sure lr_scheduler step is rightly placed based on the scheduler type
        
        if args.rank==0:
            logger.add_line('saving model')    
            # commit_state(args, model, optimizer, lr_scheduler, epoch, amp, rng, logger)
            save_checkpoint(args, model, optimizer, lr_scheduler, epoch, amp, logger)
            
        # test
        if dense:
            vid_top1, vid_top5 = run_phase('test_dense', test_loader, model, optimizer, epoch, args, logger, tb_writter, wandb_writter, print_freq=cfg['progress']['print_freq'])
            top1, top5 = vid_top1, vid_top5
        else:
            clip_top1, clip_top5 = run_phase('test', test_loader, model, optimizer, epoch, args, logger, tb_writter, wandb_writter, print_freq=cfg['progress']['print_freq'])
            top1, top5 = clip_top1, clip_top5

        if top1>best_top1:
            best_top1=top1
            best_top5=top5
            best_epoch=epoch
            # Save checkpoint
            if args.rank==0:
                torch.save(model, os.path.join(args.ckpt_dir, "best_model.pth.tar"))
           
        if tb_writter is not None:
            tb_writter.add_scalar('fine_tune_epoch/tr_top1', tr_top1, epoch)
            if dense:
                tb_writter.add_scalar('fine_tune_epoch/vid_top1', vid_top1, epoch)
            else:
                tb_writter.add_scalar('fine_tune_epoch/clip_top1', clip_top1, epoch)
           
            
        if wandb_writter is not None:
            wandb_writter.log({'fine_tune_epoch/tr_top1': tr_top1, 'custom_step': epoch})
            if dense:
                wandb_writter.log({'fine_tune_epoch/vid_top1': vid_top1, 'custom_step': epoch})
            else:
                wandb_writter.log({'fine_tune_epoch/clip_top1}': clip_top1, 'custom_step': epoch})
          
    if args.rank==0:
        logger.add_line(f'Final Acc - top1: {top1} - top5: {top5}')
        logger.add_line(f'Best Acc at epoch {best_epoch} - best_top1: {best_top1} - best_top5: {best_top5}')
    if wandb_writter is not None:
        wandb_writter.log({'fine_tune_epoch/vid_top1_best': best_top1, 'custom_step': 0})
        wandb_writter.log({'fine_tune_epoch/vid_top5_best': best_top5, 'custom_step': 0})

    torch.cuda.empty_cache()
    if wandb_writter is not None:
        wandb_writter.finish()
        
    return

def run_phase(phase, loader, model, optimizer, epoch, args, logger, tb_writter, wandb_writter, print_freq):
    
    logger.add_line('\n {}: Epoch {}'.format(phase, epoch))
    batch_time = AverageMeter('Time', ':6.3f', 100)
    data_time = AverageMeter('Data', ':6.3f', 100)
    loss_meters = AverageMeter('Loss', ':.4e', 0)
    top1_meters = AverageMeter('Acc@1', ':6.2f', 0)
    top5_meters = AverageMeter('Acc@5', ':6.2f', 0)
    progress = ProgressMeter(len(loader), meters=[batch_time, data_time, loss_meters, top1_meters, top5_meters], phase=phase, epoch=epoch, logger=logger)


    if phase == 'train':
        model.train()
    else:
        model.eval()

    end = time.time()
    criterion = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)
    for it, sample in enumerate(loader):
        data_time.update(time.time() - end)

        # prepare data
        video = sample['frames']
        target = sample['label'].cuda()
        if args.gpu is not None:
            video = video.cuda(args.gpu, non_blocking=True)

        if phase == 'test_dense':
            batch_size, clips_per_sample = video.shape[0], video.shape[1]
            video = video.flatten(0, 1).contiguous()

        # compute outputs
        if phase == 'train':
            model.zero_grad()         
            # optimizer.zero_grad()
            logits = model(video)
        else:
            with torch.no_grad():
                logits = model(video)

        # compute loss and measure accuracy
        if phase == 'test_dense':
            confidence = softmax(logits).view(batch_size, clips_per_sample, -1).mean(1)
            target_tiled = target.unsqueeze(1).repeat(1, clips_per_sample).view(-1)
            loss = criterion(logits, target_tiled)
        else:
            confidence = softmax(logits)
            loss = criterion(logits, target)

        with torch.no_grad():
            acc1, acc5 = accuracy(confidence, target, topk=(1, 5))
            loss_meters.update(loss.item(), target.size(0))
            top1_meters.update(acc1[0].item(), target.size(0))
            top5_meters.update(acc5[0].item(), target.size(0))

        # compute gradient and do SGD step
        if phase == 'train':
            loss.backward()
            optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        step = epoch * len(loader) + it
        if (it + 1) % print_freq == 0 or it == 0 or it + 1 == len(loader):
            progress.display(it+1)
            if tb_writter is not None:
                tb_writter.add_scalar('fine_tune_iter/LR', optimizer.param_groups[0]['lr'], step)
                for meter in progress.meters:
                    tb_writter.add_scalar('fine_tune_iter/{}'.format(meter.name), meter.val, step)
            
            if wandb_writter is not None and phase == 'train':
                wandb_writter.log({'fine_tune_iter/LR': optimizer.param_groups[0]['lr'], 'custom_step': step})
                for meter in progress.meters:
                     wandb_writter.log({'fine_tune_iter/{}'.format(meter.name): meter.val, 'custom_step': step})
            

    if args.distributed:
        progress.synchronize_meters(args.gpu)
        progress.display(len(loader) * args.world_size)
            
    torch.cuda.empty_cache()
    return top1_meters.avg, top5_meters.avg
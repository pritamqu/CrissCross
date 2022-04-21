# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""
import torch
import time
import os
from evaluate import get_optimizer, AverageMeter, ProgressMeter, accuracy
from datasets.augmentations import get_aud_aug
from datasets import get_dataset, dataloader, FetchSubset
from tools import environment as environ
from models import get_backbone, AudioClassifier, AudioFCtune

def fctune(args, cfg, backbone_state_dict, ngpus_per_node, logger, tb_writter, wandb_writter, dense=False):

    if args.debug:
        cfg, args = environ.set_debug_mode(cfg, args)

    # get backbone
    model = get_backbone(cfg['model']['audio_backbone'])(cfg['model']['audio_backbone_args']['depth'])
    classifier = AudioClassifier(**cfg['model']['classifier'])
    model = AudioFCtune(model, classifier,
                        feat_op=cfg['model']['audio_backbone_args']['feat_op'],
                        )
    ## load weights
    model.backbone.load_state_dict(backbone_state_dict)
    model.backbone.eval() # when extracting features it's important to set in eval mode
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)

    # transformations
    train_transformations = get_aud_aug(name=cfg['dataset']['aud_transform'],
                                    audio_fps=cfg['dataset']['audio_fps'],
                                    n_mels=cfg['dataset']['n_mels'],
                                    duration=cfg['dataset']['audio_clip_duration'],
                                    n_fft=cfg['dataset']['n_fft'],
                                    hop_length=cfg['dataset']['hop_length'],
                                    mode=cfg['dataset']['train']['aug_mode'],
                                    aug_kwargs=cfg['dataset']['train']['aud_aug_kwargs'])

    val_transformations = get_aud_aug(name=cfg['dataset']['aud_transform'],
                                    audio_fps=cfg['dataset']['audio_fps'],
                                    n_mels=cfg['dataset']['n_mels'],
                                    duration=cfg['dataset']['audio_clip_duration'],
                                    n_fft=cfg['dataset']['n_fft'],
                                    hop_length=cfg['dataset']['hop_length'],
                                    mode=cfg['dataset']['test_dense']['aug_mode'],
                                    aug_kwargs=cfg['dataset']['test_dense']['aud_aug_kwargs'])

    # dataset
    train_dataset = get_dataset(root=args.data_dir,
                                dataset_kwargs=cfg['dataset'],
                                audio_transform=train_transformations,
                                split='train')

    if dense:
        val_dataset = get_dataset(root=args.data_dir,
                                dataset_kwargs=cfg['dataset'],
                                audio_transform=val_transformations,
                                split='test_dense')
    else:
        val_dataset = get_dataset(root=args.data_dir,
                                dataset_kwargs=cfg['dataset'],
                                audio_transform=val_transformations,
                                split='test')

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

    # Start training
    start_epoch = 0
    end_epoch = cfg['optimizer']['num_epochs']
    logger.add_line('='*30 + ' Training Started FC Finetune' + '='*30)
    best_top1, best_top5 = 0, 0

    for epoch in range(start_epoch, end_epoch):

        # train
        tr_top1, tr_top5 = linear_one_epoch('train', train_loader, model, optimizer, lr_scheduler, epoch, args, logger, tb_writter, wandb_writter, print_freq=cfg['progress']['print_freq'])
        lr_scheduler.step()

        # test
        if dense:
            vid_top1, vid_top5 = linear_one_epoch('test_dense', test_loader, model, optimizer, lr_scheduler, epoch, args, logger, tb_writter, wandb_writter, print_freq=cfg['progress']['print_freq'])
            top1, top5 = vid_top1, vid_top5
        else:
            clip_top1, clip_top5 = linear_one_epoch('test', test_loader, model, optimizer, lr_scheduler, epoch, args, logger, tb_writter, wandb_writter, print_freq=cfg['progress']['print_freq'])
            top1, top5 = clip_top1, clip_top5

        if top1>best_top1:
             best_top1=top1
             best_top5=top5
             best_epoch=epoch
             # Save checkpoint
             if args.rank==0:
                 torch.save(model, os.path.join(args.ckpt_dir, "best_model.pth.tar"))

        if tb_writter is not None:
            tb_writter.add_scalar('fc_tune_epoch/tr_top1', tr_top1, epoch)
            if dense:
                tb_writter.add_scalar('fc_tune_epoch/vid_top1', vid_top1, epoch)
            else:
                tb_writter.add_scalar('fc_tune_epoch/clip_top1', clip_top1, epoch)

        if wandb_writter is not None:
            wandb_writter.log({'fc_tune_epoch/tr_top1': tr_top1, 'custom_step': epoch})
            if dense:
                wandb_writter.log({'fc_tune_epoch/vid_top1': vid_top1, 'custom_step': epoch})
            else:
                wandb_writter.log({'fc_tune_epoch/clip_top1': clip_top1, 'custom_step': epoch})

    if args.rank==0:
        logger.add_line(f'Final Acc - top1: {top1} - top5: {top5}')
        logger.add_line(f'Best Acc at epoch {best_epoch} - best_top1: {best_top1} - best_top5: {best_top5}')

    torch.cuda.empty_cache()
    if wandb_writter is not None:
        wandb_writter.finish()

    return

def linear_one_epoch(phase, loader, model, optimizer, lr_scheduler, epoch, args, logger, tb_writter, wandb_writter, print_freq=10):

    logger.add_line('\n {}: Epoch {}'.format(phase, epoch))
    batch_time = AverageMeter('Time', ':6.3f', 100)
    data_time = AverageMeter('Data', ':6.3f', 100)
    loss_meters = AverageMeter('Loss', ':.4e', 0)
    top1_meters = AverageMeter('Acc@1', ':6.2f', 0)
    top5_meters = AverageMeter('Acc@5', ':6.2f', 0)
    progress = ProgressMeter(len(loader), meters=[batch_time, data_time, loss_meters, top1_meters, top5_meters], phase=phase, epoch=epoch, logger=logger)

    # switch to train/test mode
    if phase == 'train':
        model.classifier.train()
    else:
        model.classifier.eval()

    end = time.time()
    criterion = torch.nn.CrossEntropyLoss()
    softmax = torch.nn.Softmax(dim=1)
    for it, sample in enumerate(loader):
        data_time.update(time.time() - end)

        video = sample['audio']
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
            # lr_scheduler.step()
            # if want to step here change milestones in to iters,
            # but not required, keep it simple.

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

    if args.rank==0:
        logger.add_line(f'aud_phase: {phase} - epoch: {epoch} - top1: {top1_meters.avg} - top5: {top5_meters.avg}')

    return top1_meters.avg, top5_meters.avg
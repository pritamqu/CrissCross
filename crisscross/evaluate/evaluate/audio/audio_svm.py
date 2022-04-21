# -*- coding: utf-8 -*-
"""
@author: pritam sarkar
@web: www.pritamsarkar.com
"""

import torch
import numpy as np
from sklearn.svm import LinearSVC
from sklearn import preprocessing
from evaluate import set_grad, Feature_Bank
from datasets.augmentations import get_aud_aug
from datasets import get_dataset, dataloader
from tools import environment as environ
from models import get_backbone, Aud_Wrapper


def linear_svm(args, cfg, backbone_state_dict, logger, tb_writter, wandb_writter):

    if args.debug:
        cfg, args = environ.set_debug_mode(cfg, args)

    # get backbone
    model = get_backbone(cfg['model']['audio_backbone'])(cfg['model']['audio_backbone_args']['depth'])
    model = Aud_Wrapper(model,
                        feat_op=cfg['model']['audio_backbone_args']['feat_op'],
                        feat_dim=cfg['model']['audio_backbone_args']['feat_dim'],
                        l2_norm=cfg['model']['audio_backbone_args']['l2_norm'],
                        use_bn=cfg['model']['audio_backbone_args']['use_bn']
                        )
    ## load weights
    model.backbone.load_state_dict(backbone_state_dict)
    # set grad false
    set_grad(model, requires_grad=False)
    model.eval() # when extracting features it's important to set in eval mode
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # model = torch.nn.DataParallel(model)

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
                                    mode=cfg['dataset']['test']['aug_mode'],
                                    aug_kwargs=cfg['dataset']['test']['aud_aug_kwargs'])

    # dataset
    train_dataset = get_dataset(root=args.data_dir,
                                dataset_kwargs=cfg['dataset'],
                                audio_transform=train_transformations,
                                split='train')

    val_dataset = get_dataset(root=args.data_dir,
                                dataset_kwargs=cfg['dataset'],
                                audio_transform=val_transformations,
                                split='test')
    # dataloader
    train_loader = dataloader.make_dataloader(dataset=train_dataset,
                                              batch_size=cfg['dataset']['batch_size'],
                                              use_shuffle=False,
                                              drop_last=False,
                                              num_workers=cfg['num_workers'],
                                              distributed=False)

    val_loader = dataloader.make_dataloader(dataset=val_dataset,
                                          batch_size=cfg['dataset']['batch_size'],
                                          use_shuffle=False,
                                          drop_last=False,
                                          num_workers=cfg['num_workers'],
                                          distributed=False)

    feat_bank = Feature_Bank(args.world_size, args.distributed, model, logger, mode='aud', l2_norm=False) # setting l2-norm false, as we added l2-norm in Aud_Wrapper.

    logger.add_line("computing features")
    train_features, train_labels, train_indexs = feat_bank.fill_memory_bank(train_loader)
    val_features, val_labels, val_indexs = feat_bank.fill_memory_bank(val_loader)

    train_features, train_labels, train_indexs = train_features.numpy(), train_labels.numpy(), train_indexs.numpy()
    val_features, val_labels, val_indexs = val_features.numpy(), val_labels.numpy(), val_indexs.numpy()

    best_top1=0.0
    logger.add_line("Running SVM...")
    logger.add_line(f"train_feat size: {train_features.shape}")
    logger.add_line(f"val_feat size: {val_features.shape}")
    if isinstance(cfg['model']['svm']['cost'], list): # sweep a list of cost values
        for cost in cfg['model']['svm']['cost']:
            clip_top1, clip_top5 = _compute(cost, cfg, logger,
             train_features, train_labels, train_indexs,
             val_features, val_labels, val_indexs,)

            if tb_writter is not None:
                tb_writter.add_scalar('Epoch/aud_svm_top1', clip_top1, cost)
                tb_writter.add_scalar('Epoch/aud_svm_top5', clip_top5, cost)

            if wandb_writter is not None:
                wandb_writter.log({'Epoch/aud_svm_top1': clip_top1, 'custom_step': cost})
                wandb_writter.log({'Epoch/aud_svm_top5': clip_top5, 'custom_step': cost})

            # show the best one
            if clip_top1 >= best_top1:
                best_top1 = clip_top1
                best_top5 = clip_top5
    else:
        cost = cfg['model']['svm']['cost']
        best_top1, best_top5 = _compute(cost, cfg, logger,
             train_features, train_labels, train_indexs,
             val_features, val_labels, val_indexs,)

    logger.add_line(f'Best Acc: top1: {best_top1} - top5: {best_top5}')
    if tb_writter is not None:
        tb_writter.add_scalar('Epoch/aud_svm_best1', best_top1, 1)
        tb_writter.add_scalar('Epoch/aud_svm_best5', best_top5, 1)

    if wandb_writter is not None:
        wandb_writter.log({'Epoch/aud_svm_best1': best_top1, 'custom_step': 1})
        wandb_writter.log({'Epoch/aud_svm_best5': best_top5, 'custom_step': 1})

    torch.cuda.empty_cache()
    return

def _compute(cost, cfg, logger,
             train_features, train_labels, train_indexs,
             val_features, val_labels, val_indexs,
             test_phase='test_dense'):

    # normalize
    if cfg['model']['svm']['scale_features']:
        scaler = preprocessing.StandardScaler().fit(train_features)
        train_features = scaler.transform(train_features)
        # val_features = scaler.transform(val_features)

    classifier = LinearSVC(C=cost, max_iter=cfg['model']['svm']['iter'])
    classifier.fit(train_features, train_labels.ravel())
    pred_train = classifier.decision_function(train_features)
    # for test dense, assuming this is default test case
    # reshape the data video --> cips
    if test_phase=='test_dense':
        total_samples, clips_per_sample = val_features.shape[0], val_features.shape[1]
        val_features = val_features.reshape(total_samples*clips_per_sample, -1)
    # scale if true
    if cfg['model']['svm']['scale_features']:
        val_features = scaler.transform(val_features)
    # predict
    pred_test = classifier.decision_function(val_features)
    if test_phase=='test_dense':
        pred_test = pred_test.reshape(total_samples, clips_per_sample, -1).mean(1)

    metrics = compute_accuracy_metrics(pred_train, train_labels[:, None], prefix='train_')
    metrics.update(compute_accuracy_metrics(pred_test, val_labels[:, None], prefix='test_'))
    logger.add_line(f"Audio Linear SVM on {cfg['dataset']['name']} cost: {cost}")
    for metric in metrics:
        logger.add_line(f"{metric}: {metrics[metric]}")

    return metrics['test_top1'], metrics['test_top5']


def compute_accuracy_metrics(pred, gt, prefix=''):
  order_pred = np.argsort(pred, axis=1)
  assert len(gt.shape) == len(order_pred.shape) == 2
  top1_pred = order_pred[:, -1:]
  top5_pred = order_pred[:, -5:]
  top1_acc = np.mean(top1_pred == gt)
  top5_acc = np.mean(np.max(top5_pred == gt, 1))
  return {prefix + 'top1': top1_acc*100,
          prefix + 'top5': top5_acc*100}
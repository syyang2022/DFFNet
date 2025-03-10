# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import json
import random
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from bialoss import BiasLoss
from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, load_pretrained, save_checkpoint, NativeScalerWithGradNormCount, auto_resume_helper, \
    reduce_tensor, FeatureLoss
from sklearn.metrics import f1_score, precision_score, recall_score
# pytorch major version (1.x or 2.x)
PYTORCH_MAJOR_VERSION = int(torch.__version__.split('.')[0])
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--disable_amp', action='store_true', help='Disable pytorch amp')
    parser.add_argument('--amp-opt-level', type=str, choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used (deprecated!)')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    # for pytorch >= 2.0, use `os.environ['LOCAL_RANK']` instead
    # (see https://pytorch.org/docs/stable/distributed.html#launch-utility)
    if PYTORCH_MAJOR_VERSION == 1:
        parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    # for acceleration
    parser.add_argument('--fused_window_process', action='store_true',
                        help='Fused window shift & window partition, similar for reversed part.')
    parser.add_argument('--fused_layernorm', action='store_true', help='Use fused layernorm.')
    ## overwrite optimizer in config (*.yaml) if specified, e.g., fused_adam/fused_lamb
    parser.add_argument('--optim', type=str,
                        help='overwrite optimizer if provided, can be adamw/sgd/fused_adam/fused_lamb.')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    logger.info(str(model))

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    if hasattr(model, 'flops'):
        flops = model.flops()
        logger.info(f"number of GFLOPs: {flops / 1e9}")

    model.cuda()
    model_without_ddp = model

    optimizer = build_optimizer(config, model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False,find_unused_parameters=True)
    loss_scaler = NativeScalerWithGradNormCount()

    if config.TRAIN.ACCUMULATION_STEPS > 1:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train) // config.TRAIN.ACCUMULATION_STEPS)
    else:
        lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_accuracy = 0.0

    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_accuracy = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, loss_scaler, logger)
        acc1, acc5, loss, f1, precision, recall = validate(config, data_loader_val, model)
        logger.info(f"Validation Results - Acc@1: {acc1:.1f}%, Acc@5: {acc5:.1f}%, Loss: {loss:.4f}")
        logger.info(f"Validation Metrics - F1-score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        acc1, acc5, loss, f1, precision, recall = validate(config, data_loader_val, model)
        logger.info(f"Validation Results - Acc@1: {acc1:.1f}%, Acc@5: {acc5:.1f}%, Loss: {loss:.4f}")
        logger.info(f"Validation Metrics - F1-score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    if config.THROUGHPUT_MODE:
        throughput(data_loader_val, model, logger)
        return

    logger.info("Start training")
    start_time = time.time()

    patience = 10  # 从配置中读取早停的 patience 参数
    wait = 0  # 记录当前精度未提升的连续 epoch 数

    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        # 训练一个 epoch
        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler,
                        loss_scaler)

        # 验证模型
        acc1, acc5, loss, f1, precision, recall = validate(config, data_loader_val, model)
        logger.info(f"Epoch {epoch + 1}/{config.TRAIN.EPOCHS} Validation Results - "
                    f"Acc@1: {acc1:.1f}%, Acc@5: {acc5:.1f}%, Loss: {loss:.4f}")
        logger.info(f"Epoch {epoch + 1}/{config.TRAIN.EPOCHS} Validation Metrics - "
                    f"F1-score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

        # 保存当前模型权重为 current.pth
        save_checkpoint(config, epoch, model_without_ddp, acc1, optimizer, lr_scheduler, loss_scaler, logger,
                        filename="current.pth")

        # 如果当前精度更高，更新最佳精度并保存为 best.pth
        if acc1 > max_accuracy:
            max_accuracy = acc1
            wait = 0  # 重置等待计数
            save_checkpoint(config, epoch, model_without_ddp, max_accuracy, optimizer, lr_scheduler, loss_scaler,
                            logger, filename="best.pth")
            logger.info(f'New best accuracy: {max_accuracy:.2f}%, saving best model...')
        else:
            wait += 1
            logger.info(f"No improvement in accuracy for {wait} epochs.")

        logger.info(f'Max accuracy so far: {max_accuracy:.2f}%')


    # 最终验证并输出结果
    acc1, acc5, loss, f1, precision, recall = validate(config, data_loader_val, model)
    logger.info(f"Final Validation Results - Acc@1: {acc1:.1f}%, Acc@5: {acc5:.1f}%, Loss: {loss:.4f}")
    logger.info(f"Final Validation Metrics - F1-score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")

    # 记录总训练时间
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))



def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, loss_scaler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()
    scaler_meter = AverageMeter()

    start = time.time()
    end = time.time()

    for idx,(samples_regular, samples_dct, targets) in enumerate(data_loader):
        samples_regular = samples_regular.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        samples_dct = samples_dct.cuda(non_blocking=True)

        if mixup_fn is not None:
            samples_regular, targets = mixup_fn(samples_regular, targets)
            samples_dct, _ = mixup_fn(samples_dct, targets)
        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            outputs = model(samples_regular,samples_dct)
            loss = criterion(outputs, targets)

            loss = loss / config.TRAIN.ACCUMULATION_STEPS

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=config.TRAIN.CLIP_GRAD,
                                parameters=model.parameters(), create_graph=is_second_order,
                                update_grad=(idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0)
        if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
            optimizer.zero_grad()
            lr_scheduler.step_update((epoch * num_steps + idx) // config.TRAIN.ACCUMULATION_STEPS)
        loss_scale_value = loss_scaler.state_dict()["scale"]

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets.size(0))
        if grad_norm is not None:  # loss_scaler return None if not update
            norm_meter.update(grad_norm)
        scaler_meter.update(loss_scale_value)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            wd = optimizer.param_groups[0]['weight_decay']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t wd {wd:.4f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'loss_scale {scaler_meter.val:.4f} ({scaler_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
@torch.no_grad()
def validate(config, data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    all_targets = []
    all_preds = []
    end = time.time()

    for idx, (images_regular, images_dct, target) in enumerate(data_loader):
        images_regular = images_regular.cuda(non_blocking=True)
        images_dct = images_dct.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        with torch.cuda.amp.autocast(enabled=config.AMP_ENABLE):
            output = model(images_regular,images_dct)

        loss = criterion(output, target)
        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        acc1 = reduce_tensor(acc1)
        acc5 = reduce_tensor(acc5)
        loss = reduce_tensor(loss)

        loss_meter.update(loss.item(), target.size(0))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        preds = torch.argmax(output, dim=1)
        all_targets.extend(target.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())

        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(data_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')

    # 计算 F1-score、Precision 和 Recall
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    f1 = f1_score(all_targets, all_preds, average="weighted")
    precision = precision_score(all_targets, all_preds, average="weighted")
    recall = recall_score(all_targets, all_preds, average="weighted")
    class_names = ['complex', 'frog_eye_leaf_spot', 'healthy', 'powdery_mildew', 'rust','scab']  # 替换成你的类别名称0:complex 1:frog_eye_leaf_spot 2:healthy 3:powdery_mildew 4:rust 5:scab

    # 生成混淆矩阵
    cm = confusion_matrix(all_targets, all_preds)
    # cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 去掉归一化的部分
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')

    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')
    logger.info(f' * F1-score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')

    return acc1_meter.avg, acc5_meter.avg, loss_meter.avg, f1, precision, recall


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images_regular,images_dct, _) in enumerate(data_loader):
        # 确保数据加载到 GPU
        images_regular = images_regular.cuda(non_blocking=True)
        images_dct = images_dct.cuda(non_blocking=True)
        batch_size = images_regular.shape[0]

        # 测试 images_regular 的吞吐量
        logger.info("Testing throughput for regular images")
        for i in range(50):  # 预热阶段
            model(images_regular)
        torch.cuda.synchronize()

        logger.info(f"Throughput testing starts for regular images with batch_size {batch_size}")
        tic1 = time.time()
        for i in range(30):  # 测试阶段
            model(images_regular)
        torch.cuda.synchronize()
        tic2 = time.time()
        regular_throughput = 30 * batch_size / (tic2 - tic1)
        logger.info(f"Regular images throughput: {regular_throughput:.2f} samples/sec")

        # 测试 images_dct 的吞吐量
        logger.info("Testing throughput for DCT images")
        for i in range(50):  # 预热阶段
            model(images_dct)
        torch.cuda.synchronize()

        logger.info(f"Throughput testing starts for DCT images with batch_size {batch_size}")
        tic1 = time



if __name__ == '__main__':
    args, config = parse_option()

    if config.AMP_OPT_LEVEL:
        print("[warning] Apex amp has been deprecated, please use pytorch amp instead!")

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())
    logger.info(json.dumps(vars(args)))

    main(config)

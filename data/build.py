import os
import torch
import numpy as np
from torchvision import transforms, datasets
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup, create_transform
from torch.utils.data import DataLoader
import torch.distributed as dist
from data.cvtransformer import SubsetDCT, RandomHorizontalFlip, ImageJitter, RandomResizedCrop, Aggregate, Upscale, \
    TransformUpscaledDCT
from .cached_image_folder import CachedImageFolder
from .imagenet22k_dataset import IN22KDATASET, CUBDataset, PlantvillageDataset
from .samplers import SubsetRandomSampler
from .tools import train_upscaled_static_std, train_upscaled_static_mean
try:
    from torchvision.transforms import InterpolationMode


    def _pil_interp(method):
        if method == 'bicubic':
            return InterpolationMode.BICUBIC
        elif method == 'lanczos':
            return InterpolationMode.LANCZOS
        elif method == 'hamming':
            return InterpolationMode.HAMMING
        else:
            # default bilinear, do we want to allow nearest?
            return InterpolationMode.BILINEAR


    import timm.data.transforms as timm_transforms

    timm_transforms._pil_interp = _pil_interp
except:
    from timm.data.transforms import _pil_interp

# 定义 DCT 数据增强变换
def build_transform_dct(is_train, config, filter_size=8,jitter_param = dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
    """
    针对 DCT 数据的图像变换
    """
    from data.cvtransformer import Compose, Resize, CenterCrop, GetDCT, UpScaleDCT, ToTensorDCT, NormalizeDCT

    if is_train:
        # transform = transforms.Compose([
        #     Resize(512),
        #     CenterCrop(448),
        #     Upscale(upscale_factor=2),
        #     TransformUpscaledDCT(),
        #     ToTensorDCT(),
        #     Aggregate(),
        #     NormalizeDCT(
        #         train_upscaled_static_mean,
        #         train_upscaled_static_std,
        #     )
        # ])
        transform = Compose([
             Resize(int(filter_size * 56 * 1.15)),
             CenterCrop(filter_size * 56),
             GetDCT(filter_size),
             UpScaleDCT(size=56),
             ToTensorDCT(),
             SubsetDCT(channels=192),
             Aggregate(),
             NormalizeDCT(
                train_upscaled_static_mean,
                train_upscaled_static_std,
                channels=192
            )
        ])
    else:
        transform = Compose([
            RandomResizedCrop(filter_size * 56),
            ImageJitter(jitter_param),
            RandomHorizontalFlip(),
            GetDCT(filter_size),
            UpScaleDCT(size=56),
            ToTensorDCT(),
            SubsetDCT(channels=192),
            Aggregate(),
            NormalizeDCT(
                train_upscaled_static_mean,
                train_upscaled_static_std,
                channels=192
            )
        ])

    return transform


# 常规数据增强

def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def build_dataset(is_train, config):
    """
    根据训练/验证数据集选择数据加载方式
    """
    # 构建常规和 DCT 数据变换
    transform_regular = build_transform(is_train, config)  # 常规图像变换
    transform_dct = build_transform_dct(is_train, config)  # DCT 图像变换



    if config.DATA.DATASET == 'imagenet':
        # 处理 ImageNet 数据集
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform_regular, transform_dct,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform_regular)
        nb_classes = 1000

    elif config.DATA.DATASET == 'imagenet22K':
        # 处理 ImageNet22K 数据集
        prefix = 'ILSVRC2011fall_whole'
        if is_train:
            ann_file = prefix + "_map_train.txt"
        else:
            ann_file = prefix + "_map_val.txt"
        dataset = IN22KDATASET(config.DATA.DATA_PATH, ann_file, transform_regular, transform_dct)
        nb_classes = 21841

    elif config.DATA.DATASET == 'CUB':
        # 处理 CUB 数据集
        prefix = 'train' if is_train else 'test'
        ann_file = f"{config.DATA.DATA_PATH}/{prefix}.txt"
        dataset = PlantvillageDataset(config.DATA.DATA_PATH, ann_file, transform_regular, transform_dct)
        nb_classes = 200  # CUB 数据集有 200 个类别

    elif config.DATA.DATASET == 'Plant':
        # 处理 CUB 数据集
        prefix = 'train' if is_train else 'test'
        ann_file = prefix + '.txt'  # 根据 is_train 来选择 train.txt 或 test.txt
        dataset = CUBDataset(config.DATA.DATA_PATH, ann_file, transform_regular, transform_dct)
        nb_classes = 6  # CUB 数据集有 200 个类别

    else:
        raise NotImplementedError(f"Unsupported dataset: {config.DATA.DATASET}")

    return dataset, nb_classes

 
def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    dataset_val, _ = build_dataset(is_train=False, config=config)

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
    if config.TEST.SEQUENTIAL:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_val = torch.utils.data.distributed.DistributedSampler(
            dataset_val, shuffle=config.TEST.SHUFFLE
        )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )
    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn
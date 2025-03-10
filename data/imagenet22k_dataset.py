import os
import json
import torch.utils.data as data
import numpy as np
from PIL import Image

import warnings

warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


class IN22KDATASET(data.Dataset):
    def __init__(self, root, ann_file='', transform=None, target_transform=None):
        super(IN22KDATASET, self).__init__()

        self.data_path = root
        self.ann_path = os.path.join(self.data_path, ann_file)
        self.transform = transform
        self.target_transform = target_transform
        # id & label: https://github.com/google-research/big_transfer/issues/7
        # total: 21843; only 21841 class have images: map 21841->9205; 21842->15027
        self.database = json.load(open(self.ann_path))

    def _load_image(self, path):
        try:
            im = Image.open(path)
        except:
            print("ERROR IMG LOADED: ", path)
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        idb = self.database[index]

        # images
        images = self._load_image(self.data_path + '/' + idb[0]).convert('RGB')
        if self.transform is not None:
            images = self.transform(images)

        # target
        target = int(idb[1])
        if self.target_transform is not None:
            target = self.target_transform(target)

        return images, target

    def __len__(self):
        return len(self.database)


class CUBDataset(data.Dataset):
    def __init__(self, root, ann_file='train.txt', transform_regular=None, transform_dct=None,target_transform=None):
        """
        Args:
            root (str): CUB数据集的根目录
            ann_file (str): 数据集的标注文件，如 'train.txt' 或 'test.txt'
            transform (callable, optional): 图像的变换（预处理）操作
            target_transform (callable, optional): 目标标签的变换（如映射到不同的标签空间）
        """
        super(CUBDataset, self).__init__()

        self.data_path = root  # 数据集根目录
        self.ann_path = os.path.join(self.data_path, ann_file)  # 标注文件路径
        self.transform_regular = transform_regular
        self.transform_dct = transform_dct
        self.target_transform = target_transform

        # 加载标注文件中的数据
        self.image_paths, self.labels = self._load_annotations()

    def _load_annotations(self):
        """
        从标注文件中加载图像路径和对应的标签
        """
        image_paths = []
        labels = []

        with open(self.ann_path, 'r') as f:
            for line in f:
                image_path, label = line.strip().split()
                image_paths.append(image_path)
                labels.append(int(label))

        return image_paths, labels


    def _load_image(self, path):
        """
        加载并返回图像
        """
        try:
            im = Image.open(path)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # 返回一个随机生成的图片（填充错误的图像）
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        获取指定索引的图像和标签
        """
        image_path = self.image_paths[index]
        label = self.labels[index]

        # 加载图像
        image = self._load_image(os.path.join(self.data_path, image_path)).convert('RGB')

        # 应用预处理变换
        if self.transform_regular or self.transform_dct is not None:
            image_regular = self.transform_regular(image)
            image_dct = self.transform_dct(image)


        # 应用目标标签变换
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image_regular, image_dct, label

    def __len__(self):
        return len(self.image_paths)

class PlantvillageDataset(data.Dataset):
    def __init__(self, root, ann_file='train.txt', transform_regular=None, transform_dct=None,target_transform=None):
        """
        Args:
            root (str): CUB数据集的根目录
            ann_file (str): 数据集的标注文件，如 'train.txt' 或 'test.txt'
            transform (callable, optional): 图像的变换（预处理）操作
            target_transform (callable, optional): 目标标签的变换（如映射到不同的标签空间）
        """
        super(PlantvillageDataset, self).__init__()

        self.data_path = root  # 数据集根目录
        self.ann_path = os.path.join(self.data_path, ann_file)  # 标注文件路径
        self.transform_regular = transform_regular
        self.transform_dct = transform_dct
        self.target_transform = target_transform

        # 加载标注文件中的数据
        self.image_paths, self.labels = self._load_annotations()

    def _load_annotations(self):
        """
        从标注文件中加载图像路径和对应的标签
        """
        image_paths = []
        labels = []

        with open(self.ann_path, 'r') as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) != 2:
                    print(f"Warning: Skipping malformed line {i + 1}: {line.strip()}")
                    continue
                image_path, label = parts
                image_paths.append(image_path)
                labels.append(int(label))

        return image_paths, labels

    def _load_image(self, path):
        """
        加载并返回图像
        """
        try:
            im = Image.open(path)
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            # 返回一个随机生成的图片（填充错误的图像）
            random_img = np.random.rand(224, 224, 3) * 255
            im = Image.fromarray(np.uint8(random_img))
        return im

    def __getitem__(self, index):
        """
        获取指定索引的图像和标签
        """
        image_path = self.image_paths[index]
        label = self.labels[index]

        # 加载图像
        image = self._load_image(os.path.join(self.data_path, image_path)).convert('RGB')

        # 应用预处理变换
        if self.transform_regular or self.transform_dct is not None:
            image_regular = self.transform_regular(image)
            image_dct = self.transform_dct(image)


        # 应用目标标签变换
        if self.target_transform is not None:
            label = self.target_transform(label)

        return image_regular, image_dct, label

    def __len__(self):
        return len(self.image_paths)
import math
from PIL import Image
import cv2
import torch
import argparse
import numpy as np
from config import get_config
from data.tools import train_upscaled_static_mean, train_upscaled_static_std
from models import build_model
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from data.cvtransformer import Compose, Resize, CenterCrop, GetDCT, UpScaleDCT, ToTensorDCT, NormalizeDCT, SubsetDCT, Aggregate
import os
from torchvision import transforms
from utils import GradCAM, show_cam_on_image, center_crop_img

# DCT 预处理流水线
def build_transform_dct(filter_size=8):
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
    return transform

# Grad-CAM reshape_transform
class ResizeTransform:
    def __init__(self, im_h: int, im_w: int):
        self.height = self.feature_size(im_h)
        self.width = self.feature_size(im_w)

    @staticmethod
    def feature_size(s):
        s = math.ceil(s / 4)  # PatchEmbed
        s = math.ceil(s / 2)  # PatchMerging1
        s = math.ceil(s / 2)  # PatchMerging2
        s = math.ceil(s / 2)  # PatchMerging3
        return s

    def __call__(self, x):
        # result = x.reshape(x.size(0),
        #                    self.height,
        #                    self.width,
        #                    x.size(2))
        # result = result.permute(0, 3, 1, 2)
        return x

# 解析命令行参数
def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='Path to config file')
    parser.add_argument('--data-folder', type=str, required=True, help='Path to input image folder')
    parser.add_argument('--output-folder', type=str, default='cam_results', help='Path to save CAM visualizations')
    parser.add_argument('--pretrained', help='Pretrained weight from checkpoint')
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs.",
        default=None,
        nargs='+',
    )
    args, unparsed = parser.parse_known_args()
    config = get_config(args)
    return args, config


if __name__ == "__main__":

    img_size = 448
    assert img_size % 32 == 0

    # 加载配置与参数
    args, config = parse_option()
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载模型
    model = build_model(config)
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location="cpu")
    # checkpoint["model"] = {k: v for k, v in checkpoint["model"].items() if not k.startswith("head.")}
    model.load_state_dict(checkpoint["model"])
    model.eval().to(DEVICE)

    target_layers = [model.cmf]  # 选择 Swin Transformer 和 DCT 特征的融合层
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transform_dct = build_transform_dct()

    # 创建输出文件夹
    output_folder = args.output_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历文件夹中的图片
    for img_name in os.listdir(args.data_folder):
        img_path = os.path.join(args.data_folder, img_name)
        if not os.path.isfile(img_path):
            continue

        # 加载图片
        img = Image.open(img_path).convert('RGB')
        img = np.array(img, dtype=np.uint8)
        img = center_crop_img(img, img_size)

        # [C, H, W]
        img_tensor = data_transform(img)
        input_tensor = torch.unsqueeze(img_tensor, dim=0)  # 扩展为批次维度

        def custom_forward(input_tensor):
            # 从输入张量生成 DCT 图像
            rgb_img_for_dct = input_tensor.cpu().numpy().transpose(0, 2, 3, 1)  # 转换为 HWC 格式
            dct_img = transform_dct(rgb_img_for_dct[0])  # 对第一个批次进行 DCT 处理
            dct_img = torch.tensor(dct_img).unsqueeze(0).to(DEVICE)  # 转换为张量并移动到设备

            # 将 RGB 图像 (input_tensor) 和 DCT 图像 (dct_img) 一起传递给模型
            return model(input_tensor, dct_img)


        with torch.no_grad():
            rgb_img_for_dct = input_tensor.cpu().numpy().transpose(0, 2, 3, 1)  # 转换为 HWC 格式
            dct_img = transform_dct(rgb_img_for_dct[0])  # 对第一个批次进行 DCT 处理
            dct_img = torch.tensor(dct_img).unsqueeze(0).to(DEVICE)  # 转换为张量并移动到设备
            input_tensor = input_tensor.to(DEVICE)  # 确保输入张量在正确设备上
            outputs = model(input_tensor, dct_img)  # 对输入张量进行预测
            probs = torch.nn.functional.softmax(outputs, dim=1)  # 计算类别概率
            predicted_category = torch.argmax(probs, dim=1).item()  # 获取预测类别索引
            print(f"Predicted category: {predicted_category}, Probabilities: {probs}")

        # 设置 Grad-CAM 的 forward_func
        cam = GradCAM(
            model=model,
            target_layers=target_layers,
            use_cuda=torch.cuda.is_available(),
            reshape_transform=ResizeTransform(im_h=img_size, im_w=img_size),
            forward_func=custom_forward  # 自定义 forward 函数
        )
        # Grad-CAM 热力图
        grayscale_cam = cam(input_tensor=input_tensor, target_category=predicted_category)
        grayscale_cam = grayscale_cam[0, :]

        # 生成可视化
        visualization = show_cam_on_image(img / 255., grayscale_cam, use_rgb=True)
        output_path = os.path.join(output_folder, f"cam_{os.path.splitext(img_name)[0]}.jpg")

        # 保存热力图
        plt.imsave(output_path, visualization)
        print(f"Saved CAM visualization: {output_path}")
#0:complex 1:frog_eye_leaf_spot 2:healthy 3:powdery_mildew 4:rust 5:scab


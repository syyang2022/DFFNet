import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from typing import Optional
import cv2
import math
import torch.nn.init as init  # 添加这一行
from einops import rearrange
from torch.autograd import Variable

class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]

def hard_sigmoid(x, inplace: bool = False):
    if inplace:
        return x.add_(3.).clamp_(0., 6.).div_(6.)
    else:
        return F.relu6(x + 3.) / 6.


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class SqueezeExcite(nn.Module):
    def __init__(self, in_chs, se_ratio=0.25, reduced_base_chs=None,
                 act_layer=nn.ReLU, gate_fn=hard_sigmoid, divisor=4, **_):
        super(SqueezeExcite, self).__init__()
        self.gate_fn = gate_fn
        reduced_chs = _make_divisible((reduced_base_chs or in_chs) * se_ratio, divisor)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_reduce = nn.Conv2d(in_chs, reduced_chs, 1, bias=True)
        self.act1 = act_layer(inplace=True)
        self.conv_expand = nn.Conv2d(reduced_chs, in_chs, 1, bias=True)

    def forward(self, x):
        x_se = self.avg_pool(x)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        x = x * self.gate_fn(x_se)
        return x

class GhostBottleneck(nn.Module):
    """ Ghost bottleneck w/ optional SE"""

    def __init__(self, in_chs, mid_chs, out_chs, dw_kernel_size=3,
                 stride=1,  se_ratio=0.):
        super(GhostBottleneck, self).__init__()
        has_se = se_ratio is not None and se_ratio > 0.
        self.stride = stride

        # Point-wise expansion
        self.ghost1 = GhostModule(in_chs, mid_chs, relu=True)

        # Depth-wise convolution
        if self.stride > 1:
            self.conv_dw = nn.Conv2d(mid_chs, mid_chs, dw_kernel_size, stride=stride,
                                     padding=(dw_kernel_size - 1) // 2,
                                     groups=mid_chs, bias=False)
            self.bn_dw = nn.BatchNorm2d(mid_chs)

        # Squeeze-and-excitation
        if has_se:
            self.se = SqueezeExcite(mid_chs, se_ratio=se_ratio)
        else:
            self.se = None

        # Point-wise linear projection
        self.ghost2 = GhostModule(mid_chs, out_chs, relu=False)

        # shortcut
        if (in_chs == out_chs and self.stride == 1):
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_chs, in_chs, dw_kernel_size, stride=stride,
                          padding=(dw_kernel_size - 1) // 2, groups=in_chs, bias=False),
                nn.BatchNorm2d(in_chs),
                nn.Conv2d(in_chs, out_chs, 1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_chs),
            )

    def forward(self, x):
        # if torch.isnan( x).any() or torch.isinf( x).any():
        #     print("NaN or Inf detected in  d_freq output!")
        residual = x

        # 1st ghost bottleneck
        x = self.ghost1(x)

        # Depth-wise convolution
        if self.stride > 1:
            x = self.conv_dw(x)
            x = self.bn_dw(x)

        # Squeeze-and-excitation
        if self.se is not None:
            x = self.se(x)

        # 2nd ghost bottleneck
        x = self.ghost2(x)

        x += self.shortcut(residual)
        return x

class GhostNetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ghost_bottlenecks = nn.Sequential(
            # 第一层 s=1
            GhostBottleneck(192, 192, 192, stride=1),
            # 第二层 s=2
            GhostBottleneck(192, 384, 384, stride=2),
            # 4 个 s=1 的层
            *[GhostBottleneck(384, 384, 384, stride=1) for _ in range(4)],
            # 最后一层 s=2
            GhostBottleneck(384, 768, 768, stride=2)
        )

    def forward(self, x):
        return self.ghost_bottlenecks(x)

def get_freq_indices(method):
    assert method in ['top1','top2','top4','top8','top16','top32',
                      'bot1','bot2','bot4','bot8','bot16','bot32',
                      'low1','low2','low4','low8','low16','low32']
    num_freq = int(method[3:])
    if 'top' in method:
        all_top_indices_x = [0,0,6,0,0,1,1,4,5,1,3,0,0,0,3,2,4,6,3,5,5,2,6,5,5,3,3,4,2,2,6,1]
        all_top_indices_y = [0,1,0,5,2,0,2,0,0,6,0,4,6,3,5,2,6,3,3,3,5,1,1,2,4,2,1,1,3,0,5,3]
        mapper_x = all_top_indices_x[:num_freq]
        mapper_y = all_top_indices_y[:num_freq]
    elif 'low' in method:
        all_low_indices_x = [0,0,1,1,0,2,2,1,2,0,3,4,0,1,3,0,1,2,3,4,5,0,1,2,3,4,5,6,1,2,3,4]
        all_low_indices_y = [0,1,0,1,2,0,1,2,2,3,0,0,4,3,1,5,4,3,2,1,0,6,5,4,3,2,1,0,6,5,4,3]
        mapper_x = all_low_indices_x[:num_freq]
        mapper_y = all_low_indices_y[:num_freq]
    elif 'bot' in method:
        all_bot_indices_x = [6,1,3,3,2,4,1,2,4,4,5,1,4,6,2,5,6,1,6,2,2,4,3,3,5,5,6,2,5,5,3,6]
        all_bot_indices_y = [6,4,4,6,6,3,1,4,4,5,6,5,2,2,5,1,4,3,5,0,3,1,1,2,4,2,1,1,5,3,3,3]
        mapper_x = all_bot_indices_x[:num_freq]
        mapper_y = all_bot_indices_y[:num_freq]
    else:
        raise NotImplementedError
    return mapper_x, mapper_y


class MultiSpectralDCTLayer(nn.Module):
    def __init__(self, height, width, mapper_x, mapper_y, channel):
        super(MultiSpectralDCTLayer, self).__init__()

        assert len(mapper_x) == len(mapper_y)
        assert channel % len(mapper_x) == 0

        self.num_freq = len(mapper_x)

        # fixed DCT init
        self.register_buffer('weight', self.get_dct_filter(height, width, mapper_x, mapper_y, channel))

    def forward(self, x):
        assert len(x.shape) == 4, 'x must been 4 dimensions, but got ' + str(len(x.shape))
        x = x * self.weight
        result = torch.sum(x, dim=[2, 3])
        return result

    def build_filter(self, pos, freq, POS):
        result = math.cos(math.pi * freq * (pos + 0.5) / POS) / math.sqrt(POS)
        if freq == 0:
            return result
        else:
            return result * math.sqrt(2)

    def get_dct_filter(self, tile_size_x, tile_size_y, mapper_x, mapper_y, channel):
        dct_filter = torch.zeros(channel, tile_size_x, tile_size_y)

        c_part = channel // len(mapper_x)

        for i, (u_x, v_y) in enumerate(zip(mapper_x, mapper_y)):
            for t_x in range(tile_size_x):
                for t_y in range(tile_size_y):
                    dct_filter[i * c_part: (i + 1) * c_part, t_x, t_y] = self.build_filter(t_x, u_x,
                                                                                           tile_size_x) * self.build_filter(
                        t_y, v_y, tile_size_y)

        return dct_filter


class MultiSpectralAttentionLayer(torch.nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction = 16, freq_sel_method = 'top16'):
        super(MultiSpectralAttentionLayer, self).__init__()
        self.reduction = reduction
        self.dct_h = dct_h
        self.dct_w = dct_w

        mapper_x, mapper_y = get_freq_indices(freq_sel_method)
        # print(f"channel: {channel}")
        # print(f"len(mapper_x): {len(mapper_x)}")
        self.num_split = len(mapper_x)
        mapper_x = [temp_x * (dct_h // 7) for temp_x in mapper_x]
        mapper_y = [temp_y * (dct_w // 7) for temp_y in mapper_y]
        # make the frequencies in different sizes are identical to a 7x7 frequency space
        # eg, (2,2) in 14x14 is identical to (1,1) in 7x7

        self.dct_layer = MultiSpectralDCTLayer(dct_h, dct_w, mapper_x, mapper_y, channel)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.fc.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        n,c,h,w = x.shape
        x_pooled = x
        if h != self.dct_h or w != self.dct_w:
            x_pooled = torch.nn.functional.adaptive_avg_pool2d(x, (self.dct_h, self.dct_w))
        y = self.dct_layer(x_pooled)

        y = self.fc(y).view(n, c, 1, 1)
        return x * y.expand_as(x)


class MSAFBlock(nn.Module):
    def __init__(self, channel, dct_h, dct_w, reduction=16, freq_sel_method='top16'):
        super(MSAFBlock, self).__init__()
        # Attention layer for frequency-domain features (x)
        self.attention_layer = MultiSpectralAttentionLayer(channel, dct_h, dct_w, reduction, freq_sel_method)

        # Conv block for processing the image domain features (y)
        self.conv1 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel // reduction),

            nn.LeakyReLU(negative_slope=0.01, inplace=True)

        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(channel // reduction, channel, kernel_size=1, bias=False),
            nn.BatchNorm2d(channel)
        )
        self.sigmoid = nn.Sigmoid()
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)  # 适用于较宽泛的激活函数

                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                init.ones_(m.weight)  # BatchNorm 权重初始化为 1
                init.zeros_(m.bias)  # BatchNorm 偏置初始化为 0

    def forward(self, x, y):

        x_attention = self.attention_layer(x)  # x_attention has shape [b, c, h, w]

        y_conv = self.conv1(y)  # y_conv has shape [b, c//reduction, h, w]

        y_conv = self.conv2(y_conv)  # y_conv has shape [b, c, h, w]
        if x_attention.shape[2:] != y_conv.shape[2:]:
            y_conv = F.interpolate(y_conv, size=x_attention.shape[2:], mode='bilinear', align_corners=False)

        x = x_attention + y_conv
        weights = self.sigmoid(x)

        # 对输入进行重加权
        output = x_attention * weights + y_conv * weights

        return output



class GateModule192(nn.Module):
    def __init__(self, act='relu'):
        super(GateModule192, self).__init__()

        self.inp_gs = GumbleSoftmax()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.in_ch = in_ch = 192
        if act == 'relu':
            relu = nn.ReLU
        elif act == 'relu6':
            relu = nn.ReLU6
        else: raise NotImplementedError

        self.inp_gate = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(in_ch),
            relu(inplace=True),
        )
        self.inp_gate_l = nn.Conv2d(in_ch, in_ch * 2, kernel_size=1, stride=1, padding=0, groups=in_ch, bias=True)


    def forward(self, x, temperature=1.):
        hatten = self.avg_pool(x)
        hatten_d = self.inp_gate(hatten)
        hatten_d = self.inp_gate_l(hatten_d)
        hatten_d = hatten_d.reshape(hatten_d.size(0), self.in_ch, 2, 1)
        hatten_d = self.inp_gs(hatten_d, temp=temperature, force_hard=True)

        x = x * hatten_d[:, :, 1].unsqueeze(2)

        return x, hatten_d[:, :, 1]
class GumbleSoftmax(torch.nn.Module):
    def __init__(self, hard=False):
        super(GumbleSoftmax, self).__init__()
        self.hard = hard

    def sample_gumbel(self, shape, eps=1e-10):
        """Sample from Gumbel(0, 1)"""
        noise = torch.rand(shape)
        noise.add_(eps).log_().neg_()
        noise.add_(eps).log_().neg_()
        if self.gpu:
            return noise.cuda()
        else:
            return noise

    def sample_gumbel_like(self, template_tensor, eps=1e-10):
        uniform_samples_tensor = template_tensor.clone().uniform_()
        gumble_samples_tensor = - torch.log(eps - torch.log(uniform_samples_tensor + eps))
        return gumble_samples_tensor

    def gumbel_softmax_sample(self, logits, temperature):
        """ Draw a sample from the Gumbel-Softmax distribution"""
        dim = logits.size(2)
        gumble_samples_tensor = self.sample_gumbel_like(logits.data)
        gumble_trick_log_prob_samples = logits + gumble_samples_tensor
        soft_samples = F.softmax(gumble_trick_log_prob_samples / temperature, dim)
        return soft_samples

    def gumbel_softmax(self, logits, temperature, hard=False):
        """Sample from the Gumbel-Softmax distribution and optionally discretize.
        Args:
        logits: [batch_size, n_class] unnormalized log-probslibaba
        temperature: non-negative scalar
        hard: if True, take argmax, but differentiate w.r.t. soft sample y
        Returns:
        [batch_size, n_class] sample from the Gumbel-Softmax distribution.
        If hard=True, then the returned sample will be one-hot, otherwise it will
        be a probabilitiy distribution that sums to 1 across classes
        """
        y = self.gumbel_softmax_sample(logits, temperature)
        if hard:
            # block layer
            # _, max_value_indexes = y.data.max(1, keepdim=True)
            # y_hard = logits.data.clone().zero_().scatter_(1, max_value_indexes, 1)
            # block channel
            _, max_value_indexes = y.data.max(2, keepdim=True)
            y_hard = logits.data.clone().zero_().scatter_(2, max_value_indexes, 1)
            y = Variable(y_hard - y.data) + y
        return y

    def forward(self, logits, temp=1, force_hard=False):
        samplesize = logits.size()

        if self.training and not force_hard:
            return self.gumbel_softmax(logits, temperature=1, hard=False)
        else:
            return self.gumbel_softmax(logits, temperature=1, hard=True)

ELIMINATED_THRESHOLD = 0.015
class TokenProcessor:
    def __init__(self, patch_len):
        if isinstance(patch_len, (tuple, list)):
            self.PATCH_LEN = patch_len[0]  # 假设窗口是正方形
        else:
            self.PATCH_LEN = patch_len
        self.init_idx()

    # 初始化索引（代码同前）
    def eliminate_token_by_norm(self, attn_weight):
        attn_weight = F.softmax(attn_weight, dim=1)
        attn_weight = self.process_attn_weight_by_norm(attn_weight)
        min_num, max_num = attn_weight.min(-1)[0].unsqueeze(-1), attn_weight.max(-1)[0].unsqueeze(-1)
        attn_weight = (attn_weight - min_num) / (max_num - min_num)
        rm_ind = (attn_weight < 0.025)
        random_tensor = torch.rand(rm_ind.shape, device=rm_ind.device)
        # rm_ind = (rm_ind & (random_tensor < 0.9))
        return rm_ind

    def process_attn_weight_by_norm(self, attn_weight):
        # 确保 attn_weight 是四维张量
        if attn_weight.ndim != 4:
            raise ValueError(f"Expected attn_weight to be 4D, but got shape {attn_weight.shape}")

        B, H, W, C = attn_weight.shape

        # 展平 H 和 W 维度，方便索引操作
        attn_weight = attn_weight.view(B, H * W, C)  # 形状变为 (B, N, C)，其中 N = H * W

        N = H * W
        idx = torch.arange(N, device=attn_weight.device).unsqueeze(0).expand(B, -1)  # (B, N)

        # 上下索引的计算，确保不越界
        idx_above = torch.clamp(idx + 1, max=N - 1)
        idx_below = torch.clamp(idx - 1, min=0)

        # Gather 操作更新 attn_weight
        attn_weight = (
                attn_weight.gather(1, idx.unsqueeze(-1).expand(-1, -1, C)) +  # (B, N, C)
                attn_weight.gather(1, idx_above.unsqueeze(-1).expand(-1, -1, C)) +  # (B, N, C)
                attn_weight.gather(1, idx_below.unsqueeze(-1).expand(-1, -1, C))  # (B, N, C)
        )

        # 还原形状
        attn_weight = attn_weight.view(B, H, W, C)
        return attn_weight

    def eliminate_token(self, attn_weight, origin_idx, size=None):
        attn_weight = F.softmax(attn_weight, dim=1)
        attn_weight = self.process_attn_weight(attn_weight, origin_idx)
        a, idx1 = torch.sort(attn_weight, dim=1, descending=True)
        if size is None:
            min_num, max_num = attn_weight.min(-1)[0].unsqueeze(-1), attn_weight.max(-1)[0].unsqueeze(-1)
            attn_weight = (attn_weight - min_num) / (max_num - min_num)
            rm_ind = (attn_weight < ELIMINATED_THRESHOLD)
            rm_num = rm_ind.sum(-1)
            size = rm_num.min()
        random_tensor = torch.rand((size,), device=idx1.device)
        random_retain = (random_tensor >= 0.9)
        retain_idx = idx1[:, -size:][:, random_retain]
        idx1 = idx1[:, :-size]
        idx1 = torch.cat([idx1, retain_idx], dim=-1)
        idx1, _ = torch.sort(idx1, dim=1, descending=False)
        idx1 += 1
        return idx1

    def process_attn_weight(self, attn_weight, cur_idx):
        min_num = attn_weight.min(-1)[0]
        is_return_shape = False
        if attn_weight.shape[1] != self.PATCH_LEN ** 2:
            is_return_shape = True
            attn_weight = self.pad_patch(attn_weight, cur_idx, min_num)

        idx = self.idx.expand(attn_weight.shape[0], -1)
        idx_above = self.idx_above.expand(attn_weight.shape[0], -1)
        idx_below = self.idx_below.expand(attn_weight.shape[0], -1)
        idx_left = self.idx_left.expand(attn_weight.shape[0], -1)
        idx_right = self.idx_right.expand(attn_weight.shape[0], -1)
        idx_left_above = self.idx_left_above.expand(attn_weight.shape[0], -1)
        idx_left_below = self.idx_left_below.expand(attn_weight.shape[0], -1)
        idx_right_above = self.idx_right_above.expand(attn_weight.shape[0], -1)
        idx_right_below = self.idx_right_below.expand(attn_weight.shape[0], -1)

        attn_weight = (attn_weight.gather(1, idx) + attn_weight.gather(1, idx_above) + attn_weight.gather(1,
                                                                                                          idx_below) + \
                       attn_weight.gather(1, idx_left) + attn_weight.gather(1,
                                                                            idx_right) + attn_weight.gather(1,
                                                                                                            idx_left_above) + \
                       attn_weight.gather(1, idx_left_below) + attn_weight.gather(1,
                                                                                  idx_right_above) + attn_weight.gather(
                    1, idx_right_below)) / 9
        if is_return_shape:
            tmp_idx = cur_idx[:, 1:] - 1
            attn_weight = attn_weight.gather(1, tmp_idx)
        return attn_weight

    def pad_patch(self, attn_weight, cur_idx, min_num):
        cur_idx = cur_idx[:, 1:] - 1
        new_weight = min_num.clone().unsqueeze(-1).repeat(1, self.PATCH_LEN ** 2)
        new_weight.scatter_(1, cur_idx, attn_weight)
        return new_weight

    def get_from_idx(self, x, idx):
        if len(x.shape) == 3:
            reg_token = x[0, :, :].unsqueeze(0)
            vis_token = x.gather(dim=0, index=idx)
            return torch.cat([reg_token, vis_token], dim=0)
        elif len(x.shape) == 2:
            reg_token = x[:, 0].unsqueeze(-1)
            vis_token = x.gather(dim=1, index=idx)
            return torch.cat([reg_token, vis_token], dim=1)
        elif len(x.shape) == 4:
            reg_token = x[:, :, 0:1, :]
            vis_token = x.gather(dim=2, index=idx)
            return torch.cat([reg_token, vis_token], dim=2)
        else:
            raise NotImplementedError

    def get_origin_idx(self, unm_idx, src_idx, dst_idx, origin_shape):
        un_len, merge_len = unm_idx.shape[1], src_idx.shape[1]
        tot_len = un_len + merge_len
        origin_idx = torch.zeros(origin_shape[:-1]).to(unm_idx.device).long()
        dst_origin_ind = torch.arange(un_len, tot_len + un_len, 1).to(unm_idx.device)

        idx = torch.arange(1, tot_len * 2, 2)
        origin_idx[:, idx] = dst_origin_ind
        origin_unm_idx = (unm_idx * 2).squeeze(-1)
        unm_origin_ind = torch.arange(0, un_len, 1).expand_as(origin_unm_idx).to(unm_idx.device)
        origin_idx = origin_idx.scatter(dim=1, index=origin_unm_idx, src=unm_origin_ind)

        origin_src_idx = (src_idx * 2).squeeze(-1)
        dst_idx = dst_idx.squeeze(-1) + un_len
        origin_idx = origin_idx.scatter(dim=1, index=origin_src_idx, src=dst_idx)
        return origin_idx

    def init_idx(self):
        idx = torch.arange(0, self.PATCH_LEN ** 2).unsqueeze(0)
        idx_above = idx - self.PATCH_LEN
        idx_above[idx_above < 0] += self.PATCH_LEN
        idx_below = idx + self.PATCH_LEN
        idx_below[idx_below >= (self.PATCH_LEN ** 2)] -= self.PATCH_LEN
        idx_left = idx - 1
        idx_left[(idx_left % self.PATCH_LEN) == (self.PATCH_LEN - 1)] += 1
        idx_right = idx + 1
        idx_right[(idx_right % self.PATCH_LEN) == 0] -= 1
        idx_left_above = idx - (self.PATCH_LEN + 1)
        idx_left_above[idx < self.PATCH_LEN] += (self.PATCH_LEN + 1)
        idx_left_above[(idx % self.PATCH_LEN) == 0] += (self.PATCH_LEN + 1)
        idx_left_above[:, 0] -= (self.PATCH_LEN + 1)
        idx_left_below = idx + (self.PATCH_LEN - 1)
        idx_left_below[idx > self.PATCH_LEN * (self.PATCH_LEN - 1)] -= (self.PATCH_LEN - 1)
        idx_left_below[(idx % self.PATCH_LEN) == 0] -= (self.PATCH_LEN - 1)
        idx_right_above = idx - (self.PATCH_LEN - 1)
        idx_right_above[idx < (self.PATCH_LEN - 1)] += (self.PATCH_LEN - 1)
        idx_right_above[idx % self.PATCH_LEN == (self.PATCH_LEN - 1)] += (self.PATCH_LEN - 1)
        idx_right_below = idx + (self.PATCH_LEN + 1)
        idx_right_below[idx >= self.PATCH_LEN * (self.PATCH_LEN - 1)] -= (self.PATCH_LEN + 1)
        idx_right_below[idx % self.PATCH_LEN == (self.PATCH_LEN - 1)] -= (self.PATCH_LEN + 1)
        idx_right_below[:, (self.PATCH_LEN ** 2 - 1)] += (self.PATCH_LEN + 1)

        self.idx = idx.cuda()
        self.idx_above = idx_above.cuda()
        self.idx_below = idx_below.cuda()
        self.idx_left = idx_left.cuda()
        self.idx_right = idx_right.cuda()
        self.idx_left_above = idx_left_above.cuda()
        self.idx_left_below = idx_left_below.cuda()
        self.idx_right_above = idx_right_above.cuda()
        self.idx_right_below = idx_right_below.cuda()

class CascadedGroupAttention(nn.Module):
    def __init__(self, dim, key_dim, num_heads=3, attn_ratio=4, dct_channels=192, kernels=[1, 1, 1]):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = int(key_dim)
        self.attn_ratio = attn_ratio

        qkvs = []
        dws = []
        for i in range(num_heads):
            qkvs.append(nn.Sequential(
                nn.Conv2d(dim // num_heads, self.key_dim * 2 + self.d, kernel_size=1),
                nn.BatchNorm2d(self.key_dim * 2 + self.d)
            ))
            # Ensure the depthwise convolution keeps the key_dim consistent
            dws.append(nn.Sequential(
                nn.Conv2d(self.key_dim, self.key_dim, kernel_size=kernels[i], stride=1, padding=kernels[i] // 2, groups=self.key_dim),
                nn.BatchNorm2d(self.key_dim)
            ))
        self.qkvs = nn.ModuleList(qkvs)
        self.dws = nn.ModuleList(dws)
        self.proj = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.d * num_heads, dct_channels, kernel_size=1),
            nn.BatchNorm2d(dct_channels)
        )

    def forward(self, x):  # x (B, C, H, W) - DCT features
        B, C, H, W = x.shape
        feats_in = x.chunk(self.num_heads, dim=1)  # Split DCT features for each head
        feats_out = []
        feat = feats_in[0]

        for i, qkv in enumerate(self.qkvs):
            if i > 0:  # Add the previous output to the input
                feat = feat + feats_in[i]

            feat = qkv(feat)
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1)
            q = self.dws[i](q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)  # Flatten spatial dimensions
            attn = (q.transpose(-2, -1) @ k) * self.scale  # Compute attention
            attn = attn.softmax(dim=-1)  # Normalize attention weights

            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W)  # Apply attention
            feats_out.append(feat)

        # Combine all the heads' features
        x = self.proj(torch.cat(feats_out, dim=1))  # Concatenate along channels
        return x






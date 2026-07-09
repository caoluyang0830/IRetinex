import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
import math
import warnings
from torch.nn.init import _calculate_fan_in_and_fan_out
from pdb import set_trace as stx
# import cv2
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os


def save_image_with_unique_name(image, filename):
    base_name, ext = os.path.splitext(filename)
    index = 0
    while os.path.exists(filename):
        index += 1
        filename = f"{base_name}_{index}{ext}"
    image.save(filename)

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias, stride=stride)



class MRES(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()

        self.mul_att1 = MutualAttention(dim=dim, dim_head=dim_head, heads=heads)
        self.mul_att2 = MutualAttention(dim=dim, dim_head=dim_head, heads=heads)
    def forward(self, x_in, illu_fea_trans):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        # x_in=self.mul_att1(x_in,illu_fea_trans)
        # illu_fea_trans = self.mul_att2(illu_fea_trans, x_in)
        x_in = x_in.permute(0, 3, 1, 2)
        illu_fea_trans = illu_fea_trans.permute(0, 3, 1, 2)
        # out_x=self.conv1(x_in)
        # out_illu=self.conv2(illu_fea_trans)

        out_c=self.mul_att1(x_in.permute(0,2,3,1),illu_fea_trans.permute(0,2,3,1))
        out_p = self.mul_att2(illu_fea_trans.permute(0,2,3,1), x_in.permute(0,2,3,1))
        out = out_c + out_p
        # out=out.permute(0,2,3,1)

        return out

class MutualAttention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.upsample_x = nn.ConvTranspose2d(dim, dim, stride=2, kernel_size=2, padding=0,
                                            output_padding=0)
        self.upsample_illu = nn.ConvTranspose2d(dim, dim, stride=2, kernel_size=2, padding=0,
                                            output_padding=0)
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.dim = dim

    def forward(self, x_in, illu_fea_trans):
        """
        x_in: [b,h,w,c]         # input_feature
        illu_fea: [b,h,w,c]         # mask shift? 为什么是 b, h, w, c?
        return out: [b,h,w,c]
        """
        super_x_in = self.upsample_x(x_in.permute(0,3,1,2)).permute(0,2,3,1)
        super_illu_fea_trans = self.upsample_illu(illu_fea_trans.permute(0,3,1,2)).permute(0,2,3,1)
        b, h, w, c = x_in.shape
        _, s_h, s_w, _ = super_x_in.shape
        s_x = super_x_in.reshape(b, s_h*s_w, c)
        x = x_in.reshape(b, h * w, c)
        illu_k_inp = self.to_k(super_illu_fea_trans.reshape(b, s_h * s_w, c))
        q_inp = self.to_q(s_x)
        v_inp = self.to_v(x)
        illu_attn = illu_fea_trans  # illu_fea: b,c,h,w -> b,h,w,c
        q, k, v, illu_attn = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                                 (q_inp, illu_k_inp, v_inp, illu_attn.flatten(1, 2)))
        v = v * illu_attn
        # q: b,heads,hw,c

        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        out_c = self.proj(x).view(b, h, w, c)
        out = out_c

        return out


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1,
                      bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class RCMBlock(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_blocks=2,
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(nn.ModuleList([
                MRES(dim=dim, dim_head=dim_head, heads=heads),
                PreNorm(dim, FeedForward(dim=dim))
            ]))

    def forward(self, x, illu_fea):
        """
        x: [b,c,h,w]
        illu_fea: [b,c,h,w]
        return out: [b,c,h,w]
        """
        x = x.permute(0, 2, 3, 1)
        for (attn, ff) in self.blocks:
            x = attn(x, illu_fea_trans=illu_fea.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        out = x.permute(0, 3, 1, 2)

        return out,illu_fea

class ICRR(nn.Module):
    def __init__(
            self, n_fea_middle, n_fea_in=8, n_fea_out=6):  # __init__部分是内部属性，而forward的输入才是外部输入
        super(ICRR, self).__init__()

        self.conv1 = nn.Conv2d(n_fea_in, n_fea_middle, kernel_size=1, bias=True)
        self.conv1_1 = nn.Conv2d(n_fea_out * 2, n_fea_middle, kernel_size=1, bias=True)

        # self.depth_conv = ASPP(n_fea_middle,n_fea_middle,[6,12,18])
        # self.depth_conv_1 = ASPP(n_fea_middle,n_fea_middle,[6,12,18])
        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
        self.depth_conv_1 = nn.Conv2d(
            n_fea_middle, n_fea_middle, kernel_size=5, padding=2, bias=True, groups=n_fea_in)
        self.conv2 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)
        self.conv2_1 = nn.Conv2d(n_fea_middle, n_fea_out, kernel_size=1, bias=True)

        self.upsample = nn.Sequential(nn.ConvTranspose2d(n_fea_out, n_fea_out, 2, stride=2, padding=0),
                                      nn.BatchNorm2d(n_fea_out),
                                      nn.ReLU())
        self.downsample1 = nn.Sequential(nn.Conv2d(n_fea_out, n_fea_out, 4, 2, 1, bias=False),
                                         nn.BatchNorm2d(n_fea_out),
                                         nn.ReLU())
        self.downsample2 = nn.Sequential(nn.Conv2d(n_fea_out, n_fea_out, 4, 2, 1, bias=False),
                                         nn.BatchNorm2d(n_fea_out),
                                         nn.ReLU())

    def forward(self, img):
        # img:        b,c=3,h,w
        # mean_c:     b,c=1,h,w

        # illu_fea:   b,c,h,w
        # illu_map:   b,c=3,h,w

        mean_c_rgb = img[:,:3,...].mean(dim=1).unsqueeze(1)
        mean_c_hsv = img[:,-2:-1,...]

        # stx()
        input = torch.cat([img, mean_c_rgb,mean_c_hsv], dim=1)

        x_1 = self.conv1(input)
        illu_fea = self.depth_conv(x_1)
        illu_map = self.conv2(illu_fea)

        reflect_image=F.softmax(img/(illu_map+1e-8),dim=1)
        x_2 = self.conv1_1(torch.cat([img, reflect_image], dim=1))
        reflect_fea = self.depth_conv_1(x_2)
        reflect_map = self.conv2_1(reflect_fea)

        return reflect_map, illu_map

class RCM(nn.Module):
    def __init__(self, in_dim=3, out_dim=3, dim=31, level=4, num_blocks=[1, 2, 2, 4, 4]):
        super(RCM, self).__init__()
        self.dim = dim
        self.level = level
        if len(num_blocks) < level + 1:
            raise ValueError(f"num_blocks must contain at least {level + 1} values for a {level + 1}-scale RCM.")

        # Input projection
        self.embedding1 = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)
        self.embedding2 = nn.Conv2d(in_dim, self.dim, 3, 1, 1, bias=False)

        # Encoder
        self.encoder_layers = nn.ModuleList([])
        dim_level = dim
        for i in range(level):
            self.encoder_layers.append(nn.ModuleList([
                RCMBlock(
                    dim=dim_level, num_blocks=num_blocks[i], dim_head=dim, heads=dim_level // dim),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False),
                nn.Conv2d(dim_level, dim_level * 2, 4, 2, 1, bias=False)
            ]))
            dim_level *= 2

        # Bottleneck
        self.bottleneck = RCMBlock(
            dim=dim_level, dim_head=dim, heads=dim_level // dim, num_blocks=num_blocks[-1])

        # Decoder
        self.decoder_layers = nn.ModuleList([])
        for i in range(level):
            self.decoder_layers.append(nn.ModuleList([
                nn.ConvTranspose2d(dim_level, dim_level // 2, stride=2,
                                   kernel_size=2, padding=0, output_padding=0),
                nn.Conv2d(dim_level, dim_level // 2, 1, 1, bias=False),
                RCMBlock(
                    dim=dim_level // 2, num_blocks=num_blocks[level - 1 - i], dim_head=dim,
                    heads=(dim_level // 2) // dim),
            ]))
            dim_level //= 2

        # Output projection
        # self.mapping = nn.Conv2d(self.dim, out_dim, 3, 1, 1, bias=False)
        self.mapping_illu = nn.ModuleList(
            [nn.Conv2d(self.dim * (2 ** i), out_dim, 3, 1, 1, bias=False) for i in range(self.level + 1)])
        self.mapping_reflect = nn.ModuleList(
            [nn.Conv2d(self.dim * (2 ** i), out_dim, 3, 1, 1, bias=False) for i in range(self.level + 1)])
        self.super_out = nn.ModuleList(
            [nn.ConvTranspose2d(out_dim, out_dim, stride=2 ** (self.level - i),
                                   kernel_size=4, padding=0, output_padding=0) for i in range(self.level)])
        self.super_x =nn.ConvTranspose2d(out_dim, out_dim, stride=2,
                                   kernel_size=4, padding=0, output_padding=0)
        self.finalconv = nn.Conv2d(out_dim, out_dim, 4, 2, 1, bias=False)
        # self.dropout_illu = nn.Dropout2d(p=0.5, inplace=False)
        # self.dropout_reflect = nn.Dropout2d(p=0.5, inplace=False)
        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, illu_fea,img):
        """
        x:          [b,c,h,w]         x是feature, 不是image
        illu_fea:   [b,c,h,w]
        return out: [b,c,h,w]
        """

        # Embedding
        fea = self.embedding1(x)
        illu_fea = self.embedding2(illu_fea)

        # Encoder
        fea_encoder = []
        illu_fea_list = []
        illu_fea_decoder = []
        fea_decoder = []
        for (rcm_block, FeaDownSample, IlluFeaDownsample) in self.encoder_layers:
            fea, illu_fea = rcm_block(fea, illu_fea)  # bchw
            illu_fea_list.append(illu_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            illu_fea = IlluFeaDownsample(illu_fea)

        # Bottleneck
        fea, illu_fea = self.bottleneck(fea, illu_fea)
        # fea = self.dropout_reflect(fea)
        fea_decoder.append(fea)
        illu_fea_decoder.append(illu_fea)

        # Decoder
        for i, (FeaUpSample, Fution, LeWinBlcok) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fution(
                torch.cat([fea, fea_encoder[self.level - 1 - i]], dim=1))
            illu_fea = illu_fea_list[self.level - 1 - i]
            fea, illu_fea = LeWinBlcok(fea, illu_fea)
            fea_decoder.append(fea)
            illu_fea_decoder.append(illu_fea)

        # Mapping
        out = []
        for i in range(self.level + 1):
            level_idx = self.level - i
            reflect_map = self.mapping_reflect[level_idx](fea_decoder[i])
            illu_map = self.mapping_illu[level_idx](illu_fea_decoder[i])
            out_image = reflect_map * illu_map
            out_image = F.interpolate(out_image, scale_factor=2 ** level_idx, mode="nearest")
            # out_image=self.finnal(out_image)
            # print(out_image.shape)
            out.append(out_image)
        #
        #     # # Mapping
        #     # out = self.mapping(fea)*self.mapping(illu_fea)+x
        #     print(len(fea_decoder))
        #     print(torch.max(illu_fea_decoder[1]))
        #     plt.subplot(2, 2, 1)
        #     input_image_show = self.mapping_reflect[0](fea_decoder[1])
        #     input_image_show = input_image_show[0, :3, :, :]
        #     # 找到最小值和最大值
        #     min_val = input_image_show.min()
        #     max_val = input_image_show.max()
        #
        #     # 归一化张量
        #     input_image_show = (input_image_show - min_val) / (max_val - min_val)
        #     # input_image_show_rgb = torch.zeros_like(input_image_show)
        #     # input_image_show_rgb[0, :, :] = input_image_show[2, :, :]
        #     # input_image_show_rgb[1, :, :] = input_image_show[1, :, :]
        #     # input_image_show_rgb[2, :, :] = input_image_show[0, :, :]
        #     input_image_show = input_image_show.cpu()
        #     plt.imshow(input_image_show.permute(1, 2, 0).detach().numpy())  # 如果是灰度图，使用 'gray' 颜色映射
        #     plt.title("inputimage")
        #
        #     output_image_show = self.mapping_illu[0](illu_fea_decoder[1])
        #     output_image_show = output_image_show[0, :3, :, :]
        #     # output_image_show = illu_fea_decoder[1]
        #     # output_image_show = output_image_show[0, :3, :, :]
        #     # 找到最小值和最大值
        #     min_val = output_image_show.min()
        #     max_val = output_image_show.max()
        #
        #     # 归一化张量
        #     output_image_show = (output_image_show - min_val) / (max_val - min_val)
        #     output_image_show = (1 - output_image_show).cpu()
        #     plt.subplot(2, 2, 2)
        #     plt.imshow(output_image_show[0, :, :].detach().numpy(), cmap="OrRd")  # 如果是灰度图，使用 'gray' 颜色映射
        #     plt.title("outputimage")
        #     plt.subplot(2, 2, 3)
        #     plt.imshow(output_image_show[1, :, :].detach().numpy(), cmap="gray")  # 如果是灰度图，使用 'gray' 颜色映射
        #     plt.title("outputimage")
        #     plt.subplot(2, 2, 4)
        #     plt.imshow(output_image_show[2, :, :].detach().numpy(), cmap="gray")  # 如果是灰度图，使用 'gray' 颜色映射
        #     plt.title("outputimage")
        #
        #     reflect_array = (input_image_show.permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)
        #     # 将 numpy 数组转换为 PIL 图像
        #     reflect_image = Image.fromarray(reflect_array)
        #     # 保存图像
        #     save_image_with_unique_name(reflect_image, "/root/autodl-tmp/Retinexformer-test4/Enhancement/my3/reflect_image.png")
        #
        #     illu_array = (output_image_show[0, :, :].detach().numpy() * 255).astype(np.uint8)
        #     # 将 numpy 数组转换为 PIL 图像
        #     illu_image = Image.fromarray(illu_array)
        #     # 保存图像
        #     save_image_with_unique_name(illu_image, "/root/autodl-tmp/Retinexformer-test4/Enhancement/my3/illu_image.png")
        #
        #     plt.show()
        return out

class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

class DownsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat,input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat,num_feat,  4, scale, 1, bias=False))
        m.append(nn.Conv2d(num_feat,num_feat, 3, 1, 1))
        super(DownsampleOneStep, self).__init__(*m)

class IRetinexStage(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=31, level=4, num_blocks=[1, 2, 2, 4, 4]):
        super(IRetinexStage, self).__init__()
        self.icrr = ICRR(n_feat)
        self.rcm = RCM(in_dim=in_channels, out_dim=out_channels, dim=n_feat, level=level,
                       num_blocks=num_blocks)


        self.finnal = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
    def forward(self, img):

        reflect_map, illu_map = self.icrr(img)



        # # # Mapping
        # # out = self.mapping(fea)*self.mapping(illu_fea)+x
        #
        # plt.subplot(2, 2, 1)
        # input_image_show = illu_map+img
        # input_image_show = input_image_show[0, :3, :, :]
        # # 找到最小值和最大值
        # min_val = input_image_show.min()
        # max_val = input_image_show.max()
        #
        # # 归一化张量
        # input_image_show = (input_image_show - min_val) / (max_val - min_val)
        # # input_image_show_rgb = torch.zeros_like(input_image_show)
        # # input_image_show_rgb[0, :, :] = input_image_show[2, :, :]
        # # input_image_show_rgb[1, :, :] = input_image_show[1, :, :]
        # # input_image_show_rgb[2, :, :] = input_image_show[0, :, :]
        # input_image_show = input_image_show.cpu()
        # # plt.imshow(input_image_show.permute(1, 2, 0)[: :, 1].detach().numpy())  # 如果是灰度图，使用 'gray' 颜色映射
        # # plt.title("inputimage")
        #
        # output_image_show = reflect_map+img
        # output_image_show = output_image_show[0, :3, :, :]
        # # output_image_show = illu_fea_decoder[1]
        # # output_image_show = output_image_show[0, :3, :, :]
        # # 找到最小值和最大值
        # min_val = output_image_show.min()
        # max_val = output_image_show.max()
        #
        # # 归一化张量
        # output_image_show = (output_image_show - min_val) / (max_val - min_val)
        # output_image_show = output_image_show.cpu()
        # # output_image_show = (1 - output_image_show).cpu()
        # # plt.subplot(2, 2, 2)
        # # plt.imshow(output_image_show[0, :, :].detach().numpy(), cmap="OrRd")  # 如果是灰度图，使用 'gray' 颜色映射
        # # plt.title("outputimage")
        # # plt.subplot(2, 2, 3)
        # # plt.imshow(output_image_show[1, :, :].detach().numpy(), cmap="gray")  # 如果是灰度图，使用 'gray' 颜色映射
        # # plt.title("outputimage")
        # # plt.subplot(2, 2, 4)
        # # plt.imshow(output_image_show[2, :, :].detach().numpy(), cmap="gray")  # 如果是灰度图，使用 'gray' 颜色映射
        # # plt.title("outputimage")
        #
        # reflect_array = (input_image_show.permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)
        # # 将 numpy 数组转换为 PIL 图像
        # reflect_image = Image.fromarray(reflect_array)
        # # 保存图像
        # save_image_with_unique_name(reflect_image, "/root/autodl-tmp/Retinexformer-test4/Enhancement/my3/reflect_image.png")
        #
        # illu_array = (output_image_show.permute(1, 2, 0).detach().numpy() * 255).astype(np.uint8)
        # # 将 numpy 数组转换为 PIL 图像
        # illu_image = Image.fromarray(illu_array)
        # # 保存图像
        # save_image_with_unique_name(illu_image, "/root/autodl-tmp/Retinexformer-test4/Enhancement/my3/illu_image.png")
        #
        # plt.show()
        output_img = self.rcm(reflect_map, illu_map, img)

        return reflect_map, illu_map,output_img


class IRetinex(nn.Module):
    def __init__(self, in_channels=6, out_channels=6, n_feat=31, stage=3, level=4, num_blocks=[1, 2, 2, 4, 4]):
        super(IRetinex, self).__init__()
        self.stage = stage

        modules_body = [
            IRetinexStage(in_channels=in_channels, out_channels=out_channels, n_feat=n_feat, level=level,
                          num_blocks=num_blocks)
            for _ in range(stage)]

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        """
        x: [b,c,h,w]
        return out:[b,c,h,w]
        """
        reflect_map, illu_map,out = self.body(x)

        return reflect_map, illu_map,out

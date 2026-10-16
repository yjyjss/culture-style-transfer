# -*- coding: utf-8 -*-
"""
stylize.py — Culture-Aware Style Transfer (MVP)

实现文化符号结构保持的风格迁移，用于验证假设 H1a：
  显式注入"符号结构保持损失 L_ssp"能提升符号结构保持率，且不损害风格。

方法：
  - 基线 (Gatys)：L = L_content + lambda_style * L_style
  - 本文 (Ours) ：L = L_content + lambda_style * L_style + lambda_ssp * L_ssp

符号结构：用可微 Sobel 边缘算子从风格图提取符号结构图 E_style 与符号掩码 M_sym；
L_ssp 约束迁移结果在符号区域的边缘结构与风格符号结构对齐，避免符号被打散。
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
import numpy as np

# ---------------- Encoder (VGG19, frozen) ----------------
class VGGEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        self.slices = nn.ModuleDict({
            'r11': vgg[:2],    # relu1_1
            'r21': vgg[2:7],   # relu2_1
            'r31': vgg[7:12],  # relu3_1
            'r41': vgg[12:21], # relu4_1
            'r51': vgg[21:30], # relu5_1
        })
        # VGG 归一化
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = (x - self.mean) / self.std
        feats = {}
        for name, sl in self.slices.items():
            x = sl(x)
            feats[name] = x
        return feats


# ---------------- 可微符号结构提取 (Sobel) ----------------
def _sobel_kernels(device):
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1., -2., -1.],
                            [0., 0., 0.],
                            [1., 2., 1.]], device=device).view(1, 1, 3, 3)
    return sobel_x, sobel_y


def edge_map(img_rgb):
    """可微 Sobel 边缘幅度图，输入 Bx3xHxW 输出 Bx1xHxW (0~1)。"""
    gray = 0.299 * img_rgb[:, 0:1] + 0.587 * img_rgb[:, 1:2] + 0.114 * img_rgb[:, 2:3]
    sx, sy = _sobel_kernels(img_rgb.device)
    gx = F.conv2d(gray, sx, padding=1)
    gy = F.conv2d(gray, sy, padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-8)
    mag = mag / (mag.max() + 1e-8)
    return mag


def symbol_mask(edge, thresh=0.2):
    """由边缘图得到符号显著性掩码（二值化）。"""
    return (edge > thresh).float()


# ---------------- 损失函数 ----------------
def gram(x):
    b, c, h, w = x.shape
    f = x.view(b, c, -1)
    return torch.bmm(f, f.transpose(1, 2)) / (c * h * w)


def content_loss(feat_x, feat_c, layers=('r41',)):
    loss = 0.0
    for l in layers:
        loss += F.mse_loss(feat_x[l], feat_c[l])
    return loss


def style_loss(feat_x, feat_s, layers=('r11', 'r21', 'r31', 'r41', 'r51')):
    loss = 0.0
    for l in layers:
        loss += F.mse_loss(gram(feat_x[l]), gram(feat_s[l]))
    return loss


def ssp_loss(img_x, style_edge, mask_sym):
    """符号结构保持损失：结果在符号区域的边缘与风格符号边缘对齐。"""
    ex = edge_map(img_x)
    # 只在符号区域计算
    diff = (ex - style_edge) * mask_sym
    return torch.mean(diff * diff)


# ---------------- 图像处理 ----------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_img(path, size=256, device='cpu'):
    img = Image.open(path).convert('RGB')
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    t = torch.from_numpy(arr.transpose(2, 0, 1)).unsqueeze(0).to(device).contiguous()
    return t


def save_img(tensor, path):
    arr = tensor.detach().cpu().squeeze(0).numpy()
    arr = arr * IMAGENET_STD[:, None, None] + IMAGENET_MEAN[:, None, None]
    arr = np.clip(arr, 0, 1).transpose(1, 2, 0) * 255
    Image.fromarray(arr.astype(np.uint8)).save(path)


# ---------------- 迁移（Gatys 式优化） ----------------
def transfer(content_path, style_path, out_path, encoder,
             lambda_style=1e6, lambda_ssp=0.0, steps=400, lr=0.03,
             size=256, device='cpu'):
    c_img = load_img(content_path, size, device)
    s_img = load_img(style_path, size, device)

    with torch.no_grad():
        feat_c = encoder(c_img)
        feat_s = encoder(s_img)

    # 符号结构（从风格图提取）
    style_edge = edge_map(s_img)
    mask_sym = symbol_mask(style_edge)

    # 迁移结果从内容图初始化
    x = c_img.clone().contiguous().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)

    for i in range(steps):
        opt.zero_grad()
        feat_x = encoder(x)
        lc = content_loss(feat_x, feat_c)
        ls = lambda_style * style_loss(feat_x, feat_s)
        lssp = lambda_ssp * ssp_loss(x, style_edge, mask_sym)
        loss = lc + ls + lssp
        if torch.isnan(loss):
            break  # 数值发散保护：停止优化，保存当前结果
        loss.backward()
        torch.nn.utils.clip_grad_norm_([x], 5.0)
        opt.step()
        with torch.no_grad():
            x.clamp_(-2.0, 2.0)  # 约 [0,1] 图像域对应的归一化范围

    save_img(x, out_path)
    return x


# ---------------- 评估：符号结构保持率 ----------------
def symbol_preservation(img_tensor, style_edge, mask_sym):
    """符号区域边缘结构的余弦相似度（方向一致性，越高越保持）。"""
    ex = edge_map(img_tensor)
    e1 = (ex * mask_sym).reshape(-1)
    e2 = (style_edge * mask_sym).reshape(-1)
    denom = (torch.norm(e1) * torch.norm(e2)) + 1e-8
    cos = torch.dot(e1, e2) / denom
    return float(cos.item())


if __name__ == '__main__':
    import sys
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = VGGEncoder().to(dev).eval()
    content, style, out = sys.argv[1], sys.argv[2], sys.argv[3]
    use_ssp = (len(sys.argv) > 4 and sys.argv[4] == 'ssp')
    transfer(content, style, out, enc, lambda_ssp=200.0 if use_ssp else 0.0,
             steps=300, size=256, device=dev)
    print(f"done -> {out} (ssp={use_ssp}, device={dev})")

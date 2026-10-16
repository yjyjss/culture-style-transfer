# -*- coding: utf-8 -*-
"""
multiscale_v3.py — 混合方案多尺度解耦（结合 v1 色彩层 + v2 频率分解符号层）

三层：
  - 色彩层 color   ：特征空间均值/方差对齐（v1 已验证有效，去掉后变灰）
  - 符号层 symbol  ：中频带（轻模糊-强模糊）边缘结构保持（v2 已验证有效）
  - 纹理层 texture ：高频带（原图-轻模糊）能量匹配

干预实验：固定两层、仅干预一层。
"""
import os
import torch
import torch.nn.functional as F
from stylize import (VGGEncoder, load_img, save_img, edge_map, symbol_mask,
                     content_loss, style_loss)


def gaussian_kernel(size, sigma, device):
    x = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    k2d = (g[:, None] * g[None, :]).view(1, 1, size, size)
    return k2d.repeat(3, 1, 1, 1)


def gaussian_blur(img, sigma, device):
    size = int(2 * round(3 * sigma) + 1)
    if size % 2 == 0:
        size += 1
    k = gaussian_kernel(size, sigma, device)
    return F.conv2d(img, k, padding=size // 2, groups=3)


def color_loss_feat(feat_x, feat_s, layer='r41'):
    """色彩层：特征空间均值/方差对齐（v1 有效方式）。"""
    fx, fs = feat_x[layer], feat_s[layer]
    return (F.mse_loss(fx.mean(dim=[2, 3], keepdim=True), fs.mean(dim=[2, 3], keepdim=True)) +
            F.mse_loss(fx.var(dim=[2, 3], keepdim=True), fs.var(dim=[2, 3], keepdim=True)))


def symbol_loss_mid(x, style, sigma_high=1.5, sigma_low=5.0, device='cpu'):
    """符号层：中频带边缘结构保持（v2 有效方式）。"""
    b_high = gaussian_blur(x, sigma_high, device)
    b_low = gaussian_blur(x, sigma_low, device)
    mid_x = b_high - b_low
    b_high_s = gaussian_blur(style, sigma_high, device)
    b_low_s = gaussian_blur(style, sigma_low, device)
    mid_s = b_high_s - b_low_s
    ex = edge_map(mid_x)
    es = edge_map(mid_s)
    mask = symbol_mask(es)
    return torch.mean(((ex - es) * mask) ** 2)


def texture_loss_high(x, style, sigma=1.5, device='cpu'):
    """纹理层：高频带能量匹配。"""
    hx = x - gaussian_blur(x, sigma, device)
    hs = style - gaussian_blur(style, sigma, device)
    return (F.mse_loss(hx.var(dim=[2, 3]), hs.var(dim=[2, 3])))


def transfer_v3(content_path, style_path, out_path, encoder,
                w_color=0.0, w_symbol=0.0, w_texture=0.0,
                steps=400, lr=0.03, size=256, device='cpu'):
    c_img = load_img(content_path, size, device)
    s_img = load_img(style_path, size, device)
    with torch.no_grad():
        feat_c = encoder(c_img)
        feat_s = encoder(s_img)

    x = c_img.clone().contiguous().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    for i in range(steps):
        opt.zero_grad()
        feat_x = encoder(x)
        lc = content_loss(feat_x, feat_c)
        lcol = color_loss_feat(feat_x, feat_s)
        lsym = symbol_loss_mid(x, s_img, device=device)
        ltex = texture_loss_high(x, s_img, device=device)
        loss = lc + w_color * lcol + w_symbol * lsym + w_texture * ltex
        if torch.isnan(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_([x], 5.0)
        opt.step()
        with torch.no_grad():
            x.clamp_(-2.0, 2.0)
    save_img(x, out_path)
    return x


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = VGGEncoder().to(dev).eval()
    cp = 'data/content/teaset.png'
    sp = 'data/style/dunhuang_lotus.png'
    out_dir = 'output/multiscale_v3'
    os.makedirs(out_dir, exist_ok=True)

    W_COLOR, W_SYMBOL, W_TEXTURE = 1e5, 1e5, 1e3
    configs = {
        'full':       (W_COLOR, W_SYMBOL, W_TEXTURE),
        'no_color':   (0.0, W_SYMBOL, W_TEXTURE),
        'no_symbol':  (W_COLOR, 0.0, W_TEXTURE),
        'no_texture': (W_COLOR, W_SYMBOL, 0.0),
    }
    for tag, (wc, ws, wt) in configs.items():
        out = os.path.join(out_dir, f'{tag}.png')
        transfer_v3(cp, sp, out, enc, w_color=wc, w_symbol=ws,
                    w_texture=wt, steps=400, size=256, device=dev)
        print(f"[{tag}] color={wc:.0e} symbol={ws:.0e} texture={wt:.0e}")

    print("hybrid multiscale done")


if __name__ == '__main__':
    main()

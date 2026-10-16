# -*- coding: utf-8 -*-
"""
multiscale_v2.py — 频率分解的多尺度文化风格解耦（Exp-2 改进）

用频率带分解解决符号层/纹理层信息重叠问题：
  - 低频（色彩层） ：强模糊后的全局色调 -> 均值/方差对齐
  - 中频（符号层） ：轻模糊 - 强模糊 = 结构边缘（对称纹样/母题轮廓）-> 边缘结构保持
  - 高频（纹理层） ：原图 - 轻模糊 = 笔触/皴法/晕染细节 -> 高频能量匹配

三层在频率上正交，干预实验可清晰展示各层独立可控。
"""
import os
import torch
import torch.nn.functional as F
from stylize import (VGGEncoder, load_img, save_img, edge_map, symbol_mask,
                     content_loss, symbol_preservation)


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


def freq_decompose(img, sigma_high=1.5, sigma_low=5.0, device='cpu'):
    """分解为 high(纹理)/mid(符号)/low(色彩) 三带。"""
    b_high = gaussian_blur(img, sigma_high, device)
    b_low = gaussian_blur(img, sigma_low, device)
    high = img - b_high        # 高频纹理
    mid = b_high - b_low       # 中频符号结构
    low = b_low                # 低频色彩
    return high, mid, low


# ---------------- 三层损失（图像空间，频率正交） ----------------
def color_loss_low(x, style, device):
    _, _, lx = freq_decompose(x, device=device)
    _, _, ls = freq_decompose(style, device=device)
    return (F.mse_loss(lx.mean(dim=[2, 3]), ls.mean(dim=[2, 3])) +
            F.mse_loss(lx.var(dim=[2, 3]), ls.var(dim=[2, 3])))


def symbol_loss_mid(x, style, device):
    _, mx, _ = freq_decompose(x, device=device)
    _, ms, _ = freq_decompose(style, device=device)
    ex = edge_map(mx)
    es = edge_map(ms)
    mask = symbol_mask(es)
    return torch.mean(((ex - es) * mask) ** 2)


def texture_loss_high(x, style, device):
    hx, _, _ = freq_decompose(x, device=device)
    hs, _, _ = freq_decompose(style, device=device)
    return (F.mse_loss(hx.mean(dim=[2, 3]), hs.mean(dim=[2, 3])) +
            F.mse_loss(hx.var(dim=[2, 3]), hs.var(dim=[2, 3])))


def transfer_freq(content_path, style_path, out_path, encoder,
                  w_color=0.0, w_symbol=0.0, w_texture=0.0,
                  steps=400, lr=0.03, size=256, device='cpu'):
    c_img = load_img(content_path, size, device)
    s_img = load_img(style_path, size, device)
    with torch.no_grad():
        feat_c = encoder(c_img)

    x = c_img.clone().contiguous().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    for i in range(steps):
        opt.zero_grad()
        feat_x = encoder(x)
        lc = content_loss(feat_x, feat_c)
        lcol = color_loss_low(x, s_img, device)
        lsym = symbol_loss_mid(x, s_img, device)
        ltex = texture_loss_high(x, s_img, device)
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
    out_dir = 'output/multiscale_v2'
    os.makedirs(out_dir, exist_ok=True)

    W_COLOR, W_SYMBOL, W_TEXTURE = 1e4, 1e5, 1e3
    configs = {
        'full':       (W_COLOR, W_SYMBOL, W_TEXTURE),
        'no_color':   (0.0, W_SYMBOL, W_TEXTURE),
        'no_symbol':  (W_COLOR, 0.0, W_TEXTURE),
        'no_texture': (W_COLOR, W_SYMBOL, 0.0),
    }
    for tag, (wc, ws, wt) in configs.items():
        out = os.path.join(out_dir, f'{tag}.png')
        transfer_freq(cp, sp, out, enc, w_color=wc, w_symbol=ws,
                      w_texture=wt, steps=400, size=256, device=dev)
        print(f"[{tag}] color={wc:.0e} symbol={ws:.0e} texture={wt:.0e} -> {out}")

    print("freq-decompose intervention done")


if __name__ == '__main__':
    main()

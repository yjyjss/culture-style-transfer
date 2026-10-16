# -*- coding: utf-8 -*-
"""
multiscale.py — 多尺度文化风格解耦与分层控制（Exp-2，贡献 2）

将文化风格解耦为三个可独立干预的层：
  - 色彩层 color   ：全局颜色统计量对齐（AdaIN 式均值/方差）
  - 纹理层 texture ：Gram 矩阵（笔触/皴法/晕染质感）
  - 符号层 symbol  ：符号结构保持（对称纹样/母题轮廓）

干预实验：固定两层、仅干预一层，验证三层可独立控制。
"""
import os
import glob
import csv
import torch
import torch.nn.functional as F
from stylize import (VGGEncoder, load_img, save_img, edge_map, symbol_mask,
                     ssp_loss, content_loss, style_loss, gram)


def color_loss(feat_x, feat_s, layers=('r41',)):
    """色彩层损失：特征均值/方差对齐（AdaIN 思想，全局色彩控制）。"""
    loss = 0.0
    for l in layers:
        fx, fs = feat_x[l], feat_s[l]
        mx, vx = fx.mean(dim=[2, 3], keepdim=True), fx.var(dim=[2, 3], keepdim=True)
        ms, vs = fs.mean(dim=[2, 3], keepdim=True), fs.var(dim=[2, 3], keepdim=True)
        loss += F.mse_loss(mx, ms) + F.mse_loss(vx, vs)
    return loss


def transfer_multiscale(content_path, style_path, out_path, encoder,
                        w_color=0.0, w_texture=0.0, w_symbol=0.0,
                        steps=400, lr=0.03, size=256, device='cpu'):
    c_img = load_img(content_path, size, device)
    s_img = load_img(style_path, size, device)
    with torch.no_grad():
        feat_c = encoder(c_img)
        feat_s = encoder(s_img)
    style_edge = edge_map(s_img)
    mask_sym = symbol_mask(style_edge)

    x = c_img.clone().contiguous().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    for i in range(steps):
        opt.zero_grad()
        feat_x = encoder(x)
        lc = content_loss(feat_x, feat_c)
        lcol = color_loss(feat_x, feat_s)
        ltex = style_loss(feat_x, feat_s)
        lsym = ssp_loss(x, style_edge, mask_sym)
        loss = lc + w_color * lcol + w_texture * ltex + w_symbol * lsym
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
    out_dir = 'output/multiscale'
    os.makedirs(out_dir, exist_ok=True)

    # 三层权重（平衡到量级可比的尺度）
    # content_loss≈400, color_loss≈(小), texture_loss(Gram)≈巨大×1e6, symbol≈0.01
    W_COLOR, W_TEXTURE, W_SYMBOL = 1e5, 1e6, 1e5

    configs = {
        'full':       (W_COLOR, W_TEXTURE, W_SYMBOL),
        'no_symbol':  (W_COLOR, W_TEXTURE, 0.0),
        'no_color':   (0.0, W_TEXTURE, W_SYMBOL),
        'no_texture': (W_COLOR, 0.0, W_SYMBOL),
    }
    for tag, (wc, wt, ws) in configs.items():
        out = os.path.join(out_dir, f'{tag}.png')
        transfer_multiscale(cp, sp, out, enc, w_color=wc, w_texture=wt,
                            w_symbol=ws, steps=400, size=256, device=dev)
        print(f"[{tag}] color={wc:.0e} texture={wt:.0e} symbol={ws:.0e} -> {out}")

    print("multiscale intervention done")


if __name__ == '__main__':
    main()

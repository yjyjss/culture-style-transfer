# -*- coding: utf-8 -*-
"""
adain_exp.py — AdaIN 前馈基线 + L_ssp 后优化（Exp-1 正式基线对比）

复现 ICCV2017 AdaIN 前馈任意风格迁移，验证 L_ssp 在前馈方法上同样有效：
  - AdaIN          : 前馈迁移（基线）
  - AdaIN + L_ssp  : 前馈结果 + 符号结构保持后优化（本文方法）

对比两者的符号结构保持率。
"""
import os
import glob
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from stylize import (VGGEncoder, load_img, save_img, edge_map, symbol_mask,
                     ssp_loss, symbol_preservation, content_loss, style_loss)


# ---------------- AdaIN decoder (naoto0804 结构) ----------------
def build_decoder(weight_path):
    decoder = nn.Sequential(
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(512, 256, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 256, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(256, 128, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 128, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(128, 64, (3, 3)),
        nn.ReLU(),
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 64, (3, 3)),
        nn.ReLU(),
        nn.ReflectionPad2d((1, 1, 1, 1)),
        nn.Conv2d(64, 3, (3, 3)),
    )
    state = torch.load(weight_path, map_location='cpu')
    if isinstance(state, dict) and 'state_dict' in state:
        state = state['state_dict']
    decoder.load_state_dict(state)
    return decoder


def adain(content_feat, style_feat, alpha=1.0):
    c_mean = content_feat.mean(dim=[2, 3], keepdim=True)
    c_std = content_feat.std(dim=[2, 3], keepdim=True) + 1e-5
    s_mean = style_feat.mean(dim=[2, 3], keepdim=True)
    s_std = style_feat.std(dim=[2, 3], keepdim=True) + 1e-5
    norm = (content_feat - c_mean) / c_std
    return alpha * (norm * s_std + s_mean) + (1 - alpha) * content_feat


def adain_forward(content_img, style_img, encoder, decoder, alpha=1.0):
    with torch.no_grad():
        cf = encoder(content_img)['r41']
        sf = encoder(style_img)['r41']
        t = adain(cf, sf, alpha)
        out = decoder(t)
    return out


def refine_full(content_img, style_img, x0, encoder,
                lambda_style=1e6, lambda_ssp=0.0, steps=200, lr=0.03):
    """以 AdaIN 输出为初始化的 Gatys 式精化（content + style + ssp），避免过度约束。"""
    with torch.no_grad():
        feat_c = encoder(content_img)
        feat_s = encoder(style_img)
    style_edge = edge_map(style_img)
    mask_sym = symbol_mask(style_edge)
    x = x0.clone().contiguous().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    for i in range(steps):
        opt.zero_grad()
        feat_x = encoder(x)
        lc = content_loss(feat_x, feat_c)
        ls = lambda_style * style_loss(feat_x, feat_s)
        lssp = lambda_ssp * ssp_loss(x, style_edge, mask_sym)
        loss = lc + ls + lssp
        if torch.isnan(loss):
            break
        loss.backward()
        torch.nn.utils.clip_grad_norm_([x], 5.0)
        opt.step()
        with torch.no_grad():
            x.clamp_(-2.0, 2.0)
    return x


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = VGGEncoder().to(dev).eval()
    dec = build_decoder('models/decoder.pth').to(dev).eval()

    contents = sorted(glob.glob('data/content/*.*'))
    styles = sorted(glob.glob('data/style/*.*'))
    out_dir = 'output/adain'
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for cp in contents:
        cname = os.path.splitext(os.path.basename(cp))[0]
        c_img = load_img(cp, 256, dev)
        for sp in styles:
            sname = os.path.splitext(os.path.basename(sp))[0]
            s_img = load_img(sp, 256, dev)
            style_edge = edge_map(s_img)
            mask_sym = symbol_mask(style_edge)

            # AdaIN 基线（纯前馈）
            x0 = adain_forward(c_img, s_img, enc, dec)
            save_img(x0, os.path.join(out_dir, f'{cname}_{sname}_adain.png'))
            sp_adain = symbol_preservation(x0, style_edge, mask_sym)

            # AdaIN 初始化 + 精化（无 L_ssp）
            x_r = refine_full(c_img, s_img, x0, enc, lambda_ssp=0.0, steps=200)
            save_img(x_r, os.path.join(out_dir, f'{cname}_{sname}_adain_refine.png'))
            sp_refine = symbol_preservation(x_r, style_edge, mask_sym)

            # AdaIN 初始化 + 精化（含 L_ssp，本文）
            x_ssp = refine_full(c_img, s_img, x0, enc, lambda_ssp=1e5, steps=200)
            save_img(x_ssp, os.path.join(out_dir, f'{cname}_{sname}_adain_ssp.png'))
            sp_ssp = symbol_preservation(x_ssp, style_edge, mask_sym)

            rows.append({'content': cname, 'style': sname,
                         'adain': round(sp_adain, 4),
                         'refine': round(sp_refine, 4),
                         'adain_ssp': round(sp_ssp, 4)})
            print(f"{cname} x {sname}: AdaIN={sp_adain:.4f} refine={sp_refine:.4f} "
                  f"+L_ssp={sp_ssp:.4f} (Δ={sp_ssp-sp_refine:+.4f})")

    import statistics
    a = [r['adain'] for r in rows]
    rf = [r['refine'] for r in rows]
    b = [r['adain_ssp'] for r in rows]
    print(f"\nAdaIN(纯前馈) 符号保持率均值:      {statistics.mean(a):.4f}")
    print(f"AdaIN+精化(无L_ssp) 均值:         {statistics.mean(rf):.4f}")
    print(f"AdaIN+精化(含L_ssp) 均值:         {statistics.mean(b):.4f}")
    print(f"L_ssp 净提升: {statistics.mean(b)-statistics.mean(rf):+.4f}")

    with open(os.path.join(out_dir, 'adain_results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['content', 'style', 'adain',
                                          'refine', 'adain_ssp'])
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()

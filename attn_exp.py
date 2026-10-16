# -*- coding: utf-8 -*-
"""
attn_exp.py — AdaAttN 基线（Adaptive Attention Normalization, Liu 2021）

复现 AdaAttN 的"注意力归一化"作为第四种风格对齐机制（注意力建立内容-风格
局部对应 + 自适应均值/方差对齐），验证 L_ssp 与其正交：
  - AdaAttN          : 匹配注意力对齐特征（无 L_ssp）
  - AdaAttN + L_ssp  : 匹配注意力对齐特征 + 符号结构保持
"""
import os
import glob
import csv
import torch
import torch.nn.functional as F
from stylize import (VGGEncoder, load_img, save_img, edge_map, symbol_mask,
                     ssp_loss, symbol_preservation)


def attn_align(cf, sf, alpha=1.0, eps=1e-5, temperature=0.1):
    """AdaAttN 式注意力归一化：注意力加权风格特征 + 自适应均值/方差对齐。"""
    c, h, w = cf.shape[1:]
    N = h * w
    cf_f = cf.view(c, N)
    sf_f = sf.view(c, N)
    # 余弦相似度注意力（内容位置 -> 风格位置）
    cf_n = F.normalize(cf_f, dim=0)
    sf_n = F.normalize(sf_f, dim=0)
    sim = cf_n.T @ sf_n                # N x N
    attn = F.softmax(sim / temperature, dim=1)  # 每个内容位置对风格位置的注意力
    # 注意力加权风格特征
    sf_w = sf_f @ attn.T               # c x N（每个内容位置对应的加权风格特征）
    # 自适应对齐
    cm = cf_f.mean(dim=1, keepdim=True)
    cv = cf_f.var(dim=1, keepdim=True)
    sm = sf_w.mean(dim=1, keepdim=True)
    sv = sf_w.var(dim=1, keepdim=True)
    aligned = (cf_f - cm) / torch.sqrt(cv + eps) * torch.sqrt(sv + eps) + sm
    out = alpha * aligned + (1 - alpha) * cf_f
    return out.view(1, c, h, w)


def attn_transfer(content_path, style_path, out_path, encoder,
                  lambda_ssp=0.0, steps=400, lr=0.03, size=256, device='cpu'):
    c_img = load_img(content_path, size, device)
    s_img = load_img(style_path, size, device)
    with torch.no_grad():
        fc = encoder(c_img)['r41']
        fs = encoder(s_img)['r41']
        target = attn_align(fc, fs)
    style_edge = edge_map(s_img)
    mask_sym = symbol_mask(style_edge)

    x = c_img.clone().contiguous().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    for i in range(steps):
        opt.zero_grad()
        fx = encoder(x)['r41']
        lc = F.mse_loss(fx, target)                     # 匹配注意力对齐特征
        lssp = lambda_ssp * ssp_loss(x, style_edge, mask_sym)
        loss = lc + lssp
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
    out_dir = 'output/attn'
    os.makedirs(out_dir, exist_ok=True)

    contents = sorted(glob.glob('data/content/*.*'))
    styles = sorted(glob.glob('data/style/*.*'))
    rows = []
    for cp in contents:
        cname = os.path.splitext(os.path.basename(cp))[0]
        for sp in styles:
            sname = os.path.splitext(os.path.basename(sp))[0]
            s_img = load_img(sp, 256, dev)
            style_edge = edge_map(s_img)
            mask_sym = symbol_mask(style_edge)

            x0 = attn_transfer(cp, sp, os.path.join(out_dir, f'{cname}_{sname}_attn.png'),
                               enc, lambda_ssp=0.0, steps=400, size=256, device=dev)
            sp0 = symbol_preservation(x0, style_edge, mask_sym)

            x1 = attn_transfer(cp, sp, os.path.join(out_dir, f'{cname}_{sname}_attn_ssp.png'),
                               enc, lambda_ssp=3e2, steps=400, size=256, device=dev)
            sp1 = symbol_preservation(x1, style_edge, mask_sym)

            rows.append({'content': cname, 'style': sname,
                         'attn': round(sp0, 4), 'attn_ssp': round(sp1, 4)})
            print(f"{cname} x {sname}: AdaAttN={sp0:.4f} AdaAttN+L_ssp={sp1:.4f} (Δ={sp1-sp0:+.4f})")

    import statistics
    a = [r['attn'] for r in rows]
    b = [r['attn_ssp'] for r in rows]
    print(f"\nAdaAttN 符号保持率均值:     {statistics.mean(a):.4f}")
    print(f"AdaAttN+L_ssp 均值:         {statistics.mean(b):.4f}")
    print(f"L_ssp 净提升:               {statistics.mean(b)-statistics.mean(a):+.4f}")

    with open(os.path.join(out_dir, 'attn_results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['content', 'style', 'attn', 'attn_ssp'])
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()

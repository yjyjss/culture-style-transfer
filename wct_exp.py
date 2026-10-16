# -*- coding: utf-8 -*-
"""
wct_exp.py — WCT 基线（Universal Style Transfer via Feature Transforms, Li 2017）

复现 WCT 的"白化-着色"对齐作为第三种风格对齐机制（与 Gram、均值方差并列），
验证 L_ssp 与 WCT 对齐正交有效：
  - WCT          : 匹配 WCT 对齐特征（无 L_ssp）
  - WCT + L_ssp  : 匹配 WCT 对齐特征 + 符号结构保持
"""
import os
import glob
import csv
import torch
import torch.nn.functional as F
from stylize import (VGGEncoder, load_img, save_img, edge_map, symbol_mask,
                     ssp_loss, symbol_preservation)


def wct_align(cf, sf, alpha=1.0, eps=1e-5):
    """白化内容特征、着色到风格协方差，返回对齐目标特征。"""
    c, h, w = cf.shape[1:]
    cf = cf.squeeze(0).view(c, -1)
    sf = sf.squeeze(0).view(c, -1)
    n = cf.shape[1]
    # 白化内容
    cf_c = cf - cf.mean(dim=1, keepdim=True)
    cov_c = cf_c @ cf_c.T / (n - 1)
    U_c, S_c, _ = torch.linalg.svd(cov_c)
    k = min(c, n)
    D_c = torch.diag(1.0 / torch.sqrt(S_c[:k] + eps))
    whitened = U_c[:, :k] @ D_c @ U_c[:, :k].T @ cf_c
    # 着色风格
    sf_c = sf - sf.mean(dim=1, keepdim=True)
    cov_s = sf_c @ sf_c.T / (n - 1)
    U_s, S_s, _ = torch.linalg.svd(cov_s)
    D_s = torch.diag(torch.sqrt(S_s[:k] + eps))
    colored = U_s[:, :k] @ D_s @ U_s[:, :k].T @ whitened
    colored = colored + sf.mean(dim=1, keepdim=True)
    aligned = alpha * colored + (1 - alpha) * cf
    return aligned.view(1, c, h, w)


def wct_transfer(content_path, style_path, out_path, encoder,
                 lambda_ssp=0.0, steps=400, lr=0.03, size=256, device='cpu'):
    c_img = load_img(content_path, size, device)
    s_img = load_img(style_path, size, device)
    with torch.no_grad():
        fc = encoder(c_img)['r41']
        fs = encoder(s_img)['r41']
        target = wct_align(fc, fs)
    style_edge = edge_map(s_img)
    mask_sym = symbol_mask(style_edge)

    x = c_img.clone().contiguous().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=lr)
    for i in range(steps):
        opt.zero_grad()
        fx = encoder(x)['r41']
        lc = F.mse_loss(fx, target)                     # 匹配 WCT 对齐特征
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
    out_dir = 'output/wct'
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

            x0 = wct_transfer(cp, sp, os.path.join(out_dir, f'{cname}_{sname}_wct.png'),
                              enc, lambda_ssp=0.0, steps=400, size=256, device=dev)
            sp0 = symbol_preservation(x0, style_edge, mask_sym)

            x1 = wct_transfer(cp, sp, os.path.join(out_dir, f'{cname}_{sname}_wct_ssp.png'),
                              enc, lambda_ssp=3e2, steps=400, size=256, device=dev)
            sp1 = symbol_preservation(x1, style_edge, mask_sym)

            rows.append({'content': cname, 'style': sname,
                         'wct': round(sp0, 4), 'wct_ssp': round(sp1, 4)})
            print(f"{cname} x {sname}: WCT={sp0:.4f} WCT+L_ssp={sp1:.4f} (Δ={sp1-sp0:+.4f})")

    import statistics
    a = [r['wct'] for r in rows]
    b = [r['wct_ssp'] for r in rows]
    print(f"\nWCT 符号保持率均值:     {statistics.mean(a):.4f}")
    print(f"WCT+L_ssp 均值:         {statistics.mean(b):.4f}")
    print(f"L_ssp 净提升:           {statistics.mean(b)-statistics.mean(a):+.4f}")

    with open(os.path.join(out_dir, 'wct_results.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['content', 'style', 'wct', 'wct_ssp'])
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    main()

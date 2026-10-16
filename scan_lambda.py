# -*- coding: utf-8 -*-
"""scan_lambda.py — 补 λ_ssp 消融：在 15 对全量上跑 λ=1e4 与 λ=1e6 的 ours，
得到与主实验（results.csv）同口径的符号保持率，用于 Table 8。"""
import os, glob, csv, torch, sys
from stylize import (VGGEncoder, transfer, edge_map, symbol_mask, symbol_preservation, load_img)

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[device] {dev}")
enc = VGGEncoder().to(dev).eval()

contents = sorted(glob.glob('data/content/*.*'))
styles = sorted(glob.glob('data/style/*.*'))

LAMBDAS = [1e4, 1e6]
rows = []
for lam in LAMBDAS:
    for cp in contents:
        for sp in styles:
            cname = os.path.splitext(os.path.basename(cp))[0]
            sname = os.path.splitext(os.path.basename(sp))[0]
            s_img = load_img(sp, 256, dev)
            with torch.no_grad():
                style_edge = edge_map(s_img)
                mask_sym = symbol_mask(style_edge)
            out_path = f"output/scan_{int(lam)}_{cname}_{sname}.png"
            x = transfer(cp, sp, out_path, enc, lambda_style=1e6, lambda_ssp=lam,
                         steps=400, size=256, device=dev)
            sp_score = symbol_preservation(x, style_edge, mask_sym)
            rows.append({'lambda_ssp': int(lam), 'content': cname, 'style': sname,
                         'symbol_preservation': round(sp_score, 4)})
            print(f"[lam={int(lam)}] {cname} x {sname}: sym_preserve={sp_score:.4f}", flush=True)

csv_path = 'output/scan_lambda_results.csv'
with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=['lambda_ssp', 'content', 'style', 'symbol_preservation'])
    w.writeheader(); w.writerows(rows)
print(f"[done] -> {csv_path}")

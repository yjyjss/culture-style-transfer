# -*- coding: utf-8 -*-
"""
run_exp.py — 主实验：验证 H1a（符号结构保持损失的有效性）

对每对 (内容图, 风格图) 分别运行：
  - baseline：L = L_content + lambda_style * L_style
  - ours    ：L = ... + lambda_ssp * L_ssp

计算指标：
  - symbol_preservation：符号结构保持率（越高越好）
  - style_loss（迁移后）：风格匹配程度
  - content_loss（迁移后）：内容保持程度
结果写入 results.csv，图片保存到 output/。
"""
import os
import glob
import csv
import torch
import argparse
from stylize import (VGGEncoder, transfer, edge_map, symbol_mask,
                     symbol_preservation, content_loss, style_loss,
                     load_img, save_img)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--content_dir', default='data/content')
    ap.add_argument('--style_dir', default='data/style')
    ap.add_argument('--out_dir', default='output')
    ap.add_argument('--steps', type=int, default=400)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--lambda_style', type=float, default=1e6)
    ap.add_argument('--lambda_ssp', type=float, default=1e5)
    args = ap.parse_args()

    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[device] {dev}")
    enc = VGGEncoder().to(dev).eval()

    contents = sorted(glob.glob(os.path.join(args.content_dir, '*.*')))
    styles = sorted(glob.glob(os.path.join(args.style_dir, '*.*')))
    os.makedirs(args.out_dir, exist_ok=True)

    rows = []
    for cp in contents:
        for sp in styles:
            cname = os.path.splitext(os.path.basename(cp))[0]
            sname = os.path.splitext(os.path.basename(sp))[0]

            s_img = load_img(sp, args.size, dev)
            with torch.no_grad():
                style_edge = edge_map(s_img)
                mask_sym = symbol_mask(style_edge)
                feat_s = enc(s_img)

            for tag, lssp in [('baseline', 0.0), ('ours', args.lambda_ssp)]:
                out_path = os.path.join(args.out_dir, f"{cname}_{sname}_{tag}.png")
                x = transfer(cp, sp, out_path, enc,
                             lambda_style=args.lambda_style, lambda_ssp=lssp,
                             steps=args.steps, size=args.size, device=dev)
                # 评估
                sp_score = symbol_preservation(x, style_edge, mask_sym)
                with torch.no_grad():
                    fx = enc(x)
                    fc = enc(load_img(cp, args.size, dev))
                    sl = style_loss(fx, feat_s).item()
                    cl = content_loss(fx, fc).item()
                rows.append({
                    'content': cname, 'style': sname, 'method': tag,
                    'symbol_preservation': round(sp_score, 4),
                    'style_loss': round(sl, 2),
                    'content_loss': round(cl, 4),
                })
                print(f"[{tag}] {cname} x {sname}: sym_preserve={sp_score:.4f} "
                      f"style_loss={sl:.1f} content_loss={cl:.4f}")

    # 写 CSV
    csv_path = os.path.join(args.out_dir, 'results.csv')
    fields = ['content', 'style', 'method', 'symbol_preservation',
              'style_loss', 'content_loss']
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"\n[results] -> {csv_path}")


if __name__ == '__main__':
    main()

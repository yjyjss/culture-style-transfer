# -*- coding: utf-8 -*-
"""
evaluate.py — 感知保真度评估（补充 Table 3）

对已有的迁移结果，计算与内容图的感知距离（VGG 多层特征 MSE，即简化 LPIPS），
以及风格匹配度。对比 baseline 与 ours，验证 L_ssp 不损害风格/内容感知质量。

感知距离越低 → 内容保真越好；baseline 与 ours 应接近。
"""
import os
import glob
import csv
import torch
import torch.nn.functional as F
from stylize import VGGEncoder, load_img

PERCEPTUAL_LAYERS = ['r11', 'r21', 'r31', 'r41', 'r51']


def perceptual_distance(x, y, encoder):
    """VGG 多层特征距离（感知相似度，越低越相似）。"""
    with torch.no_grad():
        fx = encoder(x)
        fy = encoder(y)
    d = 0.0
    for l in PERCEPTUAL_LAYERS:
        d += float(F.mse_loss(fx[l], fy[l]).item())
    return d


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = VGGEncoder().to(dev).eval()

    out_dir = 'output'
    content_dir = 'data/content'
    results = []
    contents = sorted(glob.glob(os.path.join(content_dir, '*.*')))
    c_imgs = {os.path.splitext(os.path.basename(p))[0]: load_img(p, 256, dev)
              for p in contents}

    # 匹配 output 下的 *_baseline.png 和 *_ours.png
    base_files = sorted(glob.glob(os.path.join(out_dir, '*_baseline.png')))
    for bf in base_files:
        of = bf.replace('_baseline.png', '_ours.png')
        name = os.path.basename(bf)
        cname = name.split('_')[0]  # 内容图名
        sname = name.replace('_baseline.png', '').replace(cname + '_', '')
        if cname not in c_imgs or not os.path.exists(of):
            continue
        c_img = c_imgs[cname]
        xb = load_img(bf, 256, dev)
        xo = load_img(of, 256, dev)
        db = perceptual_distance(xb, c_img, enc)
        do = perceptual_distance(xo, c_img, enc)
        results.append({'content': cname, 'style': sname,
                        'perceptual_base': round(db, 2),
                        'perceptual_ours': round(do, 2)})
        print(f"{cname} x {sname}: perceptual base={db:.1f} ours={do:.1f} "
              f"(Δ={do-db:+.1f})")

    # 汇总
    import statistics
    db_all = [r['perceptual_base'] for r in results]
    do_all = [r['perceptual_ours'] for r in results]
    print(f"\n感知距离 基线均值: {statistics.mean(db_all):.1f}")
    print(f"感知距离 ours 均值: {statistics.mean(do_all):.1f}")
    print(f"相对变化: {(statistics.mean(do_all)/statistics.mean(db_all)-1)*100:+.2f}%")

    csv_path = os.path.join(out_dir, 'perceptual_results.csv')
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['content', 'style',
                                          'perceptual_base', 'perceptual_ours'])
        w.writeheader()
        w.writerows(results)
    print(f"[saved] {csv_path}")


if __name__ == '__main__':
    main()

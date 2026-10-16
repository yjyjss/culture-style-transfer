# -*- coding: utf-8 -*-
"""
cfm.py — 文化保真度指标 CFM 客观子项计算（贡献 3 / H1b）

对每个迁移结果计算 CFM 的两个客观子项：
  1. 符号保真度 Symbol Fidelity     ：符号区域边缘结构余弦相似度
  2. 文化相关性 Cultural Relevance  ：迁移结果相对目标风格 vs 其他风格的 Gram 距离比
（第三子项"文化评判"由用户研究收集，见协议文档）

输出 cfm_results.csv，用于与人类文化评判做相关性分析。
"""
import os
import glob
import csv
import torch
import torch.nn.functional as F
from stylize import VGGEncoder, load_img, edge_map, symbol_mask, gram

STYLE_LAYERS = ['r11', 'r21', 'r31', 'r41', 'r51']


def gram_distance(feat_a, feat_b):
    d = 0.0
    for l in STYLE_LAYERS:
        d += float(F.mse_loss(gram(feat_a[l]), gram(feat_b[l])).item())
    return d


def symbol_fidelity(x, style_edge, mask_sym):
    from stylize import edge_map as _em
    ex = _em(x)
    e1 = (ex * mask_sym).reshape(-1)
    e2 = (style_edge * mask_sym).reshape(-1)
    denom = (torch.norm(e1) * torch.norm(e2)) + 1e-8
    return float((torch.dot(e1, e2) / denom).item())


def cultural_relevance(x, encoder, target_style_img, other_style_imgs):
    with torch.no_grad():
        fx = encoder(x)
        d_target = gram_distance(fx, encoder(target_style_img))
        d_others = sum(gram_distance(fx, encoder(s)) for s in other_style_imgs) / len(other_style_imgs)
    return d_others / (d_target + d_others + 1e-8)  # 越接近 1 越贴近目标风格


def main():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = VGGEncoder().to(dev).eval()

    style_dir = 'data/style'
    style_files = sorted(glob.glob(os.path.join(style_dir, '*.*')))
    style_imgs = {os.path.splitext(os.path.basename(p))[0]: load_img(p, 256, dev)
                  for p in style_files}
    style_edges = {k: edge_map(v) for k, v in style_imgs.items()}
    style_masks = {k: symbol_mask(v) for k, v in style_edges.items()}

    # 动态获取内容图名（去掉扩展名），遍历 内容×风格×方法 组合
    content_names = [os.path.splitext(os.path.basename(p))[0]
                     for p in glob.glob(os.path.join('data/content', '*.*'))]
    rows = []
    for cname in content_names:
        for style_name in style_imgs:
            for method in ['baseline', 'ours']:
                rf = f'output/{cname}_{style_name}_{method}.png'
                if not os.path.exists(rf):
                    continue
                x = load_img(rf, 256, dev)
                target = style_imgs[style_name]
                others = [v for k, v in style_imgs.items() if k != style_name]

                sf = symbol_fidelity(x, style_edges[style_name], style_masks[style_name])
                cr = cultural_relevance(x, enc, target, others)
                rows.append({'content': cname, 'style': style_name, 'method': method,
                             'symbol_fidelity': round(sf, 4),
                             'cultural_relevance': round(cr, 4)})
                print(f"{cname}_{style_name}_{method}: sf={sf:.4f} cr={cr:.4f}")

    with open('output/cfm_results.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['content', 'style', 'method',
                                          'symbol_fidelity', 'cultural_relevance'])
        w.writeheader()
        w.writerows(rows)

    # 汇总 baseline vs ours
    import statistics
    b = [r for r in rows if r['method'] == 'baseline']
    o = [r for r in rows if r['method'] == 'ours']
    print(f"\n符号保真度   baseline={statistics.mean([r['symbol_fidelity'] for r in b]):.4f} "
          f"ours={statistics.mean([r['symbol_fidelity'] for r in o]):.4f}")
    print(f"文化相关性   baseline={statistics.mean([r['cultural_relevance'] for r in b]):.4f} "
          f"ours={statistics.mean([r['cultural_relevance'] for r in o]):.4f}")


if __name__ == '__main__':
    main()

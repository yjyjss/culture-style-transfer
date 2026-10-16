# -*- coding: utf-8 -*-
"""
analysis.py — CFM 与人类文化评判的相关性分析（H1b）

输入：
  1. output/cfm_results.csv      —— CFM 客观子项（cfm.py 生成）
  2. output/human_ratings.csv    —— 人类评分（用户研究收集，格式见下）

human_ratings.csv 格式（每行一位评分者对一张图的打分）：
  content,style,method,symbol_score,cultural_score,aesthetic_score,rater_id
  teaset,dunhuang_lotus,ours,6,6,5,r1
  ...

输出：Pearson/Spearman 相关性、ICC 评分者一致性、配对 t 检验、指标对比。
"""
import os
import csv
import statistics
import numpy as np
from scipy import stats


def load_cfm(path='output/cfm_results.csv'):
    rows = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r['content'], r['style'], r['method'])
            rows[key] = {'symbol_fidelity': float(r['symbol_fidelity']),
                         'cultural_relevance': float(r['cultural_relevance'])}
    return rows


def load_human(path='output/human_ratings.csv'):
    # 返回 { (content,style,method): [ (symbol_score, cultural_score, aesthetic_score, rater_id), ... ] }
    data = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (r['content'], r['style'], r['method'])
            data.setdefault(key, []).append((
                float(r['symbol_score']), float(r['cultural_score']),
                float(r['aesthetic_score']), r.get('rater_id', 'anon')))
    return data


def icc2_1(ratings):
    """Shrout-Fleiss ICC(2,1)：评分者绝对一致性。ratings: n_samples x k_raters。"""
    ratings = np.asarray(ratings, dtype=float)
    n, k = ratings.shape
    grand = ratings.mean()
    ss_rows = np.sum((ratings.mean(axis=1) - grand) ** 2) * k
    ss_cols = np.sum((ratings.mean(axis=0) - grand) ** 2) * n
    ss_total = np.sum((ratings - grand) ** 2)
    ss_err = ss_total - ss_rows - ss_cols
    ms_rows = ss_rows / (n - 1) if n > 1 else 0
    ms_cols = ss_cols / (k - 1) if k > 1 else 0
    ms_err = ss_err / ((n - 1) * (k - 1)) if (n > 1 and k > 1) else 0
    icc = (ms_rows - ms_err) / (ms_rows + (k - 1) * ms_err +
                                k * (ms_cols - ms_err) / n)
    return icc


def main():
    cfm = load_cfm()
    if not os.path.exists('output/human_ratings.csv'):
        print("未找到 output/human_ratings.csv。请先开展用户研究并填入评分数据。")
        print("格式见本文件头部注释。")
        return
    human = load_human()

    # 对每个样本，聚合人类评分（跨评分者平均）
    samples = []
    for key, ratings in human.items():
        sym = statistics.mean([r[0] for r in ratings])
        cul = statistics.mean([r[1] for r in ratings])
        aes = statistics.mean([r[2] for r in ratings])
        if key in cfm:
            samples.append({'key': key, 'symbol_fidelity': cfm[key]['symbol_fidelity'],
                            'cultural_relevance': cfm[key]['cultural_relevance'],
                            'sym': sym, 'cul': cul, 'aes': aes})

    sf = np.array([s['symbol_fidelity'] for s in samples])
    sym = np.array([s['sym'] for s in samples])
    cr = np.array([s['cultural_relevance'] for s in samples])
    cul = np.array([s['cul'] for s in samples])

    print(f"样本数: {len(samples)}")

    # 1. 相关性
    r_pearson, p_pearson = stats.pearsonr(sf, sym)
    rho, p_spearman = stats.spearmanr(sf, sym)
    print(f"\n[相关性] 符号保真度 vs 人类符号保真评分:")
    print(f"  Pearson r = {r_pearson:.3f} (p={p_pearson:.4f})")
    print(f"  Spearman ρ = {rho:.3f} (p={p_spearman:.4f})")

    # 指标对比：文化相关性 vs 人类评分
    r_cr, _ = stats.pearsonr(cr, cul)
    print(f"\n[指标对比] 文化相关性 vs 人类文化得体评分: Pearson r = {r_cr:.3f}")

    # 2. ICC（符号保真评分的评分者一致性）
    # 构建 n_samples x k_raters 矩阵
    keys = sorted(human.keys())
    raters = sorted(set(r[3] for v in human.values() for r in v))
    mat = []
    for k in keys:
        row = [r[0] for r in sorted(human[k], key=lambda x: x[3])]
        mat.append(row)
    if len(raters) > 1:
        icc = icc2_1(mat)
        print(f"\n[评分者一致性] ICC(2,1) = {icc:.3f} （评分者数={len(raters)}）")

    # 3. 配对 t 检验（baseline vs ours 的人类符号保真评分）
    base_sym = [s['sym'] for s in samples if s['key'][2] == 'baseline']
    ours_sym = [s['sym'] for s in samples if s['key'][2] == 'ours']
    if base_sym and ours_sym:
        t, p = stats.ttest_rel(ours_sym, base_sym)
        print(f"\n[方法差异] 配对 t 检验（ours vs baseline 符号保真评分）:")
        print(f"  baseline 均值={statistics.mean(base_sym):.2f}, "
              f"ours 均值={statistics.mean(ours_sym):.2f}, "
              f"t={t:.3f}, p={p:.4f}")

    print("\n判据：r≥0.7 且 ICC≥0.7 且配对t p<0.05 且 文化相关性 r<0.3 → H1b 成立。")


if __name__ == '__main__':
    main()

from pathlib import Path
import json,re,hashlib,shutil,sys,subprocess,os
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qa_deps'))
import pymupdf as fitz
R=Path(r'C:\Users\Jia\Nutstore\1\Nutstore\yoga\music papers\杨晓燕\06_论文初稿\paper_v3_acm')
D=R/'revision_v3'/'full_revision';Q=D/'qa';HERE=Path(__file__).resolve().parent
skill=Path(r'C:\Users\Jia\.codex\skills\nature-figure\scripts')
figdir=R/'figures'/'v3'
for name in ['Fig3_actual_comparison','Fig4_actual_interventions']:
    p=fitz.open(figdir/(name+'.pdf'))[0]
    info=p.get_image_info()
    panels=[]
    for i,v in enumerate(info):
        ncol=3 if 'Fig3' in name else 4
        col=i%ncol;row=i//ncol
        # Reference-image aspect ratios are deliberately preserved; compare only matched output panels.
        if ncol==3 and col==0:continue
        x0,y0,x1,y1=v['bbox'];h=p.rect.height
        panels.append({'id':f'p{i}','bbox_pt':[x0,h-y1,x1,h-y0],'grid_id':'image_grid','row_start':row,'row_stop':row+1,'col_start':col,'col_stop':col+1})
    layout={'schema_version':1,'figure':{'width_pt':p.rect.width,'height_pt':p.rect.height},'panels':panels,'notes':'Reference column excluded from comparable image-area geometry because original aspect ratio is preserved; all baseline/constrained outputs and all intervention panels are compared.'}
    path=Q/(name+'_layout.json');path.write_text(json.dumps(layout,indent=2),encoding='utf-8')
    z=subprocess.run([sys.executable,'-Xutf8',str(skill/'audit_panel_alignment.py'),str(path),'--json-out',str(Q/(name+'_alignment.json'))],capture_output=True,text=True,encoding='utf-8');print(name,'alignment',z.returncode,z.stdout[:150]);assert z.returncode==0
z=subprocess.run([sys.executable,'-Xutf8',str(skill/'audit_svg_schematic.py'),str(figdir/'Fig2_framework_v3.svg'),'--min-pt','5','--max-pt','13','--max-width-mm','200','--json'],capture_output=True,text=True,encoding='utf-8')
(Q/'Fig2_svg_audit.json').write_text(z.stdout,encoding='utf-8');assert z.returncode==0
text=r'''# Nature-skills 全文修订说明（v3）

## 已完成的工作

本轮已修订标题、摘要、Introduction、Related Work、Method、Experiments、Discussion、Conclusion、Data and Code Availability，并增加完整15对结果附录及复现要求附录。全文是证据边界明确的工作稿；写作修订完成不等于实验已经补齐，也不等于保证达到任一期刊的接收标准。

新稿为 `main_v3.tex` / `main_v3.pdf`，参考文献为 `refs_v3.bib`。原 `main.tex` / `main.pdf` 保留，前一轮引言修订版另有时间戳备份。当前 v3 是全文新版，不再是“仅引言完成”的版本。

## 逐节修改

| 部分 | 修订重点 | 主要证据或边界 |
|---|---|---|
| 标题/摘要 | 收窄到陶瓷产品图像；给出主要计算结果，撤回语义有效性与完全独立控制声索 | 15对配对结果；无真人评价；实现偏差需重跑 |
| Introduction | 应用、表示、控制、测量问题形成研究问题链 | 与实际结果和讨论一致，不宣称全领域空白 |
| Related Work | 按统计对齐、局部对应、尺度控制、文化评价组织 | 承认 Gatys 2017、AdaAttN、LapStyle 等直接先例；13项实际引用，原文献库另存保留 |
| Method | 写清输入、变量、归一化、均值/方差、掩码分母、频带重叠和优化步骤 | 逐项对应本地保存的服务器代码；不把代码理想化 |
| Experiments | 从CSV汇总表格；以边缘一致性及代价为主，保留所有15对 | 无额外推理或训练；仅做描述性汇总；共享参考/载体不视为独立总体抽样 |
| Discussion | 解释同源评价、结构/载体冲突和属性耦合，区分局部实现与官方系统 | 给出影响具体结论的限制及可区分的后续检验 |
| Conclusion | 压缩为实际计算贡献及边界 | 不新增结果，不把准备材料计为证据 |
| 附录/可用性 | 全15对表、软件/权重/权限与复现缺口 | 不虚构公开代码地址、图像许可证、作者信息或伦理审批 |

## 数据与图示修正

1. 主比较仍为全部15对：边缘一致性均值0.6615→0.9211；平均内容损失393.76→398.28，约+1.15%。不再写“内容完全不损失”。
2. AdaIN表中0.7034是“无SSP的精化控制”，纯前馈初始化为0.6023。新表和方法说明已纠正。
3. 感知距离CSV只有5行、全部来自tea-set。新文稿明确分母，不再宣称覆盖15对；没有凭空补上缺失10对。
4. FID未计算，VGG特征MSE不是标准LPIPS。删除关于FID/LPIPS“盲区已被证实”的结论。
5. CFM使用同源Sobel边缘；相对Gram比值接近饱和，不能推出独立文化语义效度。模拟评分不进入结果或统计检验。
6. 多尺度仅1对、4配置，无独立性定量证据。保留真实输出，撤回“每项只改自己的属性”。
7. Figure 2修正颜色项：实际为完整图的VGG特征均值/方差，而非低通图输入；中频边缘使用自己的掩码；纹理项为方差。
8. 新Figure 3/4由服务器原始保存图组装，未修图、未裁切、未合成实验结果；各输入文件哈希见图目录source_manifest.json。原Fig1仍是示意图，图注已明确。

## 结果分配与主文纪律

| 证据 | 功能 | 放置 |
|---|---|---|
| 全15对边缘改善及内容代价 | 核心计算观察 | 摘要概述；Results正文与主表给出一次详细数字 |
| 同源指标、双重归一化、5对感知子集 | 改变解释的限定 | 主文Method/Results/Discussion保留，不能藏入SI |
| 不同对齐实现 | 兼容性支撑 | 一张主表；实现差别写方法，不冒充官方SOTA对比 |
| 两个额外lambda | 敏感性 | 一张简表；不宣称最优设置 |
| 单例干预 | 探索性证据 | 原图及有限解释，不作为通用解耦证明 |
| 全部15对逐行数值 | 追溯细节 | 附录；原始CSV保留 |
| 模拟人类评分 | 分析流水线材料 | 不进入主文经验结果；仅明确其不可用作验证 |
| 环境/权重/终止状态缺失 | 可复现性限制 | 主文说明影响；附录列出所需记录 |

## 尚未达到投稿完备性的关键问题

**A. 标准实现复现尚缺。** load_img和VGGEncoder各自进行ImageNet归一化。新稿准确报告已保存实现，但要消除该限制，必须另建标准预处理版本，成对重跑，不能只改文字。AdaIN权重来源/兼容性、本地WCT和attention控制与官方实现差异也需要真实复现。

**B. 独立评价尚缺。** 需要真正的、可追溯的人类识别/文化得体性或独立标注证据；现有模拟评分、同源边缘分数和准备好的问卷不能替代。不存在可据此声称的真实人类相关系数或ICC。

**C. 范围与权利记录尚缺。** 5参考×3陶瓷载体、单例多尺度干预不足以支持跨文化/跨材料泛化；每张图的来源、许可、权重哈希、环境和终止状态还应补齐。作者邮箱和投稿模板元数据仍需作者确认。

本轮没有未经请求招募参与者、联系专家、启动新训练或更改历史结果。后续若要推进投稿，优先修复并版本化重跑实现和缺失评价；增加篇幅无法解决这些问题。

## 验证说明

- LaTeX完整编译并渲染逐页检查；最终页数与警告见qa/document_checks.json。
- 三张新图执行最终PDF碰撞检查，报告保存在qa。
- 两个实验图板从最终PDF中的图像矩形执行对齐检查；参考图保留原始比例，不与方形输出强行裁切成等大小。
- Figure 2在159毫米实际显示宽度下，最小字形约5.32pt；SVG审计的唯一宽度提示是Nature预设89/183mm与当前ACM版面159mm不同，并非超出版面。
- 图文、数值、引用与问题清单进行了最终一致性检查。报告说明的是检查范围，不是期刊录用保证。
'''
(D/'全文修订说明与投稿缺口.md').write_text(text,encoding='utf-8')
for p in HERE.glob('*'):
    if p.is_file():shutil.copy2(p,D/p.name)
shutil.copy2(HERE.parent/'v3_images_20261016.tar.gz',D/'source_images_snapshot.tar.gz')
# Final scientific prose and bibliography are authoritative after layout edits.
shutil.copy2(R/'main_v3.tex',D/'final_manuscript_snapshot.tex')
shutil.copy2(R/'refs_v3.bib',D/'final_references_snapshot.bib')
print(D/'全文修订说明与投稿缺口.md')

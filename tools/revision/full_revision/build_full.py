from pathlib import Path
from datetime import datetime
import csv,hashlib,json,re,shutil

HERE=Path(__file__).resolve().parent
R=Path(r'C:\Users\Jia\Nutstore\1\Nutstore\yoga\music papers\杨晓燕\06_论文初稿\paper_v3_acm')
D=R/'revision_v3'/'full_revision';D.mkdir(parents=True,exist_ok=True)
old=(R/'main_v3.tex').read_text(encoding='utf-8')
B=R/'backups'/('before_full_v3_'+datetime.now().strftime('%Y%m%d_%H%M%S'));B.mkdir(parents=True)
for name in ['main_v3.tex','main_v3.pdf','refs_v3.bib']:
    if (R/name).exists():shutil.copy2(R/name,B/name)
head=old[:old.index('\\section{Introduction}')]
head=re.sub(r'\\title\{[^\n]*\}',r'\\title{Reference-Edge Constraints and Multi-Scale Control in Ceramic Product Image Stylization}',head)
abstract=r'''Neural style transfer can adapt a cultural reference to a product image, but appearance matching does not uniquely specify the contours of the reference motif. We examine an explicit reference-edge constraint and complementary multi-scale control terms for ceramic product-image stylization. The base formulation adds masked Sobel-edge matching to content and Gram-style objectives. A separate formulation weights full-image feature moments, middle-frequency edges, and high-frequency variance. We analyze an archived collection of five references paired with three ceramic tea-ware images. Adding the edge constraint increases mean reference-edge agreement from 0.6615 to 0.9211 across all 15 pairs, while the mean content objective increases by 1.15\%. Additional local refinement and reconstruction controls show positive changes in the same edge diagnostic. A single multi-scale loss-removal case illustrates the control interface but does not establish semantic independence of its terms. The evaluation explicitly distinguishes objective satisfaction from independent cultural validity: the structure score shares the training prior, the available VGG-distance results cover only five pairs, and verified human judgments are absent. The archived pipeline also contains a repeated-normalization implementation that requires standard-preprocessing replication. These results establish a bounded computational effect of a transparent reference-structure constraint and identify the measurements needed to evaluate its cultural and perceptual usefulness.'''
head=re.sub(r'\\begin\{abstract\}.*?\\end\{abstract\}',lambda m:'\\begin{abstract}\n'+abstract+'\n\\end{abstract}',head,flags=re.S)
head=re.sub(r'\\keywords\{[^\n]*\}',r'\\keywords{neural style transfer, ceramic product visualization, reference-edge constraints, multi-scale control, diagnostic evaluation}',head)
intro=old[old.index('\\section{Introduction}'):old.index('\\section{Related Work}')]
intro=intro.replace('Section 2 positions the approach relative to prior work. Section 3 specifies the objectives and diagnostics, Section 4 reports the available comparisons and interventions, and Section 5 discusses their interpretation and limitations.', 'Section~\\ref{sec:related} positions the approach relative to prior work. Section~\\ref{sec:method} specifies the objectives and diagnostics, Section~\\ref{sec:experiments} reports the available comparisons and interventions, and Section~\\ref{sec:discussion} discusses their interpretation and limitations before the conclusion.')
summary=json.loads((HERE/'summary.json').read_text())
names={'blue_porcelain':'Porcelain (full)','blue_porcelain_ring':'Porcelain (ring)','dunhuang_deer':'Dunhuang deer','dunhuang_lotus':'Dunhuang lotus','ink_landscape':'Ink landscape'}
rows=[]
for key,val in summary['style'].items():
    s=val['symbol_preservation'];c=val['content_loss']
    rows.append(f"{names[key]} & {s['baseline']:.4f} & {s['ours']:.4f} & {s['ours']-s['baseline']:.4f} & {c['baseline']:.2f} & {c['ours']:.2f}\\\\")
exp=(HERE/'04_experiments.tex').read_text().replace('@@STYLE_ROWS@@','\n'.join(rows))
tail='\n'.join([(HERE/'02_related.tex').read_text(),(HERE/'03_method.tex').read_text(),exp,(HERE/'05_discussion.tex').read_text()])
appendix=r'''
\appendix
\section{Paired Structural Results and Traceability}
\label{sec:appendix}
Table~\ref{tab:allpairs} preserves the full set of paired structural scores used in the main comparison. Values are those saved to four decimal places, not newly generated experimental outputs. The revision archive also includes the source CSV files and deterministic summary code. No empirical human-rating result enters the tables or conclusions.
\begin{table}[h]
\centering
\caption{All 15 content--reference pairs in the base comparison. The shared references and carriers mean that rows are not independent samples from a cultural population.}
\label{tab:allpairs}
\begin{tabular}{@{}llrrr@{}}
\toprule
Carrier & Reference & Base & Constrained & Difference\\
\midrule
@@PAIRS@@
\bottomrule
\end{tabular}
\end{table}

\section{Implementation and Reporting Requirements}
\label{sec:repro}
The inspected scripts specify architecture, loss terms, image size, and nominal optimization budgets, but not a complete environment lockfile, measured per-image costs, or checkpoint provenance. A reproducible replication should record library and accelerator versions, checkpoint hashes, image-source metadata, all termination states, and exact preprocessing. The double-normalization pipeline should be retained as a historical configuration and compared with a separately versioned standard-preprocessing configuration; its archived results should not be relabeled after changing code.

The saved perceptual evaluation uses only the tea-set subset. A complete evaluator should match carrier names against the known input inventory rather than truncating filenames at an underscore. That correction requires evaluation of the omitted outputs and a new result file, not insertion of inferred values into the existing table. Higher-precision Gram losses and separate runtime records are also needed before claims of equal appearance quality or efficiency can be tested.

Independent perceptual and cultural validation requires a finalized protocol, appropriate ethical procedures, documented recruitment, prespecified endpoints, and genuine raw responses. Prepared questionnaires or simulated scores cannot establish those endpoints. Data/source rights, author details, and a stable code-access route must be resolved before a submission-ready package is declared complete.
'''
S=HERE.parent/'snapshot'/'output'
rr=list(csv.DictReader((S/'results.csv').open()));b={(x['content'],x['style']):x for x in rr if x['method']=='baseline'};o={(x['content'],x['style']):x for x in rr if x['method']=='ours'}
cn={'tea_cup':'Tea cup','teaset':'Tea set','white_cup':'White cup'}
pairrows=[]
for key in sorted(b):
    a=float(b[key]['symbol_preservation']);z=float(o[key]['symbol_preservation'])
    pairrows.append(f'{cn[key[0]]} & {names[key[1]]} & {a:.4f} & {z:.4f} & {z-a:.4f}\\\\')
appendix=appendix.replace('@@PAIRS@@','\n'.join(pairrows))
new=head+intro+tail+appendix+'\n\\bibliographystyle{ACM-Reference-Format}\n\\bibliography{refs_v3}\n\\end{document}\n'
if '\\setlength{\\emergencystretch}{1em}' not in new:
    new=new.replace('\\usepackage{booktabs}','\\usepackage{booktabs}\n\\setlength{\\emergencystretch}{1em}')
new=new.replace('\\begin{figure}[t]','\\begin{figure}[!htbp]')
new=new.replace('\\includegraphics[width=\\linewidth]{figures/v3/Fig3_actual_comparison.pdf}','\\includegraphics[width=0.88\\linewidth]{figures/v3/Fig3_actual_comparison.pdf}')
# Only entries cited by the revised argument are exported. Original library is preserved in backups.
bib=(R/'refs_v3.bib').read_text(encoding='utf-8')
add=r'''
@inproceedings{lapstyle2021,
 author={Lin, Tianwei and Ma, Zhuoqi and Li, Fu and He, Dongliang and Li, Xin and Ding, Errui and Wang, Nannan and Li, Jie and Gao, Xinbo},
 title={Drafting and Revision: Laplacian Pyramid Network for Fast High-Quality Artistic Style Transfer},
 booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
 year={2021},pages={5141--5150},
 url={https://openaccess.thecvf.com/content/CVPR2021/html/Lin_Drafting_and_Revision_Laplacian_Pyramid_Network_for_Fast_High-Quality_Artistic_CVPR_2021_paper.html}
}
@inproceedings{zhang2018lpips,
 author={Zhang, Richard and Isola, Phillip and Efros, Alexei A. and Shechtman, Eli and Wang, Oliver},
 title={The Unreasonable Effectiveness of Deep Features as a Perceptual Metric},
 booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
 year={2018},pages={586--595},
 url={https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html}
}
@misc{heusel2017fid,
 author={Heusel, Martin and Ramsauer, Hubert and Unterthiner, Thomas and Nessler, Bernhard and Hochreiter, Sepp},
 title={GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium},
 year={2017},eprint={1706.08500},archivePrefix={arXiv},howpublished={arXiv:1706.08500},url={https://arxiv.org/abs/1706.08500}
}
'''
entries={}
for ent in re.split(r'(?=^@)',bib+add,flags=re.M):
    m=re.match(r'@\w+\{([^,]+),',ent)
    if m:entries[m.group(1)]=ent.strip()
keys=set(k.strip() for x in re.findall(r'\\cite\{([^}]+)\}',new) for k in x.split(','))
assert keys<=entries.keys(),keys-entries.keys()
# Use the verified CVF proceedings version consistently for the AdaAttN citation.
entries['liu2021']=re.sub(r'pages\s*=\s*\{[^}]+\}', 'pages = {6649--6658}', entries['liu2021'])
entries['liu2021']=re.sub(r'\s*doi\s*=\s*\{[^}]+\},?', '', entries['liu2021'])
entries['liu2021']=entries['liu2021'].rstrip()[:-1]+'  url = {https://openaccess.thecvf.com/content/ICCV2021/html/Liu_AdaAttN_Revisit_Attention_Mechanism_in_Arbitrary_Neural_Style_Transfer_ICCV_2021_paper.html},\n}'
entries['qadri2024']=entries['qadri2024'].replace('Aroussia and','Aroussiak and')
(R/'refs_v3.bib').write_text('\n\n'.join(entries[k] for k in sorted(keys))+'\n',encoding='utf-8')
(R/'main_v3.tex').write_text(new,encoding='utf-8')
for fn,txt in [('01_introduction.tex',intro),('04_experiments_resolved.tex',exp),('06_appendix.tex',appendix),('00_abstract.tex',abstract)]:
    (D/fn).write_text(txt,encoding='utf-8')
for p in HERE.glob('*'):
    if p.is_file():shutil.copy2(p,D/p.name)
shutil.copytree(HERE.parent/'snapshot',D/'evidence_snapshot',dirs_exist_ok=True)
def wc(t):return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*",re.sub(r'\\(?:cite|ref|label)\{[^}]*\}','',t)))
counts={}
for m in re.finditer(r'\\section\{([^}]+)\}',new):
    end=re.search(r'\\section(?:\*?)\{',new[m.end():]);stop=m.end()+end.start() if end else len(new)
    counts[m.group(1)]=wc(new[m.end():stop])
(D/'build_manifest.json').write_text(json.dumps({'backup':str(B),'sections_word_tokens':counts,'references':sorted(keys),'master_sha256':hashlib.sha256(new.encode()).hexdigest(),'status':'All text sections revised; empirical limitations remain explicit.'},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'backup':str(B),'word_tokens':counts,'references':len(keys)},ensure_ascii=False))

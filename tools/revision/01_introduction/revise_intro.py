from pathlib import Path
from datetime import datetime
import csv, hashlib, json, re, shutil, statistics

ROOT = Path(r'C:\Users\Jia\Nutstore\1\Nutstore\yoga\music papers\杨晓燕\06_论文初稿\paper_v3_acm')
OUT = ROOT / 'revision_v3' / '01_introduction'
OUT.mkdir(parents=True, exist_ok=True)
old = (ROOT / 'main.tex').read_text(encoding='utf-8')
assert not (ROOT / 'main_v3.tex').exists(), 'Existing v3 working manuscript requires inspection before overwrite'
backup = ROOT / 'backups' / ('before_nature_v3_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
backup.mkdir(parents=True)
for name in ['main.tex','main.pdf','refs.bib']:
    shutil.copy2(ROOT / name, backup / name)
start = old.index('\\section{Introduction}')
end = old.index('\\section{Related Work}')
intro = r'''\section{Introduction}

Cultural and creative product design brings visual references from cultural heritage into new application contexts. For image-based design exploration, the practical task is to combine the appearance of a reference artwork with a product carrier while retaining visual features that identify the reference. A lotus contour, a figurative motif, and the palette of a porcelain pattern can each contribute to that identity, but they impose different requirements on image synthesis. Transferring a palette does not necessarily preserve a motif, and reproducing a motif does not by itself establish its cultural appropriateness. This distinction matters when computational tools are used to propose product visualizations rather than to reproduce an artwork. Cultural-heritage resources, such as those developed for Dunhuang, make this broader computational setting possible, while culturally situated evaluation emphasizes the role of artists and domain knowledge in interpreting generated imagery~\cite{yu2022,qadri2024}.

Neural style transfer provides a starting point for this task by combining content representations with appearance constraints. Gatys et al.\ match correlations between convolutional features using Gram matrices~\cite{gatys2016}. Adaptive instance normalization (AdaIN) aligns channel-wise means and variances, whereas whitening and coloring transforms (WCT) align feature covariance~\cite{huang2017,li2017}. These formulations capture useful appearance statistics, but a statistic aggregated over feature locations does not uniquely specify the arrangement of a reference motif. Consequently, a low style-matching loss alone cannot guarantee that the desired reference contours remain in the synthesized image. Attention-based approaches address part of this difficulty: AdaAttN uses spatial attention and local feature statistics to improve local stylization~\cite{liu2021}. The remaining question for the present application is therefore more specific than whether style transfer can use spatial information: it is how an explicit reference-structure constraint interacts with appearance matching and product-content preservation.

Controllability is also an established concern in neural style transfer. Previous work introduced controls over spatial location, color, and spatial scale, including combinations of appearance information from multiple references~\cite{gatys2017control}. It would therefore be inaccurate to treat the separation of perceptual factors as a new problem. The issue examined here is whether a compact, explicitly weighted formulation can expose useful adjustments for a fixed set of cultural-reference and product-image pairs. In this setting, a designer may wish to strengthen a motif contour without making the entire image more textured, or adjust overall appearance without imposing the same change on every structural detail. Such adjustments require identifiable loss terms and intervention tests; giving terms different names or different weights does not establish that their visual effects are independent.

The structure constraint itself introduces a trade-off that needs to be made explicit. The content image determines the carrier layout, whereas the style reference supplies a different arrangement of edges. Enforcing the reference arrangement at the same image coordinates can compete with preservation of the carrier, particularly when the motif and product occupy different regions. An edge detector offers a reproducible way to specify the reference structure, but it also responds to decorative texture and other intensity boundaries. It supplies neither a semantic segmentation of cultural symbols nor a geometric correspondence between a motif and a three-dimensional surface. We consequently formulate the present task as two-dimensional visual-structure transfer under competing image losses, with cultural interpretation remaining a separate question.

Evaluation must distinguish these same levels of evidence. An edge-similarity score can determine whether a synthesis satisfies an edge-based constraint, while a content-feature distance measures a different property. Neither quantity, in isolation, establishes that people recognize the intended symbol or judge the design culturally appropriate. This limitation is especially consequential when the optimization loss and the reported structure score use the same Sobel operator and reference mask. Their agreement is informative about objective satisfaction, but is not independent validation. We therefore distinguish computational diagnostics from human judgment, consistent with the need for context-sensitive assessment of culturally situated creative tools~\cite{qadri2024}. We also distinguish the implemented VGG feature-distance diagnostic from the standard learned LPIPS metric and from distributional measures such as FID; uncomputed metrics cannot support an empirical comparison.

To examine these issues, we study a structure-preserving and layer-decoupled style-transfer framework with a frozen VGG-19 encoder. In the base formulation, a symbol-structure-preservation loss, $L_{ssp}$, aligns output edge magnitudes with the reference inside a thresholded reference-edge mask. The word ``symbol'' names the intended design target; the implemented mask remains an edge-based proxy. A complementary multi-scale formulation assigns separate weights to feature-moment appearance matching, middle-frequency edge alignment, and high-frequency energy matching. These terms provide operational controls for color, symbol, and texture, respectively, without assuming a one-to-one correspondence between frequency bands and semantic attributes. The Cultural Fidelity Metric (CFM) is retained as a diagnostic framework that separates reference-edge agreement, relative style matching, and a human-judgment component that has not yet been collected.

The study is organized around three questions. \textbf{RQ1}: on the available content--style pairs, does adding $L_{ssp}$ improve reference-edge agreement, and what changes accompany it in the measured content and appearance objectives? \textbf{RQ2}: when one multi-scale loss term is disabled, which visual attributes change and which remain coupled? \textbf{RQ3}: what can the implemented computational diagnostics establish, and which interpretations require independent human evidence? These questions separate an optimization effect, a controllability test, and an evaluation boundary. In particular, an increase in a training-related score is not treated as sufficient evidence of cultural fidelity, and a small change in another loss is not treated as a formal demonstration of equivalence.

Our experimental scope is deliberately limited. The main benchmark contains five reference images, representing Dunhuang lotus and deer motifs, two blue-and-white porcelain references, and an ink landscape, paired with three ceramic tea-ware images at $256\times256$ resolution. The resulting 15 pairs support a pilot comparison of the base objectives. Additional experiments examine AdaIN-initialized refinement and local optimization-based implementations of covariance and attention alignment; these distinguish implementation-level observations from claims about the corresponding original systems. The multi-scale intervention study examines the available tea-set/lotus example. These experiments do not establish generalization to other product materials, reliable transfer to three-dimensional surfaces, or a validated measure of cultural meaning.

Within this scope, the paper makes three concrete contributions:
\begin{itemize}
\item An explicit reference-edge constraint for product-image stylization, together with comparisons that expose its interaction with content and appearance objectives.
\item A multi-scale formulation with separately weighted appearance, structure, and texture terms, examined through loss-removal interventions rather than assumed semantic independence.
\item A diagnostic evaluation that separates training-related structure agreement from other computational measurements and identifies the need for independent cultural and perceptual assessment.
\end{itemize}
Section 2 positions the approach relative to prior work. Section 3 specifies the objectives and diagnostics, Section 4 reports the available comparisons and interventions, and Section 5 discusses their interpretation and limitations.

\begin{figure}[t]
\includegraphics[width=\linewidth]{figures/Fig1_concept.png}
\caption{Illustrative comparison of product-image stylization without and with the reference-edge constraint. The intended target is visual motif structure; cultural appropriateness requires separate assessment.}\label{fig:concept}
\end{figure}

'''
new = old[:start] + intro + old[end:]
assert '\\bibliography{refs}' in new
new = new.replace('\\bibliography{refs}', '\\bibliography{refs_v3}')
(ROOT / 'main_v3.tex').write_text(new, encoding='utf-8')
(OUT / 'Introduction_v3.tex').write_text(intro, encoding='utf-8')
(OUT / 'Introduction_before.tex').write_text(old[start:end], encoding='utf-8')
bib = (ROOT / 'refs.bib').read_text(encoding='utf-8')
bib += r'''

@inproceedings{gatys2017control,
  title = {Controlling Perceptual Factors in Neural Style Transfer},
  author = {Gatys, Leon A. and Ecker, Alexander S. and Bethge, Matthias and Hertzmann, Aaron and Shechtman, Eli},
  booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages = {3985--3993},
  year = {2017},
  url = {https://openaccess.thecvf.com/content_cvpr_2017/html/Gatys_Controlling_Perceptual_Factors_CVPR_2017_paper.html}
}
'''
(ROOT / 'refs_v3.bib').write_text(bib, encoding='utf-8')
snapshot = Path(__file__).resolve().parent / 'snapshot'
summary = {}
for p in (snapshot / 'output').rglob('*.csv'):
    rows = list(csv.DictReader(p.open(encoding='utf-8-sig')))
    item = {'rows': len(rows), 'columns': list(rows[0]) if rows else []}
    if p.name == 'results.csv':
        item['means'] = {m: {k: statistics.mean(float(r[k]) for r in rows if r['method'] == m) for k in ['symbol_preservation','content_loss','style_loss']} for m in ['baseline','ours']}
    if p.name == 'adain_results.csv':
        item['means'] = {k: statistics.mean(float(r[k]) for r in rows) for k in ['adain','refine','adain_ssp']}
    if p.name == 'perceptual_results.csv':
        item['content_names'] = sorted({r['content'] for r in rows})
    summary[str(p.relative_to(snapshot))] = item
def words(s):
    s = re.sub(r'\\(?:cite|ref|label)\{[^}]*\}', '', s)
    return len(re.findall(r"[A-Za-z]+(?:[-'][A-Za-z]+)*", s))
manifest = {'scope':'Introduction only, including its Figure 1 caption; remaining sections preserved', 'source_backup':str(backup), 'before_words_including_figure':words(old[start:end]), 'after_words_including_figure':words(intro), 'server_archive_sha256':'d2699f32b92a55e5d8dbf4cf1b97df84830043f8b2b2f6d74193b0aa9c80deb7','code_sha256':{str(p.relative_to(snapshot)):hashlib.sha256(p.read_bytes()).hexdigest() for p in snapshot.rglob('*.py')},'csv_summary':summary}
(OUT / 'evidence_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
shutil.copy2(Path(__file__), OUT / 'revise_intro.py')
print(json.dumps({'output':str(ROOT / 'main_v3.tex'),'backup':str(backup),'before_words':manifest['before_words_including_figure'],'after_words':manifest['after_words_including_figure'],'csv_summary':summary},ensure_ascii=False,indent=2))

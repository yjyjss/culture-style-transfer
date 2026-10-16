from pathlib import Path
import csv,json,statistics as st
S=Path(__file__).resolve().parents[1]/'snapshot'/'output'
def rows(p):return list(csv.DictReader((S/p).open(encoding='utf-8-sig')))
r=rows('results.csv');keys=sorted({(x['content'],x['style']) for x in r})
b={(x['content'],x['style']):x for x in r if x['method']=='baseline'}
o={(x['content'],x['style']):x for x in r if x['method']=='ours'}
assert len(keys)==len(b)==len(o)==15
def mean(rs,k):return st.mean(float(x[k]) for x in rs)
summary={'base':{},'style':{},'alignment':{},'sensitivity':{},'cfm':{}}
for k in ['symbol_preservation','content_loss','style_loss']:
    v=[float(b[z][k]) for z in keys];w=[float(o[z][k]) for z in keys];diff=[y-x for x,y in zip(v,w)]
    summary['base'][k]={'baseline':st.mean(v),'constrained':st.mean(w),'baseline_sd':st.stdev(v),'constrained_sd':st.stdev(w),'difference':st.mean(diff),'difference_sd':st.stdev(diff),'gain_pairs':sum(x>0 for x in diff),'loss_pairs':sum(x<0 for x in diff)}
for s in sorted({x['style'] for x in r}):
    summary['style'][s]={k:{m:mean([x for x in r if x['style']==s and x['method']==m],k) for m in ['baseline','ours']} for k in ['symbol_preservation','content_loss']}
for p,cols in [('adain/adain_results.csv',['adain','refine','adain_ssp']),('wct/wct_results.csv',['wct','wct_ssp']),('attn/attn_results.csv',['attn','attn_ssp'])]:
    rr=rows(p);assert len(rr)==15
    summary['alignment'][p]={c:mean(rr,c) for c in cols}
for lam in ['10000','1000000']:
    rr=[x for x in rows('scan_lambda_results.csv') if x['lambda_ssp']==lam];assert len(rr)==15
    summary['sensitivity'][lam]={'n':len(rr),'mean':mean(rr,'symbol_preservation')}
for m in ['baseline','ours']:
    rr=[x for x in rows('cfm_results.csv') if x['method']==m];assert len(rr)==15
    summary['cfm'][m]={k:mean(rr,k) for k in ['symbol_fidelity','cultural_relevance']}
rr=rows('perceptual_results.csv');summary['perceptual']={'n':len(rr),'contents':sorted({x['content'] for x in rr}),'baseline':mean(rr,'perceptual_base'),'constrained':mean(rr,'perceptual_ours')}
Path(__file__).with_name('summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))

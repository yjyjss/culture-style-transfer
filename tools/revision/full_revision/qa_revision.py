from pathlib import Path
import sys,json,subprocess,os,re
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'qa_deps'))
import fitz
from PIL import Image,ImageOps,ImageDraw
R=Path(r'C:\Users\Jia\Nutstore\1\Nutstore\yoga\music papers\杨晓燕\06_论文初稿\paper_v3_acm')
Q=R/'revision_v3'/'full_revision'/'qa';Q.mkdir(exist_ok=True)
doc=fitz.open(R/'main_v3.pdf');print('pages',len(doc))
thumbs=[];report=[]
for i,p in enumerate(doc):
    pix=p.get_pixmap(matrix=fitz.Matrix(.65,.65));im=Image.frombytes('RGB',[pix.width,pix.height],pix.samples)
    im=ImageOps.expand(im,border=(2,20,2,2),fill='white');ImageDraw.Draw(im).text((8,3),f'Page {i+1}',fill='black');thumbs.append(im)
    page_record={'page':i+1,'clipped_text':[]}
    for b in p.get_text('dict')['blocks']:
        for line in b.get('lines',[]):
            for s in line['spans']:
                x0,y0,x1,y1=s['bbox']
                if x0<0 or y0<0 or x1>p.rect.width+.5 or y1>p.rect.height+.5:page_record['clipped_text'].append(s['text'])
    report.append(page_record)
    p.get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(str(Q/f'page_{i+1:02}.png'))
for start in range(0,len(thumbs),6):
    ts=thumbs[start:start+6];w=max(t.width for t in ts);h=max(t.height for t in ts);sheet=Image.new('RGB',(3*w,2*h),'#dddddd')
    for n,t in enumerate(ts):sheet.paste(t,((n%3)*w,(n//3)*h))
    sheet.save(Q/f'contact_{start//6+1}.png')
log=(R/'main_v3.log').read_text(errors='replace')
warn=[x for x in log.splitlines() if any(k in x for k in ['Overfull','undefined','Missing character'])]
(Q/'document_checks.json').write_text(json.dumps({'pages':len(doc),'page_checks':report,'tex_warnings':warn},indent=2),encoding='utf-8')
print('tex warnings',warn);print('clipped',sum(len(x['clipped_text']) for x in report))
env=os.environ.copy();env['PYTHONPATH']=str(Path(__file__).resolve().parents[1]/'qa_deps')
skill=Path(r'C:\Users\Jia\.codex\skills\nature-figure\scripts')
for name in ['Fig2_framework_v3','Fig3_actual_comparison','Fig4_actual_interventions']:
    out=Q/(name+'_collision.json')
    p=subprocess.run([sys.executable,'-Xutf8',str(skill/'audit_figure_collisions.py'),str(R/'figures'/'v3'/(name+'.pdf')),'--json-out',str(out),'--overlay-pdf',str(Q/(name+'_collision.pdf'))],env=env,capture_output=True,text=True,encoding='utf-8')
    (Q/(name+'_collision.log')).write_text(p.stdout+p.stderr,encoding='utf-8');print(name,p.returncode,p.stdout[:300])
print(Q)

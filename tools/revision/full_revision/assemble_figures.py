from pathlib import Path
import hashlib,json,re,runpy
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
import pypdfium2 as pdfium

R=Path(r'C:\Users\Jia\Nutstore\1\Nutstore\yoga\music papers\杨晓燕\06_论文初稿\paper_v3_acm')
OUT=R/'figures'/'v3';OUT.mkdir(parents=True,exist_ok=True)
HERE=Path(__file__).resolve().parent
IMG=HERE.parent/'images'
src=Path('work/figure2_redesign/draw_figure2.py').read_text(encoding='utf-8')
src=src.replace("OUT=Path(r'C:\\Users\\Jia\\Nutstore\\1\\Nutstore\\yoga\\music papers\\杨晓燕\\06_论文初稿\\paper_v3_acm\\figures\\Fig2_redesign')",f'OUT=Path({str(OUT)!r})')
src=src.replace('Fig2_framework_v2','Fig2_framework_v3')
src=src.replace('Gaussian scale decomposition','Multi-scale control terms')
src=src.replace('Color  |  low frequency','Color  |  feature moments')
src=src.replace('High-band energy statistics','High-band channel variance')
src=src.replace('Layer-controlled synthesis','Weighted image synthesis')
line="rich(72,102.2,[ital('I'),sub('low'),norm(' = '),ital('B'),sub('σ'),('l',.55,1.7,'Arial'),norm('('),ital('I'),norm(')')],8.2,maxw=51)"
assert line in src
src=src.replace(line,"rich(72,102.2,[norm('Full-image '),ital('F'),sub('4,1'),norm('('),ital('I'),norm(')')],8.2,maxw=51)")
src=src.replace("('h',.55", "('h',.68").replace("('l',.55", "('l',.68")
src=src.replace("arrow([(47,83.8),(69,83.8)],RUST)", '# Full-reference mask belongs to the base branch only.')
src=src.replace("rich(58,81,[ital('E'),sub('S'),norm(', '),ital('M'),sub('sym')],7.9,RUST,'middle',20)", '')
(HERE/'draw_framework_v3.py').write_text(src,encoding='utf-8')
runpy.run_path(str(HERE/'draw_framework_v3.py'),run_name='__main__')
from pypdf import PdfReader,PdfWriter
from pypdf.generic import DecodedStreamObject,NameObject
pdfpath=OUT/'Fig2_framework_v3.pdf'
reader=PdfReader(pdfpath);writer=PdfWriter();writer.clone_document_from_reader(reader)
stream=DecodedStreamObject();stream.set_data(writer.pages[0].get_contents().get_data())
writer.pages[0][NameObject('/Contents')]=writer._add_object(stream.flate_encode())
normalized=pdfpath.with_suffix('.normalized.pdf');writer.write(normalized);normalized.replace(pdfpath)
# SVG records the actual manuscript display width; the PDF is scaled by LaTeX.
svgpath=OUT/'Fig2_framework_v3.svg'
svgtext=svgpath.read_text(encoding='utf-8').replace('width="200mm" height="148mm"','width="159mm" height="117.66mm"')
svgpath.write_text(svgtext,encoding='utf-8')

manifest={'image_archive_sha256':'f6b60fa475ea7de93d04c6ba1302fe14aaded38642f521c8e1525774decd4ef9','processing':'All source images retained uncropped. Uniform aspect-preserving display only; no pixel retouching or synthetic results.','figures':{}}
def plate(name,columns,rows,width=450,cell=128,gap=9,left=6,top=23,labelheight=13):
    height=top+len(rows)*(cell+labelheight+gap)+5
    width=left*2+len(columns)*cell+(len(columns)-1)*gap
    c=canvas.Canvas(str(OUT/(name+'.pdf')),pagesize=(width,height))
    c.setTitle(name);c.setFillColorRGB(1,1,1);c.rect(0,0,width,height,fill=1,stroke=0)
    c.setFont('Helvetica-Bold',10)
    for j,title in enumerate(columns):c.setFillColorRGB(.14,.20,.25);c.drawCentredString(left+j*(cell+gap)+cell/2,height-14,title)
    panels=[]
    for i,(label,paths) in enumerate(rows):
        for j,p in enumerate(paths):
            im=Image.open(p).convert('RGB');x=left+j*(cell+gap);y=height-top-(i+1)*(cell+labelheight+gap)+labelheight+gap
            c.drawImage(ImageReader(im),x,y,width=cell,height=cell,preserveAspectRatio=True,anchor='c')
            c.setFillColorRGB(.14,.20,.25);c.setFont('Helvetica',8)
            if j==0:c.drawString(x,y-11,label)
            panels.append({'row':i,'column':j,'rectangle_pt':[x,y,cell,cell],'file':str(p.relative_to(IMG)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'dimensions':list(im.size)})
    c.showPage();c.save();manifest['figures'][name]={'page_pt':[width,height],'panels':panels}
    d=pdfium.PdfDocument(str(OUT/(name+'.pdf')));d[0].render(scale=2).to_pil().save(OUT/(name+'_preview.png'));d.close()

refs=[('Lotus','dunhuang_lotus'),('Porcelain ring','blue_porcelain_ring'),('Ink landscape','ink_landscape')]
rows=[]
for label,key in refs:
    styles=list((IMG/'data'/'style').glob(key+'.*'));assert len(styles)==1
    rows.append((label,[styles[0],IMG/'output'/f'teaset_{key}_baseline.png',IMG/'output'/f'teaset_{key}_ours.png']))
plate('Fig3_actual_comparison',['Reference','Base objective','With edge constraint'],rows)
plate('Fig4_actual_interventions',['Full objective','No color term','No symbol term','No texture term'],[('Tea-set / lotus case',[IMG/'output'/'multiscale_v3'/f'{tag}.png' for tag in ['full','no_color','no_symbol','no_texture']])])
(OUT/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print(OUT)

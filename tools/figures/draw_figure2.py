"""Vector Figure 2 redrawn from main.tex Sections 3.1-3.5.
No generated experimental images; original manuscript and figures are untouched.
Shared drawing operations produce editable SVG and embedded-font PDF.
"""
from pathlib import Path
from xml.sax.saxutils import escape
import math, json
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import pypdfium2 as pdfium

OUT=Path(r'C:\Users\Jia\Nutstore\1\Nutstore\yoga\music papers\杨晓燕\06_论文初稿\paper_v3_acm\figures\Fig2_redesign')
OUT.mkdir(parents=True,exist_ok=True)
W,H=200,148
PT=72/25.4
for name,file in [('Arial','arial.ttf'),('ArialBold','arialbd.ttf'),('ArialItalic','ariali.ttf')]:
    pdfmetrics.registerFont(TTFont(name,str(Path('C:/Windows/Fonts')/file)))
c=canvas.Canvas(str(OUT/'Fig2_framework_v2.pdf'),pagesize=(W*PT,H*PT),pageCompression=1,initialFontName='Arial')
c.setTitle('Structure-preserving and layer-decoupled style transfer: framework')
c.setAuthor('Figure redrawn from the supplied manuscript')
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}mm" height="{H}mm" viewBox="0 0 {W} {H}">', '<title>Structure-preserving and layer-decoupled style-transfer framework</title>', '<desc>Reference encoding, Gaussian frequency-band constraints, masked Sobel structure prior, two separately labeled optimization formulations, and a separate diagnostic branch. Human judgment has not yet been collected.</desc>',f'<rect width="{W}" height="{H}" fill="white"/>']
INK='#233342'; MUTED='#566777'; BLUE='#39658B'; BORDER='#B7C6D2'
TEAL='#247D80'; RUST='#AF5844'; GOLD='#A57B29'
checks=[]
def col(s):
    from reportlab.lib.colors import toColor
    return toColor(s)
def line(points,color=BORDER,width=.8,dash=None):
    c.setStrokeColor(col(color));c.setLineWidth(width);c.setDash(dash or [])
    p=c.beginPath();p.moveTo(points[0][0]*PT,(H-points[0][1])*PT)
    for x,y in points[1:]:p.lineTo(x*PT,(H-y)*PT)
    c.drawPath(p)
    d=' '.join(('M' if i==0 else 'L')+f'{x:g},{y:g}' for i,(x,y) in enumerate(points))
    a=f' stroke-dasharray="{','.join(str(x/PT) for x in dash)}"' if dash else ''
    svg.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width/PT:g}" stroke-linecap="round" stroke-linejoin="round"{a}/>')
    c.setDash([])
def polygon(points,color):
    c.setFillColor(col(color));p=c.beginPath();p.moveTo(points[0][0]*PT,(H-points[0][1])*PT)
    for x,y in points[1:]:p.lineTo(x*PT,(H-y)*PT)
    p.close();c.drawPath(p,fill=1,stroke=0)
    svg.append(f'<polygon points="{' '.join(f'{x:g},{y:g}' for x,y in points)}" fill="{color}"/>')
def arrow(points,color=BLUE,width=.85,dashed=False):
    line(points,color,width,[3,2] if dashed else None)
    (x0,y0),(x,y)=points[-2:];dx=x-x0;dy=y-y0;n=math.hypot(dx,dy);ux,uy=dx/n,dy/n
    length=1.65;half=.67
    polygon([(x,y),(x-length*ux+half*uy,y-length*uy-half*ux),(x-length*ux-half*uy,y-length*uy+half*ux)],color)
def box(x,y,w,h,fill='white',stroke=BORDER,width=.8,r=1.4):
    c.setFillColor(col(fill));c.setStrokeColor(col(stroke));c.setLineWidth(width);c.roundRect(x*PT,(H-y-h)*PT,w*PT,h*PT,r*PT,fill=1,stroke=1)
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{width/PT}"/>')
def txt(x,y,text,size=8.3,font='Arial',color=INK,anchor='start',maxw=None):
    return rich(x,y,[(text,1,0,font)],size,color,anchor,maxw)
def rich(x,y,parts,size=8.3,color=INK,anchor='start',maxw=None):
    size *= 1.20  # Body labels remain >=7 pt at the manuscript's ~159-mm width.
    lengths=[pdfmetrics.stringWidth(t,font,size*scale)/PT for t,scale,dy,font in parts]
    total=sum(lengths)
    if maxw is not None:
        assert total<=maxw+.05,(parts,total,maxw)
        checks.append({'text':''.join(t for t,*_ in parts),'width_mm':round(total,3),'available_mm':maxw,'size_pt':size})
    x=x-total/2 if anchor=='middle' else x-total if anchor=='end' else x
    for (t,scale,dy,font),length in zip(parts,lengths):
        c.setFont(font,size*scale);c.setFillColor(col(color));c.drawString(x*PT,(H-y-dy)*PT,t)
        weight='700' if font=='ArialBold' else '400';style='italic' if font=='ArialItalic' else 'normal'
        svg.append(f'<text x="{x:g}" y="{y+dy:g}" fill="{color}" font-family="Arial, Helvetica, sans-serif" font-weight="{weight}" font-style="{style}" font-size="{size*scale/PT:g}">{escape(t)}</text>')
        x+=length
    return total
def norm(t):return (t,1,0,'Arial')
def ital(t):return (t,1,0,'ArialItalic')
def sub(t):return (t,.73,1.02,'Arial')
def sup(t):return (t,.73,-1.2,'Arial')
def L(name):return [ital('L'),sub(name)]
def weight(prefix,name):return [ital(prefix),sub(name)]

# Panel headings and fine rules.
txt(5,6.8,'(a) Inputs and structure',9.2,'ArialBold',maxw=52)
txt(64,6.8,'(b) Feature and layer constraints',9.2,'ArialBold',maxw=68)
txt(146,6.8,'(c) Optimization and output',9.2,'ArialBold',maxw=51)
for x,w in [(5,42),(64,65),(147,48)]:line([(x,10),(x+w,10)],BORDER,.65)

# Actual image roles; no schematic image is presented as an experiment.
box(5,14,42,13,'#F0F5FA',BLUE)
rich(8,19.5,[norm('Content image '),ital('C')],8.8,maxw=36)
txt(8,24.3,'Product carrier',7.9,color=MUTED,maxw=36)
box(5,34,42,13,'#F0F5FA',BLUE)
rich(8,39.5,[norm('Style reference '),ital('S')],8.8,maxw=36)
txt(8,44.3,'Cultural reference',7.9,color=MUTED,maxw=36)

# Frozen feature stream.
box(64,14,65,13,'#F5F8FB',BLUE)
txt(67,19.5,'Frozen VGG-19 encoder',8.8,'ArialBold',maxw=59)
txt(67,24.3,'Content and style representations',7.9,color=MUTED,maxw=59)
arrow([(47,20),(64,20)])
arrow([(47,38),(55,38),(55,24),(64,24)])
arrow([(129,20),(147,20)])

# A hierarchical frequency module: the bands constrain synthesis, not outputs.
box(64,34,65,78,'#FAFBFC',BORDER)
txt(67,39.6,'Gaussian scale decomposition',8.7,'ArialBold',maxw=59)
rich(67,44.4,[ital('σ'),sub('h'),norm(' = 1.5;  '),ital('σ'),sub('l'),norm(' = 5.0')],7.9,color=MUTED,maxw=59)
arrow([(47,43),(64,43)])
line([(66,48),(127,48)],BORDER,.6)

box(69,52,57,18,'#EFF7F6',TEAL)
txt(72,57.1,'Texture  |  high frequency',8.5,'ArialBold',TEAL,maxw=51)
rich(72,62.2,[ital('I'),sub('high'),norm(' = '),ital('I'),norm(' − '),ital('B'),sub('σ'),('h',.55,1.7,'Arial'),norm('('),ital('I'),norm(')')],8.2,maxw=51)
txt(72,67,'High-band energy statistics',7.9,color=MUTED,maxw=54)

box(69,72,57,18,'#FCF2EF',RUST)
txt(72,77.1,'Symbol  |  mid frequency',8.5,'ArialBold',RUST,maxw=51)
rich(72,82.2,[ital('I'),sub('mid'),norm(' = '),ital('B'),sub('σ'),('h',.55,1.7,'Arial'),norm('('),ital('I'),norm(') − '),ital('B'),sub('σ'),('l',.55,1.7,'Arial'),norm('('),ital('I'),norm(')')],8.2,maxw=51)
txt(72,87,'Sobel edge-structure matching',7.9,color=MUTED,maxw=54)

box(69,92,57,18,'#FBF6E9',GOLD)
txt(72,97.1,'Color  |  low frequency',8.5,'ArialBold',GOLD,maxw=51)
rich(72,102.2,[ital('I'),sub('low'),norm(' = '),ital('B'),sub('σ'),('l',.55,1.7,'Arial'),norm('('),ital('I'),norm(')')],8.2,maxw=51)
txt(72,107,'Feature mean / variance',7.9,color=MUTED,maxw=51)

# Sobel branch is explicitly a geometric structure prior, not a semantic detector.
box(5,70,42,33,'#FCF2EF',RUST)
txt(8,76,'Symbol-structure prior',7.8,'ArialBold',RUST,maxw=36)
txt(8,81.5,'Grayscale → Sobel',7.9,maxw=36)
rich(8,87,[norm('Normalize '),ital('E'),sub('S'),norm(' to [0, 1]')],7.9,color=MUTED,maxw=36)
rich(8,93,[ital('M'),sub('sym'),norm(' = 1['),ital('E'),sub('S'),norm(' > '),ital('τ'),norm(']')],8.8,maxw=36)
rich(8,99,[ital('τ'),norm(' = 0.2')],8,color=MUTED,maxw=36)
arrow([(26,47),(26,70)],RUST)
txt(29,61,'Structure prior',7.6,color=RUST,maxw=25)
arrow([(47,83.8),(69,83.8)],RUST)
rich(58,81,[ital('E'),sub('S'),norm(', '),ital('M'),sub('sym')],7.9,RUST,'middle',20)

# Optimization alternatives are kept distinct because the text gives two losses.
box(147,14,48,96,'#F5F8FB',BLUE,1.0)
txt(151,20,'Image optimization',9.2,'ArialBold',maxw=40)
rich(151,26,[norm('Initialize '),ital('x'),norm(' ← '),ital('C')],8.3,maxw=40)
txt(151,31,'Adam · 400 steps',8,color=MUTED,maxw=40)
line([(151,35),(191,35)],BORDER,.6)
txt(151,41,'Base formulation',8.1,'ArialBold',color=BLUE,maxw=40)
rich(151,48,L('content')+[norm(' + ')]+weight('λ','style')+L('style'),8.6,maxw=40)
rich(151,54,[norm('+ ')]+weight('λ','ssp')+L('ssp'),8.6,maxw=40)
line([(151,59),(191,59)],BORDER,.6)
txt(151,65,'Layer-wise formulation',8.1,'ArialBold',color=BLUE,maxw=40)
rich(151,72,L('content'),8.6,maxw=40)
for y,name in [(78,'texture'),(84,'symbol'),(90,'color')]:
    rich(151,y,[norm('+ ')]+weight('w',name)+[norm(' ')]+L(name),8.6,maxw=40)
txt(151,99,'Adjust layer weights',8,color=MUTED,maxw=40)
txt(151,104,'during synthesis',8,color=MUTED,maxw=40)
for y,name,color in [(61,'texture',TEAL),(81,'symbol',RUST),(101,'color',GOLD)]:
    arrow([(126,y),(147,y)],color)
    rich(138,y-2.2,weight('w',name),8,color,'middle',17)

# Direct masked-edge constraint supports the base formulation.
arrow([(26,103),(26,118),(138,118),(138,108),(147,108)],RUST,.85)
rich(84,116.3,[norm('Masked edge constraint  ')]+L('ssp'),8,RUST,'middle',100)

# Output and diagnostic: no feedback arrow that could imply optimizing CFM.
box(147,122,48,18,'#315D80','#315D80',.8)
rich(171,129.2,[norm('Stylized image '),ital('x')],9,'white','middle',42)
txt(171,135,'Layer-controlled synthesis',7.8,color='white',anchor='middle',maxw=44)
arrow([(171,110),(171,122)],BLUE,1.0)
box(5,122,124,18,'#F7F7FA','#B7BAC8',.7)
txt(8,127.5,'CFM diagnostic  |  separate from optimization',8.6,'ArialBold',maxw=118)
rich(8,132.9,[norm('Compare '),ital('x'),norm(' with '),ital('S'),norm(':  '),ital('f'),sub('sym'),norm(' (masked edges),  '),ital('f'),sub('rel'),norm(' (relative Gram distance)')],8,maxw=118)
rich(8,137.5,[norm('Human judgment '),ital('f'),sub('jud'),norm(': not yet collected')],7.9,color=MUTED,maxw=118)
arrow([(147,131),(129,131)],MUTED,.85,True)
txt(5,145,'Solid arrows: synthesis constraints. Dashed arrow: diagnostic evaluation.',7.6,color=MUTED,maxw=190)

c.showPage();c.save();svg.append('</svg>');(OUT/'Fig2_framework_v2.svg').write_text('\n'.join(svg),encoding='utf8')
doc=pdfium.PdfDocument(str(OUT/'Fig2_framework_v2.pdf'));page=doc[0]
for dpi,suffix in [(300,'preview_300dpi'),(600,'600dpi')]:
    im=page.render(scale=dpi/72).to_pil().convert('RGB');im.save(OUT/f'Fig2_framework_v2_{suffix}.png',dpi=(dpi,dpi))
    if dpi==600:im.save(OUT/'Fig2_framework_v2_600dpi.tiff',compression='tiff_lzw',dpi=(dpi,dpi))
page.close();doc.close()
(OUT/'layout_checks.json').write_text(json.dumps({'page_mm':[W,H],'text_width_checks':checks,'raster_dpi':[300,600],'original_files_modified':False},indent=2),encoding='utf8')
print(OUT)

"""Replay figure/table arithmetic from archived outputs; no model inference or fitting."""
from pathlib import Path
import json, hashlib, math
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'data'; ORIG=DATA/'original'; FD=DATA/'figures'; FIG=ROOT/'figures/standalone'; TAB=ROOT/'tables'
for d in [FD,FIG,TAB]:d.mkdir(parents=True,exist_ok=True)
PROV=[]

def csv(name, df):
    df.to_csv(FD/(name+'.csv'),index=False,float_format='%.17g')
    return '../../data/figures/'+name+'.csv'

def save(name, body, sources, description, axes=True):
    (FIG/(name+'.tex')).write_text('\\documentclass[border=2pt]{standalone}\n\\input{../figure_style.tex}\n\\begin{document}\n\\begin{tikzpicture}\n'+body+'\n\\end{tikzpicture}\n\\end{document}\n')
    PROV.append(dict(panel=name,source_files=sources,transformation=description))

def axis(name, opts, body, sources, desc):
    save(name,'\\begin{axis}[\n'+opts+'\n]\n'+body+'\n\\end{axis}',sources,desc)

def plt(path,x,y,style='',err=None):
    opts=style
    spec=f'x={x},y={y},col sep=comma'
    if err:
        opts+=',error bars/.cd,x dir=both,x explicit'
        spec+=f',x error minus={err[0]},x error plus={err[1]}'
    return f'\\addplot[{opts}] table[{spec}] {{{path}}};\n'

STYLE=r'''\usepackage{newtxtext,newtxmath}
\usepackage{tikz,pgfplots,amsmath}
\usetikzlibrary{arrows.meta,positioning,calc,shapes.geometric}
\pgfplotsset{compat=1.18}
\definecolor{navy}{HTML}{24465E}
\definecolor{teal}{HTML}{327F83}
\definecolor{coral}{HTML}{BF594C}
\definecolor{ochre}{HTML}{AC7A28}
\definecolor{slate}{HTML}{67737D}
\definecolor{mist}{HTML}{E4E9EC}
\definecolor{pale}{HTML}{F4F7F8}
\definecolor{wine}{HTML}{885878}
\definecolor{lightteal}{HTML}{E3F0EF}
\tikzset{every node/.style={font=\sffamily\fontsize{8}{9.5}\selectfont},
 block/.style={draw=slate!65,fill=pale,rounded corners=1.5pt,inner sep=5pt,align=center},
 link/.style={-{Latex[length=1.8mm,width=1.15mm]},draw=slate,line width=.65pt}}
\pgfplotsset{every axis/.append style={
 font=\sffamily\fontsize{8}{9.5}\selectfont,
 tick label style={font=\sffamily\fontsize{7}{8.3}\selectfont,/pgf/number format/fixed,/pgf/number format/precision=3},
 label style={font=\sffamily\fontsize{8}{9.5}\selectfont},
 axis line style={slate!70,line width=.45pt},tick style={slate!70},
 grid style={mist,line width=.35pt},
 legend style={font=\sffamily\fontsize{7}{8.2}\selectfont,draw=none,fill=white,cells={anchor=west}},
 scaled ticks=false,clip=true,mark size=1.6pt}}
'''
(ROOT/'figures/figure_style.tex').write_text(STYLE)

# Compact conceptual panels: fixed geometry; not experimental data.
save('fig1a_matched_audit',r'''
\path[use as bounding box] (-3.1,-.65) rectangle (3.1,3.05);
\node[block,text width=5.2cm,minimum height=.65cm] (q) at (0,2.45)
 {\textbf{Matched tests}\\Same input and response map};
\node[block,text width=5.2cm,minimum height=.65cm] (f) at (0,1.30)
 {\textbf{Fitting sample}\\One fixed peer combination};
\node[block,text width=5.2cm,minimum height=.65cm,fill=lightteal,draw=teal] (e) at (0,.15)
 {\textbf{Evaluation sample}\\Residuals on held-out inputs};
\draw[link] (q.south)--(f.north);
\draw[link] (f.south)--(e.north);
''',[], 'Conceptual matched-audit schematic; no observations.')
save('fig1b_peer_projection',r'''
\path[use as bounding box] (-.65,-.7) rectangle (5.55,2.88);
\filldraw[draw=teal,fill=lightteal,line width=.8pt] (0,0)--(2.4,0)--(2.4,1.9)--(0,1.9)--cycle;
\foreach \x/\y in {0/0,2.4/0,2.4/1.9,0/1.9}{\fill[teal] (\x,\y) circle (2pt);}
\node[text=teal,align=center,text width=2.0cm] at (1.2,1.0) {Convex peer\\responses};
\node[circle,draw=navy,fill=white,inner sep=0pt,minimum size=6pt] (h) at (2.4,.80) {};
\node[circle,draw=coral,fill=coral!15,inner sep=0pt,minimum size=7pt] (t) at (4.55,.80) {};
\draw[-{Latex[length=2mm]},coral,line width=.95pt] (h.east)--node[above=4pt,text=coral] {$R_t=Y_t-h_t^\star$} (t.west);
\draw[navy!65,line width=.4pt] (h.south east)--(2.78,.40);
\node[anchor=north west,align=left,text=navy] at (2.82,.43) {Fitted response\\$h_t^\star$};
\node[anchor=south,align=center,text=coral] at (4.55,1.1) {Target\\$Y_t$};
\node[anchor=north,text=slate] at (1.2,-.12) {Points: individual peers};
''',[], 'Conceptual projection onto a rectangle; target projection is on the boundary.')

# All modern LLM displays use supplied CSV extracts, not reconstructed coordinates.
G=pd.read_csv(DATA/'table_v23_distributed_convex_coverage.csv')
S=pd.read_csv(DATA/'table_v23_sibling_coverage.csv')
E=pd.read_csv(DATA/'table_v23_endpoint_claim_comparison.csv')
models=['Qwen/Qwen2.5-7B-Instruct','TIGER-Lab/General-Reasoner-Qwen2.5-7B','allenai/OLMo-2-1124-13B-Instruct','google/gemma-3-12b-it','ibm-granite/granite-3.3-8b-instruct','microsoft/Phi-4-reasoning-plus','microsoft/phi-4','mistralai/Mistral-Nemo-Instruct-2407']
short=['Qwen2.5','GR','OLMo','Gemma','Granite','Phi-R','phi-4','Mistral']; ms=dict(zip(models,short))
fams=['content_deletion','irrelevant_context']
rows=[]
body=r'\path[use as bounding box](-2.25,-.6) rectangle (4.3,4.05);'+'\n'
for c,label in enumerate(['D / MSE','D / MAE','I / MSE','I / MAE']):
    body+=fr'\node[anchor=south,font=\sffamily\fontsize{{7}}{{8}}\selectfont] at ({c*1.04+.5},3.65) {{{label}}};'+'\n'
for i,m in enumerate(models):
    y=3.38-i*.46
    body+=fr'\node[anchor=east] at (-.12,{y}) {{{ms[m]}}};'+'\n'
    for c,(fam,loss) in enumerate([(f,l) for f in fams for l in ['MSE','MAE']]):
        r=G[(G.target==m)&(G.family==fam)].iloc[0]
        v=100*(1-r[f'perm_{loss}_convex_error']/r[f'perm_{loss}_single_error'])
        lo,hi=json.loads(r[f'common_bootstrap_{loss}_improvement_interval'])
        rows.append(dict(target=m,family=fam,loss=loss,gain_pct=v,ci_low=lo,ci_high=hi))
        fill=f'teal!{min(76,max(3,int(v*2.2)))}!white' if v>0 else 'mist'
        body+=fr'\fill[{fill}] ({c*1.04},{y-.21}) rectangle ({c*1.04+1},{y+.21});'+'\n'
        body+=fr'\node[font=\sffamily\fontsize{{8}}{{9}}\selectfont] at ({c*1.04+.5},{y}) {{{v:.1f}\%}};'+'\n'
body+=r'\node[anchor=north,text=slate,font=\sffamily\fontsize{7}{8}\selectfont] at (1.5,-.35) {Relative held-out error reduction};'
csv('llm_all_gains',pd.DataFrame(rows))
save('fig2a_group_gain',body,['data/table_v23_distributed_convex_coverage.csv'],'100*(single-group)/single. All eight targets; pointwise CI classification stored in llm_all_gains.csv.')

baseopt=r'width=7.1cm,height=4.45cm,grid=major'
qs=S[S.target.isin(models[:2])].copy();qs['y']=np.arange(len(qs))[::-1]+1
qs['lo']=qs.permutation_averaged_inflation-qs.bootstrap_lower;qs['hi']=qs.bootstrap_upper-qs.permutation_averaged_inflation
p=csv('qwen_removal',qs)
body=plt(p,'permutation_averaged_inflation','y','only marks,mark=*,teal',('lo','hi'))
body+=r'\addlegendentry{Sibling removal}'+'\n'+plt(p,'largest_non_sibling_inflation','y','only marks,mark=square*,coral')+r'\addlegendentry{Largest non-sibling mean}'
axis('fig2b_sibling_removal',baseopt+r',xmin=-.003,xmax=.065,ymin=.4,ymax=4.7,xlabel={Increase in held-out residual},ytick={1,2,3,4},yticklabels={GR / I,GR / D,Qwen / I,Qwen / D},legend style={at={(.5,1.04)},anchor=south,legend columns=1}',body,['data/table_v23_sibling_coverage.csv'],'Stored mean removal effects and their recorded intervals; no inferred per-peer intervals.')

rows=[]
for i,(m,fam) in enumerate([(m,f) for m in models[:2] for f in fams]):
 r=G[(G.target==m)&(G.family==fam)].iloc[0]
 rows.append(dict(y=4-i,single=r.perm_MSE_single_error,group=r.perm_MSE_convex_error))
p=csv('qwen_direction',pd.DataFrame(rows));body=plt(p,'single','y','only marks,mark=square*,slate')+r'\addlegendentry{Selected single peer}'+'\n'+plt(p,'group','y','only marks,mark=*,teal')+r'\addlegendentry{Fixed peer group}'
axis('fig2c_directional_coverage',baseopt+r',xmin=.078,xmax=.11,ymin=.4,ymax=4.7,xlabel={Held-out MAE},ytick={1,2,3,4},yticklabels={GR / I,GR / D,Qwen / I,Qwen / D},legend style={at={(.5,1.04)},anchor=south,legend columns=1}',body,['data/table_v23_distributed_convex_coverage.csv'],'MSE-fit single and convex held-out MAE, both targets and both families.')
rows=[]
for i,(m,fam) in enumerate([(m,f) for m in [models[-1],models[5]] for f in fams]):
 r=G[(G.target==m)&(G.family==fam)].iloc[0]
 for loss,off in [('MSE',.1),('MAE',-.1)]:
  v=r[f'perm_{loss}_single_error']-r[f'perm_{loss}_convex_error'];lo,hi=json.loads(r[f'common_bootstrap_{loss}_improvement_interval'])
  rows.append(dict(y=4-i+off,loss=loss,gain=v,lo=v-lo,hi=hi-v))
b=''
for loss,col,mark in [('MSE','teal','*'),('MAE','coral','square*')]:
 p=csv('distributed_'+loss,pd.DataFrame(rows).query('loss==@loss'))
 b+=plt(p,'gain','y',f'only marks,mark={mark},{col}',('lo','hi'))+fr'\addlegendentry{{{loss} fit}}'+'\n'
axis('fig2d_distributed_gain',baseopt+r',xmin=0,xmax=.055,ymin=.4,ymax=4.7,xlabel={Single-peer error minus group error},ytick={1,2,3,4},yticklabels={Phi-R / I,Phi-R / D,Mistral / I,Mistral / D},legend style={at={(.5,1.04)},anchor=south,legend columns=2}',b,['data/table_v23_distributed_convex_coverage.csv'],'Absolute matched-objective gains; CIs copied from archived improvement intervals.')

D=pd.read_csv(DATA/'v22/table_2_estimand_decomposition.csv')
for fam,name,xlab in [('irrelevant_context','fig3a_context_trajectory','Added context (words)'),('content_deletion','fig3b_deletion_trajectory','Deleted-word fraction')]:
 b=''
 for m,col,mark in [(models[0],'teal','*'),(models[6],'coral','square*'),(models[2],'navy','triangle*'),(models[7],'ochre','diamond*')]:
  d=D[(D.target==m)&(D.family==fam)].sort_values('dose')
  p=csv(name+'_'+ms[m],d[['dose','trackwise_fit_trackwise_eval']]);b+=plt(p,'dose','trackwise_fit_trackwise_eval',f'{col},mark={mark},line width=.8pt')+fr'\addlegendentry{{{ms[m]}}}'+'\n'
 axis(name,r'width=5.05cm,height=4.1cm,grid=major,xlabel={'+xlab+r'},ylabel={Mean residual},legend style={at={(.5,1.03)},anchor=south,legend columns=2},yticklabel style={/pgf/number format/fixed,/pgf/number format/precision=2}',b,['data/v22/table_2_estimand_decomposition.csv'],'Canonical five-dose trackwise-fit/trackwise-evaluation means. Values already averaged across splits in source table.')
rows=[]
for i,m in enumerate([models[0],models[6],models[2],models[7]]):
 for fam,off in [('irrelevant_context',.13),('content_deletion',-.13)]:
  r=E[(E.target==m)&(E.family==fam)].iloc[0];v=r.v23_permutation_averaged_delta
  rows.append(dict(family=fam,y=4-i+off,value=v,lo=v-r.bootstrap_lower,hi=r.bootstrap_upper-v))
b=r'\draw[slate!60,dashed] (axis cs:0,.4)--(axis cs:0,4.6);'+'\n'
for fam,label,col,mark in [('irrelevant_context','I','teal','*'),('content_deletion','D','coral','square*')]:
 p=csv('stress_effects_'+label,pd.DataFrame(rows).query('family==@fam'))
 b+=plt(p,'value','y',f'only marks,mark={mark},{col}',('lo','hi'))+fr'\addlegendentry{{{label}}}'+'\n'
axis('fig3c_balanced_effects',r'width=5.2cm,height=4.1cm,grid=major,xmin=-.038,xmax=.051,ymin=.4,ymax=4.6,xlabel={Balanced $\Delta U_t$},ytick={1,2,3,4},yticklabels={Mistral,OLMo,phi-4,Qwen2.5},legend style={at={(.5,1.03)},anchor=south,legend columns=2},xtick={-.03,0,.03}',b,['data/table_v23_endpoint_claim_comparison.csv'],'Balanced endpoint changes and pointwise bootstrap intervals for named headline cases; all eight retained in appendix table.')

# Vision: distinct ecosystems are not averaged together.
V=pd.read_csv(ORIG/'tables/exp10_imagenet_adv_pier.csv'); b=''
for m,label,col,mark in [('ResNet50','RN-50','teal','*'),('ConvNeXtTiny','ConvNeXt','coral','square*'),('RobustResNet50','Robust RN','navy','triangle*'),('ResNet18','RN-18','slate','none'),('EfficientNetB0','EffNet','ochre','none'),('ViT_B16','ViT','wine','none')]:
 d=V[V.TargetModel==m].sort_values('Dose').copy();d['relative']=d.MeanPIER/d.iloc[0].MeanPIER
 p=csv('vision_adv_'+m,d);b+=plt(p,'Dose','relative',f'{col},mark={mark},line width=.8pt')+fr'\addlegendentry{{{label}}}'+'\n'
axis('fig4a_vision_stress',r'width=5.1cm,height=4.3cm,grid=major,xlabel={Adversarial dose},ylabel={Residual / clean residual},xmin=0,xmax=.1,ymin=.42,ymax=1.32,xtick={0,.05,.1},xticklabel style={/pgf/number format/fixed,/pgf/number format/precision=2},legend style={at={(.5,1.03)},anchor=south,legend columns=3,font=\sffamily\fontsize{6}{7}\selectfont}',b,['data/original/tables/exp10_imagenet_adv_pier.csv'],'All six archived curves, normalized by model-specific dose-zero residual. No CI reconstruction.')
C=pd.read_csv(ORIG/'tables/exp11_shape_texture_pier.csv')
vm=['ResNet50','EfficientNetB0','ConvNeXtTiny','ViT_B16','ShapeResNet50_SIN','ShapeResNet50_SININ','ShapeResNet50_ShapeResNet']
vl=['RN-50','EffNet','ConvNeXt','ViT','SIN','SIN+IN','SIN+IN to IN'];rows=[];b=''
for i,m in enumerate(vm):
 g=C[C.TargetModel==m].set_index('ContextLabel');nat=float(g.loc['texture_natural','MeanPIER'])*1000;sh=float(g.loc['shape_bias','MeanPIER'])*1000;y=7-i
 rows.append(dict(y=y,natural=nat,shape=sh));b+=fr'\draw[slate!45,line width=.6pt] (axis cs:{nat:.17g},{y})--(axis cs:{sh:.17g},{y});'+'\n'
p=csv('vision_context',pd.DataFrame(rows));b+=plt(p,'natural','y','only marks,mark=o,teal')+r'\addlegendentry{Natural}'+'\n'+plt(p,'shape','y','only marks,mark=square*,coral')+r'\addlegendentry{Shape-biased}'
axis('fig4b_vision_context',r'width=5.3cm,height=4.3cm,grid=major,xlabel={Mean residual ($\times10^{-3}$)},xmin=0,xmax=.88,ymin=.4,ymax=7.6,ytick={1,2,3,4,5,6,7},yticklabels={'+','.join(reversed(vl))+r'},xtick={0,.4,.8},legend style={at={(.5,1.03)},anchor=south,legend columns=1,font=\sffamily\fontsize{6.5}{7.5}\selectfont}',b,['data/original/tables/exp11_shape_texture_pier.csv'],'Seven-model context audit; no averaging over separated shape-variant ecosystems.')

# Geometry preserves full precision; each target/context has its own basis.
geom=[]
geo_names={}
for m,lab in [('ShapeResNet50_SIN','SIN'),('ShapeResNet50_SININ','SIN+IN'),('ShapeResNet50_ShapeResNet',r'SIN+IN$\to$IN')]:
 for ctx in ['texture_natural','shape_bias']:
  f=ORIG/'artifacts/exp13'/f'exp13_geometry_{m}_{ctx}.npz';z=np.load(f,allow_pickle=False);coords=z['coords'];peer_mask=z['peer_mask'];ti=int(z['target_idx']);mi=int(z['mix_idx']);ev=z['explained_variance'];w=z['w_hat'];labels=list(map(str,z['labels']))
  name=('fig4c_geometry_natural' if ctx=='texture_natural' else 'fig4d_geometry_shape') if m=='ShapeResNet50_SIN' else 'figD_geometry_'+('SININ' if m.endswith('SININ') else 'C')+'_'+('natural' if ctx=='texture_natural' else 'shape')
  geo_names[m,ctx]=name
  df=pd.DataFrame(dict(x=coords[:,0]*1000,y=coords[:,1]*1000,label=labels));p=csv(name,df)
  pts=coords*1000;extent=np.ptp(pts,axis=0);low=pts.min(0)-extent*np.array([.22,.35]);high=pts.max(0)+extent*np.array([.22,.38])
  hull=pts[list(z['hull_vertices'])];hp=csv(name+'_hull',pd.DataFrame(np.vstack([hull,hull[:1]]),columns=['x','y']))
  pp=csv(name+'_peers',pd.DataFrame(pts[peer_mask],columns=['x','y']))
  tp=csv(name+'_target',pd.DataFrame([pts[ti]],columns=['x','y']));mp=csv(name+'_mix',pd.DataFrame([pts[mi]],columns=['x','y']))
  b=plt(hp,'x','y','draw=teal!85,fill=lightteal,line width=.65pt')+plt(pp,'x','y','only marks,mark=*,teal')
  b+=fr'\draw[coral,line width=.8pt] (axis cs:{pts[mi,0]:.17g},{pts[mi,1]:.17g})--(axis cs:{pts[ti,0]:.17g},{pts[ti,1]:.17g});'+'\n'
  b+=plt(mp,'x','y','only marks,mark=diamond*,mark size=2.4pt,ochre')+plt(tp,'x','y','only marks,mark=square*,mark size=2pt,coral')
  # Numeric peer labels prevent dense model-name collisions. Full key in caption.
  offsets=[('south east',-2,4),('north east',-2,-4),('south west',2,4),('south west',4,1)]
  if ctx=='shape_bias':offsets=[('south east',-4,7),('north west',1,-5),('south',0,5),('south west',4,1)]
  for i in range(4):
   anchor,dx,dy=offsets[i];x,y=pts[i]
   b+=fr'\node[font=\sffamily\fontsize{{6.5}}{{7.5}}\selectfont,anchor={anchor},xshift={dx}pt,yshift={dy}pt,text=teal] at (axis cs:{x:.17g},{y:.17g}) {{{i+1}}};'+'\n'
  x,y=pts[ti];vx,vy=pts[ti]-pts[mi]
  to_right=x<(low[0]+high[0])/2;anchor=('north' if vy<0 else 'south')+' '+('west' if to_right else 'east');dx=3 if to_right else -3;dy=-4 if vy<0 else 4
  b+=fr'\node[anchor={anchor},xshift={dx}pt,yshift={dy}pt,text=coral,font=\sffamily\fontsize{{7}}{{8}}\selectfont] at (axis cs:{x:.17g},{y:.17g}) {{Target}};'+'\n'
  axis(name,fr'width=5.1cm,height=4.3cm,grid=major,clip=false,xmin={low[0]:.17g},xmax={high[0]:.17g},ymin={low[1]:.17g},ymax={high[1]:.17g},xlabel={{PC1 ({100*ev[0]:.1f}\%)}},ylabel={{PC2 ({100*ev[1]:.1f}\%)}},title={{{lab}}},xticklabel style={{/pgf/number format/fixed,/pgf/number format/precision=0}},yticklabel style={{/pgf/number format/fixed,/pgf/number format/precision=0}}',b,[str(f.relative_to(ROOT))],'Stored PCA coordinates multiplied by 1000 for axes only; stored hull, mixture and target. No recomputed PCA or refit.')
  geom.append(dict(target=m,context=ctx,n_fit=len(z['fit_ids']),l2_dist=float(z['l2_dist']),ev=float(ev.sum()),min_weight=float(w.min()),sum_error=float(w.sum()-1),**{str(p):float(v) for p,v in zip(z['peers'],w)}))
geo=pd.DataFrame(geom);csv('geometry_support',geo)

rr={};ss={};mass=[];over=[]
for ctx in ['texture_natural','shape_bias']:
 rr[ctx]=np.abs(np.load(ORIG/'artifacts/exp11'/f'exp11_{ctx}_ConvNeXtTiny_residuals.npz')['residuals'])
 ss[ctx]=np.abs(np.load(ORIG/'artifacts/exp11'/f'exp11_{ctx}_ShapeResNet50_ShapeResNet_residuals.npz')['residuals'])
for ctx in rr:
 a=rr[ctx];n=len(a);s=np.sort(a)[::-1];cs=np.r_[0,np.cumsum(s)/s.sum()]
 for k,v in enumerate(cs):mass.append(dict(context=ctx,pct=k*100/n,mass=v*100))
 ia=np.argsort(-a,kind='stable');ib=np.argsort(-ss[ctx],kind='stable')
 for k in range(1,41):over.append(dict(context=ctx,k=k,overlap=len(set(ia[:k])&set(ib[:k]))/k))
b='';b2=''
for ctx,label,col in [('texture_natural','Natural','teal'),('shape_bias','Shape-biased','coral')]:
 p=csv('residual_mass_'+ctx,pd.DataFrame(mass).query('context==@ctx'));b+=plt(p,'pct','mass',f'{col},line width=1pt,no marks')+fr'\addlegendentry{{{label}}}'+'\n'
 p=csv('residual_overlap_'+ctx,pd.DataFrame(over).query('context==@ctx'));b2+=plt(p,'k','overlap',f'{col},line width=.8pt,no marks')+fr'\addlegendentry{{{label}}}'+'\n'
axis('fig4e_residual_mass',r'width=5.3cm,height=4.3cm,grid=major,xmin=0,xmax=100,ymin=0,ymax=103,xlabel={Highest-residual inputs (\%)},ylabel={Residual mass (\%)},xtick={0,25,50,75,100},legend style={at={(.97,.05)},anchor=south east}',b,['data/original/artifacts/exp11/exp11_texture_natural_ConvNeXtTiny_residuals.npz','data/original/artifacts/exp11/exp11_shape_bias_ConvNeXtTiny_residuals.npz'],'Exact cumulative sorted residual mass at all 401 integer counts; zero included.')
b2+=r'\draw[slate!50,dotted] (axis cs:20,0)--(axis cs:20,1);'
axis('fig4f_residual_overlap',r'width=5.1cm,height=4.3cm,grid=major,xmin=1,xmax=40,ymin=0,ymax=1.03,xlabel={Top-$k$ residual inputs},ylabel={Intersection / $k$},xtick={1,10,20,30,40},legend style={at={(.97,.45)},anchor=east}',b2,[str(f.relative_to(ROOT)) for f in sorted((ORIG/'artifacts/exp11').glob('*residuals.npz')) if 'ConvNeXt' in f.name or 'ShapeResNet50_ShapeResNet' in f.name],'Matched array-position overlap at every k=1..40; stable index tie-breaking, no fabricated intermediate fractions.')

# Traffic: preserve all values, show raw stored fits, do not label feasible convex certificates.
T=pd.read_csv(ORIG/'tables/exp14_multicity_tabular_summary.csv');W=pd.read_csv(ORIG/'tables/exp14_multicity_tabular_summary_peer_weights_long.csv')
T['rel_pier']=100*T.PIER/T.MeanFlow_Test;T['impact']=100*T.Delta_Router/T.Local_MAE;T['single']=100*T.ClosestPeerMeanAbsDiff/T.MeanFlow_Test
T['min_weight']=T.City.map(W.groupby('City').Weight.min());T['flag']=T.min_weight < -1e-3
p=csv('traffic_all',T);b=r'\draw[slate!50,dashed] (axis cs:0,0)--(axis cs:64,0);'+'\n';b2=r'\addplot[slate!70,dashed,domain=0:64,no marks] {x};'+'\n'
for flag,col,mark,label in [(False,'teal','*',r'$\min w\geq-10^{-3}$'),(True,'coral','triangle*',r'$\min w<-10^{-3}$')]:
 d=T[T.flag==flag];p=csv('traffic_'+str(flag),d)
 b+=plt(p,'rel_pier','impact',f'only marks,{col},mark={mark}')+fr'\addlegendentry{{{label}}}'+'\n'
 b2+=plt(p,'single','rel_pier',f'only marks,{col},mark={mark}')
for city,anchor,dy in [('paris','north west',-3),('bern','south east',3),('luzern','south west',4)]:
 r=T[T.City==city].iloc[0];b+=fr'\node[font=\sffamily\fontsize{{7}}{{8}}\selectfont,anchor={anchor},yshift={dy}pt] at (axis cs:{r.rel_pier:.17g},{r.impact:.17g}) {{{city.title()}}};'+'\n'
b2+=r'\node[anchor=north west,font=\sffamily\fontsize{7}{8}\selectfont] at (rel axis cs:.04,.96) {26/31 group errors below single};'
axis('fig5a_traffic_utility',r'width=7.1cm,height=4.2cm,grid=major,xmin=0,xmax=64,ymin=-40,ymax=140,xlabel={PIER / mean flow (\%)},ylabel={MAE change / local MAE (\%)},legend style={at={(.98,.50)},anchor=east}',b,['data/original/tables/exp14_multicity_tabular_summary.csv','data/original/tables/exp14_multicity_tabular_summary_peer_weights_long.csv'],'All 31 exact city points; stated normalizations; flag only highlights numerical feasibility, no rows excluded.')
axis('fig5b_traffic_group',r'width=7.1cm,height=4.2cm,grid=major,xmin=0,xmax=64,ymin=0,ymax=64,xlabel={Closest single difference / mean flow (\%)},ylabel={Stored group residual / mean flow (\%)}',b2,['data/original/tables/exp14_multicity_tabular_summary.csv'],'All 31 archived points, identity reference; 26 improve, five do not. No claim of training-only single selection.')

# Supplementary exact diagnostics.
p=csv('interface_effects',E);b=r'\addplot[slate!60,dashed,domain=-.032:.046,no marks]{x};'+'\n'+plt(p,'v23_endpoint_raw_delta','v23_permutation_averaged_delta','only marks,teal,mark=*')
axis('figA_interface_change',r'width=7cm,height=4.4cm,grid=major,xlabel={Canonical endpoint change},ylabel={Balanced endpoint change},xmin=-.035,xmax=.048,ymin=-.035,ymax=.048',b,['data/table_v23_endpoint_claim_comparison.csv'],'All 16 matched-design endpoint effects; Spearman 0.93235, not the eight-target error-ranking correlation.')
d=D.copy();v1=d.trackwise_fit_aggregate_eval.mean();v2=d.trackwise_fit_trackwise_eval.mean();p=csv('cancellation_fixed',pd.DataFrame({'x':[1,2],'value':[v1,v2]}))
b=plt(p,'x','value','ybar,bar width=17pt,fill=teal!65,draw=teal')
axis('figA_cancellation',r'width=7cm,height=4.4cm,grid=major,ymin=0,ymax=.23,xmin=.4,xmax=2.6,xtick={1,2},xticklabels={{Magnitude after track mean},{Mean trackwise magnitude}},x tick label style={align=center,text width=2.4cm},ylabel={Mean absolute residual}',b,['data/v22/table_2_estimand_decomposition.csv'],'Mean across all rows of trackwise_fit_aggregate_eval versus trackwise_fit_trackwise_eval; same fitted weights.')

B=pd.read_csv(ORIG/'tables/exp6_bert_multicontext_pier.csv');bc=pd.read_csv(ORIG/'tables/exp4_bert_disco_dosesplit.csv')
for dose,name in [(0,'figF_bert_baseline'),(.5,'figF_bert_mid')]:
 d=B[np.isclose(B.Dose,dose)].pivot(index='TargetModel',columns='ContextLabel',values='MeanPIER');f=pd.DataFrame({'model':d.index,'length':d['len_long_(>15)']-d['len_medium_(8-15)'],'negation':d.has_negation-d.no_negation}).reset_index(drop=True);p=csv(name,f)
 b=r'\draw[slate!45,dashed] (axis cs:0,-.03)--(axis cs:0,.085);\draw[slate!45,dashed] (axis cs:-.032,0)--(axis cs:.029,0);'+'\n'+plt(p,'length','negation','only marks,teal,mark=*')
 for _,r in f.iterrows():
  anchors=({'ALBERT':('south east',-2,4),'BERT':('south west',2,4),'DistilBERT':('north west',2,-4),'RoBERTa':('north west',2,-4),'XLNet':('south east',-2,4)} if dose==0 else {'ALBERT':('north east',-2,-3),'BERT':('south west',2,4),'DistilBERT':('north east',-2,-4),'RoBERTa':('south east',-2,4),'XLNet':('north east',-2,-4)})
  anc,dx,dy=anchors[r.model]
  b+=fr'\node[font=\sffamily\fontsize{{6.5}}{{7.5}}\selectfont,anchor={anc},xshift={dx}pt,yshift={dy}pt] at (axis cs:{r.length:.17g},{r.negation:.17g}) {{{r.model}}};'+'\n'
 axis(name,r'width=5.25cm,height=4.2cm,grid=major,clip=false,xmin=-.036,xmax=.034,ymin=-.039,ymax=.098,xlabel={Long minus medium-length PIER},ylabel={Negation minus no-negation PIER},xtick={-.02,0,.02},ytick={-.02,0,.02,.04,.06,.08}',b,['data/original/tables/exp6_bert_multicontext_pier.csv'],f'Subgroup-mean differences at dose {dose}; no causal interpretation or invented intervals.')
b=''
for m,label,col,mark in [('Perfect Redundancy (Clone)','Clone','slate','*'),('Architectural Divergence (RoBERTa)','RoBERTa','teal','square*'),('Parametric Divergence (Finetuned)','Fine-tuned','coral','triangle*')]:
 d=bc[bc.Model==m].sort_values('Dose');p=csv('bert_control_'+label.replace('-',''),d);b+=plt(p,'Dose','PIER',f'{col},mark={mark},line width=.8pt')+fr'\addlegendentry{{{label}}}'+'\n'
axis('figF_bert_controls',r'width=5.25cm,height=4.2cm,grid=major,xlabel={Masking intensity},ylabel={Mean PIER},xmin=0,xmax=.9,ymin=0,ymax=.32,legend style={at={(.02,.98)},anchor=north west}',b,['data/original/tables/exp4_bert_disco_dosesplit.csv'],'All 30 exact dose-response values, including 1e-8 clone values.')

# Main and supplementary typeset tables from the same numeric sources.
lines=[r'\begin{table}[ht]',r'\caption{Balanced endpoint changes with pointwise 95\% common-question bootstrap intervals.}',r'\label{tab:endpoints-full}\centering\small',r'\begin{tabular}{lrr}\toprule',r'Target & Irrelevant context & Content deletion\\\midrule']
for m in models:
 vals=[]
 for fam in ['irrelevant_context','content_deletion']:
  r=E[(E.target==m)&(E.family==fam)].iloc[0];vals.append(f'${r.v23_permutation_averaged_delta:+.4f}\\;[{r.bootstrap_lower:+.4f},{r.bootstrap_upper:+.4f}]$')
 lines.append(ms[m]+' & '+' & '.join(vals)+r'\\')
lines += [r'\bottomrule\end{tabular}\end{table}'];(TAB/'endpoint_effects.tex').write_text('\n'.join(lines))
lines=[r'\begin{table}[ht]',r'\caption{Relative held-out group gains (percent). D: deletion; I: irrelevant context. Absolute-improvement intervals are preserved in the accompanying CSV.}',r'\label{tab:all-gains}\centering\small',r'\begin{tabular}{lrrrr}\toprule',r'Target & D / MSE & D / MAE & I / MSE & I / MAE\\\midrule']
for m in models:
 vals=[]
 for fam in fams:
  r=G[(G.target==m)&(G.family==fam)].iloc[0]
  for l in ['MSE','MAE']:vals.append(f'{100*(1-r[f"perm_{l}_convex_error"]/r[f"perm_{l}_single_error"]):.2f}')
 lines.append(ms[m]+' & '+' & '.join(vals)+r'\\')
lines+=[r'\bottomrule\end{tabular}\end{table}'];(TAB/'all_gains.tex').write_text('\n'.join(lines))
lines=[r'\begin{table}[ht]',r'\caption{Declared sibling-removal effects. The comparison column is the largest mean effect of deleting one non-sibling.}',r'\label{tab:sibling-full}\centering\small',r'\begin{tabular}{llrr}\toprule',r'Target & Family & Sibling effect [95\% CI] & Non-sibling max\\\midrule']
for _,r in S.iterrows():
 lines.append(f'{ms[r.target]} & '+('D' if r.family==fams[0] else 'I')+f' & ${r.permutation_averaged_inflation:.4f}\\;[{r.bootstrap_lower:.4f},{r.bootstrap_upper:.4f}]$ & {r.largest_non_sibling_inflation:.4f}'+r'\\')
lines+=[r'\bottomrule\end{tabular}\end{table}'];(TAB/'sibling_effects.tex').write_text('\n'.join(lines))
# Generation table use existing exact summary columns names supplied.
GD=pd.read_csv(DATA/'table_v23_generation_validation.csv')
# print columns for audit; tolerate consistent source schema only
print('generation columns',GD.columns.tolist())
num=GD.select_dtypes('number').columns
agg=GD.groupby('model')[list(num)].mean() if 'model' in GD else None
if agg is not None:
 agcol=next(c for c in agg if 'all_rotation_score_generation_agreement' in c)
 mf=next(c for c in agg if 'malformed_rate' in c)
 lines=[r'\begin{table}[ht]',r'\caption{Archived generation diagnostics, averaged across the three conditions. Values depend on the permissive answer parser and are not verified final-answer accuracy.}',r'\label{tab:generation}\centering\small',r'\begin{tabular}{lrr}\toprule',r'Model & Per-rotation agreement & Malformed fraction\\\midrule']
 for m in models:
  r=agg.loc[m];lines.append(ms[m]+f' & {r[agcol]:.3f} & {r[mf]:.3f}'+r'\\')
 lines+=[r'\bottomrule\end{tabular}\end{table}'];(TAB/'generation_diagnostics.tex').write_text('\n'.join(lines))

lines=[r'\begin{table}[ht]',r'\caption{Archived fitting geometry and support coefficients. Coefficients are rounded only for display; exact values and feasibility residuals remain in the data.}',r'\label{tab:geometry-support}\centering\small',r'\begin{tabular}{llrrrrr}\toprule',r'Target & Context & RN-50 & EffNet & ConvNeXt & ViT & $L_2$ residual\\\midrule']
for _,r in geo.iterrows():
 lab='SIN' if r.target.endswith('_SIN') else ('SIN+IN' if r.target.endswith('_SININ') else r'SIN+IN$\to$IN')
 ctxt='Natural' if r.context=='texture_natural' else 'Shape'
 lines.append(lab+' & '+ctxt+' & '+' & '.join(f'{r[c]:.3f}' for c in ['ResNet50','EfficientNetB0','ConvNeXtTiny','ViT_B16'])+f' & {r.l2_dist:.5f}'+r'\\')
lines+=[r'\bottomrule\end{tabular}\end{table}'];(TAB/'geometry_support.tex').write_text('\n'.join(lines))
lines=[r'\begin{figure}[!t]\centering']
for i,((m,ctx),name) in enumerate(geo_names.items()):
 lab='SIN' if m.endswith('_SIN') else ('SIN+IN' if m.endswith('_SININ') else r'SIN+IN$\to$IN')
 ctxt='natural' if ctx=='texture_natural' else 'shape-biased'
 lines.append(r'\begin{subfigure}[t]{.485\linewidth}\centering\includegraphics[width=\linewidth]{figures/pdf/'+name+r'.pdf}\caption{'+lab+' / '+ctxt+r'.}\end{subfigure}')
 lines.append(r'\hfill' if i%2==0 else r'\par\vspace{4pt}')
lines+=[r'\caption{All archived PCA views, fitted separately. Peer labels: 1 ResNet-50; 2 EfficientNet-B0; 3 ConvNeXt-T; 4 ViT-B/16. Gold diamonds denote the archived fitted mixture. Coordinates in each panel are multiplied by $10^3$ for readability; bases differ, so displacement across panels is not geometrically comparable.}',r'\label{fig:geometry-all}\end{figure}']
(TAB/'geometry_panels.tex').write_text('\n'.join(lines))

checks={
 'llm_six_targets':int(sum(all(float(json.loads(r[f'common_bootstrap_{loss}_improvement_interval'])[0])>0 for _,r in G[G.target==m].iterrows() for loss in ['MSE','MAE']) for m in models)),
 'endpoint_direction_matches':int(sum(np.sign(E.v23_endpoint_raw_delta)==np.sign(E.v23_permutation_averaged_delta))),
 'endpoint_change_spearman':float(spearmanr(E.v23_endpoint_raw_delta,E.v23_permutation_averaged_delta).statistic),
 'error_ranking_spearman':float(spearmanr(G.groupby('target').raw_MSE_convex_error.mean(),G.groupby('target').perm_MSE_convex_error.mean()).statistic),
 'traffic_improved':int(sum(T.PIER<T.ClosestPeerMeanAbsDiff)),
 'traffic_raw_corr':float(pearsonr(T.PIER,T.Delta_Router).statistic),
 'traffic_normalized_corr':float(pearsonr(T.rel_pier,T.impact).statistic),
 'traffic_min_weight':float(W.Weight.min()),
 'traffic_flagged_cities':T.loc[T.flag,'City'].tolist(),
 'natural_top20':int(round(pd.DataFrame(over).query('context=="texture_natural" and k==20').overlap.iloc[0]*20)),
 'shape_top20':int(round(pd.DataFrame(over).query('context=="shape_bias" and k==20').overlap.iloc[0]*20)),
 'fixed_fit_aggregate_mean':float(v1),'fixed_fit_trackwise_mean':float(v2),
}
(ROOT/'qa/numeric_checks.json').write_text(json.dumps(checks,indent=2))
(ROOT/'qa/figure_provenance.json').write_text(json.dumps(PROV,indent=2))
print(json.dumps(checks,indent=2))
print('Generated',len(PROV),'standalone panels.')

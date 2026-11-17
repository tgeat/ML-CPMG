"""v18 print-size artwork. Frozen arrays/measurements; no solver reruns."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
from pathlib import Path
import json,argparse,sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,SymLogNorm,ListedColormap,Normalize
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
from matplotlib.text import Text
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import MathTextParser
from mpl_toolkits.mplot3d.art3d import Poly3DCollection,Line3DCollection
ROOT=Path(__file__).resolve().parents[3];TASK=ROOT/'results/paper_revision_v18_fig10_median'
SOURCE=ROOT/'results/paper_revision_v18/data';DATA=ROOT/'results/paper_revision_v16/data'
OUT=TASK/'output/v18_English_CAS/figures';QA=TASK/'qa'
WIDTH=6.45
COLORS={'ML-CPMG':'#215F9A','Min-threshold':'#D88B54','Single-median':'#46877B','Algebraic-only':'#55545C','SA-0':'#939F41','SA-0.04':'#9565A6','SA-0.1':'#BD5364','RS-0.25':'#A47C38'}
SHORT={'ML-CPMG':'ML-CPMG','Min-threshold':'Min-threshold','Single-median':'Single-median','Algebraic-only':'Algebraic-only','SA-0':'SA (0)','SA-0.04':'SA (0.04)','SA-0.1':'SA (0.1)','RS-0.25':'RS (0.25)'}
PRIMARY=['ML-CPMG','Min-threshold','Single-median','Algebraic-only','SA-0.04','RS-0.25']
FAM=['multilevel_nested_lobes','multilevel_braided_channels','multilevel_layered_barriers','multilevel_anisotropic_streaks','unplanted_random','smooth_gaussian']
NAMES=['Nested lobes','Braided channels','Layered barriers','Anisotropic streaks','Unplanted random','Gaussian hill']
plt.rcParams.update({'font.family':'Arial','mathtext.fontset':'custom','mathtext.rm':'Times New Roman','mathtext.it':'Times New Roman:italic','mathtext.bf':'Times New Roman:bold','mathtext.sf':'Arial','mathtext.tt':'Courier New','mathtext.cal':'Times New Roman:italic','mathtext.fallback':'cm','font.size':8,'axes.titlesize':8.5,'axes.labelsize':8,'xtick.labelsize':7.5,'ytick.labelsize':7.5,'legend.fontsize':7.5,'axes.linewidth':.6,'xtick.major.width':.5,'ytick.major.width':.5,'xtick.major.size':2.5,'ytick.major.size':2.5,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path','savefig.facecolor':'white','figure.facecolor':'white'})
AUDIT={}

def export(fig,name):
    OUT.mkdir(parents=True,exist_ok=True);QA.mkdir(exist_ok=True)
    # Mathematical scripts are 70% of base size: 9 pt -> 6.3 pt.
    for _ in range(3):
        fig.canvas.draw()
        for t in fig.findobj(Text):
            if not t.get_visible() or not t.get_text().strip():continue
            t.set_fontsize(max(t.get_fontsize(),9 if '$' in t.get_text() else 7.5))
    renderer=fig.canvas.get_renderer();parser=MathTextParser('path');records=[];outside=[];skip_ticks=set()
    for ax in fig.axes:
        for axis in [ax.xaxis,ax.yaxis]+([ax.zaxis] if hasattr(ax,'zaxis') else []):
            if not ax.axison and not getattr(ax,'_axis3don',False):
                skip_ticks.update(id(t) for t in axis.findobj(Text));continue
            active={id(t) for t in axis._update_ticks()}
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if id(tick) not in active:skip_ticks.update([id(tick.label1),id(tick.label2)])
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip() or id(t) in skip_ticks:continue
        s=t.get_text();bb=t.get_window_extent(renderer)
        if bb.x1<0 or bb.y1<0 or bb.x0>fig.bbox.width or bb.y0>fig.bbox.height:continue # invisible out-of-range ticks
        if bb.x0<1 or bb.y0<1 or bb.x1>fig.bbox.width-1 or bb.y1>fig.bbox.height-1:outside.append(s)
        minimum=t.get_fontsize()
        if '$' in s:
            minimum=min(g[1] for g in parser.parse(s,dpi=72,prop=t.get_fontproperties()).glyphs)
        records.append({'text':s,'base_pt':t.get_fontsize(),'min_glyph_pt':minimum,'bbox_px':list(bb.bounds)})
    assert min(r['min_glyph_pt'] for r in records)>=6,(name,records)
    assert not outside,(name,outside)
    for ext in ['pdf','svg','png']:fig.savefig(OUT/f'{name}.{ext}',dpi=600)
    with plt.rc_context({'svg.fonttype':'none'}):fig.savefig(OUT/f'{name}_editable.svg')
    fig.savefig(QA/f'{name}.png',dpi=180)
    AUDIT[name]={'width_mm':WIDTH*25.4,'height_mm':fig.get_figheight()*25.4,'text':records,'min_glyph_pt':min(r['min_glyph_pt'] for r in records),'no_canvas_overflow':True}
    p=QA/'figure_audit.json';old=json.loads(p.read_text()) if p.exists() else {};old.update(AUDIT);p.write_text(json.dumps(old,indent=2),encoding='utf8')
    plt.close(fig);print('Exported',name,flush=True)

def clean(ax):
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',lw=.35,color='#dddddd');ax.set_axisbelow(True)
def panel(ax,label):ax.set_title(label,loc='left',pad=7)
def labels_cmap(n):
    from matplotlib.colors import hsv_to_rgb
    c=[(.92,.93,.94,1)]
    for i in range(n):c.append((*hsv_to_rgb(((.57+i*.61803398875)%1,.46+.18*(i%3)/2,.72+.19*(i%4)/3)),1))
    return ListedColormap(c)

def stencil():
    fig=plt.figure(figsize=(WIDTH,3.2));a=fig.add_axes([.04,.28,.44,.58]);b=fig.add_axes([.56,.28,.40,.58])
    for ax in [a,b]:ax.axis('off')
    a.set_xlim(-.5,5.3);a.set_ylim(-1.2,2.8)
    for x,w,c in [(0,2,'#E1ECF4'),(2,3,'#F1E8D9')]:a.add_patch(Rectangle((x,0),w,2,fc=c,ec='#303840',lw=.7))
    a.plot([2,2],[0,2],c='#BA4A35',lw=1.6)
    a.plot([1,3.5],[1,1],'o',c='#263D50',ms=3)
    a.annotate('',xy=(3.45,1),xytext=(1.05,1),arrowprops=dict(arrowstyle='->',color=COLORS['ML-CPMG'],lw=1))
    a.text(1,1.5,r'$u,\ p_u,\ k_{n,u}$',ha='center');a.text(3.5,1.5,r'$v,\ p_v,\ k_{n,v}$',ha='center')
    a.text(2.6,.65,r'$q_{uv}$',ha='center',color=COLORS['ML-CPMG'])
    a.annotate(r'$A_f$',xy=(2,2),xytext=(2.45,2.45),arrowprops=dict(arrowstyle='-',lw=.5),ha='left')
    for x in [1,2,3.5]:a.plot([x,x],[0,-.42],c='#666666',lw=.5,ls='--')
    for x0,x1,s in [(1,2,r'$d_u$'),(2,3.5,r'$d_v$')]:
        a.annotate('',xy=(x0,-.32),xytext=(x1,-.32),arrowprops=dict(arrowstyle='<->',lw=.5));a.text((x0+x1)/2,-.61,s,ha='center')
    for xy in [(.45,-.92),(-.2,-.25)]:a.annotate('',xy=xy,xytext=(-.2,-.92),arrowprops=dict(arrowstyle='->',lw=.6))
    a.text(.57,-.92,'$x$',va='center');a.text(-.2,-.12,'$z$',ha='center')
    b.set_xlim(-.4,4.35);b.set_ylim(-1.0,4.75)
    for left,zoff,col in [(0,0,'#E1ECF4'),(2,1,'#F1E8D9')]:
        for c in range(2):
            for r in range(3):b.add_patch(Rectangle((left+c,zoff+r),1,1,fc=col,ec='#303840',lw=.55))
        for r in range(3):b.plot([left+.5,left+1.5],[zoff+r+.5]*2,c='#9AA0A4',lw=.5)
        for c in range(2):b.plot([left+c+.5]*2,[zoff+.5,zoff+2.5],c='#9AA0A4',lw=.5)
        for c in range(2):
            for r in range(3):b.plot(left+c+.5,zoff+r+.5,'o',c='#303840',ms=2.8)
    b.plot([2,2],[-.25,4.35],'--',c='#444444',lw=.65)
    b.plot([1.5,2.5],[1.5,1.5],c='#BA4A35',lw=1.8,marker='o',ms=4)
    b.annotate('NNC',xy=(1.9,1.5),xytext=(.65,-.65),ha='center',va='center',fontsize=8,
               arrowprops=dict(arrowstyle='-',lw=.55,shrinkA=3,shrinkB=3),annotation_clip=False)
    b.annotate(r'$u:(1,1)$',xy=(1.5,1.5),xytext=(.0,3.45),arrowprops=dict(arrowstyle='-',lw=.5))
    b.annotate(r'$v:(2,0)$',xy=(2.5,1.5),xytext=(2.7,.25),arrowprops=dict(arrowstyle='-',lw=.5))
    b.annotate('Fault',xy=(2,4.25),xytext=(2.5,4.55),arrowprops=dict(arrowstyle='-',lw=.5))
    fig.text(.04,.94,'(a) Orthogonal TPFA',fontsize=8.5);fig.text(.56,.94,'(b) Faulted active graph',fontsize=8.5)
    fig.text(.26,.17,r'$T_{uv}=A_f(d_u/k_{n,u}+d_v/k_{n,v})^{-1}$',ha='center',fontsize=9)
    fig.text(.26,.065,r'$q_{uv}=T_{uv}(p_u-p_v)$',ha='center',fontsize=9)
    fig.text(.76,.17,r'$\|\mathbf{x}_u-\mathbf{x}_v\|_1=2\ne1$',ha='center',fontsize=9)
    fig.text(.76,.065,r'$T_{uv}>0$',ha='center',fontsize=9)
    AUDIT['stencil_logic']={'u':[1,1],'v':[2,0],'logical_l1':2,'fault_gap':0,'unequal_half_widths':[1,1.5]}
    export(fig,'fig01')

def block():
    fig=plt.figure(figsize=(WIDTH,3.3));ax=fig.add_axes([.055,.065,.56,.87]);ax.set_xlim(-.15,8.15);ax.set_ylim(-.15,8.15);ax.set_aspect('equal');ax.axis('off')
    labels=np.zeros((8,8),int)
    for r,c in [(0,2),(0,3),(0,4),(1,2),(2,0),(2,1),(2,2)]:labels[r,c]=1
    for r,c in [(3,3),(3,4),(3,5),(4,5),(5,5),(5,6),(5,7)]:labels[r,c]=2
    cols=['#F0F1F2','#8999B5','#EE987C']
    for r in range(8):
        for c in range(8):
            v=labels[r,c];ax.add_patch(Rectangle((c,7-r),1,1,fc=cols[v],ec='#A6ADB4',lw=.35))
            if v or (r,c) in [(2,3),(3,2)]:ax.text(c+.5,7-r+.5,['0','A','B'][v],ha='center',va='center',fontsize=9)
    for k in range(0,9,2):ax.plot([k,k],[0,8],c='#42494F',lw=.8);ax.plot([0,8],[k,k],c='#42494F',lw=.8)
    ax.add_patch(Rectangle((2,4),2,2,fc='none',ec='#162D47',lw=1.6,zorder=5))
    ax.annotate(r'$B_b$',xy=(4,6),xytext=(4.7,6.7),arrowprops=dict(arrowstyle='-',lw=.6))
    legend=fig.add_axes([.66,.12,.30,.78]);legend.axis('off');legend.set_xlim(0,1);legend.set_ylim(0,1)
    for y,lw,label in [(.91,.35,'Cells'),(.79,.8,'Blocks')]:legend.plot([0,.20],[y,y],c='#42494F',lw=lw);legend.text(.28,y,label,va='center')
    for i,(s,col) in enumerate(zip([r'$C_A$',r'$C_B$',r'$C_0$'],[cols[1],cols[2],cols[0]])):
        y=.62-i*.14;legend.add_patch(Rectangle((.0,y-.045),.14,.09,fc=col,ec='#444444',lw=.5));legend.text(.28,y,s,va='center')
    legend.text(0,.13,r'$G_{b,c}=B_b\cap C_c$',fontsize=9)
    legend.text(0,.015,'3 coarse DOFs',fontsize=8)
    AUDIT['block_logic']={'selected_labels':labels[2:4,2:4].tolist(),'nonempty_intersections':3,'labels':labels.tolist()}
    export(fig,'fig02')

def fields():
    d=np.load(SOURCE/'synthetic.npz');fig,axs=plt.subplots(2,3,figsize=(WIDTH,4.45))
    for i,(ax,fam) in enumerate(zip(axs.flat,FAM)):
        im=ax.imshow(d[fam+'_logk'][32].T,origin='lower',extent=(0,1,0,1),cmap='viridis',vmin=0,vmax=6,interpolation='nearest')
        panel(ax,f'({chr(97+i)}) {NAMES[i]}');ax.set_xlabel('$y$',labelpad=1);ax.set_ylabel('$z$',labelpad=1);ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1])
    fig.subplots_adjust(left=.07,right=.88,bottom=.12,top=.92,wspace=.38,hspace=.5)
    ca=fig.add_axes([.915,.20,.017,.62]);cb=fig.colorbar(im,cax=ca,ticks=[0,2,4,6]);cb.set_label(r'$\log_{10}k$',labelpad=5)
    export(fig,'fig03')

def segmentation():
    d=np.load(SOURCE/'synthetic.npz');fig=plt.figure(figsize=(WIDTH,5.9))
    # Reference layout: repeated column headings, square panels, local bars, row names below.
    width=.17;height=WIDTH*width/5.9;row_y=[.745,.425,.105]
    axs=np.array([[fig.add_axes([x,y,width,height]) for x in [.065,.335,.555,.775]] for y in row_y])
    for r,fam in enumerate([FAM[0],FAM[1],FAM[-1]]):
        for c,ax in enumerate(axs[r]):
            if c==0:
                im=ax.imshow(d[fam+'_logk'][32].T,origin='lower',extent=(0,1,0,1),cmap='viridis',vmin=0,vmax=6,interpolation='nearest')
            else:
                lab=d[fam+'_'+['Min-threshold','Single-median','ML-CPMG'][c-1]];n=int(lab.max())
                ax.imshow(lab[32].T,origin='lower',extent=(0,1,0,1),cmap=labels_cmap(n),vmin=-.5,vmax=n+.5,interpolation='nearest')
            ax.set_xticks([0,.5,1]);ax.set_yticks([0,.5,1]);ax.set_xlabel('$y$',labelpad=0);ax.set_ylabel('$z$',labelpad=0)
            ax.tick_params(pad=1.5)
            ax.set_title([r'$\log_{10}k$','Min-threshold','Single-median','ML-CPMG'][c],loc='center',pad=6,fontsize=8.5,fontweight='bold')
        ca=fig.add_axes([.247,row_y[r],.011,height]);fig.colorbar(im,cax=ca,ticks=[0,2,4,6]);ca.tick_params(pad=2)
        fig.text(.505,row_y[r]-.080,['Nested lobes','Braided channels','Gaussian hill'][r],ha='center',va='center',fontsize=8.5,fontweight='bold')
    AUDIT['segmentation_layout']={'subfigure_letters':False,'background_legend_in_figure':False,'colorbar':'three local vertical bars, one beside each permeability panel, same 0-6 limits','row_names':'centred below each row','column_titles':'repeated above every row','categorical_labels_unchanged':True}
    export(fig,'figA')

def performance(S):
    fig,axs=plt.subplots(2,2,figsize=(WIDTH,5.15));xs=np.arange(6);width=.125
    metrics=[('iterations','Krylov iterations',False),('total_seconds','Cold time (s)',False),('operator_complexity',r'$C_{\mathrm{op}}$',False),('kappa_est','Ritz condition estimate',True)]
    for j,(metric,label,log) in enumerate(metrics):
        ax=axs.flat[j]
        for i,m in enumerate(PRIMARY):
            vals=[S['Synthetic-'+f,m][metric] for f in FAM];loc=xs+(i-2.5)*width
            ax.bar(loc,vals,width*.9,color=COLORS[m],edgecolor='white',lw=.2,label=SHORT[m])
        if log:ax.set_yscale('log')
        if metric=='operator_complexity':ax.axhline(3,c='#333333',ls='--',lw=.7);ax.set_ylim(0,4);ax.text(.98,3.35,'Complexity cap',transform=ax.get_yaxis_transform(),ha='right',fontsize=7.5)
        clean(ax);ax.set_ylabel(label,labelpad=3);ax.set_xticks(xs,['Nested','Braided','Layered','Streaks','Random','Hill'],rotation=38,ha='right');panel(ax,f'({chr(97+j)})')
    h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=3,frameon=False,bbox_to_anchor=(.53,.997),columnspacing=1.3,handlelength=1.4)
    fig.subplots_adjust(top=.84,bottom=.115,left=.105,right=.985,hspace=.60,wspace=.40)
    AUDIT['synthetic_time_display']={'panel':'b','statistic':'median','repetitions':3,'range_errorbars':False}
    export(fig,'fig04')

def spe_axes(ax,j,white=False):
    ext=[(0,2200,0,1200),(0,1200,0,170),(0,2200,0,170)][j]
    ax.set_xlim(ext[:2]);ax.set_ylim(ext[2:] if j==0 else ext[:1:-1]);ax.set_aspect(1 if j==0 else 7)
    ax.set_xlabel(['Y (ft)','X (ft)','Y (ft)'][j],labelpad=1);ax.set_ylabel('X (ft)' if j==0 else 'Z (ft)',labelpad=1)
    ax.set_xticks([0,2200] if j!=1 else [0,1200]);ax.set_yticks([0,1200] if j==0 else [0,170]);ax.tick_params(pad=1.5)
    color='#FFFFFF' if white else '#222222'
    # Dark exterior rim keeps well symbols visible even where they cross white margins.
    import matplotlib.patheffects as pe
    rim=[pe.withStroke(linewidth=1.5,foreground='#222222')] if white else []
    if j==0:
        ax.scatter([1095],[590],marker='o',s=17,fc='none',ec=color,lw=.75,zorder=5,path_effects=rim)
        ax.scatter([5,2195,5,2195],[10,10,1190,1190],marker='^',s=18,fc='none',ec=color,lw=.7,zorder=5,clip_on=False,path_effects=rim)
    else:
        x=590 if j==1 else 1095;ax.axvline(x,ls='--',lw=.65,c=color,path_effects=rim)
        ax.scatter([x],[85],marker='o',s=15,fc='none',ec=color,lw=.7,zorder=5,path_effects=rim)
        ax.scatter([10,1190] if j==1 else [5,2195],[85,85],marker='^',s=18,fc='none',ec=color,lw=.7,clip_on=False,zorder=5,path_effects=rim)

def spe_grid(rows,height):
    fig,axs=plt.subplots(rows,3,figsize=(WIDTH,height),gridspec_kw={'width_ratios':[2200,1200,2200]})
    fig.subplots_adjust(left=.115,right=.845,bottom=.085,top=.89,wspace=.42,hspace=.50)
    for j,ax in enumerate(axs[0]):ax.set_title(['K = 43','J = 110\nVE = 7','I = 30\nVE = 7'][j],fontsize=8,pad=9)
    return fig,axs

def spe10():
    d=np.load(SOURCE/'spe10_sections.npz');n=int(d['label_count']);ext=[(0,2200,0,1200),(0,1200,0,170),(0,2200,0,170)]
    fig,axs=spe_grid(3,4.5)
    norms=[SymLogNorm(linthresh=1e-4,vmin=-1,vmax=1,base=10),Normalize(-4,5),Normalize(-.5,n+.5)];maps=['RdBu_r','viridis',labels_cmap(n)]
    for r,prefix in enumerate(['p','k','labels']):
        for j in range(3):
            ax=axs[r,j];im=ax.imshow(d[f'{prefix}{j}'],origin='lower',extent=ext[j],interpolation='nearest',cmap=maps[r],norm=norms[r]);spe_axes(ax,j,white=r==1)
            ax.text(.03,.91,f'({chr(97+3*r+j)})',transform=ax.transAxes,va='top',fontsize=7.5,bbox=dict(fc='white',ec='none',alpha=.8,pad=.4))
        center=np.mean([axs[r,0].get_position().y0,axs[r,0].get_position().y1]);fig.text(.022,center,['Pressure',r'$K_x$','Basin labels'][r],rotation=90,va='center',ha='center')
        if r<2:
            pos=axs[r,0].get_position();ca=fig.add_axes([.87,pos.y0,.015,pos.height]);cb=fig.colorbar(im,cax=ca)
            if r==0:cb.set_ticks([-1,-1e-3,0,1e-3,1],labels=['−1',r'$-10^{-3}$','0',r'$10^{-3}$','1']);cb.minorticks_off();cb.set_label(r'$p/\|p_{\mathrm{ref}}\|_\infty$',labelpad=2)
            else:cb.set_ticks([-4,0,4]);cb.set_label(r'$\log_{10}(K_x/\mathrm{mD})$',labelpad=2)
        else:fig.text(.873,center,'IDs\n1–256',va='center',fontsize=7.5)
    export(fig,'fig05')
    fig,axs=spe_grid(4,5.75);maximum=float(d['error_max']);emax=10**np.ceil(np.log10(maximum));emin=max(1e-16,emax*1e-5)
    for r,m in enumerate(['ML-CPMG','Single-median','Algebraic-only','SA-0.04']):
        for j in range(3):
            ax=axs[r,j];im=ax.imshow(np.maximum(d[f'error{r}_{j}'],emin),origin='lower',extent=ext[j],interpolation='nearest',cmap='magma',norm=LogNorm(emin,emax));spe_axes(ax,j,white=True)
            ax.text(.03,.91,f'({chr(97+3*r+j)})',transform=ax.transAxes,va='top',fontsize=7.5,color='white')
        pos=axs[r,0].get_position();fig.text(.022,(pos.y0+pos.y1)/2,SHORT[m],rotation=90,ha='center',va='center')
    ca=fig.add_axes([.875,.20,.017,.58]);cb=fig.colorbar(im,cax=ca,ticks=[1e-16,1e-14,1e-12,1e-11]);cb.minorticks_off();cb.set_label(r'$|p-p_{\mathrm{ref}}|/\|p_{\mathrm{ref}}\|_\infty$',labelpad=2)
    AUDIT['spe10_logic']={'error_scale':[emin,emax],'observed_error_max':maximum,'slice_indices':[43,110,30],'extents_ft':ext,'vertical_exaggeration':7,'array_labels':n}
    export(fig,'fig06')

def reservoir(S):
    methods=list(COLORS);fig,axs=plt.subplots(2,2,figsize=(WIDTH,5.45));x=np.arange(3)
    for col,(prefix,days) in enumerate([('SPE10',['0','520.04','2000']),('Brugge',['0','1000','2000'])]):
        ms=PRIMARY if col==0 else methods;width=.82/len(ms)
        for r,metric in enumerate(['total_seconds','iterations']):
            ax=axs[r,col]
            for i,m in enumerate(ms):
                vals=[S[f'{prefix}-{d}d',m][metric] for d in days];loc=x+(i-(len(ms)-1)/2)*width
                ax.bar(loc,vals,width*.91,color=COLORS[m],ec='white',lw=.2,label=SHORT[m])
            clean(ax);ax.set_xticks(x,days);ax.set_xlabel('State (d)',labelpad=2);ax.set_ylabel('Cold time (s)' if r==0 else 'Krylov iterations',labelpad=3);ax.margins(y=.13);panel(ax,f'({chr(97+2*r+col)}) {prefix}')
    h,l=axs[0,1].get_legend_handles_labels();fig.legend(h,l,ncol=4,frameon=False,loc='upper center',bbox_to_anchor=(.53,.99),columnspacing=.9,handlelength=1.2,handletextpad=.4)
    fig.subplots_adjust(top=.83,bottom=.09,left=.10,right=.985,hspace=.52,wspace=.34)
    AUDIT['reservoir_time_display']={'panel_a':'median only','panel_b':'median only','repetitions':3,'range_errorbars_a':False,'range_errorbars_b':False}
    export(fig,'fig07')

def pareto():
    rows=[r for r in json.loads((DATA/'calibrate.json').read_text()) if 'operator_complexity' in r];assert len(rows)==14
    fig,axs=plt.subplots(1,2,figsize=(WIDTH,3.7))
    for ax in axs:ax.axvspan(1.35,3,color='#EDF3F8');ax.axvline(3,c='#555555',ls='--',lw=.6);clean(ax);ax.set_xlabel(r'$C_{\mathrm{op}}$');ax.set_ylabel('Cold time (s)')
    panel(axs[0],'(a) All configurations');panel(axs[1],'(b) Feasible ML-CPMG')
    for row in rows:
        m=row['method'];xx=row['operator_complexity'];yy=row['total_seconds'];ml=m=='ML-CPMG'
        axs[0].scatter(xx,yy,s=23,c=COLORS[m],marker='o' if ml else 's',ec='white',lw=.4,zorder=4)
        if ml and xx<=3:
            axs[1].scatter(xx,yy,s=25,c=COLORS[m],marker='o' if row['stall']==.33 else 'D',ec='white',lw=.4,zorder=4)
            off={32:(5,8),64:(5,-13),128:(4,7),256:(-5,23)}[row['cap']]
            if row['stall']==.2:off=(6,8)
            axs[1].annotate(f"({row['cap']}, {row['stall']:.2f})",(xx,yy),xytext=off,textcoords='offset points',fontsize=7.5,ha='right' if row['cap']==256 else 'left')
    selected=next(r for r in rows if r['method']=='ML-CPMG' and r['cap']==256 and r['stall']==.33)
    for ax in axs:ax.scatter(selected['operator_complexity'],selected['total_seconds'],s=62,marker='*',c='#BA4A35',zorder=6,ec='white',lw=.35)
    axs[0].annotate('Algebraic\nfallback',(5.624,13.72),xytext=(6.6,29),fontsize=7.5,arrowprops=dict(arrowstyle='-',lw=.5))
    axs[0].set_yscale('log');axs[0].set_xlim(1.25,12.1);axs[0].set_ylim(10,350);axs[0].set_yticks([10,30,100,300],labels=['10','30','100','300']);axs[0].set_xticks([2,6,10])
    axs[1].set_xlim(2.06,3.08);axs[1].set_ylim(35.6,45.1);axs[1].set_xticks([2.2,2.6,3]);axs[1].set_yticks([36,40,44])
    feasible=sorted([r for r in rows if r['method']=='ML-CPMG' and r['operator_complexity']<=3],key=lambda r:r['operator_complexity']);front=[];best=float('inf')
    for r in feasible:
        if r['total_seconds']<best:front.append(r);best=r['total_seconds']
    axs[1].plot([r['operator_complexity'] for r in front],[r['total_seconds'] for r in front],ls=':',lw=.7,c=COLORS['ML-CPMG'],zorder=1)
    methods=['ML-CPMG','Min-threshold','Single-median','SA-0','SA-0.04','SA-0.1','RS-0.25'];handles=[Line2D([],[],ls='',marker='o' if m=='ML-CPMG' else 's',ms=4,c=COLORS[m],label=SHORT[m]) for m in methods]
    handles.append(Line2D([],[],ls='',marker='*',ms=6,c='#BA4A35',label='Selected'))
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.52,.995),ncol=4,frameon=False,columnspacing=.9,handletextpad=.35,handlelength=1)
    fig.subplots_adjust(left=.09,right=.985,bottom=.15,top=.77,wspace=.35)
    AUDIT['pareto_logic']={'completed_configs':len(rows),'selected':[selected['cap'],selected['stall']],'separate_protocol':True}
    export(fig,'figC')

def brugge():
    d=np.load(SOURCE/'brugge_geometry.npz');corners=d['corners'];centers=corners.mean(axis=1);xyz=d['xyz'];tail=d['tail'];head=d['head']
    lookup={tuple(x):i for i,x in enumerate(xyz)};defs=[((0,0,-1),[0,1,2,3]),((0,0,1),[4,5,6,7]),((-1,0,0),[0,3,7,4]),((1,0,0),[1,2,6,5]),((0,-1,0),[0,1,5,4]),((0,1,0),[3,2,6,7])]
    faces=[];ids=[]
    for off,ix in defs:
        inds=np.flatnonzero(np.asarray([tuple(v+off) not in lookup for v in xyz]));faces.append(corners[inds][:,ix]);ids.append(inds)
    faces=np.concatenate(faces);ids=np.concatenate(ids);origin=corners[:,:,:2].min(axis=(0,1))
    def tr(a):
        b=a.copy();b[...,:2]=(b[...,:2]-origin)/1000;return b
    f=tr(faces);cc=tr(centers);seg=np.stack((cc[tail],cc[head]),axis=1)
    fig=plt.figure(figsize=(WIDTH,4.25))
    # A real four-cell patch straddles a positive-transmissibility NNC.
    # No invented block labels, exploded coordinates, or synthetic geology.
    edge=len(tail)//2;u=int(tail[edge]);v=int(head[edge])
    selected=[u,lookup[tuple(xyz[u]+[1,0,0])],v,lookup[tuple(xyz[v]+[1,0,0])]]
    assert len(set(selected))==4 and np.all(d['trans']>0)
    assert np.all(np.abs(xyz[tail]-xyz[head]).sum(axis=1)!=1)
    patch=corners[selected].copy();patch_origin=patch.min(axis=(0,1));patch-=patch_origin
    assert np.allclose(patch+patch_origin,corners[selected],rtol=0,atol=1e-10)
    pc=patch.mean(axis=1);pmin=patch.min(axis=(0,1));pmax=patch.max(axis=(0,1));pspan=pmax-pmin
    patch_axes=[fig.add_axes([x,.555,.405,.365],projection='3d',computed_zorder=False) for x in [.035,.535]]
    face_ix=[ix for _,ix in defs];edge_ix=[(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    pillar_segments=[]
    for cell in patch:
        for j in range(4):
            p,q=cell[j],cell[j+4];delta=q-p
            pillar_segments.append([p-.18*delta,q+.18*delta])
    patch_axes[0].add_collection3d(Line3DCollection(pillar_segments,colors='#32383D',lw=.65,zorder=2))
    for n,cell in enumerate(patch):
        color='#AEC6D6' if n<2 else '#DDD0B8'
        patch_axes[1].add_collection3d(Poly3DCollection(cell[face_ix],facecolors=color,edgecolors='none',alpha=.62,zorder=1))
        patch_axes[1].add_collection3d(Line3DCollection(cell[np.array(edge_ix)],colors='#384149',lw=.55,zorder=2))
    for ax in patch_axes:
        points=np.unique(patch.reshape(-1,3),axis=0)
        ax.scatter(*points.T,s=8,facecolors='white',edgecolors='#303840',linewidths=.5,depthshade=False,zorder=4)
        ax.set_xlim(-.06*pspan[0],1.06*pspan[0]);ax.set_ylim(-.06*pspan[1],1.06*pspan[1]);ax.set_zlim(1.18*pspan[2],-.18*pspan[2])
        ax.set_box_aspect((pspan[0],pspan[1],7*pspan[2]),zoom=1.22);ax.view_init(elev=25,azim=-58);ax.set_proj_type('ortho');ax.set_axis_off()
    patch_axes[1].plot(*pc[[0,2]].T,c='#BA4A35',lw=1.5,zorder=6)
    patch_axes[1].scatter(*pc[[0,2]].T,s=13,c='#BA4A35',depthshade=False,zorder=7)
    # Orthographic patches extend into their allocated whitespace. Disable
    # the internal square 3-D axes clip, while retaining the figure margins.
    for ax in patch_axes:
        for artist in [*ax.collections,*ax.lines]:artist.set_clip_on(False)
    # Short axes, separated from the cell artwork, follow the reference convention.
    for x in [.050,.550]:
        from mpl_toolkits.mplot3d import proj3d
        tri=fig.add_axes([x,.59,.065,.09]);tri.set_xlim(-.20,1.55);tri.set_ylim(-1.35,1.2);tri.axis('off')
        mat=patch_axes[0].get_proj();zero=np.asarray(proj3d.proj_transform(0,0,0,mat)[:2])
        for i,label in enumerate(['X','Y','Z']):
            endpoint=np.eye(3)[i]*pspan[i]*.2;direction=np.asarray(proj3d.proj_transform(*endpoint,mat)[:2])-zero
            direction/=np.linalg.norm(direction);tri.plot([0,direction[0]],[0,direction[1]],c='#303840',lw=.65)
            tri.text(*(direction*1.28),label,ha='center',va='center')
    a=fig.add_axes([.015,.035,.385,.43],projection='3d',computed_zorder=False);b=fig.add_axes([.52,.035,.385,.43],projection='3d',computed_zorder=False)
    a.add_collection3d(Poly3DCollection(f,fc='#D4D9DD',ec='#59636C',lw=.10,zorder=1,rasterized=True))
    # All NNC edges are overlaid through the shell for visibility.
    a.add_collection3d(Line3DCollection(seg,colors='#BA4A35',lw=.65,zorder=5))
    vals=d['pressure']/np.max(abs(d['pressure']));norm=Normalize(0,1);cmap=plt.get_cmap('viridis');b.add_collection3d(Poly3DCollection(f,fc=cmap(norm(vals[ids])),ec='#394650',lw=.065,zorder=2,rasterized=True))
    mins=corners.min(axis=(0,1));maxs=corners.max(axis=(0,1));span=maxs-mins
    for ax in [a,b]:
        ax.set_xlim(0,span[0]/1000);ax.set_ylim(0,span[1]/1000);ax.set_zlim(maxs[2],mins[2]);ax.set_xticks([0,8,16]);ax.set_yticks([0,4,8]);ax.set_zticks([1600,1800,2000])
        ax.set_box_aspect((span[0],span[1],7*span[2]));ax.view_init(elev=25,azim=-58);ax.set_proj_type('ortho')
        ax.set_xlabel('X (km)',labelpad=5);ax.set_ylabel('Y (km)',labelpad=3);ax.set_zlabel('Depth (m)',labelpad=0);ax.tick_params(labelsize=7.5,pad=0);ax.grid(False)
        for axis in [ax.xaxis,ax.yaxis,ax.zaxis]:axis.pane.fill=False
    for x,y,s in [(.04,.955,'(a) Native pillars and corners'),(.54,.955,'(b) Corner cells and NNC'),(.04,.505,'(c) Active grid and NNC'),(.54,.505,'(d) ML-CPMG pressure')]:fig.text(x,y,s,fontsize=8.5)
    ca=fig.add_axes([.65,.435,.22,.015]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=ca,orientation='horizontal',ticks=[0,.5,1]);ca.tick_params(pad=1)
    fig.text(.55,.439,r'$p/\|p\|_\infty$',ha='left',va='center')
    AUDIT['brugge_logic']={'active':len(xyz),'nnc':len(tail),'nnc_l1_min':int(np.abs(xyz[tail]-xyz[head]).sum(axis=1).min()),'ve':7,'pressure':'stored 0 d pressure above producer datum','local_patch_active_ids':selected,'local_patch_logical_indices':xyz[selected].tolist(),'local_nnc_edge_index':edge,'local_nnc_transmissibility':float(d['trans'][edge]),'local_nnc_l1':int(abs(xyz[u]-xyz[v]).sum()),'local_geometry':'native four-cell patch, translated only, no exploded offsets','wireframe':'native external cell edges','nnc_display':'all 152 centre-to-centre edges overlaid through exterior shell','panel_count':4}
    export(fig,'figB')

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--only',nargs='*');args=parser.parse_args();rows=json.loads((DATA/'summary.json').read_text());S={(r['case'],r['method']):r for r in rows}
    funcs={'stencil':stencil,'block':block,'fields':fields,'segmentation':segmentation,'performance':lambda:performance(S),'spe10':spe10,'reservoir':lambda:reservoir(S),'pareto':pareto,'brugge':brugge}
    for name,fn in funcs.items():
        if not args.only or name in args.only:fn()
    p=QA/'figure_audit.json';old=json.loads(p.read_text()) if p.exists() else {};old.update(AUDIT);p.write_text(json.dumps(old,indent=2),encoding='utf8')
if __name__=='__main__':main()

"""Plot two existing result tables without refitting the study models."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def main():
    table=ROOT/'results/acdc_ablation.csv'
    if not table.is_file():raise SystemExit('Run the documented analysis stages first; results/acdc_ablation.csv is missing.')
    data=pd.read_csv(table)
    fig,axes=plt.subplots(1,2,figsize=(9.6,3.5),layout='constrained')
    styles={'A1 +anchors':('#99BADF','o'),'A2 +shift':('#D8E7CA','s'),'A3 +pooling':('#C7A3CC','D')}
    for name,(color,marker) in styles.items():
        sub=data[data.variant==name].sort_values('n_cal')
        for ax,key in zip(axes,['DC_MAE','cov_param']):
            ax.plot(sub.n_cal,sub[key],color=color,marker=marker,markeredgecolor='#444444',label=name)
    for ax in axes:
        ax.set_xlabel('Reference anchors');ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Simulated-drape MAE (percentage points)')
    axes[1].set_ylabel('Empirical parameter coverage');axes[1].axhline(.9,color='#777777',ls=':',lw=1)
    axes[1].legend(frameon=False,fontsize=8)
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    for ext in ['png','pdf']:fig.savefig(out/f'calibration_summary.{ext}',dpi=300)
    plt.close(fig)
if __name__=='__main__':main()

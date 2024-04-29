##% LIBRARY IMPORT
import xarray as xr
import numpy as np
import scipy as sc
import sklearn as skl
import skimage as ski
from scipy.linalg import norm
from scipy.spatial.distance import euclidean, jensenshannon, correlation
from scipy.stats import wasserstein_distance, ecdf
import skgstat as skg
import pysteps
from pysteps.verification.spatialscores import fss, intensity_scale
from pysteps.verification.salscores import sal
from pysteps.verification.detcontscores import det_cont_fct
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib as mpl
import argparse as ap
print('modules loaded')
##%
parser = ap.ArgumentParser()
parser.add_argument('--var', type=str, required=True)
args = parser.parse_args()
mm=int(args.var)
print('settings loaded')
##% SETTINGS
datapath='/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/MF_ML_PREDICT/season_2020/'
models=['graphcast_','graphcast-oper_','pangu_','pangu-oper_','ifs_']#'fcnv2_dawn','ifs']
labels=['graphcast','graphcast-oper','pangu','pangu-oper','ifs']
references=['USA_era5_convseason_2020.nc','USA_init_convseason_2020.nc']
rlabels=['era5','ifs-init']
figpath='/users/mfeldman/figs/season/'
c1='#648fff' #lightblue
c2='#785ef0' #indigo
c3='#dc267f' #magenta
c4='#fe6100' #orange
c5='#ffb000' #gold
c6='#000000' #black
c7='#808080' #grey
colors=[c1,c2,c4,c3,c6]
##% EVALUATION
era_ref=xr.open_dataset(datapath+references[0]).sortby('latitude').fillna(0)
cape=era_ref.cape.squeeze().values
wms=(cape * 2)**0.5 * era_ref.bs_06
era_ref=era_ref.assign(wms=lambda era_ref: wms)
vars=['cape','cin','bs_06','bs_03','bs_01','wms']

#for mm in range(len(vars))[:]:
var=vars[mm]
tit=['CAPE [J kg$^{-1}$]','CIN [J kg$^{-1}$]','0-6km shear [m s$^{-1}$]','0-3km shear [m s$^{-1}$]',
     '0-1km shear [m s$^{-1}$]','wmax-shear [m$^2$ s$^{-2}$]'][mm]
l1=[300,-100,10,5,5,300][mm]
l2=[1000,-300,20,10,10,500][mm]
f1=[1,-1,1,1,1,1][mm]
fig,axes = plt.subplots(1,4,figsize=(16, 5))
fig2,axes2 = plt.subplots(1,3,figsize=(12, 5))
for nn in range(len(models))[:]:
    kw='latitude'
    kw2='prediction_timedelta'
    model=models[nn]
    label=labels[nn]
    color=colors[nn]
    ii=[1,1,1,1,0][nn]
    print(model)
    files=sorted(glob(datapath+'*USA_conv_'+model+'*.nc'))
    sal_s=np.zeros([41,len(files)]); sal_s[:]=np.nan
    sal_a=np.zeros([41,len(files)]); sal_a[:]=np.nan
    sal_l=np.zeros([41,len(files)]); sal_l[:]=np.nan

    fss_eval_300=np.zeros([41,len(files)]); fss_eval_300[:]=np.nan
    fss_eval_1000=np.zeros([41,len(files)]); fss_eval_1000[:]=np.nan
    rmse=np.zeros([41,len(files)]); rmse[:]=np.nan
    bias=np.zeros([41,len(files)]); bias[:]=np.nan

    for file in range(len(files)-22):
        print(file)
        model_set=xr.open_dataset(files[file]).sortby(kw).fillna(0).squeeze()
        if var=='wms':
            capem=model_set.cape.squeeze().values
            wmsm=(capem * 2)**0.5 * model_set.bs_06
            model_set=model_set.assign(wms=lambda model_set: wmsm )
        for tstep in range(len(model_set[kw2])):
            ref=era_ref.sel(time=(model_set.time+model_set.prediction_timedelta))[var].values[tstep,:,:]*f1
            mod=model_set[var].values[tstep,:,:]*f1

            fss_eval_300[tstep+ii,file] = fss(mod, ref, l1, scale=4)
            fss_eval_1000[tstep+ii,file] = fss(mod, ref, l2, scale=4)
            rmse[tstep+ii,file] = np.nanmean((mod - ref)**2)**0.5
            bias[tstep+ii,file] = np.nanmean(mod - ref)
            
            ref[f1*ref<l1]=0
            mod[f1*mod<l1]=0
            
            (sal_s[tstep+ii,file],sal_a[tstep+ii,file],sal_l[tstep+ii,file]) = sal(mod,ref)

            
            
    ds = xr.Dataset(
        data_vars=dict(
            structure=(["leadtime","date"], sal_s[ii:,:-1]),
            amplitude=(["lead time","date"], sal_a[ii:,:-1]),
            location=(["lead time","date"], sal_l[ii:,:-1]),
            rmse=(["lead time","date"], rmse[ii:,:-1]),
            bias=(["lead time","date"], bias[ii:,:-1]),
            fss_low=(["lead time","date"], fss_eval_300[ii:,:-1]),
            fss_high=(["lead time","date"], fss_eval_1000[ii:,:-1]),
        ),
        coords=dict(
            date=(["date"], np.arange(sal_a[ii:,:-1].shape[1])),
            leadtime=(["leadtime"], np.arange(sal_a[ii:,:-1].shape[0])*6),
        ),
        attrs=dict(description="Evaluation scores"),
    )
    ds.to_netcdf(datapath+model+var+'_eval_scores.nc')
    ds.close()
    
    axes2[0].plot(np.arange(41)*6,np.nanmean(sal_s,axis=1),c=color,label=label)
    axes2[0].plot([0,240],np.zeros(2),c='grey')
    #axes2[0].fill_between(np.arange(i+2*len(files))*6,np.nanmax(sal_s,axis=1),np.nanmin(sal_s,axis=1),facecolor=color,alpha=0.3)
    axes2[0].set_title('structure')
    axes2[0].set_xlabel('leadtime [h]')
    axes2[0].set_xticks(np.arange(24,258,24))
    axes2[0].set_ylabel('score [-2,2]')
    axes2[0].legend()
    axes2[1].plot(np.arange(41)*6,np.nanmean(sal_a,axis=1),c=color,label=label)
    axes2[1].plot([0,240],np.zeros(2),c='grey')
    #axes2[1].fill_between(np.arange(i+2*len(files))*6,np.nanmax(sal_a,axis=1),np.nanmin(sal_a,axis=1),facecolor=color,alpha=0.3)
    axes2[1].set_title('amplitude')
    axes2[1].set_xlabel('leadtime [h]')
    axes2[1].set_xticks(np.arange(24,258,24))
    axes2[1].set_ylabel('score [-2,2]')
    #axes2[1].legend()
    axes2[2].plot(np.arange(41)*6,np.nanmean(sal_l,axis=1),c=color)
    axes2[2].plot([0,240],np.zeros(2),c='grey')
    #axes2[2].fill_between(np.arange(i+2*len(files))*6,np.nanmax(sal_l,axis=1),np.nanmin(sal_l,axis=1),facecolor=color,alpha=0.3)
    axes2[2].set_title('location')
    axes2[2].set_xlabel('leadtime [h]')
    axes2[2].set_xticks(np.arange(24,258,24))
    axes2[2].set_ylabel('score [0,2]')
    fig2.suptitle('SAL score '+str(l1)+' '+tit)


    axes[0].plot(np.arange(41)*6,np.nanmean(fss_eval_300,axis=1),c=color,label=model)
    #axes[0].fill_between(np.arange(i+2*len(files))*6,np.nanmax(fss_eval_300,axis=1),np.nanmin(fss_eval_300,axis=1),facecolor=color,alpha=0.3)
    axes[0].set_title('FSS '+str(l1))
    axes[0].set_xlabel('leadtime [h]')
    axes[0].set_xticks(np.arange(24,258,24))
    axes[0].set_ylabel('fraction []')
    axes[0].legend()
    axes[1].plot(np.arange(41)*6,np.nanmean(fss_eval_1000,axis=1),c=color,label=model)
    #axes[1].fill_between(np.arange(i+2*len(files))*6,np.nanmax(fss_eval_1000,axis=1),np.nanmin(fss_eval_1000,axis=1),facecolor=color,alpha=0.3)
    axes[1].set_title('FSS '+str(l2))
    axes[1].set_xlabel('leadtime [h]')
    axes[1].set_xticks(np.arange(24,258,24))
    axes[2].set_ylabel('fraction []')
    axes[2].plot(np.arange(41)*6,np.nanmean(rmse,axis=1),c=color)
    #axes[2].fill_between(np.arange(i+2*len(files))*6,np.nanmax(rmse,axis=1),np.nanmin(rmse,axis=1),facecolor=color,alpha=0.3)
    axes[2].set_title('RMSE')
    axes[2].set_xlabel('leadtime [h]')
    axes[2].set_xticks(np.arange(24,258,24))
    axes[2].set_ylabel(tit)
    axes[3].plot(np.arange(41)*6,np.nanmean(bias,axis=1),c=color)
    #axes[3].fill_between(np.arange(i+2*len(files))*6,np.nanmax(bias,axis=1),np.nanmin(bias,axis=1),facecolor=color,alpha=0.3)
    axes[3].set_title('BIAS')
    axes[3].set_xlabel('leadtime [h]')
    axes[3].set_xticks(np.arange(24,258,24))
    axes[3].set_ylabel(tit)
    fig.suptitle(tit)


fig2.tight_layout()
#fig2.show()
fig2.savefig(figpath+var+'_sal_scores.png')

fig.tight_layout()
fig.show()
fig.savefig(figpath+var+'_err_scores.png')

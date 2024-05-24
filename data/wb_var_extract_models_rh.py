import sys, os
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import numpy.matlib
import copy
import wrf
import time
import cartopy.crs as ccrs
import datetime
import glob
import datetime as dt
import seaborn as sns
import metpy
from metpy.units import units
import argparse as ap
print('modules loaded')
##%
parser = ap.ArgumentParser()
#parser.add_argument('--model', type=str, required=True)
parser.add_argument('--region', type=str, required=True)
args = parser.parse_args()
nn=1#int(args.model)
event_ID=int(args.region)
print('settings loaded')
##%
figpath='/users/mfeldman/figs/'
era5='gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr'
ifs_init='gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr'
mlpath='/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/MF_ML_PREDICT/season_2020/'

#EVENT SELECTION ERA5
if event_ID==0:
    ###USA
    year=2020; month=np.arange(3,10); 
    latslice=slice(20,50); lonslice=slice(250,300)
    lon_conv=False
    flag='USA'
elif event_ID==1:
    ###ARGENTINA
    year=2020; month=(9,10,11,12,1,2)
    latslice=slice(-40,-25); lonslice=slice(290,310)
    lon_conv=False
    flag='ARG'
elif event_ID==2:
    ###AUSTRALIA
    year=2020; month=(9,10,11,12,1,2)
    latslice=slice(-35,-20); lonslice=slice(140,155)
    lon_conv=False
    flag='AUS'
elif event_ID==3:
    ###CHINA
    year=2020; month=np.arange(4,10)
    latslice=slice(20,35); lonslice=slice(100,120)
    lon_conv=False
    flag='CHN'
elif event_ID==4:
    ###EUROPE
    year=2020; month=np.arange(4,10)
    latslice=slice(35,55); lonslice=slice(-10,30)
    lon_conv=True
    flag='EUR'

savepath='/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/MF_ML_PREDICT/season_2020_'+flag+'/'

#%%
#FOURCASTNET

era5='gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr'
xr_era5=xr.open_zarr(era5).sortby('latitude')
if lon_conv:
    xr_era5.coords['longitude'] = (xr_era5.coords['longitude'] + 180) % 360 - 180
    xr_era5 = xr_era5.sortby(xr_era5.longitude)
xr_era5=xr_era5.sel(latitude=latslice,longitude=lonslice)
xr_era5=xr_era5.sel(time=xr_era5.time.dt.year.isin([year]))
xr_era5=xr_era5.sel(time=xr_era5.time.dt.month.isin([month]))
zsurf_c=xr_era5['geopotential_at_surface']
#LOADING FORECAST DATASET
models=['sfno','sfno-oper']
model=models[nn]
path='/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/AI-milton/Monika_4castnet_2020/IFS_init/'
files=[]

for mon in month:
    mon=str(mon).zfill(2) 
    files+=sorted(glob.glob(path+'*2020.'+mon+'*00h00m_*.nc')+glob.glob(path+'*2020.'+mon+'*12h00m_*.nc'))

for file in files[:]:
    xr_model=xr.open_dataset(file)
    strfind1=file.find('fcnet_')+len('fcnet_')
    strfind2=file.find('_max')
    init=file[strfind1:strfind2].replace('-','').replace('T','').replace('h','').replace('m','').replace('.','')
    init2=xr_model.time.values
    # xr_model=xr_model.rename({'lat':'latitude'})
    # xr_model=xr_model.rename({'lon':'longitude'})
    xr_model=xr_model.rename({'leadtime_hours':'prediction_timedelta'})
    if lon_conv:
        xr_model.coords['longitude'] = (xr_model.coords['longitude'] + 180) % 360 - 180
        xr_model = xr_model.sortby(xr_model.longitude)
    #xr_model=xr_model.squeeze(dim='history')
    xr_model=xr_model.sortby('latitude').sel(latitude=latslice,longitude=lonslice)
    plevel=[50,100,150,200,250,300,400,500,600,700,850,925,1000]
    ulevel = xr_model.u #xr_model['__xarray_dataarray_variable__'][:,8:21,:]
    vlevel = xr_model.v #['__xarray_dataarray_variable__'][:,21:34,:]
    zlevel = xr_model.z #['__xarray_dataarray_variable__'][:,34:47,:]
    tlevel = xr_model.t #['__xarray_dataarray_variable__'][:,47:60,:]
    rlevel = xr_model.r #['__xarray_dataarray_variable__'][:,60:73,:]
    psurf = xr_model.sp #['__xarray_dataarray_variable__'][:,5,:]
    tsurf = xr_model.t2m #['__xarray_dataarray_variable__'][:,4,:]
    usurf = xr_model.u10 #['__xarray_dataarray_variable__'][:,0,:]
    vsurf = xr_model.v10 #['__xarray_dataarray_variable__'][:,1,:]

    plevel_dim=np.ones([1,len(plevel),1,1])
    for n1 in range(len(plevel)):
        plevel_dim[0,n1,0,0]=plevel[n1]
    plevel_exp=(tlevel/tlevel)*plevel_dim
    
    dplevel=metpy.calc.dewpoint_from_relative_humidity(tlevel.values* units('K'), rlevel.values* units.percent)
    qlevel=metpy.calc.specific_humidity_from_dewpoint(plevel_exp.values* units('hPa'), dplevel).magnitude
    #.to(units('kg/kg')#.magnitude

    # zsurf = zsurf_c.expand_dims(dim={"prediction_timedelta": psurf.prediction_timedelta}, axis=0)
    #zsurf=zsurf.sel(latitude=latslice,longitude=slice(250,300))
    
    zsurf = zsurf_c.expand_dims(dim={"time": psurf.time}, axis=0)
    zsurf = zsurf.expand_dims(dim={"prediction_timedelta": psurf.prediction_timedelta}, axis=1)
     
    zs=zsurf/9.81
    zl=zlevel/9.81

    #psurf=pmsl * (tsurf/ (tsurf - (zs) * 0.0065))**(-1)
    ps=psurf/100

    

    du_01=ulevel.sel(level=850)-usurf; dv_01=vlevel.sel(level=850)-vsurf
    du_03=ulevel.sel(level=700)-usurf; dv_03=vlevel.sel(level=700)-vsurf
    du_06=ulevel.sel(level=500)-usurf; dv_06=vlevel.sel(level=500)-vsurf
    
    bs_01=( du_01**2 + dv_01**2 )**0.5
    bs_03=( du_03**2 + dv_03**2 )**0.5
    bs_06=( du_06**2 + dv_06**2 )**0.5
    
    du_01=du_01.to_dataset(name='du_01')
    dv_01=dv_01.to_dataset(name='dv_01')
    bs_01=bs_01.to_dataset(name='bs_01')
    du_03=du_03.to_dataset(name='du_03')
    dv_03=dv_03.to_dataset(name='dv_03')
    bs_03=bs_03.to_dataset(name='bs_03')
    du_06=du_06.to_dataset(name='du_06')
    dv_06=dv_06.to_dataset(name='dv_06')
    bs_06=bs_06.to_dataset(name='bs_06')
    
    mod_params=xr.merge([du_01,dv_01,bs_01,du_03,dv_03,bs_03,du_06,dv_06,bs_06],compat='override')
    q1level=copy.deepcopy(tlevel)
    q1level.data=qlevel
    mod_params['q500']=q1level.sel(level=500)*1000
    mod_params['r500']=rlevel.sel(level=500)
    mod_params['t500']=tlevel.sel(level=500)
    mod_params['q925']=q1level.sel(level=925)*1000
    mod_params['r925']=rlevel.sel(level=925)
    mod_params['t925']=tlevel.sel(level=925)
    mod_inst=wrf.cape_2d(pres_hpa=plevel_exp, tkel=tlevel, qv=qlevel, height=zl, terrain=zs, psfc_hpa=ps, ter_follow=False)
    mod_inst=mod_inst.assign_coords(longitude=psurf.longitude.values);
    mod_inst=mod_inst.assign_coords(latitude=psurf.latitude.values);
    mod_inst=mod_inst.assign_coords(prediction_timedelta=psurf.prediction_timedelta.values)
    #mod_params=mod_params.squeeze(dim='level')
    mod_params['cape']=mod_inst.sel(mcape_mcin_lcl_lfc='mcape')
    mod_params['cin']=mod_inst.sel(mcape_mcin_lcl_lfc='mcin')
    mod_params['lcl']=mod_inst.sel(mcape_mcin_lcl_lfc='lcl')
    mod_params['lfc']=mod_inst.sel(mcape_mcin_lcl_lfc='lfc')#.squeeze(dim='mcape_mcin_lcl_lfc')
    mod_params = mod_params.squeeze()
    mod_params = mod_params.drop(['mcape_mcin_lcl_lfc','level'])
    
    
    mod_params.to_netcdf(savepath+flag+'_conv_'+model+'_'+init+'a.nc')
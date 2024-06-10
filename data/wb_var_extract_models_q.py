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
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--region', type=str, required=True)
args = parser.parse_args()
nn=int(args.model)
event_ID=int(args.region)
print('settings loaded')
##%
figpath='/users/mfeldman/figs/'
era5='gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr'
ifs_init='gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr'
dataset='gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr'
#dataset='gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr'
# dataset='gs://weatherbench2/datasets/fuxi/2020-1440x721.zarr'
# dataset='gs://weatherbench2/datasets/keisler/2020-360x181.zarr'
# dataset='gs://weatherbench2/datasets/sphericalcnn/2020-240x121_equiangular_with_poles.zarr'
mlpath='/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/MF_ML_PREDICT/season_2020/'
#mlpath='/work/FAC/FGSE/IDYST/tbeucler/default/raw_data/ML_PREDICT/panguweather/Data_Monika/'
model='graphcast_dawn'
#%%

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
models=['pangu','pangu-oper','ifs','graphcast','graphcast-oper','graphcast-red']
d1='gs://weatherbench2/datasets/pangu/2018-2022_0012_0p25.zarr'
d2='gs://weatherbench2/datasets/pangu_hres_init/2020_0012_0p25.zarr'
d3='gs://weatherbench2/datasets/graphcast/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr'
d4='gs://weatherbench2/datasets/graphcast_hres_init/2020/date_range_2019-11-16_2021-02-01_12_hours_derived.zarr'
d5='gs://weatherbench2/datasets/hres/2016-2022-0012-1440x721.zarr'
paths=[d1,d2,d5,d3,d4,d3]
dataset=paths[nn]
model=models[nn]
if nn>2:
    xr_model=xr.open_zarr(dataset)
    xr_model=xr_model.rename({'lat': 'latitude','lon': 'longitude'})
    if lon_conv:
        xr_model.coords['longitude'] = (xr_model.coords['longitude'] + 180) % 360 - 180
        xr_model = xr_model.sortby(xr_model.longitude)
    xr_model=xr_model.sortby('latitude').sel(latitude=latslice,longitude=lonslice)
else:    
    xr_model=xr.open_zarr(dataset).sortby('latitude')
    if lon_conv:
        xr_model.coords['longitude'] = (xr_model.coords['longitude'] + 180) % 360 - 180
        xr_model = xr_model.sortby(xr_model.longitude)
    xr_model = xr_model.sel(latitude=latslice,longitude=lonslice)
xr_model=xr_model.sel(time=xr_model.time.dt.year.isin([year]))
xr_model=xr_model.sel(time=xr_model.time.dt.month.isin([month]))
if nn==5: xr_model.sel(level=[  50,  100,  150,  200,  250,  300,  400,  500,  600,  700,  850,  925,
       1000]) #test to reduce graphcast model levels, disable to keep all model levels
for t in xr_model.time[:]:
    xr_dataset=xr_model.sel(time=t)
    fcst_init=xr_dataset.time
    init=fcst_init.dt.strftime('%Y%m%d%H').values; print(init)
    #xr_dataset=xr_dataset.sel(prediction_timedelta=slice(start-fcst_init,end-fcst_init))
    xr_dataset = xr_dataset.sortby('level', ascending=False)
    
    plevel=copy.deepcopy(xr_dataset.level.values)
    
    xr_dataset['pressure'] = (('level'), plevel)
    plevel=xr_dataset.pressure
    #xr_dataset = xr_dataset.assign_coords(level=np.arange(len(plevel)))
    
    tlevel=xr_dataset.temperature
    qlevel=xr_dataset.specific_humidity
    plevel_dim=np.ones([1,len(plevel),1,1])
    zlevel=xr_dataset.geopotential
    ulevel=xr_dataset.u_component_of_wind
    vlevel=xr_dataset.v_component_of_wind
    for n1 in range(len(plevel.data)):
        plevel_dim[0,n1,0,0]=plevel[n1]
    plevel_exp=(tlevel/tlevel)*plevel_dim
    
    pmsl=xr_dataset.mean_sea_level_pressure
    tsurf=xr_dataset['2m_temperature']
    #zsurf = zsurf_c.expand_dims(dim={"time": psurf.time}, axis=0)
    #zsurf = zsurf_c.expand_dims(dim={"time": pmsl.time}, axis=0)
    zsurf = zsurf_c.expand_dims(dim={"prediction_timedelta": pmsl.prediction_timedelta}, axis=0)
    
    zs=zsurf/9.81
    zl=zlevel/9.81

    psurf=pmsl * (tsurf/ (tsurf - (zs) * 0.0065))**(-1)
    ps=psurf/100
    print('with surface pressure in hPa')
    mod_inst=wrf.cape_2d(pres_hpa=plevel_exp, tkel=tlevel, qv=qlevel, height=zl, terrain=zs, psfc_hpa=ps, ter_follow=False)
    mod_inst=mod_inst.assign_coords(longitude=psurf.longitude.values);
    mod_inst=mod_inst.assign_coords(latitude=psurf.latitude.values);
    mod_inst=mod_inst.assign_coords(prediction_timedelta=psurf.prediction_timedelta.values)

    du_01=ulevel.sel(level=850)-xr_dataset['10m_u_component_of_wind']; dv_01=vlevel.sel(level=850)-xr_dataset['10m_v_component_of_wind']
    du_03=ulevel.sel(level=700)-xr_dataset['10m_u_component_of_wind']; dv_03=vlevel.sel(level=700)-xr_dataset['10m_v_component_of_wind']
    du_06=ulevel.sel(level=500)-xr_dataset['10m_u_component_of_wind']; dv_06=vlevel.sel(level=500)-xr_dataset['10m_v_component_of_wind']
    
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
    #mod_params=mod_params.squeeze(dim='level')
    mod_params['cape']=mod_inst.sel(mcape_mcin_lcl_lfc='mcape')
    mod_params['cin']=mod_inst.sel(mcape_mcin_lcl_lfc='mcin')
    mod_params['lcl']=mod_inst.sel(mcape_mcin_lcl_lfc='lcl')
    mod_params['lfc']=mod_inst.sel(mcape_mcin_lcl_lfc='lfc')#.squeeze(dim='mcape_mcin_lcl_lfc')
    rlevel=metpy.calc.relative_humidity_from_specific_humidity(plevel_exp.values* units('hPa'), tlevel.values* units('K'), qlevel.values* units('kg/kg')).to('%').magnitude
    r1level=copy.deepcopy(tlevel)
    r1level.data=rlevel
    mod_params['q500']=qlevel.sel(level=500)
    mod_params['r500']=r1level.sel(level=500)
    mod_params['t500']=tlevel.sel(level=500)
    mod_params['q925']=qlevel.sel(level=925)
    mod_params['r925']=r1level.sel(level=925)
    mod_params['t925']=tlevel.sel(level=925)
    mod_params = mod_params.squeeze()
    mod_params = mod_params.drop(['level','mcape_mcin_lcl_lfc'])
    
    
    mod_params.to_netcdf(savepath+flag+'_conv_'+model+'_'+init+'a.nc')
   

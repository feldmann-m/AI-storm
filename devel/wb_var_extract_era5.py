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
#nn=int(args.model)
event_ID=int(args.region)
print('settings loaded')
##%
figpath='/users/mfeldman/figs/'
era5='gs://weatherbench2/datasets/era5/1959-2022-6h-1440x721.zarr'
ifs_init='gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr'

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

plevel=copy.deepcopy(xr_era5.level.values)
        
xr_era5['pressure'] = (('level'), plevel)
plevel=xr_era5.pressure
#xr_dataset = xr_era5.assign_coords(level=np.arange(len(plevel)))
tlevel=xr_era5.temperature
qlevel=xr_era5.specific_humidity
plevel_dim=np.ones([1,len(plevel),1,1])
zlevel=xr_era5.geopotential
ulevel=xr_era5.u_component_of_wind
vlevel=xr_era5.v_component_of_wind
for n1 in range(len(plevel.data)):
    plevel_dim[0,n1,0,0]=plevel[n1]
plevel_exp=(tlevel/tlevel)*plevel_dim
pmsl=xr_era5.mean_sea_level_pressure
tsurf=xr_era5['2m_temperature']
#zsurf = zsurf_c.expand_dims(dim={"time": psurf.time}, axis=0)
zsurf = zsurf_c.expand_dims(dim={"time": pmsl.time}, axis=0)
#zsurf = zsurf.expand_dims(dim={"prediction_timedelta": pmsl.prediction_timedelta}, axis=1)

zs=zsurf/9.81
zl=zlevel/9.81

psurf=pmsl * (tsurf/ (tsurf - (zs) * 0.0065))**(-1)
ps=psurf/100

era_inst=wrf.cape_2d(pres_hpa=plevel_exp, tkel=tlevel, qv=qlevel, height=zl, terrain=zs, psfc_hpa=ps, ter_follow=False)
era_inst=era_inst.assign_coords(longitude=psurf.longitude.values);
era_inst=era_inst.assign_coords(latitude=psurf.latitude.values);
era_inst=era_inst.assign_coords(time=psurf.time.values)

du_01=ulevel.sel(level=850)-xr_era5['10m_u_component_of_wind']; dv_01=vlevel.sel(level=850)-xr_era5['10m_v_component_of_wind']
du_03=ulevel.sel(level=700)-xr_era5['10m_u_component_of_wind']; dv_03=vlevel.sel(level=700)-xr_era5['10m_v_component_of_wind']
du_06=ulevel.sel(level=500)-xr_era5['10m_u_component_of_wind']; dv_06=vlevel.sel(level=500)-xr_era5['10m_v_component_of_wind']

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

era_params=xr.merge([du_01,dv_01,bs_01,du_03,dv_03,bs_03,du_06,dv_06,bs_06],compat='override')

xr_era5_rh = xr.open_zarr('gs://weatherbench2/datasets/era5/1959-2023_01_10-wb13-6h-1440x721_with_derived_variables.zarr').sortby('latitude')
if lon_conv:
    xr_era5_rh.coords['longitude'] = (xr_era5_rh.coords['longitude'] + 180) % 360 - 180
    xr_era5_rh = xr_era5_rh.sortby(xr_era5_rh.longitude)
xr_era5_rh=xr_era5_rh.sel(latitude=latslice,longitude=lonslice)
xr_era5_rh=xr_era5_rh.sel(time=xr_era5_rh.time.dt.year.isin([year]))
xr_era5_rh=xr_era5_rh.sel(time=xr_era5_rh.time.dt.month.isin([month]))
xr_era5_rh.relative_humidity
rlevel=xr_era5_rh.relative_humidity
era_params['cape']=era_inst.sel(mcape_mcin_lcl_lfc='mcape')
era_params['cin']=era_inst.sel(mcape_mcin_lcl_lfc='mcin')
era_params['lcl']=era_inst.sel(mcape_mcin_lcl_lfc='lcl')
era_params['lfc']=era_inst.sel(mcape_mcin_lcl_lfc='lfc')
era_params['q500']=qlevel.sel(level=500)
era_params['r500']=rlevel.sel(level=500)
era_params['t500']=tlevel.sel(level=500)
era_params['q925']=qlevel.sel(level=925)
era_params['r925']=rlevel.sel(level=925)
era_params['t925']=tlevel.sel(level=925)
era_params = era_params.squeeze()
era_params = era_params.drop(['level','mcape_mcin_lcl_lfc'])

era_params.to_netcdf(savepath+flag+'_era5_convseason_2020.nc')

ifs_init='gs://weatherbench2/datasets/hres_t0/2016-2022-6h-1440x721.zarr'
xr_model=xr.open_zarr(ifs_init).sortby('latitude')

xr_init=xr_model.sel(time=xr_model.time.dt.year.isin([year]))
xr_init=xr_init.sel(time=xr_init.time.dt.month.isin([month]))
if lon_conv:
    xr_init.coords['longitude'] = (xr_init.coords['longitude'] + 180) % 360 - 180
    xr_init = xr_init.sortby(xr_init.longitude)
xr_init = xr_init.sel(latitude=latslice,longitude=lonslice)

print(xr_init.dims)

xr_init = xr_init.sortby('level', ascending=False)
plevel=copy.deepcopy(xr_init.level.values)
xr_init['pressure'] = (('level'), plevel)
plevel=xr_init.pressure
#xr_init = xr_init.assign_coords(level=np.arange(len(plevel)))

tlevel=xr_init.temperature
qlevel=xr_init.specific_humidity
plevel_dim=np.ones([1,len(plevel),1,1])
zlevel=xr_init.geopotential
ulevel=xr_init.u_component_of_wind
vlevel=xr_init.v_component_of_wind
for n1 in range(len(plevel.data)):
    plevel_dim[0,n1,0,0]=plevel[n1]
plevel_exp=(tlevel/tlevel)*plevel_dim

pmsl=xr_init.mean_sea_level_pressure
tsurf=xr_init['2m_temperature']
#zsurf = zsurf_c.expand_dims(dim={"time": psurf.time}, axis=0)
zsurf = zsurf_c.expand_dims(dim={"time": pmsl.time}, axis=0)
#zsurf = zsurf.expand_dims(dim={"prediction_timedelta": pmsl.prediction_timedelta}, axis=1)

zs=zsurf/9.81
zl=zlevel/9.81

psurf=pmsl * (tsurf/ (tsurf - (zs) * 0.0065))**(-1)
ps=psurf/100

init_inst=wrf.cape_2d(pres_hpa=plevel_exp, tkel=tlevel, qv=qlevel, height=zl, terrain=zs, psfc_hpa=ps, ter_follow=False)
init_inst=init_inst.assign_coords(longitude=psurf.longitude.values);
init_inst=init_inst.assign_coords(latitude=psurf.latitude.values);
init_inst=init_inst.assign_coords(time=psurf.time.values)

du_01=ulevel.sel(level=850)-xr_init['10m_u_component_of_wind']; dv_01=vlevel.sel(level=850)-xr_init['10m_v_component_of_wind']
du_03=ulevel.sel(level=700)-xr_init['10m_u_component_of_wind']; dv_03=vlevel.sel(level=700)-xr_init['10m_v_component_of_wind']
du_06=ulevel.sel(level=500)-xr_init['10m_u_component_of_wind']; dv_06=vlevel.sel(level=500)-xr_init['10m_v_component_of_wind']

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

init_params=xr.merge([du_01,dv_01,bs_01,du_03,dv_03,bs_03,du_06,dv_06,bs_06],compat='override')

init_params['cape']=init_inst.sel(mcape_mcin_lcl_lfc='mcape')
init_params['cin']=init_inst.sel(mcape_mcin_lcl_lfc='mcin')
init_params['lcl']=init_inst.sel(mcape_mcin_lcl_lfc='lcl')
init_params['lfc']=init_inst.sel(mcape_mcin_lcl_lfc='lfc')

rlevel=metpy.calc.relative_humidity_from_specific_humidity(plevel_exp.values* units('hPa'), tlevel.values* units('K'), qlevel.values* units('kg/kg')).to('%').magnitude

r1level=copy.deepcopy(tlevel)
r1level.data=rlevel
init_params['q500']=qlevel.sel(level=500)
init_params['r500']=r1level.sel(level=500)
init_params['t500']=tlevel.sel(level=500)
init_params['q925']=qlevel.sel(level=925)
init_params['r925']=r1level.sel(level=925)
init_params['t925']=tlevel.sel(level=925)
init_params = init_params.squeeze()
init_params = init_params.drop(['level','mcape_mcin_lcl_lfc'])

init_params.to_netcdf(savepath+flag+'_init_convseason_2020.nc')
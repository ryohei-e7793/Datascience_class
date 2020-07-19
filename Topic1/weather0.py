#!/usr/bin/env pythonA
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import datetime as dt
from datetime import timedelta  
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.gridspec as gridspec
from scipy.interpolate import griddata


year = '2016'
fname = './frenchWeather/NW_Ground_Stations/NW_Ground_Stations/NW_Ground_Stations_'+year+".csv"
df = pd.read_csv(fname,parse_dates=[4],infer_datetime_format=True)
date = '2016-01-01T06:00:00'
d_sub = df[df['date'] == date]

param = 't'
plt.scatter(d_sub['lon'], d_sub['lat'], c=d_sub[param], cmap='jet')
plt.colorbar()
plt.title(date+' - param '+param)
plt.show()


lllat = 46.25 
urlat = 51.896
lllon = -5.842
urlon = 2  
extent = [lllon, urlon, lllat, urlat]
fig = plt.figure(figsize=(9,5))
ax = plt.axes(projection=ccrs.PlateCarree())
plt.scatter(d_sub['lon'], d_sub['lat'], c=d_sub[param], cmap='jet')  # Plot
plt.colorbar()
plt.title(date+' - param '+param)

ax.coastlines(resolution='50m', linewidth=1)
ax.add_feature(cfeature.BORDERS.with_scale('50m'))
plt.show()


model = 'arome' 
level = '2m'    
date = dt.datetime(2016, 2, 14,0,0) 
directory = 'frenchWeather/NW_weather_models_2D_parameters_' + str(date.year) + str(date.month).zfill(2) + '/' + str(date.year) + str(date.month).zfill(2) + '/'
fname = directory + f'{model.upper()}/{level}/{model}_{level}_NW_{date.year}{str(date.month).zfill(2)}{str(date.day).zfill(2)}000000.nc'

print ('fname=',fname)
data = xr.open_dataset(fname)  
print(data)

param = 't2m'
data.isel(step=[0, 6, 12, 23])[param].plot(x='longitude',
                                           y='latitude',
                                           col='step',
                                           col_wrap=2)

coord = 'longitude'
print(data[coord])
data[coord].units
data[coord].values[0:10]
run_date = data['time']
print(run_date)
range_forecasts_dates = data['valid_time']
print(range_forecasts_dates)

d = data[param]     #param : parameter name defined at the beginning of the Notebook 
d_vals=d.values     #get the values
print(d)
print(d.dims)
print(d_vals.shape)


step = 0  
lllat = 46.25 
urlat = 51.896
lllon = -5.842
urlon = 2  
extent = [lllon, urlon, lllat, urlat]
fig=plt.figure(figsize=(9,10))
ax = plt.axes(projection=ccrs.PlateCarree())
img = ax.imshow(d_vals[step,:,:], interpolation='none', origin='upper', extent=extent)
ax.coastlines(resolution='50m', linewidth=1)
ax.add_feature(cfeature.BORDERS.with_scale('50m'))

plt.colorbar(img, orientation= 'horizontal').set_label(d.long_name+ ' (in '+d.units+ ')')
plt.title(model +" model - "+str(d['valid_time'].values[step])+" - " +"NW zone")
plt.show()

def choose_parameters_and_display(model,run_date,step,param):
    
    #open the corresponding file according to the chosen parameter    
    if param == 't2m' or param == 'd2m' or param == 'r':
        level = '2m'
    elif param == 'ws' or param =='p3031' or param == 'u10' or param == 'v10':
        level = '10m'
    elif param == 'msl':
        level = 'P_sea_level'
    else:
        level = 'PRECIP'
    directory = 'frenchWeather/NW_weather_models_2D_parameters_' + str(run_date.year) + str(run_date.month).zfill(2) + '/' + str(run_date.year) + str(run_date.month).zfill(2) + '/'    
    fname = directory + f'{model.upper()}/{level}/{model}_{level}_NW_{run_date.year}{str(run_date.month).zfill(2)}{str(run_date.day).zfill(2)}000000.nc'
    print ('fname=',fname)    
    sub_data = xr.open_dataset(fname)      
    
    lllat = 46.25  
    urlat = 51.896 
    lllon = -5.842 
    urlon = 2  
    extent = [lllon, urlon, lllat, urlat]
    fig=plt.figure(figsize=(9,10))

    ax = plt.axes(projection=ccrs.PlateCarree())
    img = ax.imshow(sub_data[param].values[step,:,:], interpolation='none', origin='upper', extent=extent)
    ax.coastlines(resolution='50m', linewidth=1)
    ax.add_feature(cfeature.BORDERS.with_scale('50m'))

    plt.colorbar(img, orientation= 'horizontal').set_label(sub_data[param].long_name+ ' (in '+sub_data[param].units+ ')')
    plt.title(model +" model - "+str(sub_data['valid_time'].values[step])+" - " +"NW zone")
    plt.show()
    
    return sub_data


model = 'arpege' 
run_date = dt.datetime(2016, 2,10,0,0) 
param = 'ws'    
step = 3   
sub_data = choose_parameters_and_display(model,run_date,step,param)


date = '2016-02-13' 
param_obs = 'dd'  
model = 'arpege' 
MODEL = 'ARPEGE' 
param_mod = 'p3031' 
algo = 'linear' #or 'nearest' 

year = date[0:4]
fname = 'frenchWeather/NW_Ground_Stations/NW_Ground_Stations/NW_Ground_Stations_'+year+".csv"
print ('fname=',fname)

def open_and_date_filtering(year,fname,date):    
    study_date = pd.Timestamp(date) 
    d_sub = df[(df['date'] >= study_date) & (df['date'] <= study_date + timedelta(days=1))]
    d_sub = d_sub.set_index('date')
    return(d_sub)

d_sub = open_and_date_filtering(year,fname,date)
station_id = 86137003

def station_lat_lon_resample(station_id,d_sub):
    d_sub = d_sub[d_sub['number_sta'] == station_id]
    lat_sta = d_sub['lat'][0]
    lon_sta = d_sub['lon'][0]
    d_sub = d_sub[param_obs].resample('H').mean()
    print('station_id:',station_id)
    print('lat/lon:',lat_sta,'/',lon_sta)
    print('weather parameter',param_obs)
    return(d_sub,station_id,lat_sta,lon_sta)

d_sub, station_id, lat_sta, lon_sta = station_lat_lon_resample(station_id,d_sub)

def open_get_values(param_mod,date):
    if param_mod == 't2m' or param_mod == 'd2m' or param_mod == 'r':
        level = '2m'
    elif param_mod == 'ws' or param_mod =='p3031' or param_mod == 'u10' or param_mod == 'v10':
        level = '10m'
    elif param_mod == 'msl':
        level = 'P_sea_level'
    else:
        level = 'PRECIP'

    year = date[0:4]
    month = date[5:7]
    day = date[8:10]

    directory = 'frenchWeather/NW_weather_models_2D_parameters_' + year + month + '/' + year + month + '/'    
    fname = directory + f'{MODEL}/{level}/{model}_{level}_NW_{year}{month}{day}000000.nc'
    print ('fname=',fname)    
    dm = xr.open_dataset(fname)
    print('dataset overview:',dm)

    grid_values = dm[param_mod].values
    grid_lat = dm['latitude'].values
    grid_lon = dm['longitude'].values
    print('shape of the forecast values array:',grid_values.shape)
    print('ten first latitudes:',grid_lat[0:10])
    print('ten first longitudes',grid_lon[0:10])
    return(grid_values,grid_lat,grid_lon)

grid_values,grid_lat,grid_lon = open_get_values(param_mod,date)

from scipy.interpolate import griddata

def interpolation(grid_values,grid_lat,grid_lon,lat_sta,lon_sta):
    model_values = []
    grid_on_points = np.empty(grid_values.shape[0], dtype = object) 
    for step in range(0,grid_values.shape[0]):
        latlon_grid = []
        val_grid = []
        latlon_obs = []

        for i in range(0,grid_lat.shape[0]):        
            for j in range(0,grid_lon.shape[0]):
                latlon_grid.append([grid_lat[i],grid_lon[j]])
                val_grid.append(grid_values[step,i,j])

        grid_latlon = np.array(latlon_grid)
        grid_val2 = np.array(val_grid)

        latlon_obs.append([lat_sta,lon_sta])
        latlon_obs = np.array(latlon_obs)

        grid_on_points[step] = griddata(grid_latlon ,grid_val2, latlon_obs,  method=algo)[0]
        print('step ',step, ' OK!')
    return(grid_on_points)

grid_on_points = interpolation(grid_values,grid_lat,grid_lon,lat_sta,lon_sta)
obs = d_sub

def preproc_output(obs,grid_on_points,param_mod):
    mod = pd.Series(grid_on_points,index=obs.index)
    print('interpolated forecasted data, param ',param_mod)
    return (mod)

mod = preproc_output(obs,grid_on_points,param_mod)

def plots(obs,mod,MODEL,param_obs,lat_sta,lon_sta):
    plt.plot(obs, label ='Observation')
    plt.plot(mod, label = MODEL +' forecast')
    plt.title('Parameter '+param_obs+' / lat='+str(lat_sta)+' and lon='+str(lon_sta))
    plt.xlabel('Time')
    plt.ylabel(param_obs)
    plt.legend()

plots(obs,mod,MODEL,param_obs,lat_sta,lon_sta)

date_obs = '2016-02-10T10:00:00' 
param_obs = 'ff'

model = 'arpege' 
MODEL = 'ARPEGE' 
date_mod = dt.datetime(2016, 2,10,10,0) 
param_mod = 'ws'

algo = 'linear' #or 'nearest' 


fname = 'frenchWeather/NW_Ground_Stations/NW_Ground_Stations/NW_Ground_Stations_'+date_obs[0:4]+".csv"
print ('fname=',fname)
study_date = pd.Timestamp(date_obs) 
d_sub = df[df['date'] == study_date]
print('observation data',d_sub)


directory = 'frenchWeather/NW_weather_models_2D_parameters_' + str(date_mod.year) + str(date_mod.month).zfill(2) + '/' + str(date_mod.year) + str(date_mod.month).zfill(2) + '/'

if param_mod == 't2m' or param_mod == 'd2m' or param_mod == 'r':
    level = '2m'
elif param_mod == 'ws' or param_mod =='p3031' or param_mod == 'u10' or param_mod == 'v10':
    level = '10m'
elif param_mod == 'msl':
    level = 'P_sea_level'
else:
    level = 'PRECIP'

fname = directory + f'{model.upper()}/{level}/{model}_{level}_NW_{date_mod.year}{str(date_mod.month).zfill(2)}{str(date_mod.day).zfill(2)}000000.nc'

print ('fname=',fname)
mod = xr.open_dataset(fname)
grid_lat = mod['latitude'].values
grid_lon = mod['longitude'].values
grid_val = mod[param_mod].values[date_mod.hour,:,:]
print('latitudes on the model grid:',grid_lat)
print('longitudes on the model grid:',grid_lon)
print('forecast values:',grid_val)

def interpolate_grid_on_points(grid_lat,grid_lon,grid_val,data_obs,algo):
    
    #initialisation
    latlon_grid = []
    latlon_obs = []
    val_grid = []
    
    #grid data preprocessing
    for i in range(0,grid_lat.shape[0]):        
        for j in range(0,grid_lon.shape[0]):
            #put coordinates (lat,lon) in list of tuples
            latlon_grid.append([round(grid_lat[i],3),round(grid_lon[j],3)])
            #put grid values into a list
            val_grid.append(grid_val[i,j])
    grid_latlon = np.array(latlon_grid)
    grid_val2 = np.array(val_grid)

    #obs data preprocessing : put coordinates (lat,lon) in list of tuples
    for i in range(0,data_obs.shape[0]):
        latlon_obs.append([data_obs['lat'].values[i],data_obs['lon'].values[i]])
    latlon_obs = np.array(latlon_obs)
    
    grid_val_on_points=griddata(grid_latlon ,grid_val2, latlon_obs,  method=algo)
    return latlon_obs,grid_val_on_points

latlon_obs,grid_val_on_points = interpolate_grid_on_points(grid_lat,grid_lon,grid_val,d_sub,algo)
print('10 first lat/lon couple per station:',latlon_obs[0:10,:])
print('associated forecast values interpolated on ground station points:',grid_val_on_points[0:10])

fig=plt.figure()
gs = gridspec.GridSpec(4, 4)

vmin_obs = d_sub[param_obs].min()
vmax_obs = d_sub[param_obs].max()
vmin_model_ori= grid_val.min()
vmax_model_ori= grid_val.max()
vmin_model_inter=grid_val_on_points.min()
vmax_model_inter=grid_val_on_points.max()
vmin=np.min([vmin_obs,vmin_model_ori,vmin_model_inter])
vmax=np.max([vmax_obs,vmax_model_ori,vmax_model_inter])

ax1 = plt.subplot(gs[:2, :2])
plt.tight_layout(pad=3.0)
im=ax1.scatter(d_sub['lon'], d_sub['lat'], c=d_sub[param_obs], cmap='jet',vmin=vmin,vmax=vmax)
ax1.set_title('Observation data')

ax2 = plt.subplot(gs[:2, 2:])
ax2.pcolor(grid_lon,grid_lat,grid_val,cmap="jet",vmin=vmin,vmax=vmax)
ax2.set_title('Weather model data (original grid)')

ax3 = plt.subplot(gs[2:4, 1:3])
im3=ax3.scatter(latlon_obs[:,1], latlon_obs[:,0], c=grid_val_on_points, cmap='jet',vmin=vmin,vmax=vmax)
ax3.set_title('Weather model data (interpolated on observation points)')

fig.colorbar(im,ax=[ax2,ax3]).set_label(mod[param_mod].long_name+ ' (in '+mod[param_mod].units+ ')')
plt.show()


def choose_and_display(model,date_mod,date_obs,param_obs,param_mod,algo,d_sub):
    study_date = pd.Timestamp(date_obs)  #study date
    d_sub = df[df['date'] == study_date]

    directory = 'frenchWeather/NW_weather_models_2D_parameters_' + str(date_mod.year) + str(date_mod.month).zfill(2) + '/' + str(date_mod.year) + str(date_mod.month).zfill(2) + '/'    

    if param_mod == 't2m' or param_mod == 'd2m' or param_mod == 'r':
        level = '2m'
    elif param_mod == 'ws' or param_mod =='p3031' or param_mod == 'u10' or param_mod == 'v10':
        level = '10m'
    elif param_mod == 'msl':
        level = 'P_sea_level'
    else:
        level = 'PRECIP'

    fname = directory + f'{model.upper()}/{level}/{model}_{level}_NW_{date_mod.year}{str(date_mod.month).zfill(2)}{str(date_mod.day).zfill(2)}000000.nc'
    print ('fname=',fname)

    mod = xr.open_dataset(fname)
    grid_lat = mod['latitude'].values
    grid_lon = mod['longitude'].values
    grid_val = mod[param_mod].values[date_mod.hour,:,:]
    
    #perform the interpolation
    latlon_obs,grid_val_on_points = interpolate_grid_on_points(grid_lat,grid_lon,grid_val,d_sub,algo)
    
    #Plot the different data
    fig=plt.figure()
    gs = gridspec.GridSpec(4, 4)

    #Min and max boundaries about colorbar
    vmin_obs = d_sub[param_obs].min()
    vmax_obs = d_sub[param_obs].max()
    vmin_model_ori= grid_val.min()
    vmax_model_ori= grid_val.max()
    vmin_model_inter=grid_val_on_points.min()
    vmax_model_inter=grid_val_on_points.max()
    vmin=np.min([vmin_obs,vmin_model_ori,vmin_model_inter])
    vmax=np.max([vmax_obs,vmax_model_ori,vmax_model_inter])

    #observation data
    ax1 = plt.subplot(gs[:2, :2])
    plt.tight_layout(pad=3.0)
    im=ax1.scatter(d_sub['lon'], d_sub['lat'], c=d_sub[param_obs], cmap='jet',vmin=vmin,vmax=vmax)
    ax1.set_title('Observation data')

    #weather model data (original grid)
    ax2 = plt.subplot(gs[:2, 2:])
    ax2.pcolor(grid_lon,grid_lat,grid_val,cmap="jet",vmin=vmin,vmax=vmax)
    ax2.set_title('Weather model data (original grid)')

    #weather model data (interpolated on observation points)
    ax3 = plt.subplot(gs[2:4, 1:3])
    im3=ax3.scatter(latlon_obs[:,1], latlon_obs[:,0], c=grid_val_on_points, cmap='jet',vmin=vmin,vmax=vmax)
    ax3.set_title('Weather model data (interpolated on observation points)')

    fig.colorbar(im,ax=[ax2,ax3]).set_label(mod[param_mod].long_name+ ' (in '+mod[param_mod].units+ ')')
    plt.show()
    
    return d_sub, mod[param_mod][date_mod.hour,:,:], latlon_obs,grid_val_on_points

date_obs = '2016-02-10T10:00:00' 
param_obs = 'hu'
model = 'arome' #weather model (arome or arpege)
date_mod = dt.datetime(2016, 2,10,10,0) # Day example 
param_mod = 'r'

algo = 'nearest' #'linear' or 'nearest' for nearest neighbors

obs_output, mod_output, latlon_obs,grid_val_on_points =  choose_and_display(model,date_mod,date_obs,param_obs,param_mod,algo,d_sub)

#!/usr/bin/env python
import os, sys
#sys.path.append(os.path.join(os.path.expanduser("~"),"CLEOPE/Trials/modules"))
#import qm
#import data_processing_S2_affine as dp
import rasterio
import rasterio.plot

import datetime
import zipfile
import xarray as xr
import pandas as pd
import rioxarray
import numpy as np
import geojson
import json
import pyproj
#import snappy_S2 as snp
import snappy_func as dp
#files = dp2.queryS2('product_list_2019.txt')

import datetime as dt
from datetime import date

import matplotlib.pyplot as plt
#%matplotlib inline

##
from pathlib import Path
from glob import glob

#from utils.cog import write_cog
#import urllib.request
# connect to the API
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt




download_unzipped_path = os.path.join(os.getcwd(), 'unzipped')  #unzipped
listfiles =[]
for item in os.listdir(download_unzipped_path):
    file_names = os.path.join(download_unzipped_path, item)
    listfiles.append(file_names)


files = os.listdir(download_unzipped_path)
print(files)


time_var = xr.Variable('time', dp.paths_to_datetimeindex(files))
print(time_var)

red =[dp.bands(file,res='10m')[3]  for file in listfiles]
#print(red)
nir = [dp.bands(file,res='10m')[4]  for file in listfiles]
#print(nir)

def paths_to_datetimeindex2(paths):
    string_slice=(7,-12) #string_slice=(45,60)
    date_strings = [os.path.basename(i)[slice(*string_slice)]
                    for i in paths]
    return pd.to_datetime(date_strings)

redlist = red
nirlist = nir


fileBbox2 = read_geojson('bboxgeo.json')
print(fileBbox2)

with open('bboxgeo.json') as f:  #Svartberget.geojson  #PA_test.geojson
    data = json.load(f)
    #print(data)
for feature in data['features']:
    print (feature['geometry']['type'])
    #print (feature['geometry']['coordinates'])
    geometry_coord =  feature['geometry']['coordinates']
    print(geometry_coord)

def get_bounding_box(geometry):
    coords = np.array(list(geojson.utils.coords(geometry)))
    return coords[:,0].min(), coords[:,0].max(),coords[:,1].min(), coords[:,1].max()
#print(get_bounding_box(geometry_coord))

#time_var = xr.Variable('time', dp.paths_to_datetimeindex(redlist))
#print(time_var)

if not os.path.exists('output'):
    os.makedir('output')


"""
nirlist= []
redlist= []

for file in files:
    #filelist.append(file)
    nir= dp.bands(file,res='10m')[4]#b8
    red= dp.bands(file,res='10m')[3]
    nirlist.append(nir)
    redlist.append(red)
    #files_b.append(dp.bands(file,res='10m')[4])
"""
time_var = xr.Variable('time', paths_to_datetimeindex2(redlist))


nir_da_gran = xr.concat([xr.open_rasterio(i,chunks={'x':512, 'y':512}) for i in nirlist],dim=time_var)

dx = 2
dy = 3

#xmin = snp.get_bounding_box(geometry_coord)[0]
lonmin = get_bounding_box(geometry_coord)[0]
print(lonmin)
lonmax = get_bounding_box(geometry_coord)[1]
latmin = get_bounding_box(geometry_coord)[2]
latmax = get_bounding_box(geometry_coord)[3]

proj_wgs84 = pyproj.Proj(init="epsg:4326")
proj_gk4 = pyproj.Proj(init="epsg:32631")
xmin, ymin = pyproj.transform(proj_wgs84, proj_gk4, lonmin, latmin)
xmax, ymax = pyproj.transform(proj_wgs84, proj_gk4, lonmax, latmax)

mask_lon_nir = (nir_da_gran.x >= (xmin-dx)) & (nir_da_gran.x <= (xmax+dx))
mask_lat_nir = (nir_da_gran.y >= (ymin-dy)) & (nir_da_gran.y <= (ymax+dy))

nir_da = nir_da_gran.where(mask_lon_nir & mask_lat_nir, drop=True)

new_nir_da = xr.DataArray(nir_da,dims=nir_da.dims,attrs= nir_da.attrs.copy())

nir_ds = new_nir_da.to_dataset('band')
nir_ds = nir_ds.rename({1: 'nir'})

print(nir_ds)
print(nir_ds['nir'].values[0,0,0])

red_da_gran = xr.concat([xr.open_rasterio(i,chunks={'x':512, 'y':512}) for i in redlist],    #
                        dim=time_var)
mask_lon_red = (red_da_gran.x >= (xmin-dx)) & (red_da_gran.x <= (xmax+dx))
mask_lat_red = (red_da_gran.y >= (ymin-dy)) & (red_da_gran.y <= (ymax+dy))

red_da = red_da_gran.where(mask_lon_red & mask_lat_red, drop=True)

new_red_ds = xr.DataArray(red_da,dims=red_da.dims,attrs= red_da.attrs.copy())

red_ds = new_red_ds.to_dataset('band')
red_ds = red_ds.rename({1: 'red'})
print(red_ds)

scl = []
for file in listfiles:
    scl_list = dp.sclbands(file)[0]     #dp2.sclbands(file)[0]  dp.sclbands(file)[1]
    #print(scl_list)
    scl.append(scl_list)

scl_da = xr.concat([xr.open_rasterio(i,chunks={'x':512, 'y':512}) for i in scl],  #
                        dim=time_var)

mask_lon_scl = (scl_da.x >= (xmin-dx)) & (scl_da.x <= (xmax+120+dx))
mask_lat_scl = (scl_da.y >= (ymin-120-dy)) & (scl_da.y <= (ymax+120+dy))

scl_da = scl_da.where(mask_lon_scl & mask_lat_scl, drop=True)

new_scl_da = xr.DataArray(scl_da,dims=scl_da.dims,attrs= scl_da.attrs.copy())

new_scl_da = new_scl_da.interp(y=red_ds["y"], x=red_ds["x"], method='nearest')
#new_scl_da
scl_ds = new_scl_da.to_dataset('band')

scl_ds = scl_ds.rename({1:'scl'})
print(scl_ds)

ds=xr.merge([nir_ds,red_ds,scl_ds])   #, compat='no_conflicts', join='left')  #broadcast_equals  #exact indexes along dimension x are not equal
#print(ds)

ndvi=(ds['nir']-ds['red'])/(ds['nir']+ds['red'])
print(ndvi)
#print(ndvi.values[0,1,1])

scl = ds['scl']
good_data = scl.where((scl == 4) | (scl == 5) | (scl == 6))
#good_data
ndvi_no_cloud = ndvi.where(good_data>=0)

#ndvi_sth=ndvi_no_cloud.rolling(time=1, min_periods=1, center=True).mean()
ndvi_sth=ndvi_no_cloud.rolling(time=5, min_periods=1, center=True).mean()
print(ndvi_sth)

y=np.squeeze(ndvi_sth.values)
print(y)
ndvi_sth.coords

x=ndvi_sth.coords['time'].values
x.sort(axis=0)
print(x)
mask = ndvi_sth.isnull()
#mask
ndvi_cl = ndvi_sth.where(~mask, other=0)

path_out  = os.path.join(os.getcwd(), 'output')
ndvi_cl.isel(time=0).rio.to_raster(os.path.join(path_out,'ndvicl.tif',dtype="float32")

#vSOS = Value at the start of season
# select timesteps before peak of season (AKA greening)
greenup = ndvi_cl.where(ndvi_cl.time < ndvi_cl.isel(time=ndvi_cl.argmax("time")).time)
# find the first order slopes
green_deriv = greenup.differentiate("time")
# find where the first order slope is postive
pos_green_deriv = green_deriv.where(green_deriv > 0)
# positive slopes on greening side
pos_greenup = greenup.where(pos_green_deriv)
print(pos_greenup)


# find the median
median = pos_greenup.median("time")
print(median)

# distance of values from median
distance = pos_greenup - median
print(distance)

da = ndvi_cl


def allNaN_arg(da, dim, stat):
    """
    Calculate da.argmax() or da.argmin() while handling
    all-NaN slices. Fills all-NaN locations with an
    float and then masks the offending cells.
    Params
    ------
    xarr : xarray.DataArray
    dim : str,
            Dimension over which to calculate argmax, argmin e.g. 'time'
    stat : str,
        The statistic to calculte, either 'min' for argmin()
        or 'max' for .argmax()
    Returns
    ------
    xarray.DataArray
    """
    # generate a mask where entire axis along dimension is NaN
    mask = da.isnull().all(dim)

    if stat == "max":
        y = da.fillna(float(da.min() - 1))
        y = y.argmax(dim=dim, skipna=True).where(~mask)
        return y

    if stat == "min":
        y = da.fillna(float(da.max() + 1))
        y = y.argmin(dim=dim, skipna=True).where(~mask)
        return y

# find index (argmin) where distance is most negative
idx = allNaN_arg(distance, "time", "min").astype("int16")
#idx

# find index (argmin) where distance is smallest absolute value
idx = allNaN_arg(xr.ufuncs.fabs(distance), "time", "min").astype("int16")
#idx.values


def _vpos(da):
    """
    vPOS = Value at peak of season
    """
    return da.max("time")

def _pos(da):
    """
    POS = DOY of peak of season
    """
    return da.isel(time=da.argmax("time")).time.dt.dayofyear


# calculate the statistics
print("Phenology")
vpos = _vpos(da)
pos = _pos(da)

print('pos')
print(pos)

#pos.rio.to_raster('pos.tif',dtype="float32")
pos.rio.to_raster(os.path.join(path_out,'pos.tif',dtype="float32"))
#vpos.rio.to_raster('vpos.tif',dtype="float32")
vpos.rio.to_raster(os.path.join(path_out,'vpos.tif',dtype="float32"))

def _trough(da):
    """
    Trough = Minimum value
    """
    return da.min("time")

def _aos(vpos, trough):
    """
    AOS = Amplitude of season
    """
    return vpos - trough

def _vsos(da, pos, method_sos="median"):
    """
    vSOS = Value at the start of season
    Params
    -----
    da : xarray.DataArray
    method_sos : str,
        If 'first' then vSOS is estimated
        as the first positive slope on the
        greening side of the curve. If 'median',
        then vSOS is estimated as the median value
        of the postive slopes on the greening side
        of the curve.
    """
    # select timesteps before peak of season (AKA greening)
    greenup = da.where(da.time < pos.time)
    # find the first order slopes
    green_deriv = greenup.differentiate("time")
    # find where the first order slope is postive
    pos_green_deriv = green_deriv.where(green_deriv > 0)
    # positive slopes on greening side
    pos_greenup = greenup.where(pos_green_deriv)
    # find the median
    median = pos_greenup.median("time")
    # distance of values from median
    distance = pos_greenup - median

    if method_sos == "first":
        # find index (argmin) where distance is most negative
        idx = allNaN_arg(distance, "time", "min").astype("int16")

    if method_sos == "median":
        # find index (argmin) where distance is smallest absolute value
        idx = allNaN_arg(xr.ufuncs.fabs(distance), "time",
                         "min").astype("int16")

    return pos_greenup.isel(time=idx)


def _sos(vsos):
    """
    SOS = DOY for start of season
    """
    return vsos.time.dt.dayofyear

vsos = _vsos(da,pos,method_sos="median")
sos = _sos(vsos)
print('sos')
print(sos)

#sos.rio.to_raster('sos.tif',dtype="float32")
sos.rio.to_raster(os.path.join(path_out,'sos.tif',dtype="float32"))

print('vsos')
print(vsos)
#vsos.rio.to_raster('vsos.tif',dtype="float32")
vsos.rio.to_raster(os.path.join(path_out,'vsos.tif',dtype="float32"))

def _veos(da, pos, method_eos="median"):
    """
    vEOS = Value at the "start" end of season
    Params
    -----
    method_eos : str
        If 'last' then vEOS is estimated
        as the last negative slope on the
        senescing side of the curve. If 'median',
        then vEOS is estimated as the 'median' value
        of the negative slopes on the senescing
        side of the curve.
    """
    # select timesteps before peak of season (AKA greening)
    senesce = da.where(da.time > pos.time)
    # find the first order slopes
    senesce_deriv = senesce.differentiate("time")
    # find where the fst order slope is postive
    neg_senesce_deriv = senesce_deriv.where(senesce_deriv < 0)
    # negative slopes on senescing side
    neg_senesce = senesce.where(neg_senesce_deriv)
    # find medians
    median = neg_senesce.median("time")
    # distance to the median
    distance = neg_senesce - median

    if method_eos == "last":
        # index where last negative slope occurs
        idx = allNaN_arg(distance, "time", "min").astype("int16")

    if method_eos == "median":
        # index where median occurs
        idx = allNaN_arg(xr.ufuncs.fabs(distance), "time",
                         "min").astype("int16")

    return neg_senesce.isel(time=idx)


def _eos(veos):
    """
    EOS = DOY for end of seasonn
    """
    return veos.time.dt.dayofyear

veos = _veos(da, pos, method_eos="median")
eos = _eos(veos)

#os.path.join('output',eos.rio.to_raster('eos.tif',dtype="float32"))
eos.rio.to_raster(os.path.join(path_out,'eos.tif',dtype="float32"))
#os.path.join('output',veos.rio.to_raster('veos.tif',dtype="float32"))
veos.rio.to_raster(os.path.join(path_out,'veos.tif',dtype="float32"))

def _los(da, eos, sos):
    """
    LOS = Length of season (in DOY)
    """
    los = eos - sos
    # handle negative values
    los = xr.where(
        los >= 0,
        los,
        da.time.dt.dayofyear.values[-1] +
        (eos.where(los < 0) - sos.where(los < 0)),
    )
    return los

def _rog(vpos, vsos, pos, sos):
    """
    ROG = Rate of Greening (Days)
    """
    return (vpos - vsos) / (pos - sos)

def _ros(veos, vpos, eos, pos):
    """
    ROG = Rate of Senescing (Days)
    """
    return (veos - vpos) / (eos - pos)

eos = _eos(veos)
los = _los(da, eos, sos)
rog = _rog(vpos, vsos, pos, sos)
ros = _ros(veos, vpos, eos, pos)

stats_dict = {
        "SOS": sos.astype(np.int16),
        "EOS": eos.astype(np.int16),
        "vSOS": vsos.astype(np.float32),
        "vPOS": vpos.astype(np.float32),
       # "Trough": trough.astype(np.float32),
        "POS": pos.astype(np.int16),
        "vEOS": veos.astype(np.float32),
        "LOS": los.astype(np.int16),
       # "AOS": aos.astype(np.float32),
        "ROG": rog.astype(np.float32),
        "ROS": ros.astype(np.float32),
    }

print(stats_dict['SOS'])

# Set up figure
fig, ax = plt.subplots(nrows=3,
                       ncols=2,
                       figsize=(12, 16),
                       sharex=True,
                       sharey=True)

# Set colorbar size
cbar_size = 0.7

# Set aspect ratios
for a in fig.axes:
    a.set_aspect('equal')

# Start of season
stats_dict['SOS'].plot(ax=ax[0, 0],
              cmap='magma_r',
              vmax=300,
              vmin=0,
              cbar_kwargs=dict(shrink=cbar_size, label='')) #label=None
              #cbar_kwargs=dict(shrink=cbar_size, label='')).figure.savefig('SOS.png') #label=None    #ax.figure.savefig('file.png')
ax[0, 0].set_title('Start of Season (DOY)')
stats_dict['vSOS'].plot(ax=ax[0, 1],
               cmap='YlGn',
               vmax=0.8,
               vmin=0,
               cbar_kwargs=dict(shrink=cbar_size, label='')) #label=None
ax[0, 1].set_title('NDVI' + ' at SOS')


# Peak of season
stats_dict['POS'].plot(ax=ax[1, 0],
              cmap='magma_r',
              vmax=365,
              vmin=0,
              cbar_kwargs=dict(shrink=cbar_size, label=''))
ax[1, 0].set_title('Peak of Season (DOY)')
stats_dict['vPOS'].plot(ax=ax[1, 1],
               cmap='YlGn',
               vmax=0.8,
               vmin=0,
               cbar_kwargs=dict(shrink=cbar_size, label=''))
ax[1, 1].set_title('NDVI' + ' at POS')


# End of season
stats_dict['EOS'].plot(ax=ax[2, 0],
              cmap='magma_r',
              vmax=365,
              vmin=0,
              cbar_kwargs=dict(shrink=cbar_size, label=None))
ax[2, 0].set_title('End of Season (DOY)')
stats_dict['vEOS'].plot(ax=ax[2, 1],
               cmap='YlGn',
               vmax=0.8,
               vmin=0,
               cbar_kwargs=dict(shrink=cbar_size, label=None))
ax[2, 1].set_title('NDVI' + ' at EOS')

fig.savefig('statsPheno.png')

nd = ndvi_cl.mean(dim=['x', 'y'])

vpos_nd = _vpos(nd)
pos_nd = _pos(nd)
veos_nd = _veos(nd,pos_nd,method_eos="median")
eos_nd = _eos(veos_nd)

vsos_nd = _vsos(nd,pos_nd,method_sos="median")
sos_nd = _sos(vsos_nd)

year = str(ds.time.dt.year.values[0]) + " "
sos_dt = dt.datetime.strptime(year + str(sos_nd.values), '%Y %j')  #stats_dict['SOS'].values
pos_dt = dt.datetime.strptime(year + str(pos_nd.values), '%Y %j')  #stats_dict['POS']
eos_dt = dt.datetime.strptime(year + str(eos_nd.values), '%Y %j')  #stats_dict['EOS'] zonal_phen.EOS.values

print('phenology metrics')
print('SOS_t:', sos_dt),print('POS_t:', pos_dt),print('EOS_t:',eos_dt),print('SOS DOY:',sos_nd.values),print('POS DOY:',pos_nd.values),print('VEOS DOY:',eos_nd.values),print('VSOS value:',vsos_nd.values),print('VPOS value:',vpos_nd.values),print('VEOS value:',veos_nd.values)



# Create plot
fig, ax = plt.subplots(figsize=(11, 4))
#ax.plot(veg_rolling.time, veg_rolling, 'b-^')
#ax.plot(ndvi_sth.time,ndvi_sth.mean(dim=['x', 'y']),'b-^')
ax.plot(ndvi_cl.time,ndvi_cl.mean(dim=['x', 'y']),'g-^')
#ndvi_sth.mean(['x', 'y'])

# Add start of season

ax.plot(sos_dt,vsos_nd,'or')
ax.annotate('SOS',
            xy=(sos_dt,vsos_nd.values),
            xytext=(-5, 10),   #xytext=(-5, 30)
            textcoords='offset points',
            arrowprops=dict(arrowstyle='-|>'))

ax.plot(pos_dt,vpos_nd,'or')
ax.annotate('POS',
            xy=(pos_dt,vpos_nd.values),
            xytext=(-5, 10),   #xytext=(-5, 30)
            textcoords='offset points',
            arrowprops=dict(arrowstyle='-|>'))

ax.plot(eos_dt, veos_nd, 'or')
ax.annotate('EOS',
            xy=(eos_dt, veos_nd.values),
            xytext=(-5, 10),  #xytext=(-10, -25)
            textcoords='offset points',
            arrowprops=dict(arrowstyle='-|>'))

os.path.join('output',fig.savefig('graph.png'))

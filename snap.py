import os,sys
#import urllib
#from urllib import request

##
from pathlib import Path
from glob import glob

##
#import urllib.request
# connect to the API
import datetime
import zipfile
import xarray as xr
import pandas as pd
from sentinelsat import SentinelAPI, read_geojson, geojson_to_wkt
from datetime import date
api = SentinelAPI('iserra', 'Creaf-21', 'https://scihub.copernicus.eu/dhus')
import rioxarray

#fileBbox = bboxgeo   #geojosn file closed linestring
#fileTemporal = sys.argv[2]   #txt file
print (bboxgeo)

ARG=json.load(open("vlabparams.json","r"))

print(str(ARG['bbox'][0])) #1
print(str(ARG['bbox'].split(","))) #['19.1031', ' 64.0410', ' 19.8981', '64.4038']

print(str(ARG['bbox'])) #19.1031, 64.0410, 19.8981,64.4038
dates[0]=ARG['bbox'].split(",")[0]
dates[1]=ARG['bbox'].split(",")[1]

"""
with open(fileTemporal) as f:
#with open('input/dates.txt') as f:
    contents = f.read()
    dates = contents.split(",")
    print(dates)
"""
print(dates[0])
print(dates[1])

"""
footprint = geojson_to_wkt(read_geojson(fileBbox))
products = api.query(footprint,
                     date = (dates[0],dates[1]),
                     platformname = 'Sentinel-2',
                     processinglevel = 'Level-2A',
                     cloudcoverpercentage = (0, 80))  #80%


print(len(products))

for i in products:
    print (i,api.get_product_odata(i)['title'])
    #print (api.get_product_odata(i)['url'])
"""

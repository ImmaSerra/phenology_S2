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
api = SentinelAPI(' ', '', 'https://scihub.copernicus.eu/dhus')
import rioxarray
import json

#fileBbox = bboxgeo   #geojosn file closed linestring

#bboxgeo = 'bboxgeo.json'
#bboxgeo = sys.argv[1]

#fileTemporal = sys.argv[2]   #txt file


ARG=json.load(open("vlabparams.json","r"))
print(str(ARG))

list = []
for item in os.listdir(os.getcwd()):
    #print(item)
    list.append(item)
print(list[1])

footprint0 = geojson_to_wkt(read_geojson('bboxgeo.json'))
print(footprint0)

fileBbox2 = read_geojson('bboxgeo.json')
print(fileBbox2)
#print(list.files(full.names=TRUE, recursive=TRUE))#
#cat("\n\n## ------------------------------------------------------------------ ##\n\n")

# Get command line values
#args <- commandArgs(trailingOnly=TRUE)


#https://www.tutorialspoint.com/python/python_command_line_arguments.htm

"""
bboxgeo
{'data1': '20210101', 'data2': '20211231', 'bbox': 'false'}
-- data1 20210101 -- data2 20211231
"""

print(str(ARG)) #f
arg=""
for k,v in ARG.items():
    if (v is False)|(v in ["False","false","F"]):
        continue
    else:
        if (v is True)|(v=="true"):
            #v=""
            v="true"
        arg+=" -- "+" ".join([k,str(v)])
print(arg)


dates =[ARG['data1'],ARG['data2']]
print(dates[0])
print(dates[1])


"""
with open(fileTemporal) as f:
#with open('input/dates.txt') as f:
    contents = f.read()
    dates = contents.split(",")
    print(dates)
"""

footprint = geojson_to_wkt(read_geojson('bboxgeo.json'))
products = api.query(footprint,
                     date = (dates[0],dates[1]),
                     platformname = 'Sentinel-2',
                     processinglevel = 'Level-2A',
                     cloudcoverpercentage = (0, 80))  #80%


print(len(products))

for i in products:
    print (i,api.get_product_odata(i)['title'])
    #print (api.get_product_odata(i)['url'])

output = "output.txt"
with open(output, "w") as outputfile:
    outputfile.write(str(len(products)))
    #outputfile.write('hola')

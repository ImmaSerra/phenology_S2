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

#footprint0 = geojson_to_wkt(read_geojson('bboxgeo.json'))
#print(footprint0)

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

if not os.path.exists('output'):
    os.makedirs('output')

output = "output.txt"

outputfile = os.path.join('output', output)

#with open(output, "w") as outputfile:
f = open(outputfile, "w")
for i in products:
    #print (i,api.get_product_odata(i)['title'],api.get_product_odata(i)['url'])
    #print (api.get_product_odata(i)['url'])
    print (api.get_product_odata(i)['title'])
    #print (api.get_product_odata(i)['url'])
    #outputfile.write(str(api.get_product_odata(i)['title'])+'\n')
    f.write(str(api.get_product_odata(i)['title'])+'\n')
f.close()

#with open(output, "w") as outputfile:
#    outputfile.write(str(len(products)))
"""
for i in products:
    #print (i,api.get_product_odata(i)['title'])
    print(api.get_product_odata(i)['title'])
    #print (api.get_product_odata(i)['url'])
"""
# convert to Pandas DataFrame
products_df = api.to_dataframe(products)
#print('df')
#print(products_df)

# sort and limit to first 1 sorted products
#products_df_sorted = products_df.sort_values(['cloudcoverpercentage', 'ingestiondate'], ascending=[True, True])
products_df_sorted = products_df.sort_values(['ingestiondate'], ascending=[True])
#products_df_sorted = products_df_sorted.head(2)


if not os.path.exists('temp'):
    os.makedirs('temp')

# download sorted and reduced products
#api.download_all(products_df_sorted.index)  #donwload products

download_path = './temp'
# download sorted and reduced products in a specific folder
api.download_all(products_df_sorted.index,directory_path=download_path)

# download all results from the search
#api.download_all(products)

#download_path4 = './temp4'
# download all results from the search in a specific folder
#api.download_all(products,directory_path=download_path4)

#ds = products_df.to_xarray()  #es un xarrayDataset
ds = products_df['title'].to_xarray()  #es un xarray.DataArray


"""
def unzip(zipped_filename):
    with zipfile.ZipFile(zipped_filename, 'r') as zip_ref:
        if not os.path.exists('unzipped'):
            os.makedirs('unzipped')
        zip_ref.extractall('./unzipped')
"""
if not os.path.exists('unzipped'):
    os.makedirs('unzipped')


download_unzipped_path = os.path.join(os.getcwd(), 'unzipped')
#print(download_unzipped_path)

extension = ".zip"
for item in os.listdir(download_path): #canviar download_unzipped_path
    print(item)
    if item.endswith(extension): # check for ".zip" extension
        file_name = os.path.join(download_path, item) # get full path of files
        zip_ref = zipfile.ZipFile(file_name) # create zipfile object
        zip_ref.extractall(download_unzipped_path) # extract file to dir
        zip_ref.close() # close file

files_unzip = os.listdir(download_unzipped_path)
print(files_unzip)

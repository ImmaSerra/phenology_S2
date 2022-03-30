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

output = "output.txt"

with open(output, "w") as outputfile:
    for i in products:
        #print (i,api.get_product_odata(i)['title'],api.get_product_odata(i)['url'])
        #print (api.get_product_odata(i)['url'])
        print (api.get_product_odata(i)['title'])
        #print (api.get_product_odata(i)['url'])
        outputfile.write(str(api.get_product_odata(i)['title'])+'\n')

#with open(output, "w") as outputfile:
#    outputfile.write(str(len(products)))

"""
with open(fileBbox) as f:  #Svartberget.geojson
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

print(get_bounding_box(geometry_coord))
#print(type(get_bounding_box(geometry_coord)))  #tupla
print(get_bounding_box(geometry_coord)[0])
"""
"""
data = json.loads(footprint)
for  i in data:
    di= data['features'][i]['geometry'] #Your first point+
    print(di)
"""


for i in products:
    #print (i,api.get_product_odata(i)['title'])
    print(api.get_product_odata(i)['title'])
    #print (api.get_product_odata(i)['url'])

# convert to Pandas DataFrame
products_df = api.to_dataframe(products)
#print('df')
#print(products_df)

# sort and limit to first 1 sorted products
products_df_sorted = products_df.sort_values(['cloudcoverpercentage', 'ingestiondate'], ascending=[True, True])
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


def paths_to_datetimeindex(paths):
    string_slice=(45,-5) #string_slice=(45,60)
    date_strings = [os.path.basename(i)[slice(*string_slice)]
                    for i in paths]
    return pd.to_datetime(date_strings)

def unzip(zipped_filename):
    with zipfile.ZipFile(zipped_filename, 'r') as zip_ref:
        if not os.path.exists('unzipped'):
            os.makedirs('unzipped')
        zip_ref.extractall('./unzipped')

if not os.path.exists('unzipped'):
    os.makedirs('unzipped')


download_unzipped_path = os.path.join(os.getcwd(), 'unzipped')
#print(download_unzipped_path)

def queryS2(file):
    """Read an input product list returning the full-path product list suitable for reading datasets.

    Parameters:
        file (str): full-path of the input file listing target products

    Return: S5P L2 full-path to files (list)
    """
    with open(file,"r") as f:
        data = f.readlines()
        list = [d.split("\n")[0] for d in data]
    products = []
    for item in list:
        if item.endswith('.zip') or item.endswith('.SAFE') :
            products.append(item)
        else:
            for file in Path(item).rglob('*.zip') or Path(item).rglob('*.SAFE') :
                products.append(str(file))
    return products


def product_level(item):
    """Check for S2 product type. This information will change the relative path to images.

    Parameters:
        item (str): full path to S2 products location

    Return: exit status (bool)
    Raise ValueError for Unrecognized product types
    """
    if "MSIL2A" in item:
        return True
    elif "MSIL1C" in item:
        return False
    else:
        raise ValueError("%s: Unrecognized S2 product type"%item)



def bands(item,res='10m'):
    """Search for target MSIL2A bands given an input resolution. This is useful for index computations.

    Parameters:
        item (str): full path to S2 products location
        res (str): resolution of S2 images; default set to `10m`; allowed options: `10m`,`20m`,`60m`

    Return: bands sorted by increasing wavelength (list)
    """
    msi = product_level(item)
    products = []
    string = '*_'+str(res)+'.jp2'
    if msi: # L2A
        for path in Path(item).rglob(string):
            products.append(str(path))
    else: # L1C
        for path in Path(item).rglob('*.jp2'):
            products.append(str(path))
    return sorted(products) # ordered bands

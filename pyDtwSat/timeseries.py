import atexit
import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt
from datetime import datetime
from osgeo import ogr,gdal
import pandas as pd
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import os




def TransformCoordonates(raster_file,referance_file):
    #Open .tif file
    raster=gdal.Open(raster_file)
    #Transformation Information
    g=raster.GetGeoTransform()
    geo_transform=gdal.InvGeoTransform(g)
    
    posi = []
    row = []
    col = []
    
    #Read Referanc file and pass all the coordinates to 
    #transform into row and columnas in array.
    df = pd.read_csv(referance_file,index_col=0)
    df['coordinates'] = list(zip(df.longitude, df.latitude))
    for i in df['coordinates']:
        X = i[0]
        Y = i[1]
        
        pos = gdal.ApplyGeoTransform(geo_transform,X,Y)
        pos = tuple(int(j) for j in pos)
        row.append(pos[0])
        col.append(pos[1])
        posi.append(pos)
    df['position'] = posi
    df['row'] = row
    df['col'] = col
    return df



def get_pos_val(raster_file , referance_file):
    #Create data set with position of point in array
    df = TransformCoordonates(raster_file, referance_file)
    position = df['position']
    raster=gdal.Open(raster_file) 
    array = raster.ReadAsArray()
    array = np.flip(array, 1)
    array = np.rot90(array, axes = (-2,-1))
    type_val = []
    
    #By using all the row and columns value extract all the
    #positional values available in array i.e NDVI, EVI
    for i in df['position']:
        try:
            val = array[i[0],i[1]]
            type_val.append(val)
        except:
            val = np.nan
            type_val.append(val)
    #print(len(type_val)) 
    df['type_value'] = type_val
    df['tuple']= list(zip( df.type_value , df.row, df.col))  
    
    #Creates catagorical data set
    type_name = list(df['label'].unique())
    type_df = []
    for i in range(len(type_name)):
        type_df.append(df.groupby(['label']).get_group(type_name[i]))
    return type_df


#Visualization Function to show all the annotated pioints on Map
def show_point_on_map(raster_file, referance_file):
    #By using get_pos_val Function get all the positional values
    type_df = get_pos_val(raster_file, referance_file)
    for df in type_df:
        position = df['position']
        raster=gdal.Open(raster_file) 
        array = raster.ReadAsArray()
        array = np.flip(array, 1)
        array = np.rot90(array, axes = (-2,-1))

        #by passing nan value to the pixel can convert pixel in to white on plot
        for i in df['position']:
            try:
                array[i[0],i[1]] = np.nan
            except:
                pass
        array = np.rot90(array , axes = (-2,-1))
        array = np.flip(array , 0)
        #plot the array    
        plt.figure()
        plt.imshow(array)

#Function to get only values of array
def val_array(raster_file, referance_file):
    type_df = get_pos_val(raster_file, referance_file)
    type_array = []
    type_value = []
    for i in type_df:
        df = i
        trc = df['tuple']
        tv = df['type_value']
        #trc (value, row,column)
        trc = np.array(trc)
        #tv only value
        tv = np.array(tv)
    
        type_array.append(trc)
        type_value.append(tv)
    return type_array, type_value  


#Take all the raster files in the folder converts 
# them into array and stack them into one array 
def stack_tif(raster_folder):
    tif_folder = os.listdir(raster_folder)
    tif_files = []
    for n, file in enumerate(tif_folder):
        if file.endswith('.tiff'):
            file_type = '.tiff'
            tif_files.append(file)
        elif file.endswith('.tif'):
            file_type = '.tif'
            tif_files.append(file)
        else:
            pass
    tif_files = sorted(tif_files)
    #print(tif_files)
    
    
    tif_array_list = []
    for i, rast in enumerate(tif_files):
        raster_file = raster_folder + '/' + rast
        raster=gdal.Open(raster_file) 
        #read raster file as array
        array = raster.ReadAsArray()
        array = np.flip(array, 1)
        array = np.rot90(array, axes = (-2,-1))
        tif_array_list.append(array)

    return np.stack(tif_array_list) 


#by using stack array converts into pixel array
def get_stack_mat(stack_array):
    S,N,M = stack_array.shape

    stack_mat = np.zeros((N,M), dtype = object)
    for i in range(N):
        for j in range(M):
            ts_pixel = np.zeros((S))
            for s in range(S):
                #takes each pixel value from stak
                ts_pixel[s] = stack_array[s,i,j]
            stack_mat[i,j] = ts_pixel
    
    return stack_mat


#Takes all the raster files and creates time series from referance
def timeseries_array(raster_folder, referance_file):
    tif_folder = os.listdir(raster_folder)
    tif_files = []
    for n, file in enumerate(tif_folder):
        if file.endswith('.tiff'):
            file_type = '.tiff'
            tif_files.append(file)
        elif file.endswith('.tif'):
            file_type = '.tif'
            tif_files.append(file)
        else:
            print(f'{file}File not Supported')
            #print('File not Supported')
    #print(tif_files)
    
    
    timeseries_array = []
    for i, rast in enumerate(tif_files):
        raster_file = raster_folder + '/' + rast
        #print(raster_file)
        time_array = val_array(raster_file, referance_file)[1]
        timeseries_array.append(time_array)
    #len_timeseries_array = len(timeseries_array)
    #len_val_array = len(timeseries_array[0])
    timeseries_array = np.array(timeseries_array)

    
    return timeseries_array


def get_mean_array(raster_folder, referance_file):
    ts_array = timeseries_array(raster_folder,referance_file)
    ts_mean_array = []
    
    for i,j in enumerate(ts_array):
        #print(len(j[i]))
        type_mean_array = []
        for k in j:
            #print(len(k))
            mean_value = np.nanmean(k)
            type_mean_array.append(mean_value)
        #print(type_mean_array)
        #type_mean_array = np.array(type_mean_array)

        #mean value for time series
        ts_mean_array.append(type_mean_array)
           
    return np.array(ts_mean_array)


def get_pattern_array(raster_folder, referance_file):
    ts_mean_array = get_mean_array(raster_folder, referance_file)
    cat_mean_array = []
    for i in ts_mean_array:
        l = range(len(i))
    for j in l:
        cat_mean_array.append(j)
    #print(cat_mean_array)
        
    #patterns = np.zeros(len(cat_mean_array)*len(ts_mean_array)).reshape(len(cat_mean_array), len(ts_mean_array))
    patterns = []
    
    for s in cat_mean_array:
        #print(s)
        ptn = []
        for a in ts_mean_array:
            #print(a[s])
            cat_val = float(a[s])
            ptn.append(cat_val)
        patterns.append(ptn)
    patterns = np.array(patterns).reshape(len(cat_mean_array), len(ts_mean_array))
    
    return patterns


def get_date(raster_folder):
    timeline = []
    tif_folder = os.listdir(raster_folder)
    tif_files = []
    for n, file in enumerate(tif_folder):
        if file.endswith('.tiff'):
            file_type = '.tiff'
            tif_files.append(file)
        elif file.endswith('.tif'):
            file_type = '.tif'
            tif_files.append(file)
        else:
            pass
            #print('File not Supported')
    tif_files = sorted(tif_files)
    
    for rast in tif_files:
        start_date = rast[:10]
        timeline.append(start_date)
    date_array = np.array(timeline)
    
    delta = np.zeros(len(date_array), dtype=int)
    date_format ='%Y-%m-%d'
    #do = datetime.strptime(date_array[0], date_format)
    for n in range(len(date_array)):
        delta[n] = (datetime.strptime(date_array[n] , date_format).month)
        
    return delta


def listOfTuples(l1, l2):
    return list(map(lambda x, y:(x,y), l1, l2))

def ArrayOfTuples(a1,a2):
    print(len(a1), len(a2))
    array = np.zeros((len(a1)+1)*len(a2)).reshape(len(a1)+1, len(a2))
    
    for i in range(len(a1)):
        array[0] = a2
        array[i+1] = a1[i]
    return array
        
        
    


def GetTemporalPatterns(patterns,raster_folder):
 
    date_array = get_date(raster_folder)
    pattern_tuple = []
    patterns_tuple = []
    print(len(patterns))
    for pattern in patterns:
        if len(date_array) == len(pattern):
            tuples = listOfTuples(pattern, date_array)
            

        
        else:
            print("date len and pattern len does not match")
    #print(">>>> patterns tuple",np.array(patterns_tuple)[0], len(np.array(patterns_tuple)[0]))
        pattern_tuple.append(tuples)
    #print(np.array(pattern_tuple))
    month_list = set(date_array)
    
    patterns_avg_month_val = []
    for i in pattern_tuple:
        avg_month_val = []
        for j in month_list:
            month_val = []
            for k in i:
                if k[1] == j:
                    month_val.append(k[0])
                else:
                    pass

            avg_month_val.append(np.mean(np.array(month_val)))
        patterns_avg_month_val.append(avg_month_val)
    return np.array(patterns_avg_month_val)


def PlotTemporalPatterns(patterns,raster_folder, 
                        smoothness_parameter = 0.3, spline_order = 5,nest = -1):
    date_array = get_date(raster_folder)
    date_array = list(set(date_array))
    temporal_patterns = GetTemporalPatterns(patterns,raster_folder)
    color = ['r','g','b','y','c','m', 'y', 'k', 'w', 'lime', 'tan','gold','pink',
         'teal','violet','navy','darkblue','indigo','coral','chocolate',
         'ivory','snow','skyblue','hotpink','beige','peru','crimson','linen']
    #label = ['Vegetation','Plants','Urban Area', 'Water']
    plt.figure()
    for i,pattern in enumerate(temporal_patterns):
        #print(len(pattern), len(date_array))
        
        t, u = splprep([date_array, pattern], s=smoothness_parameter
                       , k=spline_order, nest=nest)
        xn, yn = splev(np.linspace(0, 1, 500), t)
        plt.plot(xn, yn, color=color[i], linewidth=1, label = 'crop'+str(i))
        plt.legend(loc = 'upper left')
        plt.ylim(0.0, 1.2)
        plt.xlabel("Time scale(in months)")
        plt.ylabel("NDVI")
    return plt.show()

   


#r1 = '/home/kira/SENTINEL_2_images_NDVI/2019-01-01_2019-02-01_HoshiyarPur.tif'
#r2 = '/home/kira/SENTINEL_2_images_NDVI/2019-02-01_2019-03-01_HoshiyarPur.tif'
#referance_file = '/home/kira/SENTINEL_2_images_NDVI/referance_file.csv'
#raster_folder = '/home/kira/SENTINEL_2_images_NDVI'
#print(val_array(r1,referance_file)[1])
#print(val_array(r2,referance_file)[1])
#print(timeseries_array(raster_folder, referance_file))
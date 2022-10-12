import os
from array import array
import pandas as pd
import numpy as np
from numpy import load
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import train_test_split
from scipy.stats import randint
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn import preprocessing

from timeseries import *

GradientBoost = GradientBoostingClassifier()
RandomForest = RandomForestClassifier()
KNeighbors = KNeighborsClassifier()

def get_data_eval(array_folder,raster_file, reference_file):
    df = TransformCoordonates(raster_file,reference_file)
    array_folder_list = sorted(os.listdir(array_folder))
    array_files = []
    array_file_name = []
    for n,arr_file in enumerate(array_folder_list):
        if arr_file.endswith('.npy'):
            array_files.append(np.load(array_folder+'/'+arr_file,allow_pickle=True))
            array_file_name.append(arr_file[:-4])
        else:
            print(arr_file, 'Not Supported')
    
    for j,class_ in enumerate(array_files):
        class_ = np.array(class_, dtype =float)
        array_val = []
        for i in df['position']:
            try:
                val = class_[i[0], i[1]]
                array_val.append(val)
            except:
                val = np.nan
                array_val.append(val)
        df[array_file_name[j]] = array_val

    label_list = df['label'].unique()
    label_class = []
    for i in range(0,len(label_list)):
        label_class.append(i)
        
    df['type'] =  df['label'].replace(label_list, label_class)

    df_train = df.drop(['longitude','latitude','label','coordinates','position','row','col'], axis = 1)

    return df_train, [array_files, array_file_name]


def Land_cover_pred_plot(array_folder,raster_file, reference_file,ML_algo, plot = False):
    df_train , train_array = get_data_eval(array_folder,raster_file, reference_file)
    df_train = df_train.dropna()
    print(df_train)
    train_array = np.array(train_array, dtype=object)
    
    tile_df = pd.DataFrame()
    for i, array in enumerate(train_array[0]):
        # print(train_array[i], train_array_name[i])
        tile_df[train_array[1][i]] = np.nan_to_num(array.ravel(), copy=False)
        # print(train_array[0][i], train_array[1][i])
    X_train, X_test, y_train, y_test = train_test_split(df_train.drop('type' , axis = 1),df_train['type'],test_size = 0.1)
    print(X_train)
    ML_algo.fit(X_train,y_train)
    test_pred = ML_algo.predict(X_test)
    confusion_mat = confusion_matrix(y_test,test_pred)
    classification_repo = classification_report(y_test, test_pred)
    test_acc = accuracy_score(y_test, test_pred)
    print("Confusion Matri : \n", confusion_mat)
    print("Classification Report : \n", classification_repo)
    print("Accuracy on Test : ", test_acc)

    pred_array = ML_algo.predict(tile_df)

    mask_array = np.reshape(pred_array, train_array[0][0].shape)

    class_sum = []
    for i,j in enumerate(df_train['type'].unique()):
        sum = (mask_array == j).sum()
        class_sum.append([j,sum])
    print(class_sum)
    print(mask_array)
    if plot == True:
        arr_f = np.array(mask_array, dtype = float)
        arr_f = np.rot90(arr_f, axes=(-2,-1))
        arr_f = np.flip(arr_f,0)    
        plt.imshow(arr_f)
        plt.colorbar()

    return mask_array


# coding: utf-8

# In[1]:

import sys
import glob
import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageColor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt


# In[2]:

def modify_Weather(Weather):
    if 'Clear' in Weather:
        Weather = 'Clear'
    elif 'Cloudy' in Weather:
        Weather = 'Cloudy'
    elif 'Dizzle' in Weather:
        Weather = 'Dizzle'
    elif 'Fog' in Weather:
        Weather = 'Fog'
    elif 'Rain' in Weather:
        Weather = 'Rain'
    elif 'Snow' in Weather:
        Weather = 'Snow'
    elif 'Thunderstorms' in Weather:
        Weather = 'Thunderstorms'
    return Weather


# In[3]:

def get_img(img, imagefile):
    pixel = []
    for i in imagefile:
        if i == img:
            im = Image.open(i)
            rgb_im = im.convert('RGB')
            r,g,b = rgb_im.getpixel((1,1))
            pixel.append(r)
            pixel.append(g)
            pixel.append(b)
    return pixel


# In[4]:

def clean_on_image_pixels(pixellist):
    if pixellist == []:
        pixellist = np.NaN
    return pixellist


# In[5]:

def modify_rgb(value):
    value = str(value)[1:-1]
    return value


# In[6]:

def StandardSVM(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    model = make_pipeline(StandardScaler(), 
                          SVC(kernel='linear', C=10))
    model.fit(X_train,y_train)
    score = model.score(X_test, y_test)
    return score


# In[7]:

def MinMaxSVM(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    model = make_pipeline(MinMaxScaler(),
                          SVC(kernel='linear', C=10))
    model.fit(X_train,y_train)
    score = model.score(X_test, y_test)
    return score


# In[8]:

def NB(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    model = make_pipeline(PCA(2),
                          GaussianNB())
    model.fit(X_train,y_train)
    score = model.score(X_test,y_test)
    return score


# In[9]:

def Knn(X,y):
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    model = make_pipeline(KNeighborsClassifier(n_neighbors=150))
    model.fit(X_train,y_train)
    score = model.score(X_test, y_test)
    return score


# In[10]:

def main():
    #call sys
    datafile = sys.argv[1]
    imagefile = sys.argv[2]
    
    
    #read in csv files and create dataframe weather_data
    #datafile = glob.glob('yvr-weather/*.csv')
    
    datalist = []
    weather_data = pd.DataFrame()
    for i in datafile:
        df = pd.read_csv(i, index_col=None, header=0, skiprows=16)
        datalist.append(df)
    weather_data = pd.concat(datalist)
    
    
    #clean data in weather_data
    weather_data = weather_data[['Date/Time','Month','Day','Time','Temp (°C)','Dew Point Temp (°C)','Rel Hum (%)','Wind Dir (10s deg)','Wind Spd (km/h)','Visibility (km)','Weather']]
    
    weather_data = weather_data[weather_data['Time']>'05:00'] 
    weather_data = weather_data[weather_data['Time']<'19:00']
    
    weather_data = weather_data[weather_data['Weather'].notnull()].reset_index()
    weather_data = weather_data.drop('index',1)
    
    weather_data.loc[weather_data.Weather != '', 'Weather'] = weather_data['Weather'].str.split(',').str.get(0)
    
    #weather_data.groupby(['Weather'])['Weather'].count() #helping data
    
    weather_data['Weather'] = weather_data['Weather'].apply(modify_Weather)
    
    weather_data['Date/Time'] = weather_data['Date/Time'].str.replace('-','').str.replace(':','').str.replace(' ','')
    
    weather_data['Image Name'] = 'katkam-scaled/katkam-' + weather_data['Date/Time'].map(str) + '00.jpg'
    #weather_data.shape #helping data
    
    
    #read in image files
    #imagefile = glob.glob('katkam-scaled/*.jpg')
    
    pixels = []
    for i in range(2916):
        pixel = get_img(weather_data['Image Name'][i], imagefile)
        pixels.append(pixel)
    
    
    #combine image data and weather data, then clean data
    pixel_data = pd.Series(pixels, name='RGB')
    weather_data = weather_data.join(pixel_data)
    
    weather_data['RGB'] = weather_data['RGB'].apply(clean_on_image_pixels)
    weather_data = weather_data[weather_data['RGB'].notnull()].reset_index()
    weather_data = weather_data.drop('index',1)
    
    weather_data['Darkness'] = weather_data['RGB'].apply(lambda x: np.mean(x))
    
    weather_data['RGB'] = weather_data['RGB'].apply(lambda x: modify_rgb(x))
    
    weather_data['R'] = weather_data['RGB'].str.split(',').str.get(0)
    weather_data['G'] = weather_data['RGB'].str.split(',').str.get(1)
    weather_data['B'] = weather_data['RGB'].str.split(',').str.get(2)
    
    weather_data['Time'] = weather_data['Time'].str.replace(':00','')
    #weather_data
    
    
    #train and test split
    X = weather_data[['Month', 'Time', 'Temp (°C)', 'Visibility (km)', 'Wind Spd (km/h)','R', 'G', 'B', 'Darkness']].values
    y = weather_data['Weather'].values
    X_train, X_test, y_train, y_test = train_test_split(X,y)
    
    
    #model training
    StandardSVM_score = StandardSVM(X,y)
    #MinMaxSVM_score = MinMaxSVM(X,y)
    #NB_score = NB(X,y)
    #Knn_score = Knn(X,y)

    
    #method : standardSVM
    print(StandardSVM_score)
    #print(MinMaxSVM_score) 
    #print(NB_score)
    #print(Knn_score) 


# In[11]:

if __name__ == '__main__':
    main()





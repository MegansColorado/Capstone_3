import numpy as np 
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt 
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, ModelCheckpoint
import os
import shutil
from tensorflow.keras import metrics
import matplotlib
#from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.xception import preprocess_input
from tensorflow.keras.applications import Xception
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, RMSprop



def create_X_and_y(X_filepath, y_filepath):
    ''' imports lined images and unlined images from jpeg files into numpy arrays for use into autoencoder'''


    y_list = []
    for image in sorted(os.listdir(y_filepath)):
        file= y_filepath + str(image)
        y_image = np.array(Image.open(file))
        y_image = y_image.reshape(y_image.reshape(*y_image.shape,1))
        y_list.append(y_image)
                    
    y = np.array(y_list)  

    X_list = []

    for image in sorted(os.listdir(X_filepath)):
        
        file = X_filepath+str(image)
        X_image = np.array(Image.open(file))
        X_image = X_image.reshape(X_image.reshape(*X_image.shape,1))
        X_list.append(X_image)
        
    X = np.array(X_list)
    return X, y



if __name__ == "__main__":
    y_filepath = '../Data/y_variables/Unruled/'
    X_filepath = '../Data/X_variables/computer_generated_lines/'
    X,y = create_X_and_y(X_filepath,y_filepath)
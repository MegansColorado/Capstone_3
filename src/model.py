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
from image_creator import create_X_and_y #how do i import from another .py file in the same folder



def callbacks(model_name = 'model', early_stopping = False, patience = None):
    checkpoint_filepath = (f'./tmp/{model_name}/checkpoint')
    tensorboard = TensorBoard(log_dir=(f"./logs/{model_name}"),
                            
                            histogram_freq=2,
                            write_graph=True,
                            write_images=True,
                            update_freq="epoch",
                            profile_batch=2,
                            embeddings_freq=0,
                            embeddings_metadata=None)
    if early_stopping == True:
        early_stopping = EarlyStopping(monitor='loss',  patience=patience, restore_best_weights=True)
    model_cp = ModelCheckpoint(filepath=checkpoint_filepath, monitor = 'loss', save_best_only=True)
    return tensorboard, early_stopping, model_cp
    

def create_and_compile_sparse_model(optimizer='adam', loss='mse', metrics='accuracy'):
    ''' creates and compiles sparse layer MODEL B of capstone 3 '''

    input_img = Input(shape=(500,400,1)) 
    x = Conv2D(32, (3, 3), activation='relu')(input_img)
    x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
    x = Dense(128)(x) 
    x= Conv2DTranspose(32,(3,3), activation='relu')(x)
    decoded = Conv2DTranspose(1, (3, 3), activation='linear', padding='same')(x)

    autoencoder_B = Model(input_img, decoded)
    autoencoder_B.summary()
    return autoencoder_B.compile(optimizer=optimizer, loss=loss, metrics=metrics)

def fit_model(compiled_model, x=X, y=y, batch_size=15, epochs=50, , verbose=1, callbacks=[tensorboard, model_cp], validation_split=0.2):

    history_B = compiled_model.fit(x=X, y=y, batch_size=batch_size, epochs = epochs, verbose=verbose, callbacks=callbacks, validation_split=validation_split)
    return history_B


if __name__ == "__main__":
    y_filepath = '../Data/y_variables/Unruled/'
    X_filepath = '../Data/X_variables/computer_generated_lines/'
    X,y = create_X_and_y(X_filepath,y_filepath)
    sparse_model = create_and_compile_sparse_model()
    sparse_model_fit = fit_model(sparse_model, epochs = 300)
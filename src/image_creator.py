import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from PIL import Image
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, Conv2DTranspose
from tensorflow.keras.models import Sequential
import os
import shutil

for image in os.listdir('Data/Unruled'):
    
    y=np.array((Image.open(f'../Data/Unruled/{image}.jpg')))
    X = np.array(y)
    X[::40+np.random.randint(-2,2),:] = 80+np.random.randint(-60, 130)
    pil_X = Image.fromarray(X)
    pil_X.save(f'../Data/computer_generated_lines/{image}.png')


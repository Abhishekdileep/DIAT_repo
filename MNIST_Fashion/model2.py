import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D , Dense , AveragePooling2D , Flatten , MaxPooling2D
from tensorflow.keras import Sequential

def build_lenet_model2():
    model = Sequential()
    
    model.add(Conv2D(filters=64 , kernel_size=5 , padding="same", activation='sigmoid', input_shape=(28,28,1)))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

    model.add(Conv2D(filters=32 , kernel_size=5 , activation='sigmoid'))
    model.add(MaxPooling2D(pool_size=(2,2), strides=2, padding='same'))

    model.add(Flatten())

    model.add(Dense(120 , activation='sigmoid'))
    model.add(Dense(84,activation='sigmoid'))
    model.add(Dense(10,activation='softmax'))

    model.compile(loss='categorical_crossentropy' , 
                    optimizer='adam',
                    metrics=['accuracy'])
    return model
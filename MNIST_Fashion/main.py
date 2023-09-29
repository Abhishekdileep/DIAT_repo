import tensorflow as tf
import numpy as np 
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D , Dense , AveragePooling2D , Flatten 
from tensorflow.keras import Sequential
from model1 import build_lenet_model1
from model2 import build_lenet_model2
from evaluate import plot_accuracy , plot_loss 

#Data Preprocessing 
def Preprocessing():
    (X_train , Y_train ) , (X_test , Y_test ) = tf.keras.datasets.fashion_mnist.load_data()
    (X_train , Y_train) , (X_val ,Y_val) = (X_train[:5000] , Y_train[:5000] ) , (X_train[5000:],Y_train[5000:])

    w,h = 28,28
    X_train = X_train.reshape(X_train.shape[0] , w, h , 1)
    X_test = X_test.reshape(X_test.shape[0], w, h, 1)
    X_val = X_val.reshape(X_val.shape[0] , w, h, 1)

    Y_train = tf.keras.utils.to_categorical(Y_train , 10)
    Y_test = tf.keras.utils.to_categorical(Y_test , 10)
    Y_val = tf.keras.utils.to_categorical(Y_val , 10)

    return (X_train , X_test , X_val ) , (Y_train , Y_test , Y_val)

#model details 


if __name__ == "__main__":
    (X_train , X_test , X_val ) , (Y_train , Y_test , Y_val)  = Preprocessing()
    model1 = build_lenet_model1()
    model2 = build_lenet_model2()

    #FIT 
    hist1 = model1.fit(X_train , Y_train)
    hist2 = model2.fit(X_train , Y_train)
    
    #Test Values 
    plot_accuracy(hist1) 
    plot_loss(hist1)

    plot_accuracy(hist2)
    plot_loss(hist2)
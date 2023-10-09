import cv2
import matplotlib.pyplot as plt
from PIL import Image 
import numpy as np 

def generate_histograms(image_path):
    # Open the image using Pillow
    image_np = cv2.imread(image_path)
    # Plot the histograms
    colors = ('b' , 'g' , 'r')
    for i ,color in enumerate(colors):
        hist = cv2.calcHist([image_np],[i],None,[256],[0,256]) 
        plt.plot(hist,color = color) 
    plt.title('Image Histogram GFG') 
    plt.show()

    
image_path = 'random.jpg'
generate_histograms(image_path)

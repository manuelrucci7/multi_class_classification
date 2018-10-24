# Import required libraries
import os
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image # faster than opencv in opening images
import glob
import pandas as pd

from sklearn.metrics import confusion_matrix
import itertools

from keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Dropout, concatenate
from keras.models import Model
from keras import backend as K
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import TensorBoard


def round_f(vec):
    # This function given a input vector return a vector like [0 .... 1 ....0] in which all the elements are zeros expect one element that is 1.
    out = []
    for i in range(vec.shape[0]):
        max_index = np.argmax(vec[i,:])
        rounded_vector = np.zeros((vec[0].shape))
        rounded_vector[max_index] = 1
        out.append(rounded_vector)
    out = np.asarray(out)
    return out

def get_model():
    shape = (128,128,1)
    num_classes=5
    # Unet https://github.com/rachitk/UNet-Keras-TF/blob/master/unet_build.py
    input_img = Input(shape=shape)
    conv1 = Conv2D(4, (3, 3), activation='relu', padding='same')(input_img)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(8, (3, 3), activation='relu', padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(16, (3, 3), activation='relu', padding='same')(pool2)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    up4 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv3), conv2], axis=3)
    conv4 = Conv2D(8, (3, 3), activation='relu', padding='same')(up4)
    up5 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv4), conv1], axis=3)
    conv5 = Conv2D(4, (3, 3), activation='relu', padding='same')(up5)
    #output_layer = Conv2D(1, (1, 1), activation='sigmoid', padding='same')(conv5)
    # Fully connected layer
    x = Flatten()(conv5)
    x = Dropout(0.5)(x)
    x = Dense(100, activation='relu')(x)
    output_layer = Dense(num_classes, activation='softmax')(x)
    unet = Model(inputs=[input_img], outputs=[output_layer])
    return unet


if __name__ == "__main__":

    class_names = ["maniche_a_34" , "maniche_corte", "maniche_lunghe", "monospalla", "senza_maniche"]

    # Load an color image in grayscale
    img = cv2.imread('test_image.jpeg',1)
    temp_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    temp_gray_resize = cv2.resize(temp_gray, (128,128), interpolation=cv2.INTER_LANCZOS4)
    temp_gray_mask = (1 - temp_gray_resize/255) # background 0 , dress 1
    input_img = temp_gray_mask[np.newaxis,:,:,np.newaxis]

    # See how the image it is preprocess
    cv2.imshow("Original Image", img)
    cv2.imshow("Post Processed Image", 255*temp_gray_mask)
    cv2.waitKey(0)

    ModelName = 'model-hpa.h5'
    test_model= get_model()
    test_model.load_weights(ModelName)

    Y_hat = test_model.predict(input_img)
    # Let's round the output by setting the max of the vector =1, so if y=[0.1,0.2,0,0,0] it becames y=[0,1,0,0,0]
    Y_hat = round_f(Y_hat)
    index = np.argmax(Y_hat)

    print("THE CLASSIFIER SAYS: " + str(class_names[index]))

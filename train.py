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


class Data():

    def __init__(self):
        self.root = "dati_maniche/"
        self.img_type = ".png"
        self.class_names = ["maniche_a_34" , "maniche_corte", "maniche_lunghe", "monospalla", "senza_maniche"]
        self.chosen_num_images = 250
        self.shape_desired = (128,128)

    def get_num_classes(self):
        return int(len(self.class_names))

    def get_data(self):
        if os.path.exists('data_X.npy') and os.path.exists('data_Y.npy') and os.path.exists('data_num_classes.npy'):
            print("Loading previously saved data")
            X = np.load('data_X.npy')
            Y = np.load('data_Y.npy')
            num_images_per_class = np.load('data_num_classes.npy')
        else:
            print("Creating new data")
            images = []
            labels = []
            num_images_per_class = []
            for i in range(0,len(self.class_names)):
                temp_path = self.root + self.class_names[i] + "/*" + self.img_type
                count = 0
                for filename in glob.glob(temp_path):
                    # Counter
                    count = count + 1
                    # Get Image
                    temp_image = np.array(Image.open(filename))
                    images.append(temp_image)
                    # Get Label (one hot encoding)
                    temp_label = np.zeros(len(self.class_names))
                    temp_label[i] = 1
                    labels.append(temp_label)

                num_images_per_class.append(count)
            num_images_per_class = np.asarray(num_images_per_class)

            # The data has to be numpy array not list
            X = np.asarray(images)
            Y = np.asarray(labels)
            # Save data locally
            np.save('data_X.npy', X)
            np.save('data_Y.npy', Y)
            np.save('data_num_classes.npy', num_images_per_class)
        return X, Y, num_images_per_class

    def get_prepared_data(self, X, Y,num_images_per_class):
        X_train = []
        Y_train = []
        X_val = []
        Y_val = []
        start = 0
        stop = num_images_per_class[0]
        for i in range(0, (len(self.class_names))):
            #print("start: =",start)
            #print("stop: =",stop)
            temp_X_train = X[start:(start+self.chosen_num_images-1),:,:,:]
            temp_Y_train = Y[start:(start+self.chosen_num_images-1),:]
            temp_X_val = X[(start+self.chosen_num_images):stop,:,:,:]
            temp_Y_val = Y[(start+self.chosen_num_images):stop,:]
            X_train.append(temp_X_train)
            Y_train.append(temp_Y_train)
            X_val.append(temp_X_val)
            Y_val.append(temp_Y_val)
            if (i!=(len(self.class_names)-1)):
                start = start + num_images_per_class[i]
                stop = stop + num_images_per_class[i+1]
        X_train = np.asarray(X_train)
        Y_train = np.asarray(Y_train)
        X_val = np.asarray(X_val)
        Y_val = np.asarray(Y_val)
        X_train = np.concatenate((X_train[0],X_train[1],X_train[2],X_train[3],X_train[4]),0)
        Y_train = np.concatenate((Y_train[0],Y_train[1],Y_train[2],Y_train[3],Y_train[4]),0)
        X_val = np.concatenate((X_val[0],X_val[1],X_val[2],X_val[3],X_val[4]),0)
        Y_val = np.concatenate((Y_val[0],Y_val[1],Y_val[2],Y_val[3],Y_val[4]),0)
        return X_train, Y_train, X_val, Y_val

    def get_preprocess_data(self, X_train, X_val):
        X_train_pp = []
        X_val_pp = []
        for i in range(0, X_train.shape[0]):
            temp_gray = cv2.cvtColor(X_train[i,:,:,:], cv2.COLOR_BGR2GRAY)
            temp_gray_resize = cv2.resize(temp_gray, self.shape_desired, interpolation=cv2.INTER_LANCZOS4)
            temp_gray_mask = (1 - temp_gray_resize/255) # background 0 , dress 1
            X_train_pp.append(temp_gray_mask)
        for i in range(0, X_val.shape[0]):
            temp_gray = cv2.cvtColor(X_val[i,:,:,:], cv2.COLOR_BGR2GRAY)
            temp_gray_resize = cv2.resize(temp_gray, self.shape_desired, interpolation=cv2.INTER_LANCZOS4)
            temp_gray_mask = (1 - temp_gray_resize/255) # background 0 , dress 1
            X_val_pp.append(temp_gray_mask)
        X_train_pp = np.asarray(X_train_pp)
        X_val_pp = np.asarray(X_val_pp)
        return X_train_pp, X_val_pp

class Metric():

    def round_f(self,vec):
        # This function given a input vector return a vector like [0 .... 1 ....0] in which all the elements are zeros expect one element that is 1.
        out = []
        for i in range(vec.shape[0]):
            max_index = np.argmax(vec[i,:])
            rounded_vector = np.zeros((vec[0].shape))
            rounded_vector[max_index] = 1
            out.append(rounded_vector)
        out = np.asarray(out)
        return out

    def CM(self,y_true,y_pred):
        # Batch wise confusion matrix for multiclass classification
        # This function calculate the confusion matrix of the overall classifier. It outputs the average True Positive, True Negative,
        # False Positive and False negative.
        TP = np.sum(np.round(np.clip(y_true * y_pred, 0, 1))) # true positives looking at all the classes
        TP_plus_FP = np.sum(np.round(np.clip(y_pred, 0, 1))) # predited positives  looking at all the classes
        FP = TP_plus_FP - TP
        TP_plus_FN = np.sum(np.round(np.clip(y_true, 0, 1))) # possible positives  looking at all the classes
        FN = TP_plus_FN-TP
        TN = y_true.shape[0] - TP # true negative (len y_true or y_pred - TP)
        return TP,FP,FN,TN

    def recall(self,y_true, y_pred):
        # Recall metric only computes a batch-wise average of recall.
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) # true positives
        FN = K.sum(K.round(K.clip(y_true, 0, 1))) # possible positives
        recall = TP / (FN + K.epsilon())
        return recall

    def precision(self,y_true, y_pred):
        # Precision metric only computes a batch-wise average of precision.
        TP = K.sum(K.round(K.clip(y_true * y_pred, 0, 1))) # true positives
        FP = K.sum(K.round(K.clip(y_pred, 0, 1))) # predited positives
        precision = TP / (FP + K.epsilon())
        return precision

    def f1(self,y_true,y_pred):
        p = self.precision(y_true, y_pred)
        r = self.recall(y_true, y_pred)
        return 2*((p*r)/(p+r+K.epsilon()))

class Classifier():

    def __init__(self):
        self.MyMetric = Metric()
        self.image_shape = (128,128)
        self.class_names = ["maniche_a_34" , "maniche_corte", "maniche_lunghe", "monospalla", "senza_maniche"]
        self.num_classes = len(self.class_names)
        self.ModelName = 'model-hpa1.h5'
        self.PlotName = 'PLOT_NAME1.png'

    def plot_confusion_matrix(self,cm, classes, normalize=False, title='Confusion matrix',cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    def my_generator(self, X_train_pp, Y_train, batch_size):
        SEED=10
        data_generator = ImageDataGenerator(
                horizontal_flip=True,
                width_shift_range=0.01,
                height_shift_range=0.01,
                rotation_range=5,
                zoom_range=0.1).flow(X_train_pp, Y_train, batch_size, seed=SEED)
        while True:
            x_batch, y_batch = data_generator.next()
            yield x_batch, y_batch

    def get_model(self):
        # Unet https://github.com/rachitk/UNet-Keras-TF/blob/master/unet_build.py
        input_img = Input(shape=(self.image_shape[0],self.image_shape[1],1))
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
        output_layer = Dense(self.num_classes, activation='softmax')(x)
        unet = Model(inputs=[input_img], outputs=[output_layer])
        return unet

    def train(self, X_train_pp, Y_train, X_val_pp, Y_val):
        # Type in a terminal (tensorboard --logdir=/tmp/classification) to see tensoboard
        my_model =  self.get_model()
        my_model.compile(optimizer=Adam(lr=2e-4),loss='categorical_crossentropy' , metrics=[self.MyMetric.f1,self.MyMetric.precision,self.MyMetric.recall])

        # Early stopping
        weight_saver = ModelCheckpoint(self.ModelName,  monitor='val_f1',  verbose=1, save_best_only=True, save_weights_only=False, mode='max', period=1)

        batch_size = 10
        hist = my_model.fit_generator(self.my_generator(X_train_pp, Y_train, batch_size),
                                       steps_per_epoch = int(X_train_pp.shape[0]/batch_size),
                                       validation_data = (X_val_pp, Y_val),
                                       epochs=30, verbose=1,
                                       callbacks = [weight_saver, TensorBoard(log_dir='/tmp/classification')])

        fig, ax = plt.subplots(2,2, figsize = (15,10))
        ax[0,0].plot(hist.history['loss'], color='b',label='Training Loss')           # Training Loss
        ax[0,0].plot(hist.history['val_loss'], color='r',label='Validation Loss')     # Validation Loss
        ax[0,0].legend()
        ax[0,0].set_xlabel('Epochs')
        ax[0,1].plot(hist.history['precision'], color='b',label='Training Precision')
        ax[0,1].plot(hist.history['val_precision'], color='r', label='Validation Precision')
        ax[0,1].legend()
        ax[0,1].set_xlabel('Epochs')
        ax[1,0].plot(hist.history['recall'], color='b',label='Training Recall')
        ax[1,0].plot(hist.history['val_recall'], color='r', label='Validation Recall')
        ax[1,0].legend()
        ax[1,0].set_xlabel('Epochs')
        ax[1,1].plot(hist.history['f1'], color='b',label='Training F1')
        ax[1,1].plot(hist.history['val_f1'], color='r', label='Validation F1')
        ax[1,1].legend()
        ax[1,1].set_xlabel('Epochs')

        plt.savefig( self.PlotName ) # Save into result folder

        print("------------------------------------------------------")
        print ("Training categorical_crossentropy loss: " + str(hist.history['loss'][-1]))
        print ("Training precision: " + str(hist.history['precision'][-1]))
        print ("Training recall: " + str(hist.history['recall'][-1]))
        print ("Training f1: " + str(hist.history['f1'][-1]))
        print("------------------------------------------------------")

        print("------------------------------------------------------")
        print ("Validation categorical_crossentropy loss: " + str(hist.history['val_loss'][-1]))
        print ("Validation precision: " + str(hist.history['val_precision'][-1]))
        print ("Validation recall: " + str(hist.history['val_recall'][-1]))
        print ("Validation f1: " + str(hist.history['val_f1'][-1]))
        print("------------------------------------------------------")

    def test(self,X_val_pp, Y_val):
        model1=self.get_model()
        model1.load_weights(self.ModelName)

        Y_hat = model1.predict(X_val_pp)
        # Let's round the output by setting the max of the vector =1, so if y=[0.1,0.2,0,0,0] it becames y=[0,1,0,0,0]
        Y_hat = self.MyMetric.round_f(Y_hat)
        #Y_hat = np.round(np.clip(Y_hat,0,1))  # Soft round, if y=[0.1,0.2,0,0,0] it becomes y=[0,0,0,0,0]

        cnf_matrix = confusion_matrix(Y_val.argmax(axis=1), Y_hat.argmax(axis=1))
        plt.figure()
        self.plot_confusion_matrix(cnf_matrix, classes=self.class_names, normalize=False, title='Confusion matrix, without normalization')
        plt.savefig('Confusion_Matrix.png')

if __name__ == "__main__":

    MyData = Data()
    X,Y,num_images_per_class = MyData.get_data()
    X_train, Y_train, X_val, Y_val = MyData.get_prepared_data(X,Y,num_images_per_class)
    X_train_pp, X_val_pp = MyData.get_preprocess_data(X_train, X_val)
    X_train_pp = X_train_pp[:,:,:,np.newaxis]
    X_val_pp = X_val_pp[:,:,:,np.newaxis]

    MyClassifier = Classifier()
    MyClassifier.train(X_train_pp, Y_train, X_val_pp, Y_val)
    MyClassifier.test(X_val_pp, Y_val)

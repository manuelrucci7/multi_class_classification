# multi_class_classification
In this repository a multi class classifier algorithm  it is used to classify different types of T-Shirts

## Clone Repository ##
* sudo apt-get install git
* cd $HOME
* git clone https://github.com/manuelrucci7/multi_class_classification.git

## Dependencies ##
* Python 2.7, Keras==2.1.4 , numpy==1.11.0, opencv_python==3.4.1.15,    
 matplotlib==2.0.2, pandas==0.21.0, Pillow==5.3.0,  scikit_learn==0.20.0
* To install the packages types: $ pip install -r requirements.txt    
The requirements.txt files has been automatically generated using  
<a href="https://github.com/bndr/pipreqs">pipreqs package</a>
* To install jupyter notebook type: $ pip install jupyter
<a href=" https://jupyter.readthedocs.io/en/latest/install.html">Jupyter Information</a>


## Instruction ##
* $ cd $HOME/multi_class_classification
* Download the data and extract them inside the $HOME/multi_class_classification folder. You should have inside $HOME/multi_class_classification/ a folder called **dati_maniche**.
* Run jupyter notebook --> $ jupyter notebook
* if u are NOT interested in running the jupyter notebook you can use the python script train.py to train the model. To run it, type: $ python train.py


## Information ##
* The jupyter notebook **classification.ipynb** explains the preproccessing, the model design and evaluation steps. This is the most important file of this repository.
* The training set is composed by 250 images for each class.  This means that our training dataset will have the same amount of images per classes. This choice aims at avoiding class imbalance.
* The validation set is made up by the remaining images.  Numbers of maniche_a_34=853-250=603, Numbers of maniche_corte = 978-250=728,  Numbers of maniche_lunghe=864-250=614, Numbers of monospalla=374-250=124, Numbers of senza_maniche=811-250=561
* As preprocessing tecninques the images have been resized to 128x128, normalize in range 0,1, and converted to grayscale images.
* A Unet Model combined with a fully connected layer has been used. The model has been trained for 30 epochs (around 8 minutes), categorical_crossentropy has been chosen as loss function  and f1 score has been selected as performance metric.
* The overall model precision, recall, accuracy and F1 score are:
> Algorithm Precision: 0.9129       
  Algorithm Recall: 0.9129    
  Algorithm Accuracy: 0.9652   
  Algorithm F1 Score: 0.9129
* A python script it is also available to train the model. To run it, types $ **python train.py** after having Downloaded the data. The training average time it is around 8 minutes.
The ouput of the train.py files are: PLOT_NAME1.png, model-hpa1.h5 and Confusion_Matrix.png.  
* **test.py**  instead it is used to make a prediction of an random image using the pretrained model model-hpa1.h5 . This script takes as input a chosen image Downloaded from internet called test_image.png and it classifies it. The classification output it is shown in the terminal.


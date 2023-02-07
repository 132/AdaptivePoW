'''Import the required ibraries'''
import numpy as np
import pandas as pd
import random
import sys
import getopt
import cv2
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D 
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
# from tensorflow.keras.optimizers import SGD
from tensorflow.keras.optimizers.legacy import SGD
from tensorflow.keras import backend as k
import os



''' make a simple Model for training'''
class SimpleMLP:
    @staticmethod
    def build(shape, classes):
        model = Sequential()
        model.add(Dense(200, input_shape=(shape,)))
        model.add(Activation("relu"))
        model.add(Dense(200))
        model.add(Activation("relu"))
        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model

''' Load the local data present in the device'''
def load(paths, verbose=-1):
    data = list()
    labels = list()
    # Loop over input images
    for (i,imgpath) in enumerate(paths):
        #  load image and extract label
        im_gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)
        image = np.array(im_gray).flatten()
        label = imgpath.split(os.path.sep)[-2]
        #  scale image
        data.append(image/255)
        labels.append(label)
        #  show update
        if verbose > 0 and i > 0 and (i+1) % verbose ==  0:
            print("[INFO] processed {}/{}".format(i + 1, len(paths)))
        #  return tuple mof data and label
    return data,labels

def batch_data(data_shard, bs=32):
    '''client data shard -> tfds object
        args:
            shard
            batch size
        return
            tfds object'''
    #  sep data and label
    data, label = zip(*data_shard)
    dataset = tf.data.Dataset.from_tensor_slices((list(data),list(label)))
    return dataset.shuffle(len(label)).batch(bs)

def scale_model_weights(weight, scalar):
    '''Function for scaling a model's weights'''
    weight_final = []
    steps = len(weight)
    for i in range(steps):
        weight_final.append(scalar*weight[i])
    return weight_final


def  sum_scaled_weights(scaled_weight_list):
    '''Returns the sum of the listed scaled weights.
    scaled average of the weights'''
    avg_grad = list()
    #  get the average grad over all client gradients
    for grad_list_tupel in zip(*scaled_weight_list):
        layer_mean = tf.math.reduce_sum(grad_list_tupel, axis=0)
        avg_grad.append(layer_mean)
    return avg_grad


def client_main(folder_path='/home/nextg3/Documents/Thesis/Code/FL/local_models',img_path='/home/nextg3/Documents/Thesis/Code/FL/MNIST_training',client_name='client'):
    '''
    Main fuction called for a client for
    local model training and sharing models
    '''
    print('######################')
    print('######################')
    print('I am a client called {}'.format(client_name))
    #  declare path
    # img_path = '/home/nextg3/Documents/FederatedLearning/FL_final/MNIST_training'
    # Get path list
    image_paths = list(paths.list_images(img_path))
    print('Reading image folders at the location {}'.format(img_path))
    print('######################')
    #  apply function
    image_list, label_list = load(image_paths, verbose = 10000)
    #  Binarize the labels
    lb = LabelBinarizer()
    label_list = lb.fit_transform(label_list)
    X_train, y_train =  image_list, label_list
    batched_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(len(y_train))

    lr = 0.01
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    optimizer = SGD(learning_rate = lr,
                    momentum = 0.9
                    )
    
    smlp_local = SimpleMLP()
    local_model = smlp_local.build(784,10)
    local_model.compile(loss=loss,
                        optimizer=optimizer,
                        metrics=metrics)
    # Load weights from h5 file
    print('loading global weight at: {}'.format(os.path.join(folder_path,'global_model.h5')))
    local_model.load_weights(os.path.join(folder_path,'global_model.h5'))
    # Fit local model with client data
    local_model.fit(batched_data, epochs=1, verbose=0)
    client_name = client_name+'_weights.h5'
    print('Saving trained model weights as: {}'.format(os.path.join(folder_path,client_name)))
    local_model.save_weights(os.path.join(folder_path,client_name))
    print('######################')
    print('######################')


def miner_main(folder_path='/home/nextg3/Documents/Thesis/Code/FL/local_models'):
    # Declare Model parameters
    print('######################')
    print('######################')
    print('I am a miner')
    print('######################')
    print('######################')
    lr = 0.01
    loss = 'categorical_crossentropy'
    metrics = ['accuracy']
    optimizer = SGD(learning_rate = lr,
                    momentum = 0.9
                    )
    # Read all the h5 files from the path
    weight_path = folder_path
    # Get path list
    weight_paths = list(paths.list_files(weight_path))
    local_models  = []

    # get all models
    for p in weight_paths:
        if 'global_model' in p:
            continue
        test_model = SimpleMLP()
        local_model = test_model.build(784,10)
        local_model.compile(loss=loss,
                            optimizer=optimizer,
                            metrics=metrics)
        # set weights of the global model as weights of the local modelk

        local_model.load_weights(p)
        scaled_model_weights = scale_model_weights(local_model.get_weights(),len(weight_paths))
        local_models.append(scaled_model_weights)
        k.clear_session()

    # Final weights
    average_weights = sum_scaled_weights(local_models)
    smlp_global = SimpleMLP()
    global_model = smlp_global.build(784,10)
    global_model.set_weights(average_weights)
    print('######################')
    print('######################')
    print('Saving global weights at {}'.format(os.path.join(folder_path,'global_model.h5')))
    global_model.save_weights(os.path.join(folder_path,'global_model.h5'))
    print('######################')
    print('######################')



if __name__=='__main__':
    try:
      opts, args = getopt.getopt(sys.argv[1:],"he:p:i:",["entity=","path=","image_path="])
    except getopt.GetoptError:
        print ('script.py -e <miner/client> -p <path for shared storage> -ip <path to folder with local images>')
        sys.exit(2)
    miner = False
    for opt, arg in opts:
      if opt == '-h':
         print ('script.py -e <miner/client> -p <path for shared storage> -ip <path to folder with local images>')
         sys.exit()
      elif opt in ("-e", "--entity"):
         if arg == 'miner':
            miner=True
      elif opt in ("-p", "--path"):
         folder_path = arg
      elif opt in ("-i", "--image_path"):
         imag_path = arg
    
    if miner:
        miner_main(folder_path)
    else:
        client_main(folder_path=folder_path,img_path=imag_path,client_name='bob')





'''Import the required ibraries'''
import numpy as np
import pandas as pd
import random
import cv2
from imutils import paths
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
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


'''
    Main fuction called for a client for
    local model training and sharing models
'''
def client_main(dir_path, scaling_factor=10):
    #  declare path
    # img_path = '/home/nextg3/Documents/FederatedLearning/FL_final/MNIST_training'
    img_path = '/home/nextg3/Documents/Thesis/Code/FL/MNIST_training'
    # Get path list
    image_paths = list(paths.list_images(img_path))
    #  apply function
    image_list, label_list = load(image_paths, verbose = 10000)
    #  Binarize the labels
    lb = LabelBinarizer()
    label_list = lb.fit_transform(label_list)
    # split train test
    # X_train,X_test, y_train, y_test = train_test_split(image_list,
    #                                                 label_list,
    #                                                 test_size=0.1,
    #                                                 random_state=37)
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
    local_model.load_weights('/home/nextg3/Documents/Thesis/Code/FL/weights.h5')
    # Fit local model with client data
    local_model.fit(batched_data, epochs=1, verbose=0)
    # scale the model weights and add to the list
    # scaling_factor = weight_scaling_factor(clients_batched, client)
    scaled_weights = scale_model_weights(local_model.get_weights(),scaling_factor)
    return scaled_weights



    # scaled_local_weightsmlp_local = SimpleMLP()
    # local_model = smlp_local.build(784,10)
    # local_model.compile(loss=loss,
    #                     optimizer=optimizer,
    #                     metrics=metrics)

def miner_main(global_model, scaled_local_weight_list):
    average_weights = sum_scaled_weights(scaled_local_weight_list)
    # update global model
    global_model.set_weights(average_weights)




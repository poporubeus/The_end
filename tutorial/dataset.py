import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
from PIL import Image


path_galaxy_TRAIN = "/Users/francescoaldoventurelli/Desktop/datasets/GALAXY_DATASET/train/1/"
path_nogalaxy_TRAIN = "/Users/francescoaldoventurelli/Desktop/datasets/GALAXY_DATASET/train/0/"
path_galaxy_TEST = "/Users/francescoaldoventurelli/Desktop/datasets/GALAXY_DATASET/test/1"
path_nogalaxy_TEST = "/Users/francescoaldoventurelli/Desktop/datasets/GALAXY_DATASET/test/0"
                

'''def downsample(data_img):
    data_img= np.expand_dims(data_img, axis=-1)
    data_img = tf.image.resize(data_img, [16,16]) # if we want to resize 
    data_img = tf.squeeze(data_img, axis=-1)
    return data_img'''


def binarization(data):
    data = np.asarray(data.reshape((16,16)))
    threshold = 0.25
    for i in range(len(data)):
        for j in range(len(data)):
            if data[i][j] <= threshold:
                data[i][j] = 0.05
            else:
                data[i][j] = data[i][j]
    return data

def downsample(data_img, nfeatures):
                data_img= np.expand_dims(data_img, axis=-1)
                data_img = tf.image.resize(data_img, [nfeatures, nfeatures]) # if we want to resize 
                data_img = tf.squeeze(data_img, axis=-1)
                return data_img

def normalization(data):
    data_norm = data/255
    return data_norm


class data_mnist:
    def __init__(self, myseed) -> None:
        #self.data = data
        self.myseed = myseed

    def data_creation(self, zerotrain, onetrain, ntest, nfeatures):
            np.random.seed(self.myseed)
            (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
            x_train = x_train.astype("float32") / 255
            x_test = x_test.astype("float32") / 255
            train_filter = np.where((y_train == 0 ) | (y_train == 1))
            X_train, Y_train = x_train[train_filter], y_train[train_filter]
            idx = [i for i in range(len(X_train))]
            np.random.shuffle(idx)
            train_new  = X_train[idx, :,:]
            target_new = Y_train[idx,]
            zero_index_train = np.where(target_new == 0)[0]
            one_index_train = np.where(target_new == 1)[0]
            zero_indices_test = np.where(y_test == 0)[0]
            one_indices_test = np.where(y_test == 1)[0]
            selected_zero_indices = zero_indices_test[:ntest] ### come vedi il test set è fisso a 140 immagini
            selected_one_indices = one_indices_test[:ntest]

            selected_zero_train = zero_index_train[:zerotrain]
            selected_one_train = one_index_train[:onetrain]
            selected_indices = np.concatenate([selected_zero_indices, selected_one_indices])
            sel_train_indices = np.concatenate([selected_zero_train, selected_one_train])
            X_test = x_test[selected_indices]
            Y_test = y_test[selected_indices]
            np.random.shuffle(sel_train_indices)
            X_train = train_new[sel_train_indices]
            Y_train = target_new[sel_train_indices]
            X_train = np.expand_dims(X_train, axis=-1)
            X_train = tf.image.resize(X_train, [nfeatures, nfeatures]) # if we want to resize 
            X_train = tf.squeeze(X_train, axis=-1)
            X_test = np.expand_dims(X_test, axis=-1)
            X_test = tf.image.resize(X_test, [nfeatures, nfeatures]) # if we want to resize 
            X_test = tf.squeeze(X_test, axis=-1)
            ntrain_images = int(zerotrain+onetrain)
            return (
                np.asarray(X_train),
                np.asarray(Y_train),
                np.asarray(X_test),
                np.asarray(Y_test),
                ntrain_images,
                )
        

class Galaxy():
            def __init__(self, myseed, path_galaxy_train, path_nogalaxy_train, path_galaxy_test,
                         path_nogalaxy_test, features) -> None:
                    self.myseed = myseed
                    self.path_galaxy_train = path_galaxy_train
                    self.path_nogalaxy_train = path_nogalaxy_train
                    self.path_galaxy_test = path_galaxy_test
                    self.path_nogalaxy_test = path_nogalaxy_test
                    self.features = features
            
            def make_train(self,zerotrain, onetrain):
                    np.random.seed(self.myseed)
                    galaxy_list = []
                    nogalaxy_list = []
                    labels0 = [0 for i in range(zerotrain)]
                    labels1 = [1 for i in range(onetrain)]
                    #path_galaxy = "/Users/francescoaldoventurelli/Desktop/datasets/GALAXY_DATASET/train/1/"
                    #path_nogalaxy = "/Users/francescoaldoventurelli/Desktop/datasets/GALAXY_DATASET/train/0/"
                    galaxy_files = os.listdir(self.path_galaxy_train)
                    nogalaxy_files = os.listdir(self.path_nogalaxy_train)
                    index0 = np.array([int(i) for i in range(len(nogalaxy_files))]).flatten()
                    index1 = np.array([int(i) for i in range(len(galaxy_files))]).flatten()
                    np.random.shuffle(index0)
                    np.random.shuffle(index1)
                    train0 = list(nogalaxy_files[i] for i in index0)
                    train1 = list(galaxy_files[i] for i in index1)
                    ### select how many images you want
                    train0 = train0[:zerotrain]
                    train1 = train1[:onetrain]
                    for filename in train0:
                        file_path0 = os.path.join(self.path_nogalaxy_train, filename)
                        img0 = Image.open(file_path0).convert('L')
                        img_array0 = np.array(downsample(img0, self.features))
                        img_array0 = normalization(img_array0)
                        img_array0 = binarization(img_array0)
                        nogalaxy_list.append(img_array0.flatten())

                    for filename in train1:
                        file_path1 = os.path.join(self.path_galaxy_train, filename)
                        img1 = Image.open(file_path1).convert('L')
                        img_array1 = np.array(downsample(img1, self.features))
                        img_array1 = normalization(img_array1)
                        img_array1 = binarization(img_array1)
                        galaxy_list.append(img_array1.flatten())

                    input_combine = galaxy_list + nogalaxy_list
                    labels_combine = labels1 + labels0
                    idx = np.array([int(i) for i in range(len(input_combine))]).flatten()
                    np.random.shuffle(idx)
                    img_array_final = list(input_combine[i] for i in idx)
                    labels_final = list(labels_combine[i] for i in idx)
                    return (img_array_final, labels_final)
            def make_test(self):
                galaxy_list = []
                nogalaxy_list = []
                num0 = 50
                num1 = 50
                labels0 = [0 for i in range(num0)]
                labels1 = [1 for i in range(num1)]
                
                
                galaxy_files = os.listdir(self.path_galaxy_test)
                nogalaxy_files = os.listdir(self.path_nogalaxy_test)

                galaxy_files = [file for file in galaxy_files if not file.startswith('.')]
                nogalaxy_files = [file for file in nogalaxy_files if not file.startswith('.')]

                test0 = list(nogalaxy_files[i] for i in range(num0))
                test1 = list(galaxy_files[i] for i in range(num1))
                for filename in test0:
                    file_path0 = os.path.join(self.path_nogalaxy_test, filename)
                    img0 = Image.open(file_path0).convert('L')
                    img_array0 = np.array(downsample(img0, self.features))
                    img_array0 = normalization(img_array0)
                    img_array0 = binarization(img_array0)
                    nogalaxy_list.append(img_array0.flatten())

                for filename in test1:
                    file_path1 = os.path.join(self.path_galaxy_test, filename)
                    img1 = Image.open(file_path1).convert('L')
                    img_array1 = np.array(downsample(img1, self.features))
                    img_array1 = normalization(img_array1)
                    img_array1 = binarization(img_array1)
                    galaxy_list.append(img_array1.flatten())

                input_combine = galaxy_list + nogalaxy_list
                labels_combine = labels1 + labels0

                return (input_combine, labels_combine)
                        
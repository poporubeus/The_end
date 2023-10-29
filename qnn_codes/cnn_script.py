import tensorflow as tf
from tensorflow import keras
from keras.datasets.mnist import load_data
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os 

keras.utils.set_random_seed(5)


epochs = 20

def create_train_test(zerotrain, onetrain, zerotest, onetest,seed):
    np.random.seed(seed)
    (X_train, Y_train), (X_test, Y_test) = load_data()
    train_filter = np.where((Y_train == 0) | (Y_train == 1))
    test_filter = np.where((Y_test == 0) | (Y_test == 1))
    X_train = X_train[train_filter]
    Y_train = Y_train[train_filter]
    X_train = X_train.reshape((len(X_train),28,28,1))
    X_train_resized = np.empty((len(X_train), 12, 12, 1))

    for i in range(len(X_train)):
        resized_image = tf.image.resize(X_train[i], [12,12], method='bilinear')
        X_train_resized[i] = resized_image.numpy()

    X_test = X_test[test_filter]
    Y_test = Y_test[test_filter]
    X_test = X_test.reshape((len(X_test),28,28,1))
    X_test_resized = np.empty((len(X_test), 12,12, 1))

    for i in range(len(X_test)):
        resized_image = tf.image.resize(X_test[i], [12,12], method='bilinear')
        X_test_resized[i] = resized_image.numpy() 

    idx_train = [i for i in range(len(X_train))]
    idx_test = [i for i in range(len(X_test))]
    np.random.shuffle(idx_train)
    train_new  = X_train_resized[idx_train, :,:]
    target_new = Y_train[idx_train,]

    np.random.shuffle(idx_test)
    test_new  = X_test_resized[idx_test, :,:]
    target_test_new = Y_test[idx_test,]  

    zero_index_train = np.where(target_new == 0)[0]
    one_index_train = np.where(target_new == 1)[0]

    
    selected_zero_train = zero_index_train[:zerotrain]
    selected_one_train = one_index_train[:onetrain]
    
    sel_train_indices = np.concatenate([selected_zero_train, selected_one_train])
    
    np.random.shuffle(sel_train_indices)
    X_train_resized = train_new[sel_train_indices]
    Y_train = target_new[sel_train_indices]


    zero_index_test = np.where(target_test_new == 0)[0]
    one_index_test = np.where(target_test_new == 1)[0]

    selected_zero_test = zero_index_test[:zerotest]
    selected_one_test = one_index_test[:onetest]

    #selected_indices = np.concatenate([selected_zero_indices, selected_one_indices])
    sel_test_indices = np.concatenate([selected_zero_test, selected_one_test])
    
    X_test_resized = test_new[sel_test_indices]
    Y_test = target_test_new[sel_test_indices]


    return X_train_resized, Y_train, X_test_resized, Y_test


#############################
### GALAXY DATASET###
## IMPORTA IL DATASET

def downsample(data_img):
    data_img= np.expand_dims(data_img, axis=-1)
    data_img = tf.image.resize(data_img, [16,16]) # if we want to resize 
    data_img = tf.squeeze(data_img, axis=-1)
    return data_img


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


def normalization(data):
    data_norm = data/255
    return data_norm



seed1 = 1
seed2 = 2
seed3 = 3
seed4 = 4
seed5 = 5

tot_seed = [seed1, seed2, seed3, seed4, seed5]

def make_train(num0, num1, seed):
    np.random.seed(seed)
    galaxy_list = []
    nogalaxy_list = []
    labels0 = [0 for i in range(num0)]
    labels1 = [1 for i in range(num1)]
    path_galaxy = "/Users/francescoaldoventurelli/Desktop/datasets/GALAXY_DATASET/train/1/"
    path_nogalaxy = "/Users/francescoaldoventurelli/Desktop/datasets/GALAXY_DATASET/train/0/"
    galaxy_files = os.listdir(path_galaxy)
    nogalaxy_files = os.listdir(path_nogalaxy)
    index0 = np.array([int(i) for i in range(len(nogalaxy_files))]).flatten()
    index1 = np.array([int(i) for i in range(len(galaxy_files))]).flatten()
    np.random.shuffle(index0)
    np.random.shuffle(index1)
    train0 = list(nogalaxy_files[i] for i in index0)
    train1 = list(galaxy_files[i] for i in index1)
    ### select how many images you want
    train0 = train0[:num0]
    train1 = train1[:num1]
    for filename in train0:
        file_path0 = os.path.join(path_nogalaxy, filename)
        img0 = Image.open(file_path0).convert('L')
        img_array0 = np.array(downsample(img0))
        img_array0 = normalization(img_array0)
        img_array0 = binarization(img_array0)
        #nogalaxy_list.append(img_array0.flatten())
        #nogalaxy_list.append(img_array0)
        img_array0.reshape(16,16,1)

    for filename in train1:
        file_path1 = os.path.join(path_galaxy, filename)
        img1 = Image.open(file_path1).convert('L')
        img_array1 = np.array(downsample(img1))
        img_array1 = normalization(img_array1)
        img_array1 = binarization(img_array1)
        #galaxy_list.append(img_array1.flatten())
        #galaxy_list.append(img_array1)
        img_array1.reshape(16,16,1)

    input_combine = galaxy_list + nogalaxy_list
    labels_combine = labels1 + labels0
    idx = np.array([int(i) for i in range(len(input_combine))]).flatten()
    np.random.shuffle(idx)
    img_array_final = np.array(input_combine[i] for i in idx)
    labels_final = np.array(labels_combine[i] for i in idx)
    img_array_final.reshape((len(img_array_final),16,16,1))
    return (img_array_final, labels_final)

#### TEST SET

def make_test():
    galaxy_list = []
    nogalaxy_list = []
    num0 = 50
    num1 = 50
    labels0 = [0 for i in range(num0)]
    labels1 = [1 for i in range(num1)]
    
    path_galaxy = "/Users/francescoaldoventurelli/Desktop/datasets/GALAXY_DATASET/test/1"
    path_nogalaxy = "/Users/francescoaldoventurelli/Desktop/datasets/GALAXY_DATASET/test/0"
    
    galaxy_files = os.listdir(path_galaxy)
    nogalaxy_files = os.listdir(path_nogalaxy)

    galaxy_files = [file for file in galaxy_files if not file.startswith('.')]
    nogalaxy_files = [file for file in nogalaxy_files if not file.startswith('.')]

    test0 = list(nogalaxy_files[i] for i in range(num0))
    test1 = list(galaxy_files[i] for i in range(num1))
    for filename in test0:
        file_path0 = os.path.join(path_nogalaxy, filename)
        img0 = Image.open(file_path0).convert('L')
        img_array0 = np.array(downsample(img0))
        img_array0 = normalization(img_array0)
        img_array0 = binarization(img_array0)
        #nogalaxy_list.append(img_array0.flatten())
        nogalaxy_list.append(img_array0)

    for filename in test1:
        file_path1 = os.path.join(path_galaxy, filename)
        img1 = Image.open(file_path1).convert('L')
        img_array1 = np.array(downsample(img1))
        img_array1 = normalization(img_array1)
        img_array1 = binarization(img_array1)
        #galaxy_list.append(img_array1.flatten())
        galaxy_list.append(img_array1)

    input_combine = galaxy_list + nogalaxy_list
    labels_combine = labels1 + labels0

    return (np.array(input_combine), np.array(labels_combine))


#### Save the plots to check the balance

'''def train_plot(train_imgs, labels):
    num_train = train_imgs
    fig, axes = plt.subplots(nrows=1, ncols=10, figsize=(16, 16))
    fig.suptitle("Train set")

    for i, ax in enumerate(axes.flatten()):
        ax.imshow(train_imgs[i].reshape((16,16)), cmap="gray")
        ax.set_xlabel(labels[i])
        ax.set_xticks([])
        ax.set_yticks([])
    save1 = plt.savefig("/Users/francescoaldoventurelli/Desktop/TESI_FINAL/GALAXY/Train/trainset16_RL_WATERFALL_feat_"+str(num_train)+"run"+str(run)+".jpg")
    plt.close()
    return save1
    #plt.show()

def test_plot(test_imgs, labels):
    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(16, 16))
    fig.suptitle("Test set")
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(test_imgs[i].reshape((16,16)), cmap="gray")
        ax.set_xlabel(labels[i])
        ax.set_xticks([])
        ax.set_yticks([])
    save1 = plt.savefig("/Users/francescoaldoventurelli/Desktop/TESI_FINAL/GALAXY/Test/testset100_16_RL_WATERFALL_feat_"+str(run)+".jpg")
    plt.close()
    return save1'''
#########################################################






def CNNModel(input_features):
    model = Sequential([
    Conv2D(1, (3,3), activation='relu', input_shape=(input_features,input_features,1)),
    MaxPool2D(pool_size=(2,2), strides=3),
    Flatten(),
    Dense(1, activation='sigmoid')])
    return model

def Summary(model):
    return model.summary()

#summary = Summary(CNNModel(16))


seed0 = 0
seed1 = 1
seed2 = 2
seed3 = 3
seed4 = 4

myseed = [seed0,seed1,seed2,seed3,seed4]

#train_sets = np.array([6,10,20])

def TRAIN_EVAL_MODEL(num_train, features): 
    model = CNNModel(features)
    numzero = int(num_train/2)
    numone = int(num_train/2)
    train_eval=[]
    test_eval = []
    train_loss = []
    train_acc = []
    for i in myseed:
        print('Seed number:', i)
        #X_train, Y_train, X_test, Y_test = create_train_test(numzero, numone, 50,50,i)
        X_train, Y_train = make_train(numzero,numone,i)
        X_test,Y_test = make_test()
        model.compile('adam','binary_crossentropy',['accuracy'])
        history = model.fit(X_train, Y_train, batch_size=1, epochs=epochs)
        train_loss.append(history.history['loss'])
        train_acc.append(history.history['accuracy'])
        accuracy_ontrain = model.evaluate(X_train, Y_train, verbose=0)
        train_eval.append(accuracy_ontrain[1])
        score = model.evaluate(X_test, Y_test, verbose=0)
        test_eval.append(score[1])
    return [train_loss, train_acc, train_eval, test_eval]


'''total_loss = []
total_train_acc = []
test_accuracy_total=[]
for i in train_sets:
    total_loss.append(TRAIN_EVAL_MODEL(i, 16)[0])
    total_train_acc.append(TRAIN_EVAL_MODEL(i,16)[1])
    test_accuracy_total.append(TRAIN_EVAL_MODEL(i,16)[2])
    print('Loss:', total_loss)
    print('Train acc:', total_train_acc)
    print('Test acc:', test_accuracy_total)
#loss6_16, acc6_16, test_acc6_16 = TRAIN_EVAL_MODEL(6, 16)'''

train_loss = TRAIN_EVAL_MODEL(6,16)[0]
train_acc = TRAIN_EVAL_MODEL(6,16)[1]
train_accuracy_evaluated = TRAIN_EVAL_MODEL(6,16)[2]
test_acc = TRAIN_EVAL_MODEL(6,16)[3]

print('Loss:', train_loss)
print('Train acc.:', train_acc)

print('Evaluation on trainset:', train_accuracy_evaluated)
print('Test acc.:', test_acc)




    




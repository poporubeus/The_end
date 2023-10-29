import tensorflow as tf
from tensorflow import keras
from keras.datasets.mnist import load_data
#import time
#import torch 
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.parameter import Parameter
from qiskit import Aer, QuantumCircuit
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.utils import QuantumInstance
from torch.nn import CrossEntropyLoss
from torch.optim import LBFGS
#from qiskit_machine_learning.algorithms.classifiers import NeuralNetworkClassifier, VQC
#from qiskit.algorithms.optimizers import COBYLA, L_BFGS_B
#import pandas as pd 

from circuits import *
#from circuits import encoding, circuit15

from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


seed = 42
qi = QuantumInstance(Aer.get_backend('aer_simulator'), seed_simulator=seed,seed_transpiler=seed,shots=256)
qi.backend.set_option("seed_simulator", seed)


def mnist_data(zerotrain, onetrain, myseed):
    np.random.seed(myseed)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    

    train_filter = np.where((y_train == 0 ) | (y_train == 1))
 

    X_train, Y_train = x_train[train_filter], y_train[train_filter]
    #X_test, Y_test = x_test[test_filter], y_test[test_filter]
    idx = [i for i in range(len(X_train))]
    np.random.shuffle(idx)
    train_new  = X_train[idx, :,:]
    target_new = Y_train[idx,]
    #X_train = train_new[:num_train]
    #Y_train = target_new[:num_train]

    zero_index_train = np.where(target_new == 0)[0]
    one_index_train = np.where(target_new == 1)[0]
    zero_indices_test = np.where(y_test == 0)[0]
    one_indices_test = np.where(y_test == 1)[0]

    # Select the desired number of indices for 0s and 1s
    selected_zero_indices = zero_indices_test[:70] ### come vedi il test set è fisso a 140 immagini
    selected_one_indices = one_indices_test[:70]

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
    X_train = tf.image.resize(X_train, [16,16]) # if we want to resize 
    X_train = tf.squeeze(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)
    X_test = tf.image.resize(X_test, [16,16]) # if we want to resize 
    X_test = tf.squeeze(X_test, axis=-1)
    

    return (
        np.asarray(X_train),
        np.asarray(Y_train),
        np.asarray(X_test),
        np.asarray(Y_test),
    )


def train_plot(x_train, y_train):
    fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(16, 16))
    fig.suptitle("Train set")

    for i, ax in enumerate(axes.flatten()):
        ax.imshow(x_train[i].reshape((16,16)), cmap="gray")
        ax.set_xlabel(y_train[i])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    save = plt.savefig("/Users/francescoaldoventurelli/Desktop/TESI_FINAL/train/train"+str(num_train)+"_dist_run_WF_2L"+str(nrun+1)+".jpg")
    plt.close()
    return save #### NOME dell'immagine è: train_data + numero di train (es. 6 imm di train) + run + num_run (prima, seconda,...ripetizione == max = 5)
#plt.show()


def test_plot(x_test, y_test):
    fig, axes = plt.subplots(nrows=14, ncols=10, figsize=(16, 16))
    fig.suptitle("Test set")
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(x_test[i].reshape((16, 16)), cmap="gray")
        ax.set_xlabel(y_test[i])
    #ax.set_xlabel(y_test[i])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    save = plt.savefig("/Users/francescoaldoventurelli/Desktop/TESI_FINAL/test/test140__WF_2L"+str(num_train)+"_"+str(nrun+1)+".jpg")
    plt.close()
    return save #plt.show()



def train_distribution():
    num_0_train = np.count_nonzero(y_train==0)
    num_1_train = np.count_nonzero(y_train==1)
    
    colors = ['royalblue', 'orangered']

    unique1, counts1 = np.unique(y_train, return_counts=True)
    fig = plt.figure()
    plt.bar(unique1, counts1, color= colors)
    plt.title("Training set")
    plt.xlabel("Classes")
    plt.ylabel("Number of items")
    plt.xticks(unique1)
    legend_labels = [num_0_train, num_1_train]
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    plt.legend(legend_handles, legend_labels)
    save = plt.savefig("/Users/francescoaldoventurelli/Desktop/TESI_FINAL/train/train_dist6x6_WF_2L"+str(num_train)+"_dist_run"+str(nrun+1)+".jpg")
    plt.close()
    return save
    
def test_distribution():
    num_0_test = np.count_nonzero(y_test==0)
    num_1_test = np.count_nonzero(y_test==1)  
    colors = ['royalblue', 'orangered']

    unique2, counts2 = np.unique(y_test, return_counts=True)
    fig = plt.figure()
    plt.bar(unique2, counts2, color= colors)
    plt.title("Testing set")
    plt.xlabel("Classes")
    plt.ylabel("Number of items")
    plt.xticks(unique2)
    legend_labels = [num_0_test, num_1_test]
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in colors]
    plt.legend(legend_handles, legend_labels)
    save=plt.savefig("/Users/francescoaldoventurelli/Desktop/TESI_FINAL/test/test_dist140_6x6_WF_2L"+str(num_train)+"_"+str(nrun+1)+".jpg")
    plt.close()
    return save
    



def QNN_model(embedding_circ, PQC):
    np.random.seed(seed)
    nqubits=6
    num_inputs=256 ### FEATURES
    qc = QuantumCircuit(nqubits)

    # Encoding of initial random guessed parameters
    ### Features are 256, but I used a way of encoding them that embed each qubit (6) with 2 features per layer;
    # so I have 12 features per layer. 12x22 = 264, so I have 264 features as total. 
    # I need to guess 8 of them in a way that 264-8 = 256 the exact number of features.
    # This is done right below.
    features=[] ### features
    for i in range(num_inputs):
        features.append(Parameter('x'+str(i)))
    for i in range(8):
        features.append(np.pi/2)


    feature_map = embedding_circ(qc,features,22)

    # Optimzing circuit PQC
    ### I do the same for the parameterized quantum circuit, but now it has 3 layers, so I need to
    # pass 2 (rotations per qubit) x 6 (nqubits) x 3 (n layers)
    nlayers = 2
    param_y=[]  ### learnable paramteres
    for i in range(nqubits*2*nlayers):
        param_y.append(Parameter('θ'+str(i)))
    ansatz = PQC(qc, param_y, nlayers)

    #θ
    qc.append(feature_map, range(nqubits))
    qc.append(ansatz, range(nqubits))
    ### The model names qnn2 because I've changed the encoding circuit.

    parityyy = lambda x: "{:b}".format(x).count("1") % 2
    qnn = CircuitQNN(qc, input_params=feature_map.parameters, weight_params=ansatz.parameters, 
                  interpret=parityyy, output_shape=2, quantum_instance=qi)
    initial_weights = 0.1*(2*np.random.rand(qnn.num_weights) - 1)
    return qnn, initial_weights



def process(learning_rate):

    ### Create the vectors that feed the QNN
    x_digits = [x_train[i].flatten() for i in range(len(x_train))]
    y01_digits = [y_train[i] for i in range(len(y_train))]
    X_digits_test= [x_test[i].flatten() for i in range(len(x_test))]
    y01_digits_test= [y_test[i] for i in range(len(y_test))] 
    
    ## Set the optimizer and the loss
    #optimizer = optim.Adam(model2.parameters(), lr=learning_rate)


    optimizer = LBFGS(model2.parameters(),lr=learning_rate)

    f_loss = CrossEntropyLoss()
    model2.train()
    accuracy_f = []
    loss_values = []
    def closure():
        
        optimizer.zero_grad()
        loss = 0.0
        accuracy = 0.0
        #y_predict=[]
        for x, y_target in zip(x_digits, y01_digits): 
            output = model2(Tensor(x)).reshape(1, 2)
            loss += f_loss(output, Tensor([y_target]).long())
            correct_prediction = (output.argmax(dim=1) == y_target).item()
            accuracy += correct_prediction
            
        loss.backward()
        #optimizer.step()
        print("Train Loss:", loss.item())
        loss_values.append(loss.item())
        accuracy_f.append(accuracy/len(x_digits))
        print("Train accuray:", accuracy/len(x_digits))
        #print("Train acc:", accuracy/len(x_digits))
        return loss
    
    optimizer.step(closure)
    print("Loss on training:", loss_values)
    print("Accuracy on training:", accuracy_f)

    
    ### Train
    y_predict = []
    for x in x_digits:
        output = model2(Tensor(x))
        y_predict += [np.argmax(output.detach().numpy())]
 
    print("Predicted labels:", y_predict)
    train_acc = np.mean(y_predict == np.array(y01_digits))

    ### Test
    y_predict_test = []
    for x in X_digits_test:
        output_test = model2(Tensor(x))
        y_predict_test += [np.argmax(output_test.detach().numpy())]
    #test_acc = sum(y_predict_test == np.array(y01_digits_test))/len(np.array(y01_digits_test))
    test_acc = np.mean(y_predict_test == np.array(y01_digits_test))
    return train_acc, test_acc, y01_digits, y_predict, y01_digits_test, y_predict_test

#train_accuracy, test_accuracy, y01_train, y_predicted_train, y01_test, y_predicted_test = process(0.07)



def miss_digits_test(xtest, ypredicted):
    nrows=14
    ncols=10
    fig, axes = plt.subplots(nrows=14, ncols=10, figsize=(15, 15))
    fig.suptitle("Test set")
    for i in range(nrows):
        for j in range(ncols):
            index = i * ncols + j
            if index < len(xtest):
                ax = axes[i, j]
                ax.imshow(xtest[index].reshape((16,16)), cmap="gray")
                ax.set_title(f"Label: {ypredicted[index]}")
                ax.set_xticks([])
                ax.set_yticks([])

    # Remove any empty subplots if the number of images is less than 140
    for i in range(len(xtest), nrows * ncols):
        ax = axes.flatten()[i]
        ax.axis("off")
    plt.tight_layout()
    save = plt.savefig("/Users/francescoaldoventurelli/Desktop/TESI_FINAL/missed_images/miss_imgs_test_6x6_WF_2L"+str(num_train)+"_run_"+str(nrun+1)+".jpg")
    plt.close()
    return save

def miss_training(xtrain, ypredicted):
    nrows=5
    ncols=10
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 15))
    fig.suptitle("Train set")

    for i in range(nrows):
        for j in range(ncols):
            index = i * ncols + j
            if index < len(xtrain):
                ax = axes[i, j]
                ax.imshow(xtrain[index].reshape((16,16)), cmap="gray")
                ax.set_title(f"Label: {ypredicted[index]}")
                ax.set_xticks([])
                ax.set_yticks([])
    for i in range(len(xtrain), nrows * ncols):
        ax = axes.flatten()[i]
        ax.axis("off")
    plt.tight_layout()
    save = plt.savefig("/Users/francescoaldoventurelli/Desktop/TESI_FINAL/missed_images/miss_imgs_train__WF_2L"+str(num_train)+"_run_"+str(nrun+1)+".jpg")
    plt.close()
    return save


def count_miss(ytest, ypredicted):
    count = 0
    for i in range(len(ytest)):
        if ytest[i] != ypredicted[i]:
            count +=1
        else:
            count = count
    return count

def miss_distr(ytest, ypredicted):
    num0_missed = 0
    num1_missed = 0
    for i in range(len(ytest)):
        if (ytest[i] == 0 and ypredicted[i] == 1):
            num0_missed +=1
        elif (ytest[i] == 1 and ypredicted[i] == 0):
            num1_missed +=1
        else:
            num0_missed = num0_missed
            num1_missed = num1_missed
    total_miss = [num0_missed, num1_missed]
    return total_miss


### Confusion matrices
def confusion_m_training(true_labels, y_predicted):
    cm = confusion_matrix(true_labels, y_predicted)
    prfs = precision_recall_fscore_support(true_labels, y_predicted)
    print("Precision, recall, f_score, support:", prfs)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("/Users/francescoaldoventurelli/Desktop/TESI_FINAL/train/c_matrix/cm_train_6x6_WF_2L"+str(num_train)+"_run"+str(nrun+1)+".jpg")
    return disp

def confusion_m_testing(true_labels, y_predicted):
    cm = confusion_matrix(true_labels, y_predicted)
    prfs = precision_recall_fscore_support(true_labels, y_predicted)
    print("Precision, recall, f_score, support:", prfs)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("/Users/francescoaldoventurelli/Desktop/TESI_FINAL/test/c_matrix/cm140_6x6_WF_2L"+str(num_train)+"_"+str(nrun+1)+".jpg")
    return disp



### Choose which circuits you want
embedding = real_amp_embedding #### Encoding
#variational_ansatz = real_amp_PQC    #### My circuit as pqc
variational_ansatz = my_ansatz
qnn2, initial_weights = QNN_model(embedding, variational_ansatz)
model2 = TorchConnector(qnn2, initial_weights)
run = [0,1,2,3,4]
seed0 = 0
seed1 = 1
seed2 = 2
seed3 = 3
seed4 = 4

seedsss = [seed0,seed1,seed2,seed3,seed4]
values_train_acc = np.zeros(5)
values_test_acc = np.zeros(5)
missed_images = np.zeros(5)
miss_0s = np.zeros(5)
miss_1s = np.zeros(5)
#total_miss = []
#total_0 = []
#total_1 = []


num_train = 50
num_zero = 25
num_one = 25

for nrun,j in zip(run,seedsss):
    #start = time.time()
    print("Starting run number:", nrun+1)
    print("")

    x_train, y_train, x_test, y_test = mnist_data(num_zero, num_one, myseed=seedsss[j])

    train_accuracy, test_accuracy, y01_train, y_predicted_train, y01_test, y_predicted_test = process(0.07)
    #end = time.time()
    #train_img = train_plot(x_train, y_train)
    #test_img = test_plot(x_test, y_test)

    #train_dist = train_distribution()
    #test_dist = test_distribution()

    #vis_missing_digits = miss_digits_test(x_test, y_predicted_test)
    #vis_missing_train = miss_training(x_train, y_predicted_train)


    # Append the loss_values list for this run to the all_loss_values list

    print("Missed TEST images:", count_miss(y01_test, y_predicted_test))
    print("Number of 0s completely missed:", miss_distr(y01_test, y_predicted_test)[0])
    print("Number of 1s completely missed:", miss_distr(y01_test, y_predicted_test)[1])
    #total_miss.append(count_miss(y01_test, y_predicted_test))
    #total_0.append(miss_distr(y01_test, y_predicted_test)[0])
    #total_1.append(miss_distr(y01_test, y_predicted_test)[1])
    print("Train accuracy: ", train_accuracy)
    print("Test accuracy: ", test_accuracy)
    print("Num train images: ", num_train)
    #print("Num test images: ", 140)
    #print("")
    
    #print("Elapsed time:", end-start)

    cm_train = confusion_m_training(y01_train, y_predicted_train)
    cm_test = confusion_m_testing(y01_test, y_predicted_test)

#print("Train accuracy:", values_train_acc)
#print("Test accuracy", values_test_acc)
'''print("Images completely missed:", missed_images)
print("0 digits completely missed:", miss_0s)
print("1 digits completely missed:", miss_1s)'''

#print("Images completely missed:", total_miss)
#print("0 digits completely missed:", total_0)
#print("1 digits completely missed:", total_1)

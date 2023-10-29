import tensorflow as tf
from tensorflow import keras
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.parameter import Parameter
from qiskit import Aer, QuantumCircuit
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit_machine_learning.connectors import TorchConnector
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
from qiskit.utils import QuantumInstance
from qiskit.quantum_info import state_fidelity
from torch.nn import CrossEntropyLoss, MSELoss, NLLLoss
from torch.optim import LBFGS, Adam
from PIL import Image
from circuits import *
from qiskit.opflow import Z, StateFn
#from circuits import encoding, circuit15
from circuits import my_ansatz
#from qcnn import qcnn
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support


seed = 42
qi = QuantumInstance(Aer.get_backend('aer_simulator'), seed_simulator=seed,seed_transpiler=seed,shots=256)
qi.backend.set_option("seed_simulator", seed)


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
        nogalaxy_list.append(img_array0.flatten())

    for filename in train1:
        file_path1 = os.path.join(path_galaxy, filename)
        img1 = Image.open(file_path1).convert('L')
        img_array1 = np.array(downsample(img1))
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
        nogalaxy_list.append(img_array0.flatten())

    for filename in test1:
        file_path1 = os.path.join(path_galaxy, filename)
        img1 = Image.open(file_path1).convert('L')
        img_array1 = np.array(downsample(img1))
        img_array1 = normalization(img_array1)
        img_array1 = binarization(img_array1)
        galaxy_list.append(img_array1.flatten())

    input_combine = galaxy_list + nogalaxy_list
    labels_combine = labels1 + labels0

    return (input_combine, labels_combine)


#### Save the plots to check the balance

def train_plot(train_imgs, labels):
    num_train = train_imgs
    fig, axes = plt.subplots(nrows=5, ncols=10, figsize=(16, 16))
    fig.suptitle("Train set")

    for i, ax in enumerate(axes.flatten()):
        ax.imshow(train_imgs[i].reshape((16,16)), cmap="gray")
        ax.set_xlabel(labels[i])
        ax.set_xticks([])
        ax.set_yticks([])
    save1 = plt.savefig("/Users/francescoaldoventurelli/Desktop/TESI_FINAL/GALAXY/Train/trainset1WF_QCNN_feat_"+str(num_train)+"run"+str(run)+".jpg")
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
    save1 = plt.savefig("/Users/francescoaldoventurelli/Desktop/TESI_FINAL/GALAXY/Test/testset100_1WF_QCNN_feat_"+str(run)+".jpg")
    plt.close()
    return save1



def QNN_model(embedding_circ, PQC):
    np.random.seed(seed)
    nqubits=8
    num_inputs=256 ### FEATURES
    qc = QuantumCircuit(nqubits)
    nlayers=3
    # Encoding of initial random guessed parameters
    ### Features are 256, but I used a way of encoding them that embed each qubit (6) with 2 features per layer;
    # so I have 12 features per layer. 12x22 = 264, so I have 264 features as total. 
    # I need to guess 8 of them in a way that 264-8 = 256 the exact number of features.
    # This is done right below.
    features=[] ### features
    for i in range(num_inputs):
        features.append(Parameter('x'+str(i)))
    #for i in range(8):
    #    features.append(np.pi/2)


    feature_map = embedding_circ(nqubits,16, features)

    # Optimzing circuit PQC
    ### I do the same for the parameterized quantum circuit, but now it has 3 layers, so I need to
    # pass 2 (rotations per qubit) x 6 (nqubits) x 3 (n layers)
    '''nlayers = 4
    param_y=[]  ### learnable paramteres
    for i in range(nqubits*2*nlayers):
        param_y.append(Parameter('θ'+str(i)))'''

    param_y = []
    #for i in range(nqubits*2*nlayers):
    for i in range(42):
        param_y.append(Parameter('θ'+str(i)))
    '''for i in range(36):
        param_y.append(Parameter('θ'+str(i)))'''
    #ansatz = PQC(qc, param_y, nlayers)
    ansatz = PQC(nqubits,param_y)
    #θ
    qc.append(feature_map, range(nqubits))
    qc.append(ansatz, range(nqubits))

    #qc.measure(range(nqubits), range(nqubits))
    ### The model names qnn2 because I've changed the encoding circuit.

    parityyy = lambda x: "{:b}".format(x).count("1") % 2
    qnn = CircuitQNN(qc, input_params=feature_map.parameters, weight_params=ansatz.parameters, 
                  interpret=parityyy, output_shape=2, quantum_instance=qi)
    initial_weights = 0.1*(2*np.random.rand(qnn.num_weights) - 1)

    return qnn, initial_weights


def process(learning_rate):
    ### Create the vectors that feed the QNN
    galaxy_train = [x_train[i].flatten() for i in range(len(x_train))]
    label_train = [y_train[i] for i in range(len(y_train))]
    galaxy_test= [x_test[i].flatten() for i in range(len(x_test))]
    label_test= [y_test[i] for i in range(len(y_test))] 
    
    ## Set the optimizer and the loss
    #optimizer = optim.Adam(model2.parameters(), lr=learning_rate)

    #optimizer = Adam(model2.parameters(), lr=learning_rate)
    optimizer = LBFGS(model2.parameters(),lr=learning_rate)

    f_loss = CrossEntropyLoss()
    #f_loss = NLLLoss()

    '''def MyLossFunc(data):
        data = np.asarray(data).reshape((12,12))
        n_light_pixels = 0
        for i in range(len(data)):
            for j in range(len(data)):
                if data[i][j] != 0:
                    n_light_pixels = np.count_nonzero(data[i][j])/256
                else:
                    n_light_pixels = n_light_pixels/256
        return n_light_pixels'''

    model2.train()
    accuracy_f = []
    loss_values = []
    def closure():
        optimizer.zero_grad()
        loss = 0.0
        accuracy = 0.0
        for x, y_target in zip(galaxy_train, label_train): 
            output = model2(Tensor(x)).reshape(1, 2)
            #output = model2(Tensor(x))
            loss += f_loss(output, Tensor([y_target]).long())
            #loss += f_loss(output, Tensor([y_target]).float())
            correct_prediction = (output.argmax(dim=1) == y_target).item()
            #correct_prediction = (output.argmax(dim=0) == y_target).item()

            accuracy += correct_prediction
        loss.backward()
        #loss.backward(retain_graph=True)
        #optimizer.step()
        print("Train Loss:", loss.item())
        loss_values.append(loss.item())
        accuracy_f.append(accuracy/len(galaxy_train))
        print("Train accuray:", accuracy/len(galaxy_train))

        return loss
    

    '''epochs = 20
    for epoch in range(epochs):
        optimizer.step(closure)
        print("Loss on training:", loss_values[epoch])
        print("Accuracy on training:", accuracy_f[epoch])'''
    
    optimizer.step(closure)
    print("Loss on training:", loss_values)
    print("Accuracy on training:", accuracy_f)

    
    ### Train
    y_predict = []
    for x in galaxy_train:
        output = model2(Tensor(x))
        y_predict += [np.argmax(output.detach().numpy())]
 
    print("Predicted labels:", y_predict)
    train_acc = np.mean(y_predict == np.array(label_train))

    ### Test
    y_predict_test = []
    for x in galaxy_test:
        output_test = model2(Tensor(x))
        y_predict_test += [np.argmax(output_test.detach().numpy())]
    #test_acc = sum(y_predict_test == np.array(y01_digits_test))/len(np.array(y01_digits_test))
    test_acc = np.mean(y_predict_test == np.array(label_test))
    return train_acc, test_acc, label_train, y_predict, label_test, y_predict_test



def miss_digits_test(xtest, ypredicted):
    nrows=10
    ncols=10
    fig, axes = plt.subplots(nrows=10, ncols=10, figsize=(15, 15))
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
    save = plt.savefig("/Users/francescoaldoventurelli/Desktop/TESI_FINAL/GALAXY/Test/missed_images/te16_WF_QCNN_feat_"+str(num_train)+"_run_"+str(nrun+1)+".jpg")
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
                ax = axes[i,j]
                ax.imshow(xtrain[index].reshape((16,16)), cmap="gray")
                ax.set_title(f"Label: {ypredicted[index]}")
                ax.set_xticks([])
                ax.set_yticks([])
    for i in range(len(xtrain), nrows * ncols):
        ax = axes.flatten()[i]
        ax.axis("off")
    plt.tight_layout()
    save = plt.savefig("/Users/francescoaldoventurelli/Desktop/TESI_FINAL/GALAXY/Train/missed_images/tr1WF_QCNN_feat_"+str(num_train)+"_run_"+str(nrun+1)+".jpg")
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
    disp.plot(cmap='Blues_r')
    plt.savefig("/Users/francescoaldoventurelli/Desktop/TESI_FINAL/GALAXY/Train/cm_train/cm_1WF_QCNN_feat_"+str(num_train)+"_run"+str(nrun+1)+".jpg")
    return disp

def confusion_m_testing(true_labels, y_predicted):
    cm = confusion_matrix(true_labels, y_predicted)
    prfs = precision_recall_fscore_support(true_labels, y_predicted)
    print("Precision, recall, f_score, support:", prfs)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues_r')
    plt.savefig("/Users/francescoaldoventurelli/Desktop/TESI_FINAL/GALAXY/Test/cm_test/cm_1WF_QCNN_feat_"+str(num_train)+"_"+str(nrun+1)+".jpg")
    return disp




def qcnn(nqbits, theta):
    count = 0
    qc = QuantumCircuit(nqbits)
    
    for i in range(nqbits):
        qc.ry(theta[count],i)
        count +=1
    for i in range(0,nqbits-1,2):
        qc.cnot(i,i+1)
        qc.rz(theta[count],i+1)
        qc.cnot(i,i+1)
        count +=1
    for i in range(nqbits):
        qc.ry(theta[count],i)
        count +=1
    for i in range(0, nqbits, 2):
        qc.crz(theta[count],i+1, i)
        count +=1
    
    qc.barrier()

    for i in range(0,nqbits,2):
        qc.ry(theta[count],i)
        count +=1
    for i in range(0,nqbits-2, 4):
        qc.cnot(i,i+2)
        qc.rz(theta[count],i+2)
        qc.cnot(i,i+2)
        count +=1
    for i in range(0,nqbits-1,2):
        qc.ry(theta[count],i)
        count +=1
    for i in range(0, nqbits, 4):
        qc.crz(theta[count],i+2, i)
        count +=1
    qc.barrier()

    for i in range(0,nqbits,4):
        qc.ry(theta[count],i)
        count +=1
    for i in range(0,nqbits-4,4):
        qc.cnot(i,i+4)
        qc.rz(theta[count],i+4)
        qc.cnot(i,i+4)

        count +=1
    for i in range(0,nqbits-4,4):
        qc.ry(theta[count], i)
        count += 1
        qc.ry(theta[count], 4)  # This line doesn't increment count; it might cause issues
        count +=1
        qc.crz(theta[count], 4, 0)
        
    qc.barrier()
    return qc


def real_amp_embedding2(qbits,nlayers,theta):
    qc = QuantumCircuit(qbits)
    
    count=0
    for i in range(qbits):
        qc.h(i)
    

    for layer in range(nlayers):
        for i in range(qbits):
            qc.ry(theta[count],i)
            count +=1
        
        for j in range(qbits-1):
            for i in range(j+1, qbits):
                qc.cx(j,i)
        for i in range(qbits):
            qc.rz(theta[count],i)
            count += 1
        qc.barrier()

    #qc.measure(qr[0], cr)
    #qc.to_instruction()
    return qc

def embedding_RL(qbits, nlayers, theta):
    qc = QuantumCircuit(qbits)
    count=0
    for i in range(qbits):
        qc.h(i)
    
    for l in range(nlayers):
        for i in range(qbits):
            qc.ry(theta[count],i)
            count=count+1
        for i in range(qbits-1):
            qc.cx(i,i+1)
        
        qc.cx(qbits-1,0)
        qc.barrier()
        for i in range(qbits):
            qc.ry(theta[count],i)
            count=count+1
        
        for i in range(qbits-1):
            qc.cx(i+1,i)
        qc.cx(0,qbits-1)
        qc.barrier()
    return qc



### Choose which circuits you want
embedding = real_amp_embedding2  #### Encoding
#variational_ansatz = real_amp_PQC    #### My circuit as pqc
variational_ansatz = qcnn
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
'''missed_images = np.zeros(5)
miss_0s = np.zeros(5)
miss_1s = np.zeros(5)
total_miss = []
total_0 = []
total_1 = []'''


num_train = 50
num_zero = 25
num_one = 25

for nrun,j in zip(run,seedsss):
    #start = time.time()
    print("Starting run number:", nrun+1)
    print("")

    x_train, y_train = make_train(num_zero, num_one, seedsss[j])
    x_test, y_test = make_test()
    train_accuracy, test_accuracy, y01_train, y_predicted_train, y01_test, y_predicted_test = process(0.07)
    vis_missing_digits = miss_digits_test(x_test, y_predicted_test)
    vis_missing_train = miss_training(x_train, y_predicted_train)


    # Append the loss_values list for this run to the all_loss_values list

    print("Missed TEST images:", count_miss(y01_test, y_predicted_test))
    print("Number of 0s completely missed:", miss_distr(y01_test, y_predicted_test)[0])
    print("Number of 1s completely missed:", miss_distr(y01_test, y_predicted_test)[1])
    '''total_miss.append(count_miss(y01_test, y_predicted_test))
    total_0.append(miss_distr(y01_test, y_predicted_test)[0])
    total_1.append(miss_distr(y01_test, y_predicted_test)[1])'''
    print("Train accuracy: ", train_accuracy)
    print("Test accuracy: ", test_accuracy)
    print("Num train images: ", num_train)
    print("Num test images: ", 100)
    print("")
    
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

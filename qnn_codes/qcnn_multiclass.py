from qiskit import *
import numpy as np
import matplotlib.pyplot as plt
from qiskit.utils import QuantumInstance
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from qiskit.circuit.parameter import Parameter
from circuits import *
from qiskit_machine_learning.neural_networks import CircuitQNN, SamplerQNN
import tensorflow as tf
from tensorflow import keras
from keras.datasets.mnist import load_data

from torch import Tensor
from qiskit import Aer, QuantumCircuit

from qiskit_machine_learning.connectors import TorchConnector
from torch.nn import CrossEntropyLoss
from torch.optim import LBFGS, Adam


seed = 42
qi = QuantumInstance(Aer.get_backend('aer_simulator'), seed_simulator=seed,seed_transpiler=seed,shots=256)
qi.backend.set_option("seed_simulator", seed)


def mnist_data(zerotrain, onetrain, twotrain, threetrain, myseed):
    np.random.seed(myseed)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    train_filter = np.where((y_train == 0) | (y_train == 1) | (y_train == 2) | (y_train == 3))
    
    X_train, Y_train = x_train[train_filter], y_train[train_filter]
    
    # ... (rest of your data selection logic)
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

    two_index_train = np.where(target_new == 2)[0]
    three_index_train = np.where(target_new == 3)[0]
    two_indices_test = np.where(y_test == 2)[0]
    three_indices_test = np.where(y_test == 3)[0]
    four_index_train = np.where(target_new == 4)[0]
    five_index_train = np.where(target_new == 5)[0]
    four_indices_test = np.where(y_test == 4)[0]
    five_indices_test = np.where(y_test == 5)[0]

    # Select the desired number of indices for 0s and 1s
    selected_zero_indices = zero_indices_test[:35] ### come vedi il test set è fisso a 140 immagini
    selected_one_indices = one_indices_test[:35]
    selected_two_indices = two_indices_test[:35] ### come vedi il test set è fisso a 140 immagini
    selected_three_indices = three_indices_test[:35]

    selected_zero_train = zero_index_train[:zerotrain]
    selected_one_train = one_index_train[:onetrain]
    selected_two_train = two_index_train[:twotrain]
    selected_three_train = three_index_train[:threetrain]
    selected_indices = np.concatenate([selected_zero_indices, selected_one_indices, selected_two_indices, selected_three_indices])
    sel_train_indices = np.concatenate([selected_zero_train, selected_one_train, selected_two_train, selected_three_train])
    X_test = x_test[selected_indices]
    Y_test = y_test[selected_indices]
    
    # ... (rest of your data preprocessing)

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




'''fig, axes = plt.subplots(nrows=4, ncols=10, figsize=(16, 16))
fig.suptitle("Train set")

for i, ax in enumerate(axes.flatten()):
        ax.imshow(x_train[i].reshape((16, 16)), cmap="gray")
        ax.set_xlabel(y_train[i])
        ax.set_xticks([])
        ax.set_yticks([])
plt.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0)
plt.show()'''


def embedding(qbits, nlayers, theta):
    qc = QuantumCircuit(qbits)
    count = 0
    for i in range(qbits):
        qc.h(i)
    for layer in range(nlayers):
        for i in range(qbits):
            qc.ry(theta[count], i)
            count +=1
        for i in range(qbits-1):
            qc.cnot(i,i+1)
        for i in range(qbits):
            qc.rz(theta[count], i)
            count +=1
        qc.barrier()

    return qc


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
    qc.measure(0,0)

    return qc



def real_amp_embedding(nqubits,theta,layers,cbit):

    #cr = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(nqubits, cbit)
    
    count=0
    for i in range(nqubits):
        qc.h(i)
    

    for layer in range(layers):
        for i in range(nqubits):
            qc.ry(theta[count],i)
            count +=1
        
        for j in range(nqubits-1):
            for i in range(j+1, nqubits):
                qc.cx(j,i)
        for i in range(nqubits):
            qc.rz(theta[count],i)
            count += 1
        qc.barrier()

    #qc.measure(qr[0], cr)
    qc.to_instruction()
    return qc

def hardware_circ(nqubits, theta, layers):
    qc = QuantumCircuit(nqubits)
    #cr = ClassicalRegister(1, 'c')
    #qc = QuantumCircuit(qr, name="Ansatz")

    count=0
    for layer in range(layers):
        for i in range(nqubits):
            qc.ry(theta[count], i)
            count +=1
            qc.rz(theta[count], i)
        qc.barrier()
        for i in range(nqubits-1):
            qc.cz(i,i+1)
        qc.cz(0,-1)
        qc.barrier()
    #qc.to_instruction()
    #qc.measure(0,0)
    return qc


'''param_x=[] ### features
for i in range(42):
    param_x.append(Parameter('x'+str(i)))

qqq = qcnn(8,param_x)
qqq.draw('mpl')
plt.show()'''

def encode_labels(label, num_classes):
    encoded = [0] * num_classes
    encoded[label] = 1
    return encoded



def QNN_model(feature_map, PQC, num_classes):
    np.random.seed(seed)
    cbit = 1
    nqubits=6
    num_inputs=256 ### FEATURES
    #qc = QuantumCircuit(nqubits,cbit)
    qc2 = QuantumCircuit(nqubits)
    
    
    def multi_class_parity(x):
        num_ones = "{:b}".format(x).count("1")   #### FUNZIONA!!!
        return num_ones % 4
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

    emb_circ = feature_map(nqubits, 22,features)
    #emb_circ = feature_map(qc2, features,22)
    #feature_map = embedding(8,16,features)
    num_layers = 3
    # Optimzing circuit PQC
    ### I do the same for the parameterized quantum circuit, but now it has 3 layers, so I need to
    # pass 2 (rotations per qubit) x 6 (nqubits) x 3 (n layers)
    param_y=[]  ### learnable paramteres
    for i in range(2*nqubits*num_layers):
        param_y.append(Parameter('θ'+str(i)))
    #ansatz = PQC(6, param_y, num_layers)
    ansatz = PQC(nqubits, param_y, num_layers)
    #θ
    qc2.compose(emb_circ, inplace=True)
    qc2.compose(ansatz, inplace=True)
    qc2.add_register(ClassicalRegister(2))
    qc2.measure(0,0)
    qc2.measure(5,1)
    #qc.append(emb_circ, range(8,1))
    #qc.append(ansatz, range(8,1))
    ### The model names qnn2 because I've changed the encoding circuit.

    #parityyy = lambda x: "{:b}".format(x).count("1") % 2
    interpret=lambda x: x
    #interpreter = interpret
    '''qnn = CircuitQNN(qc2, input_params=emb_circ.parameters, weight_params=ansatz.parameters, 
                  interpret=parityyy, output_shape=num_classes, quantum_instance=qi)
    initial_weights = 0.1*(2*np.random.rand(qnn.num_weights) - 1)'''
    qnn = SamplerQNN(circuit=qc2, input_params=emb_circ.parameters, weight_params=ansatz.parameters, 
                  interpret=interpret, output_shape=num_classes)
    initial_weights = 0.1*(2*np.random.rand(qnn.num_weights) - 1)

    backward_output = qnn.backward(input, ansatz.parameters)
    print(backward_output)
    return qnn, initial_weights


def process(learning_rate, num_classes):
    ### Create the vectors that feed the QNN
    
    x_digits = [x_train[i].flatten() for i in range(len(x_train))]
    y01_digits = [y_train[i] for i in range(len(y_train))]
    X_digits_test= [x_test[i].flatten() for i in range(len(x_test))]
    y01_digits_test= [y_test[i] for i in range(len(y_test))] 

    #y_train_one_hot = tf.keras.utils.to_categorical(y_train, num_classes)
    #y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes)
    
    ## Set the optimizer and the loss
    #optimizer = optim.Adam(model2.parameters(), lr=learning_rate)

    #optimizer = Adam(model2.parameters(),lr=learning_rate)
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
        for x, y_target in zip(x_digits, y_train): 
            y_target_one_hot = np.array(tf.keras.utils.to_categorical(y_target, num_classes))
            output = model2(Tensor(x)).reshape(1, 4)
            #loss += f_loss(output, Tensor([y_target]).long())
            loss += f_loss(output, Tensor(np.array([y_target_one_hot])).float())
            #correct_prediction = (output.argmax(dim=1) == y_target).item()
            correct_prediction = (output.argmax(dim=1) == y_target_one_hot.argmax()).item()
            accuracy += correct_prediction
            
        loss.backward()
        #optimizer.step()
        print("Train Loss:", loss.item())
        loss_values.append(loss.item())
        accuracy_f.append(accuracy/len(x_digits))
        print("Train accuray:", accuracy/len(x_digits))
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
    save = plt.savefig("/Users/francescoaldoventurelli/Desktop/TESI_FINAL/missed_images/miss_imgs_test"+str(num_train)+"_run_"+str(nrun+1)+"QCNN_multiclass_MEASURE.jpg")
    plt.close()
    return save

def miss_training(xtrain, ypredicted):
    nrows=2
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
    save = plt.savefig("/Users/francescoaldoventurelli/Desktop/TESI_FINAL/missed_images/miss_imgs_train"+str(num_train)+"_run_"+str(nrun+1)+"QCNN_multiclass_MEASURE.jpg")
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
    num2_missed = 0
    num3_missed = 0
    for i in range(len(ytest)):
        if (ytest[i] == 0 and ypredicted[i] == 1):
            num0_missed +=1
        elif (ytest[i] == 1 and ypredicted[i] == 0):
            num1_missed +=1
        elif (ytest[i] == 0 and ypredicted[i] == 2):
            num0_missed +=1
        elif (ytest[i] == 2 and ypredicted[i] == 0):
            num2_missed +=1
        elif (ytest[i] == 0 and ypredicted[i] == 3):
            num0_missed +=1
        elif (ytest[i] == 3 and ypredicted[i] == 0):
            num3_missed +=1
        elif (ytest[i] == 1 and ypredicted[i] == 2):
            num1_missed +=1
        elif (ytest[i] == 2 and ypredicted[i] == 1):
            num2_missed +=1
        elif (ytest[i] == 1 and ypredicted[i] == 3):
            num1_missed +=1
        elif (ytest[i] == 3 and ypredicted[i] == 1):
            num3_missed +=1
        elif (ytest[i] == 3 and ypredicted[i] == 2):
            num3_missed +=1
        else:
            num0_missed = num0_missed
            num1_missed = num1_missed
            num2_missed = num2_missed
            num3_missed = num3_missed
    total_miss = [num0_missed, num1_missed, num2_missed, num3_missed]
    return total_miss


### Confusion matrices
def confusion_m_training(true_labels, y_predicted):
    cm = confusion_matrix(true_labels, y_predicted)
    prfs = precision_recall_fscore_support(true_labels, y_predicted)
    print("Precision, recall, f_score, support:", prfs)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("/Users/francescoaldoventurelli/Desktop/TESI_FINAL/train/c_matrix/cm_train"+str(num_train)+"_run"+str(nrun+1)+"QCNN_multiclass_MEASURE.jpg")
    return disp

def confusion_m_testing(true_labels, y_predicted):
    cm = confusion_matrix(true_labels, y_predicted)
    prfs = precision_recall_fscore_support(true_labels, y_predicted)
    print("Precision, recall, f_score, support:", prfs)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("/Users/francescoaldoventurelli/Desktop/TESI_FINAL/test/c_matrix/cm140_"+str(num_train)+"_"+str(nrun+1)+"QCNN_multiclass_MEASURE.jpg")
    return disp


### Choose which circuits you want
embed = embedding #### My circuit as embedding
#variational_ansatz = real_amp_PQC    #### My circuit as pqc
variational_ansatz = hardware_circ
qnn2, initial_weights = QNN_model(embed, variational_ansatz, 4)
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
#miss_0s = np.zeros(5)
#miss_1s = np.zeros(5)
total_miss = []
total_0 = []
total_1 = []
total_2 = []
total_3 = []

num_train = 80
num_zero = 20
num_one = 20
num_two = 20
num_three = 20

for nrun,j in zip(run,seedsss):
    print("Starting run number:", nrun+1)
    print("")

    x_train, y_train, x_test, y_test = mnist_data(num_zero, num_one, num_two, num_three, myseed=seedsss[j])

    train_accuracy, test_accuracy, y01_train, y_predicted_train, y01_test, y_predicted_test = process(0.08, 4)
    
    vis_missing_digits = miss_digits_test(x_test, y_predicted_test)
    vis_missing_train = miss_training(x_train, y_predicted_train)

    # Append the loss_values list for this run to the all_loss_values list

    print("Missed TEST images:", count_miss(y01_test, y_predicted_test))
    print("Number of 0s completely missed:", miss_distr(y01_test, y_predicted_test)[0])
    print("Number of 1s completely missed:", miss_distr(y01_test, y_predicted_test)[1])
    total_miss.append(count_miss(y01_test, y_predicted_test))
    total_0.append(miss_distr(y01_test, y_predicted_test)[0])
    total_1.append(miss_distr(y01_test, y_predicted_test)[1])
    total_2.append(miss_distr(y01_test, y_predicted_test)[2])
    total_3.append(miss_distr(y01_test, y_predicted_test)[3])
    print("Train accuracy: ", train_accuracy)
    print("Test accuracy: ", test_accuracy)
    print("Num train images: ", num_train)
    print("Num test images: ", 140)
    print("")

    cm_train = confusion_m_training(y01_train, y_predicted_train)
    cm_test = confusion_m_testing(y01_test, y_predicted_test)

#print("Train accuracy:", values_train_acc)
#print("Test accuracy", values_test_acc)


print("Images completely missed:", total_miss)
print("0 digits completely missed:", total_0)
print("1 digits completely missed:", total_1)
print("2 digits completely missed:", total_2)
print("3 digits completely missed:", total_3)
from qiskit import *
import numpy as np
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from qiskit.algorithms.optimizers import COBYLA
from qiskit.opflow import AerPauliExpectation
from qiskit.utils import QuantumInstance
from qiskit.utils import algorithm_globals
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from qiskit.circuit.parameter import Parameter
from qiskit_machine_learning.circuit.library import RawFeatureVector
from qiskit_machine_learning.neural_networks import CircuitQNN, SamplerQNN,TwoLayerQNN
from qiskit_machine_learning.algorithms import NeuralNetworkClassifier
import tensorflow as tf
from tensorflow import keras
from keras.datasets.mnist import load_data
from qiskit.circuit.library import ZZFeatureMap
from torch import Tensor
from qiskit import Aer, QuantumCircuit
from torch.optim import Adam
from qiskit_machine_learning.connectors import TorchConnector
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import LBFGS
from circuits import encoding

seed = 42
qi = QuantumInstance(Aer.get_backend('aer_simulator'), seed_simulator=seed,seed_transpiler=seed,shots=256)
qi.backend.set_option("seed_simulator", seed)



def mnist_data(zerotrain, onetrain, myseed):
    np.random.seed(myseed)
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    train_filter = np.where((y_train == 0 ) | (y_train == 1))
    #test_filter = np.where((y_test == 0) | (y_test == 1))

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
    #qc.measure(qr, cr)
    #qc.measure(qr[0],[0])

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
    #qc.measure(0,0)

    return qc


'''features=[] ### features
for i in range(32):
    features.append(Parameter('x'+str(i)))
    

#for i in range(8):
#    features.append(np.pi/2)

qqq = embedding_RL(8,2,features)
qqq.draw('mpl')
plt.show()'''
'''param_x=[] ### features
for i in range(8):
    param_x.append(Parameter('x'+str(i)))

qqq = qcnn(8,param_x)
qqq.draw('mpl')
plt.title('Qcnn ansatz')
plt.savefig('/Users/francescoaldoventurelli/Desktop/workfolder/FIGUREXPRESENTATION/qcnn_qiskit_ansatz.jpg',dpi=300)
plt.show()

print(len(qqq.parameters))'''



def qcnn222(nqbits,theta):
    count = 0
    #cr = ClassicalRegister(nqbits)
    qc = QuantumCircuit(nqbits)
    #cr = ClassicalRegister(8)
    for i in range(nqbits):
        qc.ry(theta[count],i)
        count +=1
    for i in range(0,nqbits-1,2):
        qc.cnot(i,i+1)
        qc.rz(theta[count],i+1)
        qc.cnot(i,i+1)
        count +=1
        qc.cnot(i,i+1)
        qc.ry(theta[count],i+1)
        qc.cnot(i,i+1)
        count +=1
    for i in range(nqbits):
        qc.ry(theta[count],i)
        count +=1
    for i in range(0, nqbits, 2):
        qc.crz(theta[count],i+1, i)
        count +=1
        qc.x(i)
    #qc.measure([1,3,5,7], [1,2,3,4])
    qc.barrier()
    

    for i in range(0,nqbits,2):
        qc.ry(theta[count],i)
        count +=1
    for i in range(0,nqbits-2, 4):
        qc.cnot(i,i+2)
        qc.rz(theta[count],i+2)
        qc.cnot(i,i+2)
        count +=1
        qc.cnot(i,i+2)
        qc.ry(theta[count],i+2)
        qc.cnot(i,i+2)
        count +=1
    for i in range(0,nqbits-1,2):
        qc.ry(theta[count],i)
        count +=1
        qc.x(i)
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
        qc.cnot(i,i+4)
        qc.ry(theta[count],i+4)
        qc.cnot(i,i+4)

        count +=1
    for i in range(0,nqbits-4,4):
        qc.ry(theta[count], i)
        count += 1
        qc.ry(theta[count], 4)  # This line doesn't increment count; it might cause issues
        count +=1
        qc.crz(theta[count], 4, 0)
    qc.t(i)
    qc.barrier()
    #qc.measure(0,0)

    return qc



def QNN_model(feature_map, PQC):
    np.random.seed(seed)
    qbits=8
    num_inputs=256 ### FEATURES
    qc = QuantumCircuit(qbits)

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

    #feature_map = amplitude_embedding(256)
    #feature_map = feature_map(8,16,features)
    feature_map = feature_map(qbits,16,features)

    # Optimzing circuit PQC
    ### I do the same for the parameterized quantum circuit, but now it has 3 layers, so I need to
    # pass 2 (rotations per qubit) x 6 (qbits) x 3 (n layers)
    param_y=[]  ### learnable paramteres
    #for i in range(49):  se usi qcnn222
    for i in range(42):
        param_y.append(Parameter('θ'+str(i)))
    ansatz = PQC(qbits, param_y)

    #θ
    qc.append(feature_map, range(qbits))
    qc.append(ansatz, range(qbits))
    qc.add_register(ClassicalRegister(1))
    qc.measure(0,0)
    '''measured_param = list(param_y)
    measure_result = []
    for i in range(len(measured_param)):
        print(measured_param[i])
        measure_result.append(measured_param[i])
    print(type(measure_result))
    m_param = measure_result'''
    #measured_param = measured_param[Parameter('θ'+str(41))]
    #measure_result = qc.measure(0,0)
    
    #qc.measure(0,0)
    ### The model names qnn2 because I've changed the encoding circuit.
    
    parityyy = lambda x: "{:b}".format(x).count("1") % 2
    def parity2(x):
        return x
    
    
    def binary(x):  ### questa l'aveva fatta il tipo
        return ('0'*(8-len('{:b}'.format(x, '#010b') ))+'{:b}'.format(x, '#010b'))
    
    def firsttwo(x):
        return x[0]  ### qui gli dico di prendere la prima, quella che secondo me è legata al qubit 0
    parity = lambda x: firsttwo(binary(x)).count('1') % 2   
    #print(parityyy)
    #qnn = CircuitQNN(qc, input_params=feature_map.parameters, weight_params=ansatz.parameters, 
                #interpret=parityyy, output_shape=2, quantum_instance=qi)
    #print(qnn.probabilities)
    
    
    #sampler_qnn_input = algorithm_globals.random.random(qnn.num_inputs)
    #sampler_qnn_weights = algorithm_globals.random.random(qnn.num_weights)
    #input = np.random.random(len(qc.input_parameters))
    #weights = np.random.random(len(qc.weight_parameters))
    #forward_output = qnn.forward(input, weights)
    #print(forward_output)
    #sampler_qnn_forward2 = qnn.forward(sampler_qnn_input, sampler_qnn_weights)
    #sampler_qnn_input_grad2, sampler_qnn_weight_grad2 = qnn.backward(
    #sampler_qnn_input, sampler_qnn_weights
    #)
    '''backend_sim = Aer.get_backend('aer_simulator')
    job_sim = execute(qc, backend_sim, shots=128)
    result_sim = job_sim.result()
    counts = result_sim.get_counts(qc)
    print(counts)
    results_of_counts = []
    for key in counts.keys():
            results_of_counts.append(counts[key]/128)
    #plot_histogram(counts)
    qnn = SamplerQNN(circuit=qc, input_params=feature_map.parameters, weight_params=results_of_counts, interpret=parityyy, output_shape=2, input_gradients=True)'''
    

    '''print("Forward output for SamplerQNN2:", sampler_qnn_forward2)
    print("Backward output for SamplerQNN1:", sampler_qnn_input_grad2)
    print("Backward output for SamplerQNN2:", sampler_qnn_weight_grad2)'''
    #print(ansatz.parameters)
    #print("Lenght", len(ansatz.parameters))
   
    qnn = SamplerQNN(circuit=qc, input_params=feature_map.parameters, weight_params=ansatz.parameters, interpret=parity2, output_shape=2, input_gradients=True)
    #return initial_weights
    #initial_weights = 0.1*(2*np.random.rand(qnn.num_weights)-1)
    
    #input = np.random.random(len(feature_map.parameters))
    weights = np.random.random(len(ansatz.parameters))
    #forward_output = qnn.forward(input, weights)
    #print(forward_output)

    #backward_output = qnn.backward(input, weights)
    #print(backward_output)
    return qnn, weights




def process(learning_rate):
    ### Create the vectors that feed the QNN
    x_digits = [x_train[i].flatten() for i in range(len(x_train))]
    y01_digits = [y_train[i] for i in range(len(y_train))]
    X_digits_test= [x_test[i].flatten() for i in range(len(x_test))]
    y01_digits_test= [y_test[i] for i in range(len(y_test))] 
    
    ## Set the optimizer and the loss
    #optimizer = Adam(model2.parameters(), lr=learning_rate)

    print(model2.parameters())
    optimizer = LBFGS(model2.parameters(),lr=learning_rate)

    f_loss = CrossEntropyLoss()
    #optimizer = COBYLA(maxiter=20, tol=0.01, disp=False)
    #f_loss = MSELoss()
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
        print("Train Loss:", loss.item())
        loss_values.append(loss.item())
        accuracy_f.append(accuracy/len(x_digits))
        print("Train accuray:", accuracy/len(x_digits))
        #accuracy_f.append(correct_prediction)
        #print(correct_prediction)

        #print("Train acc:", accuracy/len(x_digits))
        return loss
    '''for epoch in range(20):
        optimizer.step(closure)
        #optimizer.step(closure)
        print("Loss on training:", loss_values)
        print("Accuracy on training:", accuracy_f)'''
    optimizer.step(closure)
    
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
    save = plt.savefig("/Users/francescoaldoventurelli/Desktop/TESI_FINAL/missed_images/miss_imgs_test_RL_MEASURE1"+str(num_train)+"_run_"+str(nrun+1)+"QCNN22.jpg")
    plt.close()
    return save

def miss_training(xtrain, ypredicted):
    nrows=2
    ncols=5
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
    save = plt.savefig("/Users/francescoaldoventurelli/Desktop/TESI_FINAL/missed_images/miss_imgs_train_RL_MEASURE1"+str(num_train)+"_run_"+str(nrun+1)+"QCNN22.jpg")
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
    plt.savefig("/Users/francescoaldoventurelli/Desktop/TESI_FINAL/train/c_matrix/cm_train_RL_MEASURE1"+str(num_train)+"_run"+str(nrun+1)+"QCNN22.jpg")
    return disp

def confusion_m_testing(true_labels, y_predicted):
    cm = confusion_matrix(true_labels, y_predicted)
    prfs = precision_recall_fscore_support(true_labels, y_predicted)
    print("Precision, recall, f_score, support:", prfs)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig("/Users/francescoaldoventurelli/Desktop/TESI_FINAL/test/c_matrix/cm140_RL_MEASURE1"+str(num_train)+"_"+str(nrun+1)+"QCNN22.jpg")
    return disp



### Choose which circuits you want
embed = embedding_RL#### My circuit as embedding
#variational_ansatz = real_amp_PQC    #### My circuit as pqc
variational_ansatz = qcnn
qnn2, initial_weights = QNN_model(embed, variational_ansatz)
#initial_weights = QNN_model(embed, variational_ansatz)
#qnn_2 = TwoLayerQNN(num_qubits=8,feature_map=embedding_RL,ansatz=qcnn,exp_val=AerPauliExpectation(), quantum_instance=qi)
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
total_miss = []
total_0 = []
total_1 = []


num_train = 20
num_zero = 10
num_one = 10
for nrun,j in zip(run,seedsss):
    print("Starting run number:", nrun+1)
    print("")

    x_train, y_train, x_test, y_test = mnist_data(num_zero, num_one, myseed=seedsss[j])

    train_accuracy, test_accuracy, y01_train, y_predicted_train, y01_test, y_predicted_test = process(0.07)
    
    vis_missing_digits = miss_digits_test(x_test, y_predicted_test)
    vis_missing_train = miss_training(x_train, y_predicted_train)


    # Append the loss_values list for this run to the all_loss_values list

    print("Missed TEST images:", count_miss(y01_test, y_predicted_test))
    print("Number of 0s completely missed:", miss_distr(y01_test, y_predicted_test)[0])
    print("Number of 1s completely missed:", miss_distr(y01_test, y_predicted_test)[1])
    total_miss.append(count_miss(y01_test, y_predicted_test))
    total_0.append(miss_distr(y01_test, y_predicted_test)[0])
    total_1.append(miss_distr(y01_test, y_predicted_test)[1])
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

    
    
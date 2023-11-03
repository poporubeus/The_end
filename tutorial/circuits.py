import qiskit
#from qiskit import transpile, assemble
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, transpile
from qiskit import BasicAer, Aer, execute
#from qiskit.quantum_info import state_fidelity
from qiskit.visualization import *
from qiskit.quantum_info.operators import Operator
import numpy as np
import matplotlib.pyplot as plt
from qiskit.circuit.parameter import Parameter
import qiskit
from qiskit.visualization import *
from qiskit.circuit.parameter import Parameter
from qiskit_machine_learning.neural_networks import CircuitQNN
from qiskit.utils import QuantumInstance
#qi = QuantumInstance(Aer.get_backend('statevector_simulator'))
qi = QuantumInstance(Aer.get_backend('aer_simulator'), shots=256) 

### I work with hand digits made by 64 features (16x16) so I change the architecture

nqubits=6
def circuit15(qc, theta, layers):

    qr = QuantumRegister(nqubits)
    qc = QuantumCircuit(qr, name="PQC")
    
    count=0
    for layer in range(layers):
        for i in range(nqubits):
            qc.ry(theta[count],i)
            count=count+1
        qc.barrier()
        for i in range(nqubits-1):
            qc.cx(i,i+1)
        qc.cx(0,nqubits-1)
        qc.barrier()
        for i in range(nqubits):
            qc.ry(theta[count],i)
            count=count+1   
        qc.barrier() 
        for i in range(nqubits-1):
            qc.cx(i+1,i)
        qc.cx(nqubits-1,0)
    #qc.measure(qr[0],[0])
    qc.to_instruction()
    
    return qc

'''simulator = Aer.get_backend('statevector_simulator')
compiled_circuit = transpile(qc, simulator)
job = simulator.run(assemble(compiled_circuit))

# Get the resulting quantum state
result = job.result()
quantum_state = result.get_statevector()'''

def encoding(qc,theta,layers):

    qr = QuantumRegister(nqubits)
    #cr = ClassicalRegister(1)  # Add a classical register
    qc = QuantumCircuit(qr,name='Embed')
    count=0
    for i in range(nqubits):
        qc.h(i)
    
    for l in range(layers):
        for i in range(nqubits):
            qc.ry(theta[count],i)
            count=count+1
        for i in range(nqubits-1):
            qc.cx(i,i+1)
        
        qc.cx(nqubits-1,0)

        for i in range(nqubits):
            qc.ry(theta[count],i)
            count=count+1
        
        for i in range(nqubits-1):
            qc.cx(i+1,i)
        qc.cx(0,nqubits-1)
        qc.barrier()
    #qc.measure(qr, cr)
    #qc.measure(qr[0],[0])
        
    qc.to_instruction()
    

    return qc



def real_amp_embedding(qc,theta,layers):

    qr = QuantumRegister(nqubits)
    #cr = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(qr, name="Embedding")
    
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


def real_amp_PQC(qc, theta):

    qr = QuantumRegister(nqubits)
    #cr = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(qr, name="PQC")

    count=0
    '''for i in range(nqubits):
        #qc.h(i)
        qc.ry(theta[count], i)
        count +=1
    qc.barrier()
    for i in range(nqubits-1):
        qc.cx(0,i+1)
    for i in range(1,nqubits-1):
        qc.cx(1,i+1)
    for i in range(2, nqubits-1):
        qc.cx(2, i+1)
    for i in range(3,nqubits-1):
        qc.cx(3,i+1)
    for i in range(4,nqubits-1):
        qc.cx(4,i+1)'''
    #qc.cx(5,6)

    for i in range(nqubits):
        #qc.h(i)
        qc.ry(theta[count], i)
        count +=1
    qc.barrier()
    for j in range(1,nqubits-1):
        for i in range(j+1, nqubits):
            qc.cx(j,i)
    qc.barrier()
    for i in range(nqubits):
        qc.ry(theta[count],i)
        count += 1
    '''for i in range(nqubits):
        qc.rx(theta[count],i)
        count += 1'''
    qc.barrier()
    qc.to_instruction()
    return qc



def my_ansatz(qc, theta, layers):

    qr = QuantumRegister(nqubits)
    #cr = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(qr, name="Ansatz")

    count=0
    for layer in range(layers):

        for i in range(nqubits):
        #qc.h(i)
            qc.ry(theta[count], i)
            count +=1
        #   qc.barrier()
        for j in range(0,nqubits-1):
            for i in range(j+1, nqubits):
                qc.cx(j,i)
        qc.barrier()
        for i in range(nqubits):
            qc.rz(theta[count],i)
            count += 1
        qc.barrier()
    qc.to_instruction()
    return qc


def hardware_circ(qc, theta, layers):
    qr = QuantumRegister(nqubits)
    #cr = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(qr, name="Ansatz")

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
    qc.to_instruction()
    return qc



#### MOSTRA IL CIRCUITO !!!!
'''qc = QuantumCircuit(nqubits)
num_inputs=256
features=[]
for i in range(num_inputs):
    features.append(Parameter('x'+str(i)))
for i in range(8):
    features.append(np.pi/2)
num_layers = 3
param_y=[]
#for i in range(2*nqubits):
    #param_y.append(Parameter('θ'+str(i)))
for i in range(2*num_layers*nqubits):
    param_y.append(Parameter('θ'+str(i)))
ansatz = my_ansatz(qc, param_y,3)
ansatz2 = circuit15(qc,param_y,3)
ansatz3 = hardware_circ(qc,param_y,3)

layers_arr = np.arange(1,4,1)
depth_array = np.zeros(3)
size_array = np.zeros(3)
width_array = np.zeros(3)
param_array = np.zeros(3)
for i in range(3):
    depth_array[i] = circuit15(qc, param_y, i).depth()
    size_array[i] = circuit15(qc, param_y, i).size()
    width_array[i] = circuit15(qc,param_y,i).width()
    param_array[i] = len(circuit15(qc,param_y,i).parameters)

plt.style.use('seaborn-whitegrid')
plt.plot(layers_arr,depth_array, label='depth', marker='o', linewidth=2.5, color='chocolate', markersize=10)
plt.plot(layers_arr,size_array, label='size',marker='o', linewidth=2.5, color='darkblue', markersize=10)
plt.plot(layers_arr,width_array, label='width', marker='o', linewidth=2.5, color='gold',markersize=10)
plt.plot(layers_arr,param_array, label='parameters', marker='o', linewidth=2.5, color='mediumaquamarine',markersize=10)
plt.xticks(layers_arr)
plt.xlabel('Num of layers')
plt.ylabel('Value')
plt.legend()
plt.show()

fm_layers=np.arange(1,23,1)

depth_array_fm = np.zeros(22)
size_array_fm = np.zeros(22)
width_array_fm = np.zeros(22)
param_array_fm = np.zeros(22)
for i in range(22):
    depth_array_fm[i] = real_amp_embedding(qc, features, i).depth()
    size_array_fm[i] = real_amp_embedding(qc, features, i).size()
    width_array_fm[i] = real_amp_embedding(qc,features,i).width()
    param_array_fm[i] = len(real_amp_embedding(qc,features,i).parameters)
plt.plot(fm_layers,depth_array_fm, label='depth', marker='s', linewidth=2.5, color='chocolate', markersize=8)
plt.plot(fm_layers,size_array_fm, label='size',marker='s', linewidth=2.5, color='darkblue', markersize=8)
plt.plot(fm_layers,width_array_fm, label='width', marker='s', linewidth=2.5, color='gold',markersize=8)
plt.plot(fm_layers,param_array_fm, label='parameters', marker='s', linewidth=2.5, color='mediumaquamarine',markersize=8)
plt.xticks(fm_layers)
plt.title('Feature map Waterfall')
plt.xlabel('Num of layers')
plt.ylabel('Value')
plt.legend()
#plt.savefig()
plt.show()

#feature_map = real_amp_embedding(qc, features, 22)
#feature_map = encoding(qc, features, 22)
#####
#ansatz = my_ansatz(qc, param_y,3)
#ansatz.measure_all()'''

'''ansatz.draw('mpl')
plt.title('Waterfall ansatz')
plt.savefig("/Users/francescoaldoventurelli/Desktop/workfolder/FIGUREXPRESENTATION/wf_ansatz.jpg", dpi=300)
plt.show()

ansatz2.draw('mpl')
plt.title('Ring like ansatz')
plt.savefig("/Users/francescoaldoventurelli/Desktop/workfolder/FIGUREXPRESENTATION/rl_ansatz.jpg", dpi=300)

plt.show()

ansatz3.draw('mpl')
plt.title('CZ ansatz')
plt.savefig("/Users/francescoaldoventurelli/Desktop/workfolder/FIGUREXPRESENTATION/cz_ansatz.jpg", dpi=300)

#print(len(ansatz.parameters))
#plt.savefig("/Users/francescoaldoventurelli/Desktop/workfolder/FIGUREXPRESENTATION/hardware_ansatz.jpg")
plt.show()

print(len(ansatz.parameters))
print(len(ansatz2.parameters))
print(len(ansatz3.parameters))
#print("Depth:", ansatz.depth())
#print("Width:", ansatz.width())
#print("Size:", ansatz.size())
#print(ansatz.decompose())'''


### EMBEDDING
'''qc = QuantumCircuit(nqubits)
param_x=[]
num_inputs = 256
for i in range(num_inputs):
    param_x.append(Parameter('x'+str(i)))
for i in range(8):
    param_x.append(np.pi/2)
embedding = real_amp_embedding(qc, param_x,22)
#embedding.draw('mpl')
#plt.savefig('/Users/francescoaldoventurelli/Desktop/workfolder/FIGUREXPRESENTATION/fully_embedding.jpg')
#plt.show()
print("Depth:", embedding.depth())
print("Width:", embedding.width())
print("Size:", embedding.size())'''

def circuit11(qc,theta):
    
    qr = QuantumRegister(nqubits)
    qc = QuantumCircuit(qr, name='PQC')
    count = 0
    for i in range(nqubits):
        qc.ry(theta[count],i)
        count +=1
        qc.rz(theta[count], i)
        count +=1
    for j in range(0,nqubits-1,2):
        qc.cnot(j+1,j)
    for i in range(1,3):
        qc.ry(theta[count],i)
        count +=1
        qc.rz(theta[count], i)
        count +=1
    qc.cnot(2,1)
    return qc

### Try to visualize them

#nqubits = 6
'''qr = QuantumRegister(nqubits)  ### Se vuoi misurare questo devi fare!!!!
cr = ClassicalRegister(1)
qc = QuantumCircuit(qr, cr)'''
'''qc = QuantumCircuit(nqubits)

### Parameteres that I give as random initial guess
num_inputs=256
param_x=[]
for i in range(num_inputs):
    param_x.append(Parameter('x'+str(i)))
for i in range(8):
    param_x.append(np.pi/2)

#####
#ansatz = circuit11(qc, param_x)
#ansatz.draw('mpl')
#plt.show()
ansatz = real_amp_PQC(qc, param_x)
ansatz.draw('mpl')
plt.suptitle("Fully entangled paramterized quantum circuit")
plt.savefig("/Users/francescoaldoventurelli/Desktop/workfolder/FIGUREXPRESENTATION/Fully_PQC.jpg")
plt.show()

print("Depth:", ansatz.depth())
print("Width:", ansatz.width())
print("Size:", ansatz.size())'''
#####

'''embedding = encoding(qc, param_x, 12)
embedding.draw('mpl')
plt.show()'''
#embedding = encoding(qc, param_x, 8)
#print("Circuit's depth:", ansatz.depth())
#print("Circuit's size:", ansatz.size())
#print("Circuit's width:", ansatz.width())
#embedding.draw('mpl')
#qc.append(embedding, range(nqubits))
#qc.append(ansatz, range(nqubits))
#qc.draw('mpl')
#plt.suptitle("Fully entangled quantum circuit")
#plt.savefig("/Users/francescoaldoventurelli/Desktop/tHESIS_results_MNIST/Circuits/Fully-likeQC.jpg")
'''print("Circuit's depth:", qc.depth())
print("Circuit's size:", qc.size())
print("Circuit's width:", qc.width())'''
#plt.show()


#### To append to the circuit and show ####
#qc.append(embedding, range(nqubits))
#qc.append(ansatz, range(nqubits))
#qc.measure(qr[0],cr[0])
#qc.draw('mpl')
#plt.show()

def binary(x):
    return ('0'*(4-len('{:b}'.format(x, '#010b') ))+'{:b}'.format(x, '#010b'))
def firsttwo(x):
    return x[:2]
parity = lambda x: firsttwo(binary(x)).count('1') % 2


def qcnn_circuit(qc, theta):
    cr = ClassicalRegister(1)
    qr = QuantumRegister(nqubits)
    qc = QuantumCircuit(qr, cr, name='QCNN')
    
    count = 0

    #### CONV 1
    for i in range(nqubits):
        qc.ry(theta[count], i)
        count +=1
    for i in range(nqubits-1):
        qc.cnot(i+1,i)
    qc.barrier()


    #### POOLING 1
    for i in range(nqubits-1):
        qc.crz(theta[count],i,i+1)
        count +=1
    qc.barrier()

    #### ENTG
    qc.cnot(7,3)
    qc.cnot(5,2)
    qc.cnot(4,1)
    qc.barrier()

    #### CONV2
    for i in range(nqubits//2):
        count+=1
        qc.ry(theta[count], i)
        count +=1
    for i in range((nqubits//2)-1):
        qc.cnot(i+1,i)
    qc.barrier()


    #### POOLING2
    for i in range((nqubits//2)-1):
        qc.crz(theta[count],i,i+1)
        count +=1
    qc.barrier()

    ### ENTG
    qc.cnot(3,1)
    qc.cnot(2,1)
    qc.barrier()

    #### CONV3
    for i in range(nqubits//4):
        count+=1
        qc.ry(theta[count], i)
        count +=1
    for i in range((nqubits//4)-1):
        qc.cnot(i+1,i)
    
    qc.barrier()

    #### POOLING3
    for i in range((nqubits//4)-1):
        qc.crz(theta[count],i,i+1)
        count +=1
    qc.barrier()

    #### MEASURE
    qc.measure(qr[1],cr[0])
    return qc


#nqubits = 8

'''num_inputs=256
param_x=[]
for i in range(num_inputs):
    param_x.append(Parameter('x'+str(i)))
for i in range(8):
    param_x.append(np.pi/2)'''


'''qr = QuantumRegister(nqubits)  ### Se vuoi misurare questo devi fare!!!!
cr = ClassicalRegister(1)
qc = QuantumCircuit(qr, cr)'''

#qc.append(real_amp_embedding(qc, param_x, 12))'''
#qc = QuantumCircuit(nqubits)
#qcnn = qcnn_circuit(qc, param_x)
#qc.append(qcnn, [0,1,2,3,4,5,6,7])
#qc.measure(qr[0],cr[0])


# Count the number of parameters in the circuit
#num_parameters = len(qcnn.parameters)
#print("Total number of parameters:", num_parameters)
#qcnn.draw('mpl')
#plt.show()
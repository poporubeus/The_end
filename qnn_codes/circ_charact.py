from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute
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
from circuits import hardware_circ
#qi = QuantumInstance(Aer.get_backend('statevector_simulator'))
qi = QuantumInstance(Aer.get_backend('aer_simulator'), shots=256) 


nqubits=6
def circuit15(theta, layers):

    #qr = QuantumRegister(nqubits)
    qc = QuantumCircuit(6, name="PQC")
    
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



def encoding(theta,layers):

    #qr = QuantumRegister(nqubits)
    #cr = ClassicalRegister(1)  # Add a classical register
    qc = QuantumCircuit(6, name='Embed')

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
        qc.barrier()
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



def real_amp_embedding(theta,layers):


    #cr = ClassicalRegister(1, 'c')
    qc = QuantumCircuit(nqubits, name="Embedding")
    
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
        qc.barrier()
        for i in range(nqubits):
            qc.rz(theta[count],i)
            count += 1
        qc.barrier()

    #qc.measure(qr[0], cr)
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



def qcnn(nqbits, theta, which_layer):
    count = 0
    qc = QuantumCircuit(nqbits)
    if which_layer == 1:

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
        return qc
    elif which_layer ==2:
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
        return qc
    elif which_layer==3:
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
    else:
        mex = print("Too much layer for this circuit!")
        return mex
    

def qcnn_tot(nqbits, theta):
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



'''print("Circuits parameters and characteristics")
print("")

param_x_qcnn=[] ### features
for i in range(42):
    param_x_qcnn.append(Parameter('x'+str(i)))

qqq = qcnn(8,param_x_qcnn)

print("QCNN cnot:", len(qqq.parameters))


qc = QuantumCircuit(nqubits)
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
#ansatz3 = hardware_circ(qc,param_y,3)

print("WF ansatz 3L cnot:", len(ansatz.parameters))
print("RL ansatz 3L cnot:", len(ansatz2.parameters))
#print(len(ansatz3.parameters))'''



param_x_qcnn=[] ### features
for i in range(24):
    param_x_qcnn.append(Parameter('x'+str(i)))
qqq1 = qcnn(8,param_x_qcnn,1)
'''qqq1.draw('mpl')
plt.show()'''

param_x_qcnn2=[] ### features
for i in range(36):
    param_x_qcnn2.append(Parameter('x'+str(i)))
qqq2 = qcnn(8,param_x_qcnn2,2)
'''qqq2.draw('mpl')
plt.show()'''

param_x_qcnn_tot=[] ### features
for i in range(42):
    param_x_qcnn_tot.append(Parameter('θ'+str(i)))
qcnn_ansatz = qcnn_tot(8,param_x_qcnn_tot)
'''qcnn_ansatz.draw('mpl')
plt.show()'''


qc_dummy = QuantumCircuit(6)

nqubits = 6
num_inputs=256
features=[]
for i in range(num_inputs):
    features.append(Parameter('x'+str(i)))
for i in range(8):
    features.append(np.pi/2)
num_layers = 1
param_y=[]
for i in range(2*nqubits*num_layers):
    param_y.append(Parameter('θ'+str(i)))

ansatz3 = hardware_circ(qc_dummy,param_y,1)
ansatz3.draw('mpl')
#plt.savefig("/Users/francescoaldoventurelli/Desktop/CIRCUITSSS/hardware_cz.jpg", dpi=800)
plt.show()

#qcnn_ansatz.draw('mpl')
#plt.savefig("/Users/francescoaldoventurelli/Desktop/CIRCUITSSS/qcnn_ansatz.jpg", dpi=800)
#plt.show()


'''param_x_qcnn=[] ### features
for i in range(42):
    param_x_qcnn.append(Parameter('x'+str(i)))

qqq = qcnn(8,param_x_qcnn)'''

num_layers = 3
param_y=[]
for i in range(2*nqubits*num_layers):
    param_y.append(Parameter('θ'+str(i)))

size_wf = []
size_rl = []
size_cz = np.zeros(3)
size_cz_list = []
size_qcnn1 = qqq1.size()
size_qcnn2 = qqq2.size()
size_qcnn3 = qcnn_ansatz.size()
size_qcnn_tot = np.array([size_qcnn1,size_qcnn2,size_qcnn3])

'''for j in range(3):
    for i in range(1,num_layers):
        #size_wf[j] = my_ansatz(qc_dummy, param_y,i).size()
        #size_rl[j] = circuit15(param_y,i).size()
        s#ize_cz[j] = hardware_circ(qc_dummy, param_y,i).size()'''

for i in range(1,4):
        size_cz_list.append(hardware_circ(qc_dummy, param_y,i).size())
        size_rl.append(circuit15(param_y,i).size())
        size_wf.append(my_ansatz(qc_dummy, param_y,i).size())
print(my_ansatz(qc_dummy, param_y,1).size())




layers_array = np.arange(1,4,1)
#plt.style.use('seaborn-whitegrid')
plt.plot(layers_array,size_wf, marker='o', markersize=15, label='Waterfall', color='royalblue', fillstyle='none', linewidth=2, linestyle='none')
plt.plot(layers_array,size_rl, marker='o', markersize=15,  label='Ring-like', color='tab:orange',fillstyle='none',linewidth=2,  linestyle='none')
plt.plot(layers_array,size_cz_list, marker='o', markersize=15, label='CZ',color='magenta',fillstyle='none',linewidth=2,  linestyle='none')
plt.plot(layers_array,size_qcnn_tot, marker='o', markersize=15, label='Qcnn', color='gold',fillstyle='none',linewidth=2,  linestyle='none')
#plt.plot(layers_array, size_cz_list, '--')
plt.legend(loc='best', prop={'size':16})
plt.xlabel('N° layers', fontsize=12, fontweight='semibold')
plt.ylabel('Size', fontsize=12, fontweight='semibold')
plt.title('Size of each ansatz', fontsize=12, fontweight='semibold')
plt.xticks(layers_array)
plt.tick_params('x')
#plt.grid(True)
plt.ylim(0,max(size_wf)+10)
plt.savefig('/Users/francescoaldoventurelli/Desktop/CIRCUITSSS/size_vs_layers.jpg', dpi=800)
plt.show()




qc__ = qcnn_ansatz(8,param_x_qcnn_tot)

quantum_circ = QuantumCircuit(8,8)
quantum_circ.append(qc__, range(8))
quantum_circ.measure(0, 0)
quantum_circ.measure(1, 1)
quantum_circ.measure(2, 2)
quantum_circ.measure(3, 3)
quantum_circ.measure(4, 4)
quantum_circ.measure(5, 5)
quantum_circ.measure(6, 6)
quantum_circ.measure(7, 7)
job = execute(quantum_circ, backend = Aer.get_backend('aer_simulator'), shots=256)
result = job.result()
count = result.get_counts()
print(len(count))
plot_histogram(count)
plt.show()
import qiskit
import tensorflow as tf
from tensorflow import keras
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, execute, transpile
from qiskit import BasicAer, Aer, execute
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
from circuits import encoding, real_amp_embedding, circuit15, my_ansatz, hardware_circ





class QNN:
    def __init__(self, shots, features, learning_weights, model_nam, nqubits, nlayers_featuremap,
                 nlayers_ansatz) -> None:
        self.shots = shots
        self.model_nam = model_nam
        self.features = features
        self.learning_weights = learning_weights
        self.nqubits = nqubits
        self.nlayers_featuremap = nlayers_featuremap
        self.nlayers_ansatz = nlayers_ansatz
    
    def make_qc(self):
        qc = QuantumCircuit(self.nqubits)
        return qc 

    def feature_map(self):
        """
        Model name must be of 4 words
        """
        qc = self.make_qc()
        if self.model_nam[:2] == "wf":
            fm = real_amp_embedding(qc=qc, theta=self.features, layers=self.nlayers_featuremap)
        elif self.model_nam[:2] == "rl":
            fm = encoding(qc=qc, theta=self.features, layers=self.nlayers_featuremap)
        return fm

    def ansatz(self):
        qc = self.make_qc()
        if self.model_nam[2:4] == "wf":
            ans = my_ansatz(qc=qc, theta=self.learning_weights, layers=self.nlayers_ansatz)
        elif self.model_nam[2:4] == "rl":
            ans = circuit15(qc=qc, theta=self.learning_weights, layers=self.nlayers_ansatz)
        elif self.model_nam[2:4] == "cz":
            ans = hardware_circ(qc=qc, theta=self.learning_weights, layers=self.nlayers_ansatz)
        else:
            raise ValueError("You should insertn the correct sigla, this is a simple model!!!")
        return ans
    
    def model(self):
        qc = self.make_qc()
        feature_map = self.feature_map()
        ansatz = self.ansatz()
        qc.append(feature_map, range(self.nqubits))
        qc.append(ansatz, range(self.nqubits))
        mymodel = qc
        return mymodel
    
    def draw(self, which):
        """
        Type feature_map if you want to display the chosen feature map;
        Type ansatz if you want to display the ansatz.
        """
        #qc = self.make_qc()
        if which == "feature_map":
            fm = self.feature_map()
            drawn_fm = fm.draw("mpl")
            return drawn_fm
        elif which == "ansatz":
            ans = self.ansatz()
            drawn_ans = ans.draw("mpl")
            return drawn_ans
    

'''class QCNN:
    def __init__(self, shots, features, learning_weights, model_nam, nqubits, nlayers_featuremap,
                 nlayers_ansatz ) -> None:
        self.shots = shots
        self.features = features
        self.learning_weights = learning_weights
        self.model_nam = model_nam
        self.nqubits = nqubits
        self.nlayers_featuremap = nlayers_featuremap
        self.nlayers_ansatz = nlayers_ansatz

    def make_circuit(self):
        """
        The number of qubits must be """
        qc = QuantumCircuit(self.nqubits)
        return qc'''
    

        





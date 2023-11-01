# QUANTUM NEURAL NETWORKS FOR DATA-EFFICIENT IMAGE CLASSIFICATION
This is the repository containing my Master thesis's project on QML.

I've started from an existing project where two kinds of quantum circuits (a feature map and the corresponding ansatz) were tested on a Supervised Learning (SL) problem with a customized galaxy dataset used to perform a binary image classification. I've added multiple quantum circuits taken out from different papers, combined together to build several Quantum Neural Networks and I tested their abilities on the binary Supervised Learning (SL) problem, but introducing the MNIST (0-1 digits) in addition to the galaxy dataset.
Afterwards, I tried to build a QCNN ansatz inspired from Cong's [Quantum Convolutional Neural Networks - Cong](https://arxiv.org/pdf/1810.03787.pdf) article and I examinated it on the same problem. Eventually, I made a simple and fair classical Convolutional Neural Network (CNN) tested on the same datasets, used to make a comparison between quantum devices and classical counterparts.

The main idea of this project is based on a paper released in 2022 [Generalization in QML from few training data - Caro,Cincio,Cerezo](https://www.nature.com/articles/s41467-022-32550-3); in the article authors asked themselves about the generalization performances of a QML model and proved that such algorithms can learn faster and with less data compared to classical models. This unespected results can significantly have an impact on today's ML models where plenty of data are required for training to reach high levels of accuracy.

### What is QML?
QML stands for Quantum Machine Learning: a discipline that uses quantum models to develops Machine Learning algorithms specific for an assigned task (Supervised, Unsupervised and Reinforcement Learning) and runs each process on a quantum computer, or a quantum simulator. 
### Why QML?
People intensively use QML since it offers a possibility to represent many more functions than a classical computer thanks to the larger computational space (i.e. Hilber space) where do computations ($dim(H) = 2^{n}$, where $n$ is the number of qubits). QML exploits superposition and entanglement, two purely quantum effects responsible for representing 2 possible states with only 1 qubit (that substitutes the classical bit in classical computers) and for creating associations between parameters that are passed through qubits.
Ultimately, with the increasing amount of data that are spred out every day in the current 21th century, we need for computers able to handle plenty of data and information. Quantum computers seem to be the candidates for this task based on what we have explained few lines above.
### Sad story...disadvantages
I don't want to go deep into this argument seems I'm strongly convinced of the super-powers of QCs over the CCs, so I need to present my point in the best way possible (lol) without touching too much the disadvantages bounded to it...Anyway, to be serious, the main disadvantages are:
1) Complexity and Error Correction;
2) Scalability and qubit count;
3) Limited Applicability.
For those of you really interested in this topic I strongly recommend to have a look at [Quantum Computing Disadvantages Explained](https://originstamp.com/blog/quantum-computing-disadvantages-explained/).

## GO BACK TO MACHINE LEARNING
Before move on the analysis, a brief intro to ML.
### Supervised Learning (SL)
Think about image classification. I have the images, I know what they are (I have the labels) and I train an algorithm to match as many images to the corresponding labels as possible. In this way, once the train has finished, I move to the test phase and I ask the model to predict/generalize images it hasn't seen (obviosuly these images must be of the same class and kind of the training ones, otherwise the model will fail);
### Unsupervised Learning (UL)
Think I have thousands of data mixed together such as images of apples, peers, pumplinks, watermelons and so on...and I want to group them basing on the characteristics they have in common (for example if I have images of green, red, yellow apples, they are still apples so I want to group them under the same class of objects, in this case apple). By using a ML algorithm I can find the logic that those data have in common and make my dataset more clear.
### Reinforcement Learning (RL)
Think I want to train a ML engine to play Chess. Whenever it makes a mistake I give a "bad" feedback such as a negative weights, on the contrary, when it makes a good move I'll give it a present, for example a positive weights until it reaches a desiderable solution.
### Deep Learning (DL)
DL is not a way of learning as I have mentioned before. It's simply a new class of ML models much better and powerful of the pre-existing ones...####why? They are able to introduce the non-linearity within the models. A Neural Network (NN) is an architecture that emulates the way our brain learns from the external world. Entities called neurons incorporate the data (i.e. a vector element $\overrightarrow{x}$), then they pass each element through links assinging statistical weights until a new neuron has reached. In this new node, a function proportional to a sum of the probabilites coming from each link is applied to the data and the process repeats until a new kind of layer called "dense" or "linear" take all the remained data to make the classification. The function contained in each neuron are activiated according to non-linear functions such as ReLU, Sigmoid and other kinds of functions.
The great success of DL in applications is based on the clever idea of constructing sufficiently large nonlinear function spaces throughout the composition of layers of linear and simple nonlinear functions (in the name of activation function). The nonlinearity of a deep network results from the composition of the layers of nonlinear functions. Therefore, the impact of different activation functions on the final performance is likely linked to the model architecture (width and the depth). Apart from this quick introduction to NNs, to understand better what happens inside a NN I suggest [Quantum Neural Netoworks - K.Beer](https://arxiv.org/abs/2205.08154) where the classical part, as the quantum part is really well-explained than I did. I recommend also [Effects of the Nonlinearity in Activation Functions on the Performance of Deep Learning Models](https://arxiv.org/pdf/2010.07359.pdf) to know better the effects of the activation functions.
_Here an illustration of the Neural Network architecture_
/The_end/Images/NN.png

# Back to QUANTUM
## QNN models
QNN models (a.k.a. Quantum Neural Networks) are particular ML models that use the elements of Quantum Computing: qubits and quantum gates.
The general expression of the fundamental unit of a Quantum Computer, i.e. the qubit, is
$|\Psi \rangle = \alpha |0\rangle + \beta|1\rangle$
and as can be seen, it represents two quantum states at the same time. In this sense, the Schrodinger's cat has a probability ($\alpha$) to be in the state $|0\rangle$ - alive and a probability ($\beta$) to be dead (in the sate ($|1\rangle$).
A quantum gate is a particular operator ($\hat{U}$), i.e. a matrix, that is applied to a state and change it. Have a look at [List of quantum logic gates](https://en.wikipedia.org/wiki/List_of_quantum_logic_gates) to see all the gates we can represent in a Quantum device. These gates are unitary operators $\rightarrow U^{\dagger}U = I$. A quantum circuits is a set of unitary operations applied to qubits to solve a specific problem.
The last very important concept we need to introduce before starting is the concept of # measurement.
Suppose our friend formulates the Schrodinger's cat problem, and he asks this specific question "Where are you sure about the status of the cat?". Our simple correct answer shoul be "In the moment I measure its state!". In other words, for the Schrodinger problem we need to open the box and in order to know if the cat is really dead or alive: this operation we perform on the system (here another important concept i.e. the entanglement between the system and the measurement apparatus, will pop up, but it's a bit behind the simple illustration I want to explain here, so I won't touch it) is the measurement.
Generally a QNN is composed by two quantum circuits.

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
The great success of DL in applications is based on the clever idea of constructing sufficiently large nonlinear function spaces throughout the composition of layers of linear and simple nonlinear functions (in the name of activation function). The nonlinearity of a deep network results from the composition of the layers of nonlinear functions. Therefore, the impact of different activation functions on the final performance is likely linked to the model architecture (width and the depth). Apart from this quick introduction to NNs, to understand better what happens inside a NN I suggest [Quantum Neural Netoworks - K.Beer](https://arxiv.org/abs/2205.08154) where the classical part, as the quantum part is really well-explained than I did. I recommend also [Effects of the Nonlinearity in Activation Functions on the Performance of Deep Learning Models](https://arxiv.org/pdf/2010.07359.pdf) to know better the effects of the activation functions.\
\
#Here an illustration of the Neural Network architecture#\
\
<img src="https://github.com/poporubeus/The_end/blob/main/Images/nn_scheme.png" width="700" height="400" />\
\
# Back to QUANTUM
## QNN models
QNN models (a.k.a. Quantum Neural Networks) are particular ML models that use the elements of Quantum Computing: qubits and quantum gates.
The general expression of the fundamental unit of a Quantum Computer, i.e. the qubit, is \
$|\Psi \rangle = \alpha |0\rangle + \beta|1\rangle$ \
and as can be seen, it represents two quantum states at the same time. In this sense, the Schrodinger's cat has a probability ($\alpha$) to be in the state $|0\rangle$ - alive and a probability ($\beta$) to be dead (in the sate ($|1\rangle$).
A quantum gate is a particular operator ($\hat{U}$), i.e. a matrix, that is applied to a state and change it. Have a look at [List of quantum logic gates](https://en.wikipedia.org/wiki/List_of_quantum_logic_gates) to see all the gates we can represent in a Quantum device. These gates are unitary operators $\rightarrow U^{\dagger}U = I$. A quantum circuits is a set of unitary operations applied to qubits to solve a specific problem.
The last very important concept we need to introduce before starting is the concept of # measurement.
Suppose our friend formulates the Schrodinger's cat problem, and he asks this specific question "Where are you sure about the status of the cat?". Our simple correct answer shoul be "In the moment I measure its state!". In other words, for the Schrodinger problem we need to open the box to know if the cat is really dead or alive: this operation we perform on the system (here another important concept i.e. the entanglement between the system and the measurement apparatus, will pop up, but it's a bit behind the simple illustration I want to explain here, so I won't touch it) is the measurement. \
Measuring a qubit translates into the following expression: $\lambda = \langle \psi | \hat{O} | \psi \rangle$ \
$\lambda$ is the outcome of the measurement process, i.e. the eigenvalue of the operator $\hat{O}$ we have measured. It's obtained by starting from the usual Schrodinger equation, here in a simplified formulation, for the operator we measure: $\hat{O}|\psi\rangle = \lambda|\psi\rangle$ from which we can come back to the previous expression: known also as the expectation value of such operator $\hat{O}$. This measurement outcome is obtained with a certain probability. In fact, coming back to the Schrodinger's problem, if we have the system in the state $|\psi\rangle = \alpha|0\rangle + \beta|1\rangle$, and we measure the Pauli $\hat{Z}$ operator with respect to the orthonormal basis ${|0\rangle,|1\rangle}$ we obtain two possible outcomes $\lambda = \pm 1$. Why? We start from the definition of the Pauli $\hat{Z}$ operator that is of this expression $|0\rangle \langle 0| - |1\rangle \langle 1|$, so, by following the Schrodinger's equation I presented above we can conclude that \
\
$\hat{Z}|\psi\rangle = (|0\rangle \langle 0| - |1\rangle \langle 1|)|\psi\rangle = \lambda |\psi\rangle$ \
\
$\lambda = \dfrac{\langle\psi|\hat{Z}|\psi\rangle}{\langle\psi||\psi\rangle}$ \
\
We know that all the states are orthonormal, so terms such as $\langle 0||1\rangle$ and viceversa give 0, rather terms like \langle 0||0\rangle give 1.\
\
If we now set $|\psi\rangle = |0\rangle$ we get $\lambda_{0} = \dfrac{\langle 0|(|0\rangle \langle 0| - |1\rangle \langle 1|)|0\rangle}{\langle\psi||\psi\rangle}$\
we get $\dfrac{\langle 0||0\rangle \langle 0||0\rangle - \langle 0||1\rangle \langle 1||0\rangle}{\langle 0 ||0\rangle}$ that's $\lambda_{0} = 1$. The same is valid when the state is $|1\rangle$ and we obtain $\lambda_{1} = -1$.\
\
Arrived to this point we can ask which is the probaility of obtaining $\lambda_{0}$ for example. By introducing the measurement operators defined as follows \
\
$\sum_{m}M_{m}^{\dagger}M_{m} = I$\
\
we can write $p(\lambda_{0}) = \langle \psi|M_{0}^{\dagger}M_{0}|\psi\rangle$ and the state after the measurement will be collapsed into $|\psi'\rangle = \dfrac{M_{0}|\psi\rangle}{\sqrt{\langle \psi|M_{0}^{\dagger}M_{0}|\psi\rangle}}$. The probability of getting the state $|0\rangle$ correspondent to $\lambda_{0}$ is $|\alpha|^{2} = \langle \psi|M_{0}^{\dagger}M_{0}|\psi\rangle$.\
\
After have annoying with the complex theoretical aspectes of Quantum Mechanics that can be recovered in [Quantum Information and Quantum Computation - Niels and Chuang](https://profmcruz.files.wordpress.com/2017/08/quantum-computation-and-quantum-information-nielsen-chuang.pdf), why is the measurement important for QML? Simply because by measuring the qubit and storing the value on a classical but we have access to the variational parameters (angles for classical Q computers) that are used to minimize the loss function. But what is the loss function?\

### Again Machine Learning
We need to clarify other two aspects before diving into ML.
#### Accuracy: number of correct predictions over the total number of predictions;
#### Loss function - Cost: function that is minimized to catch the best variational parameters to make the correct predictions.
To summarize, Neural Newtorks and Quantum Neural Networks work in the "same" way: during training they learn the characteristic of data such as images and perform predictions over the train set in order to learn the best parameters that minimize the loss function and, consequently raise up the accuracy. Once the train has been finished, they are ready to generalize new unseen data. Most of the ML algorithms like CNNs (we are going to discuss later) and QNNs use gradient descent based method to minimize the loss. Roughly speaking they compute the gradient of the loss landscape and look for valleys. 

## QNN
Generally a QNN model is composed by three main parts:
1) The feature map;
2) The ansatz;
3) The classical optimization.\
### Feature map
The feature map is the circuit responsible for the representation of the classical data into quantum data, i.e. quantum states.
### Ansatz
The ansatz is the circuit that learn the parameters that undergo to successive optimization.
### Classical optimization
The classical optimization consists into a classical optimizer that compute the cost function and minimize it based on the current learned parameters.\
### Brief illustration 
\
<img src="https://github.com/poporubeus/The_end/blob/main/Images/Screenshot%202023-09-20%20alle%2017.19.30.png" width="800" height="300" />

### Circuits
Here I display the circuits I used as feature map and ansatz. The feature map is composed by 6 qubits to which multiple unitary gates are applied to induce rotations on them. The feature map is composed by 22 layers, so it's repeated that number of time to encode all the 256 features of Mnist and Galaxy images (the images I have used are 16x16 grayscale images), while the ansatz is composed by only 3 layers since is the ciruit that would be simulated and for this reason it cannot be too depth otherwise it generates errors during the process. I decided yo show only the ansatz circuits since are the shallower ones so it's easy to visualize.

### Ring-like circuit
\
<img src="https://github.com/poporubeus/The_end/blob/main/Circuits_imgs/ringlike_feature_map.jpg" width="600" height="250" />

### Waterfall circuit
\
<img src="https://github.com/poporubeus/The_end/blob/main/Circuits_imgs/waterfall_feature_map.jpg" width="600" height="250" />

### CZ circuit
\
<img src="https://github.com/poporubeus/The_end/blob/main/Circuits_imgs/hardware_cz.jpg" width="600" height="250" />

### QCNN circuit
\
<img src="https://github.com/poporubeus/The_end/blob/main/Circuits_imgs/qcnn_ansatz.jpg" width="600" height="250" />

The ring-like and the waterfall circuits are used either as feature map, all the others as ansatz. I composed many QNN architectures by combining all the circuits together, for example I used the ringlike and the waterfall combined with the CZ circuit and so on...

### Technical details and Frameworks
I've used Qiskit as the quantum interface combined with PyTorch and Tensorflow for downloading the dataset and making the classification.
As I said, Mnist dataset with {0,1} digits and a custom galaxy dataset reduced to 256 features are used to test the different models. I used the LBFGS optimizer furnished by PyTorch with only 20 epoches since it requires some time to end the process. Subsequently I made a simple and fair CNN constituted by only 36 optimizing parameters, 1 convolutional layer and 1 pooling to make a direct comparison between the quantum and classical part. 

### Look at the dataset

<img src="https://github.com/poporubeus/The_end/blob/main/Images/galaxy_binary_NON_bianrized.jpg" width="400" height="300" /> <img src="https://github.com/poporubeus/The_end/blob/main/Images/MNIST16X16.jpg" width="400" height="300" />

### Experiments
1) From less data to accuracy: The first experiment consists in training each QNN model with less images to see the generalization performance's trend when the train set is considerably enlarged. I have challenged the models by considering very few images like 6, 10, 20, 30, 40 and 50. I set 5 different seeds to make 5 different runs for each train set with always different and non-repeated images in each run, to have an average behaviour of each process. Unfortunaly, as I was expecting, the CNN reached the best performances on the test set rather than each QNN model which met several difficulties during training.

<img src="https://github.com/poporubeus/The_end/blob/main/Images/RL_train_test.jpg" width="400" height="300" />

2) From more layers to accuracy: The second simple experiment wants to show what happens to the test accuracy when we add more layers to the ansatz (from 1 to 3). As we exected, the accuracy on the test set increases. This particular simple task is not present in the project, but it's mentioned within the thesis and the tables of results are appropriately shown in it.
3) From less features to accuracy: Last experiment where I asked myself if the meaning of "less images" present in the article I cited could have been applied to features: can a QNN model reach high-levels of accuracy in the test set when the images are reduced in number of features? I downsampled each Mnist and Galaxy image from 16 to 12, 8 and 6 and I basically applied the Exp1 to see the results in generalization performance and surprisingly I've got better results also compared to the CNN!!!!
   
<img src="https://github.com/poporubeus/The_end/blob/main/Images/Test_vs_number_of_features_RLRL.jpg" width="400" height="300" /> <img src="https://github.com/poporubeus/The_end/blob/main/Images/test_acc_CNN_different_features_GALAXY.jpg" width="400" height="300" />

These are some of the plots I got. The others are collected in the folder plot.
Thanks for watching :) 

Cheers!

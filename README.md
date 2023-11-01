### The_end
This is the repository containing my Master thesis's project on QML.

I've started from an existing project where two kinds of quantum circuits (a feature map and the corresponding ansatz) were tested on a Supervised Learning (SL) problem with a customized galaxy dataset used to perform a binary image classification. I've added multiple quantum circuits taken out from different papers, combined together to build several Quantum Neural Networks and I tested their abilities on the binary Supervised Learning (SL) problem, but introducing the MNIST (0-1 digits) in addition to the galaxy dataset.
Afterwards, I tried to build a QCNN ansatz inspired from Cong's [Quantum Convolutional Neural Networks - Cong](https://arxiv.org/pdf/1810.03787.pdf) article and I examinated it on the same problem. Eventually, I made a simple and fair classical Convolutional Neural Network (CNN) tested on the same datasets, used to make a comparison between quantum devices and classical counterparts.

The main idea of this project is based on a paper released in 2022 [Generalization in QML from few training data - Caro,Cincio,Cerezo](https://www.nature.com/articles/s41467-022-32550-3); in the article authors asked themselves about the generalization performances of a QML model and proved that such algorithms can learn faster and with less data compared to classical models. This unespected results can significantly have an impact on today's ML models where plenty of data are required for training to reach high levels of accuracy.

# What is QML?
QML stands for Quantum Machine Learning: a discipline that uses quantum models to develops Machine Learning algorithms specific for an assigned task (Supervised, Unsupervised and Reinforcement Learning) and runs each process on a quantum computer, or a quantum simulator. 
# Why QML?
People intensively use QML since it offers a possibility to represent many more functions than a classical computer thanks to the larger computational space (i.e. Hilber space) where do computations ($dim(H) = 2^{n}$, where $n$ is the number of qubits). QML exploits superposition and entanglement, two purely quantum effects responsible for representing 2 possible states with only 1 qubit (that substitutes the classical bit in classical computers) and for creating associations between parameters that are passed through qubits.
Ultimately, with the increasing amount of data that are spred out every day in the current 21th century, we need for computers able to handle plenty of data and information. Quantum computers seem to be the candidates for this task based on what we have explained few lines above.
# Sad story...disadvantages
I don't want to go deep into this argument seems I'm strongly convinced of the super-powers of QCs over the CCs, so I need to present my point in the best way possible (lol) without touching too much the disadvantages bounded to it...Anyway, to be serious, the main disadvantages are:
\item Complexity and Error Correction;

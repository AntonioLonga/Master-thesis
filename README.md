
# Master-thesis Antonio Longa

Master-thesis  
_Isomorphism  
___Trees  
_____Rootedtree.py  
_Documents  
___graphKernel  
_____AaltoLecture-LearningOnGraphs  
_____A short tour of Kernel method for Graphs (Thomas Garther (Germany) - Quoc V.Le, Alex J Smola (Austria))  
_____Graph Kernel 2014 (Vishwanathan (Pordue) - Schraudolph (Australia) - Kondor (Pasadena) - Borgwardt (Germany))  
_____NSPDK (Fabrizio Costa)  
___garphNeuralNetwork  
_____How powerful are GNN - (Jure Leskovec)  
_____Graph Neural Networks with Convolutional ARMA Filters  
___Autoencoder and network(Aalto)  
___Statistical Comparisons of Classifiers(Janez Demasar)  
__TestDataset  
___AIDS (dataset folder)  
___DD (dataset folder)  
___KKI (dataset folder)  
___MIO (dataset folder)  
___MUTAG (dataset folder)  
___PROTEINS (dataset folder)  
___Import_data.ipynb (Notebook)  
___K-mean_Evaluation.ipynb (Notebook)  
___KKN_Evaluation.ipynb (Notebook)  
___KKN_UMAP_Evaluation.ipynb (Notebook)
  
  
## Rootedtree.py  
Is a python method that convert a Tree in a binary string, the idea comes from the book "Application of Graph Theory" of  Robin J.Wilson and Lowell W.Beineke.  
The algorithm generate the code starting from the root of the tree, if the tree has no root, the center is used as root.


## Import_data.ipynb (Notebook)
It contains a method to import data. The format data comes from [Dourmund Universitat](https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets).  
The method produce a Networkx OBJ


## K-mean_Evaluation.ipynb (Notebook)  
It import the data, vectorize it using NSPDK, successively it use a SVD dimensionality reduction and finally it uses **K-mean** as classifier


## KNN_Evaluation.ipynb (Notebook)  
It import the data, vectorize it using NSPDK, successively it use a SVD dimensionality reduction and finally it uses **K-nearest neighbor** as classifier


## KNN_UMPA_Evaluation.ipynb (Notebook)  
It import the data, vectorize it using NSPDK, successively it use a UMAP dimensionality reduction and finally it uses **K-nearest neighbor** as classifier


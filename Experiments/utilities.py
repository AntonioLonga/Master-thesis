from spektral.utils import conversion
from keras.utils import to_categorical
import networkx as nx
import numpy as np
from visualizator import Visualizator
from evaluator import Evaluator


from tabulate import tabulate

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from sklearn.utils import shuffle
from eden.graph import vertex_vectorize

from tabulate import tabulate
import matplotlib.pyplot as plt
import matplotlib




'''


def repeat_n_times(graphs, labels, emb, dim, times):
    n_classifiers = len(emb)
    models_names = [embedder.name for embedder in emb]
    vis = Visualizator(dim, n_classifiers = n_classifiers, models_names=models_names)


    for t in range(0,times):
        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(graphs, labels, test_size=0.3)

        print("\t iteration n:",t+1)
        for d in range(0,len(dim)):
            print("\t \t dim: ",int(dim[d]))
            dimension_embedding = int(dim[d])

            position = 0
            for embedder in emb:


                embedder.estimators[-1].n_components = dimension_embedding
                X_res = embedder.fit(X_train,y_train).transform(X_test)


                evaluator = Evaluator(KNeighborsClassifier(n_neighbors = 1))
                acc, pre, rec, f = evaluator.performance_with_kfold(X_res,y_test)


                vis.add_metrics(acc,pre,rec,f,position,d)
                position = position + 1

    vis.y_test = y_test
    return (vis)
     '''
def embed(emb,d,vis,dimension_embedding,X_train, X_test,y_train, y_test):
    
    position = 0
    for embedder in emb:

        print("\t \t \t ",embedder.name)
        embedder.estimators[-1].estimator.n_components = dimension_embedding

        #### if the modeel is a GNN reset the weights
        ##if (str(type(embedder.estimators[0].estimator)) == "<class 'embedder.Transformer_GNN'>"):
            ##embedder.estimators[0].estimator.reset_state()
        
        embedder.reset_state()

        X_res = embedder.fit(X_train,y_train).transform(X_test)
        
        

        evaluator = Evaluator(KNeighborsClassifier(n_neighbors = 1))
        acc, pre, rec, f = evaluator.performance_with_kfold(X_res,y_test)


        vis.add_metrics(acc,pre,rec,f,position,d)
        position = position + 1

def try_dimensions(dim,emb,vis,X_train,X_test,y_train,y_test):
    
    for d in range(0,len(dim)):
        print("\t \t dim: ",int(dim[d]))
        embed(emb,d,vis,int(dim[d]),X_train, X_test,y_train, y_test)
        
def repeat_n_times(graphs, labels, emb, dim, times,seed=-1,test_size=0.3):
    
    n_classifiers = len(emb)
    models_names = [embedder.name for embedder in emb]
    
    vis = Visualizator(dim, n_classifiers = n_classifiers, models_names=models_names)

    for t in range(0,times):
        if seed == -1:
            X_train, X_test, y_train, y_test = train_test_split(graphs, labels, test_size=test_size)
        else:

            X_train, X_test, y_train, y_test = train_test_split(graphs, labels, test_size=test_size,random_state=seed)
        print("\t iteration n:",t+1)
        try_dimensions(dim,emb,vis,X_train,X_test,y_train,y_test)
        
    vis.y_test = y_test
    return (vis)
    


def from_nx_to_adj(graphs,nf_keys=['vec'],ef_keys=None):
    np_adj =conversion.nx_to_numpy(graphs,nf_keys=nf_keys, ef_keys=ef_keys)
    adjs = np_adj[0]
    nodes_f = np_adj[1]
    edges_f = np_adj[2]

    return (adjs,nodes_f,edges_f)

def from_np_to_one_hot(labels):
    labels = to_categorical(labels)
    return(labels)

def from_one_hot_to_np(labels):
    res = []
    for label in labels:
        res.append(np.argmax(label, axis=None, out=None))
        
    return(res)

def vec_vertex(graph):
    X = vertex_vectorize([graph], complexity=2, nbits=5)
    x = X[0]
    x = x.A
    values = []
    count = 0
    for node in graph.nodes():
        val = x[count]
        count = count + 1
        values.append(val)

    return([values])

def degree(graph):
    values = []
    for node in graph.nodes():
        values.append(graph.degree(node))
    
    return ([values])

def clust_coefficient(graph):
    values = []
    for node in graph.nodes():
        values.append(nx.clustering(graph,node))
        
    return ([values])

def local_degree_profile(graph):
    '''Return max,min,mean and std of the degrees among all neighbours of a node.
    '''
    max_degree = []
    min_degree = []
    mean_degree = []
    std_degree = []
    for node in graph:
        neigbours = list(graph.neighbors(node))
        neigbours_degree = []
        for neighbor in neigbours:
            neigbours_degree.append(graph.degree(neighbor))
        max_degree.append(np.max(neigbours_degree))
        min_degree.append(np.min(neigbours_degree))
        mean_degree.append(np.mean(neigbours_degree))
        std_degree.append(np.std(neigbours_degree))
    return([max_degree,min_degree,mean_degree,std_degree])
    

def add_info_to_nodes(graphs,functions):
    '''applica function a ogni nodo di ogni grafo, e aggiunge il risultato come attributo del nodo.
    ''' 
    if (not type(functions) == list):
        functions = [functions]
        
    for function in functions:
        for g in graphs:
            values = function(g)
            for value in values:
                counter = 0
                for node in g.nodes():
                    vec = g.nodes[node]['vec'] 
                    g.nodes[node]['vec'] = vec + [value[counter]]
                    counter = counter + 1

    return(graphs)

def global_wlt(visualizators,metric="accuracy"):

    vis = list(visualizators.values())

    n_models = len(vis[0].models_names)
    global_wlt = np.zeros((n_models,n_models))

    for v in vis:
        tmp = wlt(v,return_matrix=True,metric=metric)
        global_wlt = global_wlt + tmp

        
    res = []
    for i in range(0,len(global_wlt)):
        tmp = [vis[0].models_names[i]]
        summation = sum(global_wlt[i])
        for j in list(global_wlt[i]):
            tmp.append(j)
        tmp.append(summation)

        res.append(tmp)



    print(tabulate(res,headers=["sum"]))

def wlt(visualizator, return_matrix=False,metric="accuracy"):

    summary_matrix = visualizator.summary(std=False,return_matrix=True, metric=metric)

    dim = visualizator.dim.copy()
    tmp = []
    for i in summary_matrix:
        tmp.append(i[1:-1])

    tmp = np.matrix(tmp)

    tmp = tmp.getA()

    size = len(visualizator.accuracy)
    wlt = np.zeros((size,size))

    for i in range(0,size):
        for j in range(0,size):
            if i != j:
                count = 0
                for k in range(0,len(dim)):
                    if(tmp[i][k] > tmp[j][k]):
                        count = count +1
                wlt[i][j] = count

    if(return_matrix==True):
        return wlt

    res = []
    for i in range(0,len(wlt)):
        tmp = [visualizator.models_names[i]]
        summation = sum(wlt[i])
        for j in list(wlt[i]):
            tmp.append(j)
        tmp.append(summation)
        
        res.append(tmp)



    print(tabulate(res,headers=["sum"]))

def global_rank(visualizators,metric="accuracy"):

    visualizators = list(visualizators.values())
    rank_matricies = []
    dim = visualizators[0].dim
    for vis in visualizators:
        rank_matrix = vis.rank(return_matrix=True,metric=metric)
        tmp = []
        for i in rank_matrix:
            tmp.append(i[1:-1])
            
        tmp = np.matrix(tmp)
        rank_matricies.append(tmp)
        

    result = np.zeros(rank_matricies[0].shape)
    for i in rank_matricies:
        result = result + i

    models_names = visualizators[0].models_names

    result = result.getA()
    ranks = []
    for i in range(0,len(result)):
        mean = np.mean(result[i])
        tmp = []
        tmp.append(models_names[i])
        for j in result[i]:
            tmp.append(j)
        tmp.append(mean)
        ranks.append(tmp)
    dims = dim.copy()
    dims.append("mean")
    print (tabulate(ranks, headers=dims))


def find_shapes(graphs):
    max_number_of_nodes = 0
    for graph in graphs:
        number_of_nodes = len(graph.nodes())
        
        if(max_number_of_nodes < number_of_nodes):
            max_number_of_nodes = number_of_nodes

    ### length feature vector
    index = list(graphs[0].node())[0]
    
    return(max_number_of_nodes, len(graphs[0].node()[index]['vec']))



def sub_sampling(graphs,labels,samples_for_class):
    
    n_classes = len(np.unique(labels))

    if (isinstance(samples_for_class, int)):
        samples_for_class = np.full(n_classes,samples_for_class)


    groups = [ [] for i in range(0,n_classes)]

    # group the classes
    for i in range(0,len(graphs)):
        groups[labels[i]].append(graphs[i])


    res_graphs = []
    res_labels = []
    

    returned_labels = np.zeros(n_classes)

    # take elements fro each group
    for i in range(0,n_classes):
        group = groups[i]
        np.random.shuffle(group)
        for item in group[0:samples_for_class[i]]:
            res_graphs.append(item)
            res_labels.append(i)
            returned_labels[i] = returned_labels[i] + 1 


    #shuffle it
    res_graphs, res_labels = shuffle(res_graphs, res_labels)
   

    
    #print the balancing of the output
    print(np.around(returned_labels/np.sum(returned_labels), decimals=2))

    return(res_graphs,res_labels)

    


def evaluate_emb_train_test(emb_test,y_test,emb_train,y_train,return_value=False):
    dim = len(emb_test[0])
    eva = Evaluator(KNeighborsClassifier(n_neighbors = 1))
    acc_test, _,_,_ = eva.performance_with_kfold(emb_test,y_test)
    acc_test = "%.3f" % np.mean(acc_test)
    
    eva = Evaluator(KNeighborsClassifier(n_neighbors = 1))
    acc_train, _,_,_ = eva.performance_with_kfold(emb_train,y_train)
    acc_train = "%.3f" % np.mean(acc_train)
    
    if (return_value == True):
        return(acc_test,acc_train)
    else:
        print (tabulate([['K.N.N.  accuracy', acc_test, acc_train]], headers=["DIM: "+str(dim), 'TEST','TRAIN']))
    

def plot_embedding_2d(embed,graphs,labels,test_size,seed):
    X_train, X_test, y_train, y_test = train_test_split(graphs, labels, test_size=0.3,random_state=11)
    res_test = embed.transform(X_test)
    res_train = embed.transform(X_train)
    acc_test,acc_train = evaluate_emb_train_test(res_test,y_test,res_train,y_train,return_value=True)
    
    
    colors = ['red','blue']
    plt.figure(figsize=(10,5))
    plt.subplot(121)
    x = res_test[:,0]
    y = res_test[:,1]
    plt.title("TEST \nacc: "+str(acc_test))
    plt.scatter(x,y,s=8,c=y_test,cmap=matplotlib.colors.ListedColormap(colors))


    plt.subplot(122)
    x = res_train[:,0]
    y = res_train[:,1]
    plt.title("TRAIN \nacc: "+str(acc_train))
    plt.scatter(x,y,s=8,c=y_train,cmap=matplotlib.colors.ListedColormap(colors))
    
    plt.show()
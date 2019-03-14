
import numpy as np
import networkx as nx
import os


"""convert a file into a list of pairs. each pairs is (Graph,label) """
def load_data(folder_name):
	
    
    A_path =folder_name+"\\"+folder_name+"_A.txt"
    GraphIndicator_path =folder_name+"\\"+folder_name+"_graph_indicator.txt"
    
    files = explore_folder(folder_name)
    #extract informations from files
    att_node = files[0]
    label_node = files[1]
    att_edge = files[2]
    label_edge = files[3]
    
    label_graph = files[4]


    ### Create a Huge graph of disconnected components
    ## each connected component is a graph
    G = nx.Graph()
    ### add edges, labels of the edges and attributes of the nodes
    file = open(A_path, "r") 
    c=0
    for t in file:
        tmp=t.split(',')
        tmp = [float(x.strip()) for x in tmp]
        G.add_node(tmp[0],label = label_node[tmp[0]], vec = att_node[tmp[0]])
        G.add_edge(tmp[0],tmp[1], label=label_edge[c], vec = att_edge[c])
        c = c + 1
    file.close()

    
    # load graph_indicator file and compute it
    file = open(GraphIndicator_path, "r") 
    # import the file in an array
    graph_indicator = []
    for t in file:
        graph_indicator.append(int(t))       
    file.close()

    ### np.unique works because the graph_indicator are sequentially
    u, indices = np.unique(graph_indicator, return_index=True)

    indices = np.append(indices +1 ,[len(graph_indicator)+1])
    indices = np.cumsum(indices[1:]-indices[0:-1])+1
    
    # split the graph in subgraphs according to graph_indicator file and store it in graphs
    graphs = []
    initial = 1
    for i in range(0,len(indices)):
        nodes = np.arange(initial,indices[i])
        initial = indices[i]
        graphs.append([G.subgraph(nodes),label_graph[i]])
        
        
    return (graphs)




### INPUT: path node attributes:
### RETURN: a dict of attributes (id : [att1, att2 ..,]) .
###### the element on position i-th is the vector of the attributes of the node with id = i
def import_node_attributes(path,n):
    
    mapping_node = {}
    if(os.path.exists(path)):
        file = open(path, "r") 
        mapping = {}
        c=1
        for t in file:
            tmp=t.split(',')
            tmp = [float(x.strip()) for x in tmp]
            mapping_node[c]= tmp
            c=c+1
        file.close()
    else:
        for i in range(1,n+1):
            mapping_node[i]=""
            
    return (mapping_node)
   


### INPUT: path node attributes:
### RETURN: a list of attributes.
###### the element on position i-th is the vector of the attributes of the node with id = i
def import_edge_attributes(path,m):
    edges_attribute = []
    if (os.path.exists(path)):
        file = open(path, "r") 
        for t in file:
            tmp=t.split(',')
            tmp = [float(x.strip()) for x in tmp]
            edges_attribute.append(tmp)
        file.close()
    else:
        edges_attribute = np.zeros(m)
    
    return (edges_attribute)






### INPUT: path node LABEL:
### RETURN: a dictionary of {id :label}.
###### the element on position i-th is a string that corrisbonds to the label of the node with id = i
def import_node_label(path,n):
    mapping_node = {}
    if(os.path.exists(path)):
        file = open(path, "r") 
        mapping = {}
        c=1
        for t in file:
            mapping_node[c]=t.rstrip()
            c=c+1
        file.close()
    else:
        for i in range(1,n+1):
            mapping_node[i]=""
        
    
    return (mapping_node)


### INPUT: path edge LABEL:
### RETURN: a list of edges labels.
###### the element on position i-th is a string that corrisbonds to the label of the edge with id = i
def import_edge_label(path,m):
    edges_label = []
    if (os.path.exists(path)):
        file = open(path, "r") 
        for t in file:
            tmp = int(t)
            edges_label.append(tmp)
        file.close()
    else:
        edges_label = np.zeros(m)
        
    return (edges_label)
	
	
	
	
#### explore folder:


def explore_folder(folder_name):
    
    A_path =folder_name+"\\"+folder_name+"_A.txt"
    GraphIndicator_path =folder_name+"\\"+folder_name+"_graph_indicator.txt"
    Graph_labels =folder_name+"\\"+folder_name+"_graph_labels.txt"

    nodeAtt_path = folder_name+"\\"+folder_name+"_node_attributes.txt"
    nodeLabel_path = folder_name+"\\"+folder_name+"_node_labels.txt"
    edgeAtt_path = folder_name+"\\"+folder_name+"_edge_attributes.txt"
    edgeLabel_path = folder_name+"\\"+folder_name+"_edge_labels.txt"

    info = [folder_name+"\\"+folder_name+"_node_attributes.txt", folder_name+"\\"+folder_name+"_node_labels.txt",
           folder_name+"\\"+folder_name+"_edge_attributes.txt", folder_name+"\\"+folder_name+"_edge_labels.txt"]
    # m = number of edges
    m = 0
    for line in open(A_path).readlines(  ): m += 1
    # n = number of nodes
    n = 0
    for line in open(GraphIndicator_path).readlines(  ): n += 1
    # import the label of the graphs
    graph_lab = []
    file = open(Graph_labels, "r") 
    for t in file:
        t = int(t)
        if (t == -1):
            t = 0
        graph_lab.append(t)
    file.close()
	

    ### import file, if they does not exist return n/m elements of ""
    node_att = import_node_attributes(nodeAtt_path,n)
    node_lab = import_node_label(nodeLabel_path,n)

    edge_att = import_edge_attributes(edgeAtt_path,m)
    edge_lab = import_edge_label(edgeLabel_path,m)

    return([node_att,node_lab,edge_att,edge_lab,graph_lab])



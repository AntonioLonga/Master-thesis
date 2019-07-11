import numpy as np
import networkx as nx
from spektral.utils import conversion
from keras.utils import to_categorical
import os
import utilities as ut


def load_data_pubchem(path):
    
    graphs = nx.read_gpickle(path+'/'+'graphs.gpickle')    
    in_labels = list(np.load(path+'/'+'labels.npy'))
    dic = ut.create_dict_labels(graphs)
    # add int(label) in vec
    for g in graphs:
        for n in g.nodes():
            if 'vec' in g.nodes()[n].keys():
                
                lab = g.nodes()[n]['label']
                vec = g.nodes()[n]['vec']
                new_vec = vec
                new_vec.append(dic[lab])
                g.nodes()[n]['vec'] = new_vec
            else:
                lab = g.nodes()[n]['label']
                g.nodes()[n]["vec"] = [dic[lab]]
    
    labels = []
    for i in in_labels:
        labels.append(int(i))
    
    labels = np.array(labels)
    return (graphs,labels)


def load_data(folder_name):
    '''Import a set of graphs.
    
    The data comes from https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets.
    Each graph has a node and edge labels and node and edge attributes.
    If such information are missing, then an empty string is added.
    
    Args:
        folder_name (String): the name of the folder
        
    Returns:
        (graphs,label): a list of graphs, and a list of labels
    
    '''
    

    path_files = os.path.join(os.getcwd(), folder_name)
    A_path = os.path.join(path_files,(folder_name + '_A.txt'))
    graphIndicator_path = os.path.join(path_files,(folder_name + "_graph_indicator.txt"))

    
    
    
    att_node, label_node, att_edge, label_edge, label_graph = explore_folder(folder_name)


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
    file = open(graphIndicator_path, "r") 
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
    labels = []
    initial = 1
    for i in range(0,len(indices)):
        nodes = np.arange(initial,indices[i])
        initial = indices[i]
        graphs.append(G.subgraph(nodes))
        labels.append(label_graph[i])
       
    graphs = add_info_to_nodes(graphs)
    
    
    return (graphs, np.array(labels))

   
    
def add_info_to_nodes(graphs):
    '''add label, degree and clustering coefficient to nodes attributes
    
  
    Args:
        graphs ([nx.graph]): an array of graphs
        
    Returns:
        [nx.graph]: a list of graphs
    
    '''   
    for g in graphs:
        for i in g.nodes():
            label = int(g.nodes[i]['label'])
            #label = g.nodes[i]['label']
            vec = g.nodes[i]['vec'] 
            if (vec == ''):
                g.nodes[i]['vec'] = [label]
            else:
                g.nodes[i]['vec'] = vec + [label]
   
    return(graphs)




### INPUT: path node attributes:
### RETURN: a dict of attributes (id : [att1, att2 ..,]) .
###### the element on position i-th is the vector of the attributes of the node with id = i
def import_node_attributes(path,n):
    '''Import attributes of the nodes.
    
    Args:
        path (os.path): the path of the file
        n : number of nodes        
        
    Returns:
        (mapping_node): a dict {node_id : "attributes", node_id : "attributes", ... }
    
    '''
    
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
   

def import_edge_attributes(path,m):
    '''Import attributes of the edges.
    
    Args:
        path (os.path): the path of the file
        m : number of edges        
        
    Returns:
        (edges_attribute): a list of attributes, where the element on position i-th 
                            is the vector of the attributes of the edge with id = i
    
    '''
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
    '''Import label of the nodes.
    
    Args:
        path (os.path): the path of the file
        n : number of nodes        
        
    Returns:
        (mapping_node): a dict {node_id : "label", node_id : "label", ... }
    
    '''
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
    '''Import label of the edges.
    
    Args:
        path (os.path): the path of the file
        m : number of edges        
        
    Returns:
        (edges_attribute): a list of labels, the element in position i-th 
                            is a string that corrisbonds to the label of the edge with id = i
    
    '''
    edges_label = []
    if (os.path.exists(path)):
        file = open(path, "r") 
        for t in file:
            tmp = int(t)
            edges_label.append(str(tmp))
        file.close()
    else:
        edges_label = np.zeros(m)
        
    return (edges_label)

#### explore folder:


def explore_folder(folder_name):
    '''Take all the files from a folder.
    
    Args:
        folder_name (os.path): the path of the folder
        
    Returns:
        (node_att,node_lab,edge_att,edge_lab,graph_lab): dict,dict,list,list,list
    
    '''

    
    path_files = os.path.join(os.getcwd(), folder_name)

    A_path = os.path.join(path_files,(folder_name + '_A.txt'))

    graphIndicator_path = os.path.join(path_files,(folder_name + "_graph_indicator.txt"))
    graph_labels = os.path.join(path_files,(folder_name + "_graph_labels.txt"))

    nodeAtt_path = os.path.join(path_files,(folder_name + "_node_attributes.txt"))
    nodeLabel_path = os.path.join(path_files,(folder_name + "_node_labels.txt"))
    edgeAtt_path = os.path.join(path_files,(folder_name + "_edge_attributes.txt"))
    edgeLabel_path = os.path.join(path_files,(folder_name + "_edge_labels.txt"))



    # m = number of edges
    m = 0
    for line in open(A_path).readlines(  ): m += 1

    # n = number of nodes
    n = 0
    for line in open(graphIndicator_path).readlines(  ): n += 1

    # import the label of the graphs
    graph_lab = []
    file = open(graph_labels, "r") 
    for t in file:
        t = int(t)
        if (t == -1):
            t = 0
        graph_lab.append(t)
    file.close()

    #### start from 0 in labeling
    min_label = np.min(graph_lab)
    if (min_label > 0):
        graph_lab = graph_lab - min_label

    ### import file, if they does not exist return n/m elements of ""
    node_att = import_node_attributes(nodeAtt_path,n)
    node_lab = import_node_label(nodeLabel_path,n)

    edge_att = import_edge_attributes(edgeAtt_path,m)
    edge_lab = import_edge_label(edgeLabel_path,m)

    return(node_att,node_lab,edge_att,edge_lab,graph_lab)

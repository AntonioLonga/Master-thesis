import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
class Graph_generator:
    
    
    def gen_set(n_graphs,n_nodes,prob_edge=0.5, n_node_lab=None,n_edge_lab=None, seed=None):
        
        g1 = Graph_generator.gen_graph(n_nodes,prob_edge=0.5, n_node_lab=None,n_edge_lab=None, seed=None)
        g2 = Graph_generator.gen_graph(n_nodes,prob_edge=0.5, n_node_lab=None,n_edge_lab=None, seed=None)
        
        if (n_node_lab == None):
            n_node_lab = n_nodes
        if (n_edge_lab == None):
            n_edge_lab = n_nodes
        
        graphs = []
        labels = []
        
        for i in range(0,int(n_graphs/2)):
            g = Graph_generator.flip_edges(g1)
            g = Graph_generator.gen_change_edge_label(g,n_edge_lab)
            g = Graph_generator.gen_change_node_label(g,n_node_lab)
            
            graphs.append(g)
            labels.append(0)
        
        for i in range(0,int(n_graphs/2)):
            g = Graph_generator.flip_edges(g2)
            g = Graph_generator.gen_change_edge_label(g,n_edge_lab)
            g = Graph_generator.gen_change_node_label(g,n_node_lab)
            
            graphs.append(g)
            labels.append(1)

        return(graphs,labels)

    def gen_change_edge_label(graph,n_label):

        new_graph = graph.copy()
        for n in graph.edges(data=True):
            lab = np.random.randint(n_label)
            new_graph.add_edge(n[0],n[1],label=lab)
        return (new_graph)

    '''
    From a given list of grphs, change randomly the label of the node.
    at each etaration only ONLY one label is changed
    '''
    def gen_change_node_label(graph,n_label):

        new_graph = graph.copy()
        for n in graph.nodes(data=True):
            lab = np.random.randint(n_label)
            new_graph.add_node(n[0],label=str(lab))
        return (new_graph)


    def myplot(G,n_label='label'):
        pos = nx.spring_layout(G,seed =4)

        nx.draw(G, pos)
        node_labels = nx.get_node_attributes(G,n_label)
        nx.draw_networkx_labels(G, pos, labels = node_labels)
        edge_labels = nx.get_edge_attributes(G,'')
        nx.draw_networkx_edge_labels(G, pos, labels = edge_labels)

        plt.show()
        
    '''
    Generate a CONNECTED graph with random node and edge labels
    return a nx graph
    '''
    def gen_graph(n_nodes,prob_edge=0.5, n_node_lab=None,n_edge_lab=None, seed=None):
        g = nx.random_geometric_graph(n_nodes,prob_edge,seed=seed)
        
        if(n_node_lab == None):
            n_node_lab = n_nodes
            
        if (n_edge_lab == None):
            n_edge_lab = len(g.edges())
        
        for n in g.nodes():
            lab = np.random.randint(n_node_lab)
            g.add_node(n,label=lab)
            del g.node[n]['pos']
        for e in g.edges():
            att1 = np.random.randint(n_edge_lab)
            g.add_edge(e[0],e[1],label=att1)
                      
        if (nx.is_connected(g)):
            return(g) # is connected, return it
        else:
            if (seed == None):  # if seed is not specified, try a random one
                seed=np.random.randint(100)
            else:
                seed = seed + 1 # increment seed and try again
            return(Graph_generator.gen_graph(n_nodes,prob_edge,n_node_lab,n_edge_lab, seed))
    
    def flip_edges(g):
        edges =list(g.edges())
        flag = True
        c = 0
        while (flag):
            g_new = g.copy()
            e1 = edges[np.random.randint(len(edges))]
            e2 = edges[np.random.randint(len(edges))]
            if (e1 != e2):
                if ((not g_new.has_edge(e1[0],e2[1])) and( not g_new.has_edge(e2[0],e1[1]))):
                    e1_lab = g[e1[0]][e1[1]]
                    e2_lab = g[e2[0]][e2[1]]
                    g_new.remove_edge(e1[0],e1[1])
                    g_new.add_edge(e1[0],e2[1],label=e1_lab['label'])
                    g_new.remove_edge(e2[0],e2[1])
                    g_new.add_edge(e2[0],e1[1],label=e2_lab['label'])
                    c = c+1

                    if (c == len(g.edges())):
                        g_new = g.copy()
                        flag = False
                    if (nx.is_connected(g_new)):
                        flag = False
                        
        return(g_new)

    def flip_edge_n_times(g,times):
        g_new = g.copy()
        for i in range(0,times):
            g_new = Graph_generator.flip_edges(g_new)
            
        return(g_new)
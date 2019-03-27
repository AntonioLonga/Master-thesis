import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from eden import display
import warnings

class Graph_Generator:
    n_graphs = None
    degree = None
    seed = None
    
    def __init__(self,n_graphs,degree=4,seed=None):        
        self.n_graphs = n_graphs
        self.degree = degree
        
        if (seed == None):
            np.random.seed(None)
        else:
            np.random.seed(seed)
            self.seed = seed
            
        
    def generate_set(self,g1,g2,
                     node_alph_g1, edge_alph_g1,
                     node_alph_g2, edge_alph_g2,
                     dept_g1 = 1,dept_g2 = 1,
                     plot = True):
        
        class_size = int(self.n_graphs/2)
        
        graphs=[]
        labels=[]
        for i in range(0,class_size):
            graphs.append(self.perturb(g1,node_alph_g1,edge_alph_g1,dept_g1))
            labels.append(0)
        
        for i in range(0,class_size):
            graphs.append(self.perturb(g2,node_alph_g2,edge_alph_g2,dept_g2))
            labels.append(1)
            
        if (plot):
            self.plot_alphabets(node_alph_g1,edge_alph_g1,node_alph_g2,edge_alph_g2)
            
        return(graphs, labels)
    
    # ****
    def perturb(self, graph,node_alph, edge_alph,n_exchange=1):
        graph = self.change_edge_label(graph,edge_alph)
        graph = self.change_node_label(graph,node_alph)
        graph = self.edge_exchange_n_times(graph, n_exchange)
        
        return(graph)
        
    # ****  
    def generate(self,n_nodes,node_alph_start,edge_alph_start,node_alph_end=None,edge_alph_end=None,seed=None):
        if (seed == None):
            seed = np.random.randint(100)
        g = nx.random_regular_graph(self.degree,n_nodes,seed)
        
        if (node_alph_end == None):
            node_alph = np.arange(node_alph_start,node_alph_start + n_nodes,1)
        else:
            node_alph = np.arange(node_alph_start,node_alph_end,1)

        if (edge_alph_end == None):
            edge_alph = np.arange(edge_alph_start,edge_alph_start+len(g.edges()),1)
        else:
            edge_alph = np.arange(edge_alph_start,edge_alph_end)
                
        if (nx.is_connected(g)):
            g = self.add_node_labels(g, node_alph)
            g = self.add_edge_labels(g, edge_alph)
            return(g,node_alph,edge_alph)
        else:
            return(self.generate())
        
    # ****
    def add_node_labels(self,g,node_alph):
        np.random.shuffle(node_alph)
        for n in g.nodes():
            c = np.random.randint(len(node_alph))
            g.node[n]['label']=node_alph[c]
        return(g)
    
    # ****    
    def add_edge_labels(self,g,edge_alph):
        np.random.shuffle(edge_alph)
        for e in g.edges():
            c = np.random.randint(len(edge_alph))
            g.add_edge(e[0],e[1],label=edge_alph[c])

        return(g)
    
    
    # ****
    def change_edge_label(self,graph,edge_alph):
        g = graph.copy()
        np.random.shuffle(edge_alph)
        for n in g.edges(data=True):
            lab = edge_alph[np.random.randint(len(edge_alph))]
            g[n[0]][n[1]]['label'] = lab
        return (g)

    # ****
    def change_node_label(self,graph, node_alph):
        g = graph.copy()
        np.random.shuffle(node_alph)
        for n in g.nodes():
            c = np.random.randint(len(node_alph))
            lab = node_alph[c]
            g.node[n]['label'] = str(lab)
            
        return (g)
    
    
    def edge_exchange_n_times(self,g,n_times):
        g_new = g.copy()
        for i in range(0,n_times):
            g_new = self.edge_exchange(g_new)
            
        return(g_new)

     
    def edge_exchange(self,g):
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
    
    def draw_graph_set(self,graphs, n, n_graphs_per_line=5):
        
        class_size = int(n/2)
        b=graphs[0:class_size]
        c=graphs[-class_size:]
        g = b + c
        warnings.filterwarnings('ignore')
        display.draw_graph_set(g,n_graphs_per_line=n_graphs_per_line,edge_label='label')
    
    
    
  

    def plot_alphabets(self,node_alph_1,edge_alph_1,node_alph_2,edge_alph_2):

        plt.figure(figsize=(8, 1))

        aa = np.zeros(len(node_alph_1))
        bb = np.zeros(len(node_alph_2))+0.1

        plt.subplot(121)
        plt.plot(node_alph_1,aa,linewidth=10, label="graph 1")
        plt.plot(node_alph_2,bb,linewidth=10, label="graph 2")
        plt.ylim(-0.1,0.3)
        plt.title("Node alphabet")


        aa = np.zeros(len(edge_alph_1))
        bb = np.zeros(len(edge_alph_2))+0.1

        plt.subplot(122)
        plt.plot(edge_alph_1,aa,linewidth=10, label="graph 1")
        plt.plot(edge_alph_2,bb,linewidth=10, label="graph 2")
        plt.ylim(-0.1,0.3)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.title("Edge alphabet")
        plt.show()

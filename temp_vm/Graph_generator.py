import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from eden import display
import warnings

class Graph_Generator:
    '''Generate set of similar graphs.

    Attributes:
        n_graphs (int): Number of graphs to generate
        degree (int)[4]: Degree of the generated graphs
    '''

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
                     pert_times_g1 = 1,pert_times_g2 = 1,
                     plot = True):
        '''Given a graph, generate a set of similar graphs.
        
        The generated graphs has the same alphabet of the originator.
              
        Args:
            g1 (nx.graph) : a graph
            g2 (nx.graph) : a graph
            node_alph_g1 ([int]) : node alphabet of graph g1
            edge_alph_g1 ([int]) : edge alphabet of graph g1
            node_alph_g2 ([int]) : node alphabet of graph g2
            edge_alph_g2 ([int]) : edge alphabet of graph g2
            pert_times_g1 (int)[1] : how many perturbation on g1 
            pert_times_g1 (int)[1] : how many perturbation on g2
            
        Returns:
            (graphs, labels)
        
        '''
        
        class_size = int(self.n_graphs/2)
        
        graphs=[]
        labels=[]
        for i in range(0,class_size):
            graphs.append(self.perturb(g1,node_alph_g1,edge_alph_g1,pert_times_g1))
            labels.append(0)
        
        for i in range(0,class_size):
            graphs.append(self.perturb(g2,node_alph_g2,edge_alph_g2,pert_times_g2))
            labels.append(1)
            
        if (plot):
            self.plot_alphabets(node_alph_g1,edge_alph_g1,node_alph_g2,edge_alph_g2)
            
        return(graphs, labels)
    
    
    def perturb(self, graph,node_alph, edge_alph,times):
        '''Given a graph, change edge labels, change node labels and exchange an edge.
              
        Args:
            graph (nx.graph) : a graph
            node_alph ([int]) : node alphabet
            edge_alph ([int]) : edge alphabet
            n_exchange (int) : how many edges are exchanged
            
        Returns:
            (graph)
        
        '''
        g = graph
        for i in range(0,times):
            g = self.change_edge_label(g,edge_alph)
            g = self.change_node_label(g,node_alph)
            g = self.edge_exchange(g)
        
        return(g)
        
        
    def generate(self,n_nodes,node_alph_end,edge_alph_end,seed=None):
        '''Generate a graph with node and edge labels
        
        If node/edge_alph_end is None the is used the number of nodes and the number of edges
                
        Args:
            n_nodes (int): number of nodes of the graph
            node_alph_end (int = None): an integer that represents the end of the node alphabet
            edge_alph_end (int = None): an integer that represents the end of the edge alphabet
            seed (int = None): seed for np.random.
            
        Returns:
            (graph, node_alphabed ([int]), edge_alphaber ([int]))
        '''
        if (seed == None):
            seed = np.random.randint(100)
        g = nx.random_regular_graph(self.degree,n_nodes,seed)
        

        node_alph = np.arange(0,node_alph_end,1)
        edge_alph = np.arange(0,edge_alph_end,1)
        
        np.random.shuffle(node_alph)
        np.random.shuffle(edge_alph)
                
        if (nx.is_connected(g)):
            g = self.add_node_labels(g, node_alph)
            g = self.add_edge_labels(g, edge_alph)
            return(g,node_alph,edge_alph)
        else:
            return(self.generate())
        
        
    def add_node_labels(self,g,node_alph):
        '''Add labels to nodes of a graph
        
        Args:
            graph (nx.graph): a graph
            node_alph ([int]): node alphabet
            
        Returns:
            a graph
        
        '''
        np.random.shuffle(node_alph)
        for n in g.nodes():
            c = np.random.randint(len(node_alph))
            g.node[n]['vec']=[node_alph[c]]
        return(g)
    
    def add_edge_labels(self,g,edge_alph):
        '''Add labels to edges of a graph
        
        Args:
            graph (nx.graph): a graph
            edge_alph ([int]): edge alphabet
        
        Returns:
            a graph
        '''
        np.random.shuffle(edge_alph)
        for e in g.edges():
            c = np.random.randint(len(edge_alph))
            g.add_edge(e[0],e[1],vec=[edge_alph[c]])

        return(g)
    

    def change_edge_label(self,graph,edge_alph):
        '''Change one random label of one edge 
        
        Args:
            graph (nx.graph): a graph
            edge_alph ([int]): edge alphabet
        
        Returns:
            a graph
        '''
        g = graph.copy()
        edges = list(graph.edges())
        
        pos_e = np.random.randint(len(edges))
        e = edges[pos_e]
        c = np.random.randint(len(edge_alph))
        lab = edge_alph[c]
        g[e[0]][e[1]]['vec'] = [lab]
        
        return (g)

    
    def change_node_label(self,graph, node_alph):
        '''Change one random label of a node 
        
        Args:
            graph (nx.graph): a graph
            node_alph ([int]): node alphabet
        
        Returns:
            a graph
        '''
        
        g = graph.copy()
        n = np.random.randint(len(graph.nodes()))
        c = np.random.randint(len(node_alph))
        lab = node_alph[c]
        g.node[n]['vec'] = [lab]
        
        return (g)

     
    def edge_exchange(self,g):
        '''Exchange two edges.
        
        Take two random edges and exchange the endpoint.
        Example: 
            random edges: e1 = (v1,u1,lab1) and e2 = (v2,u2,lab2).
            remove e1 and e2 from the graph
            add to the graph e1* = (v1,u2,lab1) and e2* = (v2,u1,lab2) 
        
        Args:
            g (nx.graph): a graph     
        
        Returns:
            a graph
        '''
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
                    g_new.add_edge(e1[0],e2[1],vec=e1_lab['vec'])
                    g_new.remove_edge(e2[0],e2[1])
                    g_new.add_edge(e2[0],e1[1],vec=e2_lab['vec'])
                    c = c+1

                    if (c == len(g.edges())):
                        g_new = g.copy()
                        flag = False
                    if (nx.is_connected(g_new)):
                        flag = False
                        
        return(g_new)
    
    def draw_graph_set(self,graphs, n, n_graphs_per_line=5):
        '''Plot n/2 graphs with target 1 and n/2 graphs with target 0.
        
        Args:
            graphs ([nx.Graph]): array of graphs
            n (int): number of graphs to plot
            n_graphs_per_line (int): number of graphs plotted in each line
        '''
        
        class_size = int(n/2)
        b=graphs[0:class_size]
        c=graphs[-class_size:]
        g = b + c
        warnings.filterwarnings('ignore')
        display.draw_graph_set(g,n_graphs_per_line=n_graphs_per_line,edge_label='vec')
    
    
    
  

    def plot_alphabets(self,node_alph_1,edge_alph_1,node_alph_2,edge_alph_2):
        ''' Show the intersection between alphabets of graphs.
        
        Args: 
            node_alph_1 ([int]): node alphabet used for graph 1
            edge_alph_1 ([int]): edge alphabet used for graph 1
            node_alph_2 ([int]): node alphabet used for graph 2
            edge_alph_2 ([int]): node alphabet used for graph 2
        
        '''

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

